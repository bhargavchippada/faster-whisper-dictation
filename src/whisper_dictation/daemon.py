"""Dictation daemon — ties together audio capture, VAD, transcription, and typing."""

from __future__ import annotations

import logging
import threading

import numpy as np

from .audio import AudioStream
from .config import Config
from .engine import TranscriptionEngine, create_engine
from .hotkey import HotkeyListener
from .notifier import notify
from .typer import type_text
from .vad import SpeechDetector

log = logging.getLogger(__name__)


class DictationDaemon:
    """Main dictation daemon.

    Listens for hotkey, captures audio, detects speech via VAD,
    transcribes via engine, and types into the focused application.
    """

    def __init__(self, config: Config):
        self.config = config
        self._engine: TranscriptionEngine = create_engine(config)
        self._vad = SpeechDetector(
            sample_rate=config.audio.sample_rate,
            threshold=config.vad.threshold,
            silence_ms=config.vad.silence_ms,
            min_speech_ms=config.vad.min_speech_ms,
            max_speech_s=config.vad.max_speech_s,
        )
        self._audio: AudioStream | None = None
        self._hotkey: HotkeyListener | None = None
        self._running = threading.Event()
        self._recording = False
        self._lock = threading.Lock()

    def _on_audio_chunk(self, audio: np.ndarray) -> None:
        """Called for each audio chunk from the microphone."""
        if not self._recording:
            return

        complete, utterance = self._vad.process_chunk(audio)
        if complete and utterance is not None:
            # Transcribe in background thread to not block audio capture
            threading.Thread(
                target=self._transcribe_and_type,
                args=(utterance,),
                daemon=True,
            ).start()

    def _transcribe_and_type(self, audio: np.ndarray) -> None:
        """Transcribe audio and type the result."""
        text = self._engine.transcribe(audio, self.config.audio.sample_rate)
        if text:
            type_text(text + " ")
            log.debug("Typed: %d chars", len(text))

    def _on_activate(self) -> None:
        """Hotkey pressed — start recording."""
        with self._lock:
            if self._recording:
                return
            self._recording = True

        log.info("Recording started")
        notify("Recording...", "Speak now")
        self._vad.reset()

        self._audio = AudioStream(self.config.audio, self._on_audio_chunk)
        self._audio.start()

    def _on_deactivate(self) -> None:
        """Hotkey released/toggled — stop recording."""
        with self._lock:
            if not self._recording:
                return
            self._recording = False

        log.info("Recording stopped")

        if self._audio is not None:
            self._audio.stop()
            self._audio = None

        # Process any remaining buffered speech
        remaining = self._vad.flush()
        if remaining is not None:
            threading.Thread(
                target=self._transcribe_and_type,
                args=(remaining,),
                daemon=True,
            ).start()

        notify("Stopped", "Dictation paused")

    def start(self) -> None:
        """Start the dictation daemon."""
        # Verify engine is available
        if not self._engine.is_available():
            engine_type = self.config.engine.type
            if engine_type == "server":
                log.error(
                    "Server not reachable at %s. Start with: docker compose up -d",
                    self.config.server.url,
                )
            else:
                log.error("Local engine not available.")
            notify("Error", "Transcription engine not available")
            raise RuntimeError("Transcription engine not available")

        # Start hotkey listener
        self._hotkey = HotkeyListener(
            binding=self.config.hotkey.binding,
            mode=self.config.hotkey.mode,
            on_activate=self._on_activate,
            on_deactivate=self._on_deactivate,
        )
        self._hotkey.start()

        self._running.set()
        mode = self.config.hotkey.mode
        binding = self.config.hotkey.binding
        engine = self.config.engine.type
        log.info(
            "Dictation daemon started: hotkey=%s, mode=%s, engine=%s",
            binding,
            mode,
            engine,
        )
        notify(
            "Dictation Ready",
            f"Press {binding} to {'start/stop' if mode == 'toggle' else 'hold and speak'}",
        )

    def stop(self) -> None:
        """Stop the dictation daemon."""
        self._running.clear()

        with self._lock:
            should_deactivate = self._recording
        if should_deactivate:
            self._on_deactivate()

        if self._hotkey is not None:
            self._hotkey.stop()
            self._hotkey = None

        self._engine.close()
        log.info("Dictation daemon stopped")
        notify("Dictation Stopped", "")

    def wait(self) -> None:
        """Block until the daemon is stopped."""
        try:
            self._running.wait()
            # Keep main thread alive while running
            while self._running.is_set():
                self._running.wait(timeout=1.0)
        except KeyboardInterrupt:
            log.info("Interrupted")
            self.stop()

    @property
    def is_running(self) -> bool:
        return self._running.is_set()
