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

    def __init__(self, config: Config, streaming: bool = False):
        self.config = config
        self.streaming = streaming
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
        self._recorded_chunks: list[np.ndarray] = []
        self._lock = threading.Lock()

    def _on_audio_chunk(self, audio: np.ndarray) -> None:
        """Called for each audio chunk from the microphone.

        Note: reads _recording without the lock for performance in the
        audio callback hot path. Safe on CPython due to the GIL.
        """
        if not self._recording:
            return

        if self.streaming:
            self._on_audio_chunk_streaming(audio)
        else:
            with self._lock:
                self._recorded_chunks.append(audio.copy())

    def _on_audio_chunk_streaming(self, audio: np.ndarray) -> None:
        """Streaming mode: VAD splits speech, each utterance transcribed immediately."""
        try:
            complete, utterance = self._vad.process_chunk(audio)
        except Exception:
            log.error("VAD processing failed", exc_info=True)
            return

        if complete and utterance is not None:
            threading.Thread(
                target=self._transcribe_and_type,
                args=(utterance,),
                daemon=True,
            ).start()

    def _transcribe_and_type(self, audio: np.ndarray) -> None:
        """Transcribe audio and type the result."""
        try:
            text = self._engine.transcribe(audio, self.config.audio.sample_rate)
            if text:
                type_text(text + " ")
                log.debug("Typed: %d chars", len(text))
            elif not self.streaming:
                log.info("No speech detected")
                notify("No speech", "Nothing was transcribed")
        except Exception:
            log.error("Transcription or typing failed", exc_info=True)

    def _on_activate(self) -> None:
        """Hotkey pressed — start recording."""
        with self._lock:
            if self._recording:
                return
            self._recording = True
            self._recorded_chunks.clear()

        log.info("Recording started")
        notify("Recording", "Speak now")
        if self.streaming:
            self._vad.reset()

        self._audio = AudioStream(self.config.audio, self._on_audio_chunk)
        try:
            self._audio.start()
        except Exception:
            log.error("Failed to start audio capture", exc_info=True)
            self._audio.stop()
            self._audio = None
            with self._lock:
                self._recording = False
            notify("Error", "Could not access microphone")

    def _on_deactivate(self) -> None:
        """Hotkey released/toggled — stop recording and transcribe."""
        with self._lock:
            if not self._recording:
                return
            self._recording = False
            chunks = list(self._recorded_chunks)
            self._recorded_chunks.clear()

        log.info("Recording stopped")

        if self._audio is not None:
            self._audio.stop()
            self._audio = None

        if self.streaming:
            # Flush any remaining VAD-buffered speech
            remaining = self._vad.flush()
            if remaining is not None:
                threading.Thread(
                    target=self._transcribe_and_type,
                    args=(remaining,),
                    daemon=True,
                ).start()
        elif chunks:
            # Batch mode: transcribe the full recording at once
            full_audio = np.concatenate(chunks)
            duration = len(full_audio) / self.config.audio.sample_rate
            log.info("Transcribing %.1fs of audio...", duration)
            notify("Transcribing", f"{duration:.0f}s of audio...")
            threading.Thread(
                target=self._transcribe_and_type,
                args=(full_audio,),
                daemon=True,
            ).start()
        else:
            notify("Stopped", "No audio recorded")

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
        notify("Dictation Stopped", "Daemon exited")

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
