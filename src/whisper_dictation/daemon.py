"""Dictation daemon — ties together audio capture, VAD, transcription, and typing."""

from __future__ import annotations

import logging
import threading
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from .audio import AudioStream
from .config import Config
from .engine import TranscriptionEngine, create_engine
from .engine.websocket import WebSocketEngine
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

        # WebSocket streaming: server handles VAD, text arrives via callback
        # Local streaming: local VAD splits utterances, REST/local engine transcribes
        self._use_ws = streaming and config.engine.type == "server"
        self._ws_engine: WebSocketEngine | None = None

        # VAD only needed for local-engine streaming or batch mode
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
        self._stop_event = threading.Event()
        self._recording = False
        self._recorded_chunks: list[np.ndarray] = []
        self._lock = threading.Lock()
        self._transcribe_pool = ThreadPoolExecutor(max_workers=1)

    def _on_audio_chunk(self, audio: np.ndarray) -> None:
        """Called for each audio chunk from the microphone.

        Note: reads _recording without the lock for performance in the
        audio callback hot path. Safe on CPython due to the GIL.
        """
        if not self._recording:
            return

        if self._use_ws and self._ws_engine is not None:
            try:
                self._ws_engine.send_audio(audio)
            except Exception:
                log.error("WS audio send failed", exc_info=True)
        elif self.streaming:
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
            self._transcribe_pool.submit(self._transcribe_and_type, utterance)

    def _on_ws_text(self, text: str) -> None:
        """Callback from WebSocket engine when transcription text arrives."""
        try:
            type_text(text + " ")
            log.debug("WS typed: %d chars", len(text))
        except Exception:
            log.error("Typing failed", exc_info=True)

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

        log.info("Recording started (use_ws=%s, streaming=%s, engine=%s)",
                 self._use_ws, self.streaming, self.config.engine.type)
        notify("Recording", "Speak now")

        if self._use_ws:
            ws_cfg = self.config.websocket
            self._ws_engine = WebSocketEngine(
                server_url=self.config.server.url,
                model=self.config.server.model,
                language=self.config.server.language,
                reconnect_attempts=ws_cfg.reconnect_attempts,
                reconnect_delay=ws_cfg.reconnect_delay,
                server_vad_silence_ms=ws_cfg.server_vad_silence_ms,
                server_vad_threshold=ws_cfg.server_vad_threshold,
                on_text=self._on_ws_text,
            )
            try:
                self._ws_engine.connect()
            except Exception:
                log.error("WebSocket connection failed", exc_info=True)
                self._ws_engine = None
                with self._lock:
                    self._recording = False
                notify("Error", "WebSocket connection failed")
                return
        elif self.streaming:
            self._vad.reset()

        audio = AudioStream(self.config.audio, self._on_audio_chunk)
        try:
            audio.start()
        except Exception:
            log.error("Failed to start audio capture", exc_info=True)
            audio.stop()
            if self._ws_engine is not None:
                self._ws_engine.close()
                self._ws_engine = None
            with self._lock:
                self._recording = False
                self._audio = None
            notify("Error", "Could not access microphone")
            return

        with self._lock:
            self._audio = audio

    def _on_deactivate(self) -> None:
        """Hotkey released/toggled — stop recording and transcribe."""
        with self._lock:
            if not self._recording:
                return
            self._recording = False
            chunks = list(self._recorded_chunks)
            self._recorded_chunks.clear()
            audio = self._audio
            self._audio = None

        log.info("Recording stopped")

        if audio is not None:
            audio.stop()

        if self._use_ws and self._ws_engine is not None:
            self._ws_engine.flush()
            # Wait for server to finish transcribing before closing
            if not self._ws_engine.wait_for_completion(timeout=5.0):
                log.debug("WS transcription did not complete within timeout")
            self._ws_engine.close()
            self._ws_engine = None
        elif self.streaming:
            # Local-engine streaming: flush remaining VAD-buffered speech
            remaining = self._vad.flush()
            if remaining is not None:
                self._transcribe_pool.submit(self._transcribe_and_type, remaining)
        elif chunks:
            # Batch mode: transcribe the full recording at once
            full_audio = np.concatenate(chunks)
            duration = len(full_audio) / self.config.audio.sample_rate
            log.info("Transcribing %.1fs of audio...", duration)
            notify("Transcribing", f"{duration:.0f}s of audio...")
            self._transcribe_pool.submit(self._transcribe_and_type, full_audio)
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
        self._stop_event.set()

        with self._lock:
            should_deactivate = self._recording
        if should_deactivate:
            self._on_deactivate()

        if self._hotkey is not None:
            self._hotkey.stop()
            self._hotkey = None

        if self._ws_engine is not None:
            self._ws_engine.close()
            self._ws_engine = None
        self._transcribe_pool.shutdown(wait=True, cancel_futures=True)
        self._engine.close()
        log.info("Dictation daemon stopped")
        notify("Dictation Stopped", "Daemon exited")

    def request_stop(self) -> None:
        """Signal the daemon to stop. Safe to call from a signal handler."""
        self._stop_event.set()

    def wait(self) -> None:
        """Block until the daemon is stopped or a stop is requested."""
        try:
            self._running.wait()
            while self._running.is_set():
                if self._stop_event.wait(timeout=1.0):
                    break
        except KeyboardInterrupt:
            log.info("Interrupted")

    @property
    def is_running(self) -> bool:
        return self._running.is_set()
