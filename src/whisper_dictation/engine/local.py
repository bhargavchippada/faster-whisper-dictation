"""Local transcription engine using faster-whisper (no server needed)."""

from __future__ import annotations

import logging
import threading

import numpy as np

from ..config import EngineConfig, ServerConfig
from .base import TranscriptionEngine

log = logging.getLogger(__name__)


class LocalEngine(TranscriptionEngine):
    """Transcription using faster-whisper locally."""

    def __init__(self, server_config: ServerConfig, engine_config: EngineConfig):
        self._model = None
        self._model_lock = threading.Lock()
        self._model_name = server_config.model
        self._language = server_config.language
        self._compute_type = engine_config.compute_type
        self._device = engine_config.device

    def _ensure_model(self):
        if self._model is not None:
            return

        with self._model_lock:
            if self._model is not None:
                return  # double-checked locking

            try:
                from faster_whisper import WhisperModel
            except ImportError:
                raise RuntimeError(
                    "faster-whisper not installed. "
                    "Install with: pip install 'whisper-dictation[local]'"
                )

            device = self._device
            compute_type = self._compute_type

            if device == "auto":
                try:
                    import torch
                    device = "cuda" if torch.cuda.is_available() else "cpu"
                except ImportError:
                    device = "cpu"

            if compute_type == "auto":
                compute_type = "float16" if device == "cuda" else "int8"

            log.info(
                "Loading model %s on %s (%s)...",
                self._model_name, device, compute_type,
            )
            self._model = WhisperModel(
                self._model_name,
                device=device,
                compute_type=compute_type,
            )
        log.info("Model loaded")

    def transcribe(self, audio: np.ndarray, sample_rate: int = 16000) -> str:
        self._ensure_model()

        # faster-whisper expects float32
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32) / 32768.0

        segments, _ = self._model.transcribe(
            audio,
            language=self._language,
            vad_filter=True,
        )

        text = " ".join(seg.text.strip() for seg in segments).strip()
        if text:
            log.debug("Transcribed: %s", text[:80])
        return text

    def is_available(self) -> bool:
        try:
            self._ensure_model()
            return True
        except Exception as e:
            log.debug("Local engine not available: %s", e)
            return False

    def close(self) -> None:
        self._model = None
