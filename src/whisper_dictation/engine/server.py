"""REST API transcription engine — connects to any OpenAI-compatible STT server."""

from __future__ import annotations

import logging

import numpy as np
import requests

from ..audio import audio_to_wav
from ..config import ServerConfig
from .base import TranscriptionEngine

log = logging.getLogger(__name__)


class ServerEngine(TranscriptionEngine):
    """Transcription via REST API (Speaches, OpenAI, Groq, etc.)."""

    def __init__(self, config: ServerConfig):
        self.config = config
        self._base_url = config.url.rstrip("/")
        self._url = f"{self._base_url}/v1/audio/transcriptions"

    def transcribe(self, audio: np.ndarray, sample_rate: int = 16000) -> str:
        wav_data = audio_to_wav(audio, sample_rate)

        try:
            resp = requests.post(
                self._url,
                files={"file": ("audio.wav", wav_data, "audio/wav")},
                data={
                    k: v
                    for k, v in {
                        "model": self.config.model,
                        "language": self.config.language,
                        "prompt": self.config.prompt or None,
                        "temperature": str(self.config.temperature),
                        "hotwords": self.config.hotwords or None,
                    }.items()
                    if v is not None
                },
                timeout=self.config.timeout,
            )
            resp.raise_for_status()
            text = resp.json().get("text", "").strip()
            if text:
                log.debug("Transcribed: %d chars", len(text))
            return text
        except requests.Timeout:
            log.error("Transcription timed out")
            return ""
        except requests.RequestException as e:
            log.error("Transcription failed: %s", e)
            return ""
        except (ValueError, KeyError) as e:
            log.error("Invalid response from server: %s", e)
            return ""

    def is_available(self) -> bool:
        try:
            resp = requests.get(
                f"{self._base_url}/health",
                timeout=3,
            )
            return resp.ok
        except requests.RequestException:
            return False
