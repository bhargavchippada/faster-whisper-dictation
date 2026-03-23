"""Base interface for transcription engines."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class TranscriptionEngine(ABC):
    """Abstract base for transcription backends."""

    @abstractmethod
    def transcribe(self, audio: np.ndarray, sample_rate: int = 16000) -> str:
        """Transcribe audio data to text.

        Args:
            audio: Audio samples as float32 or int16 numpy array.
            sample_rate: Sample rate in Hz.

        Returns:
            Transcribed text, or empty string if nothing detected.
        """

    @abstractmethod
    def is_available(self) -> bool:
        """Check if the engine is ready to transcribe."""

    def close(self) -> None:
        """Release resources. Override if needed."""
