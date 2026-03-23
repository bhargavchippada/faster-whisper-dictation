"""Tests for whisper_dictation.vad — SpeechDetector, OnnxVAD."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from whisper_dictation.vad import SpeechDetector


@pytest.fixture
def mock_vad_model():
    """Patch the global VAD model with a controllable mock."""
    mock_model = MagicMock()
    mock_model.return_value = 0.0  # default: silence
    with (
        patch("whisper_dictation.vad._model", mock_model),
        patch("whisper_dictation.vad._load_model"),
    ):
        yield mock_model


@pytest.fixture
def detector(mock_vad_model):
    """Create a SpeechDetector with default params and mocked model."""
    det = SpeechDetector(
        sample_rate=16000,
        threshold=0.5,
        silence_ms=800,
        min_speech_ms=250,
    )
    det._model_loaded = True  # skip lazy load
    return det


# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------


class TestSpeechDetectorInit:
    def test_defaults(self, mock_vad_model):
        det = SpeechDetector()
        det._model_loaded = True
        assert det.sample_rate == 16000
        assert det.threshold == 0.5
        assert det.silence_chunks == 25  # 800 // 32
        assert det.min_speech_chunks == 7  # 250 // 32 = 7 (int div)
        assert det.chunk_size == 512  # 16000 * 32 // 1000
        assert det.is_speaking is False

    def test_custom_params(self, mock_vad_model):
        det = SpeechDetector(
            sample_rate=8000,
            threshold=0.7,
            silence_ms=1600,
            min_speech_ms=500,
        )
        assert det.chunk_size == 256  # 8000 * 32 // 1000
        assert det.silence_chunks == 50  # 1600 // 32
        assert det.min_speech_chunks == 15  # 500 // 32


# ---------------------------------------------------------------------------
# reset
# ---------------------------------------------------------------------------


class TestReset:
    def test_reset_clears_state(self, detector, mock_vad_model):
        detector._is_speaking = True
        detector._silence_count = 5
        detector._speech_count = 10
        detector._speech_frames.append(np.zeros(512, dtype=np.float32))
        detector._buffer = np.ones(100, dtype=np.float32)

        mock_vad_model.reset_states = MagicMock()
        detector.reset()

        assert detector._is_speaking is False
        assert detector._silence_count == 0
        assert detector._speech_count == 0
        assert len(detector._speech_frames) == 0
        assert len(detector._buffer) == 0
        mock_vad_model.reset_states.assert_called_once()

    def test_reset_without_model_reset_states(self, detector):
        with patch("whisper_dictation.vad._model", None):
            # Should not raise even when model is None
            detector.reset()


# ---------------------------------------------------------------------------
# process_chunk — speech patterns
# ---------------------------------------------------------------------------


class TestProcessChunk:
    def test_silence_returns_no_utterance(self, detector, mock_vad_model):
        mock_vad_model.return_value = 0.1  # below threshold
        audio = np.zeros(512, dtype=np.float32)
        complete, data = detector.process_chunk(audio)
        assert complete is False
        assert data is None

    def test_speech_starts_speaking(self, detector, mock_vad_model):
        mock_vad_model.return_value = 0.9  # above threshold
        audio = np.zeros(512, dtype=np.float32)
        detector.process_chunk(audio)
        assert detector.is_speaking is True

    def test_short_speech_rejected(self, detector, mock_vad_model):
        """Speech shorter than min_speech_ms is rejected."""
        # 3 speech chunks (below min_speech_chunks=7), then silence
        results = []
        for i in range(3):
            mock_vad_model.return_value = 0.9
            audio = np.zeros(512, dtype=np.float32)
            results.append(detector.process_chunk(audio))

        # Now send enough silence to trigger end
        for i in range(30):
            mock_vad_model.return_value = 0.1
            audio = np.zeros(512, dtype=np.float32)
            complete, data = detector.process_chunk(audio)
            results.append((complete, data))

        # No completed utterance should be returned
        assert all(not r[0] for r in results)

    def test_valid_utterance_detected(self, detector, mock_vad_model):
        """Enough speech followed by silence produces a completed utterance."""
        # Send min_speech_chunks (7) + extra speech
        for _ in range(10):
            mock_vad_model.return_value = 0.9
            audio = np.zeros(512, dtype=np.float32)
            detector.process_chunk(audio)

        assert detector.is_speaking is True
        assert detector._speech_count == 10

        # Now send silence_chunks (25) silence frames
        complete = False
        utterance = None
        for _ in range(30):
            mock_vad_model.return_value = 0.1
            audio = np.zeros(512, dtype=np.float32)
            c, u = detector.process_chunk(audio)
            if c:
                complete = True
                utterance = u

        assert complete is True
        assert utterance is not None
        # Utterance should contain all speech + silence frames
        # 10 speech + 25 silence = 35 chunks * 512 samples
        assert len(utterance) == 35 * 512
        assert detector.is_speaking is False

    def test_int16_input_converted(self, detector, mock_vad_model):
        """int16 audio is converted to float32."""
        mock_vad_model.return_value = 0.1
        audio = np.zeros(512, dtype=np.int16)
        detector.process_chunk(audio)

        # The model should have been called with float32 data
        call_args = mock_vad_model.call_args[0]
        assert call_args[0].dtype == np.float32

    def test_buffering_partial_chunks(self, detector, mock_vad_model):
        """Audio smaller than chunk_size is buffered."""
        mock_vad_model.return_value = 0.1
        audio = np.zeros(256, dtype=np.float32)  # half a chunk
        complete, data = detector.process_chunk(audio)
        assert complete is False
        assert data is None
        assert len(detector._buffer) == 256

        # Send another half — should trigger model call
        detector.process_chunk(audio)
        assert len(detector._buffer) == 0
        mock_vad_model.assert_called_once()

    def test_large_chunk_processed_in_pieces(self, detector, mock_vad_model):
        """Audio larger than chunk_size is split into multiple model calls."""
        mock_vad_model.return_value = 0.1
        audio = np.zeros(512 * 5, dtype=np.float32)  # 5 chunks
        detector.process_chunk(audio)
        assert mock_vad_model.call_count == 5

    def test_empty_audio(self, detector, mock_vad_model):
        audio = np.array([], dtype=np.float32)
        complete, data = detector.process_chunk(audio)
        assert complete is False
        assert data is None
        mock_vad_model.assert_not_called()


# ---------------------------------------------------------------------------
# is_speaking property
# ---------------------------------------------------------------------------


class TestIsSpeaking:
    def test_initially_false(self, detector):
        assert detector.is_speaking is False

    def test_true_after_speech(self, detector, mock_vad_model):
        mock_vad_model.return_value = 0.9
        detector.process_chunk(np.zeros(512, dtype=np.float32))
        assert detector.is_speaking is True

    def test_false_after_silence(self, detector, mock_vad_model):
        # Start speech
        for _ in range(10):
            mock_vad_model.return_value = 0.9
            detector.process_chunk(np.zeros(512, dtype=np.float32))

        # End with silence
        for _ in range(30):
            mock_vad_model.return_value = 0.1
            detector.process_chunk(np.zeros(512, dtype=np.float32))

        assert detector.is_speaking is False


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------


class TestModelLoading:
    def test_ensure_model_calls_load(self):
        with patch("whisper_dictation.vad._load_model") as mock_load:
            det = SpeechDetector()
            det._ensure_model()
            mock_load.assert_called_once()

    def test_ensure_model_only_loads_once(self):
        with patch("whisper_dictation.vad._load_model") as mock_load:
            det = SpeechDetector()
            det._ensure_model()
            det._ensure_model()
            mock_load.assert_called_once()
