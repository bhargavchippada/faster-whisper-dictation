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

    def test_max_utterance_cap_flushes(self, mock_vad_model):
        """Cover lines 186-194: speech exceeding max_speech_s is flushed."""
        # Use a very short max_speech_s so we hit the cap quickly
        det = SpeechDetector(
            sample_rate=16000,
            threshold=0.5,
            silence_ms=800,
            min_speech_ms=250,
            max_speech_s=0.128,  # 128ms = 4 chunks of 32ms each
        )
        det._model_loaded = True
        assert det.max_speech_chunks == 4

        mock_vad_model.return_value = 0.9  # always speech
        audio = np.ones(512, dtype=np.float32)

        # Send 4 speech chunks to hit the cap exactly
        for i in range(3):
            complete, data = det.process_chunk(audio)
            assert complete is False

        # The 4th chunk should trigger the max cap flush
        complete, data = det.process_chunk(audio)
        assert complete is True
        assert data is not None
        # Should contain 4 chunks * 512 samples
        assert len(data) == 4 * 512
        # State should be reset
        assert det._is_speaking is False
        assert det._speech_count == 0
        assert len(det._speech_frames) == 0


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


# ---------------------------------------------------------------------------
# flush
# ---------------------------------------------------------------------------


class TestFlush:
    def test_flush_returns_none_when_not_speaking(self):
        det = SpeechDetector()
        det._is_speaking = False
        assert det.flush() is None

    def test_flush_returns_none_when_no_frames(self):
        det = SpeechDetector()
        det._is_speaking = True
        det._speech_frames = []
        assert det.flush() is None

    def test_flush_returns_audio_when_enough_speech(self):
        det = SpeechDetector(min_speech_ms=250)
        det._is_speaking = True
        det._speech_count = det.min_speech_chunks + 1
        det._speech_frames = [np.ones(512, dtype=np.float32)]

        result = det.flush()
        assert result is not None
        assert len(result) == 512
        assert not det._is_speaking
        assert det._speech_count == 0
        assert len(det._speech_frames) == 0

    def test_flush_returns_none_when_speech_too_short(self):
        det = SpeechDetector(min_speech_ms=250)
        det._is_speaking = True
        det._speech_count = 1  # below min_speech_chunks
        det._speech_frames = [np.ones(512, dtype=np.float32)]

        result = det.flush()
        assert result is None
        assert not det._is_speaking

    def test_flush_concatenates_multiple_frames(self):
        det = SpeechDetector(min_speech_ms=32)
        det._is_speaking = True
        det._speech_count = det.min_speech_chunks + 1
        det._speech_frames = [
            np.ones(512, dtype=np.float32),
            np.ones(512, dtype=np.float32),
        ]

        result = det.flush()
        assert result is not None
        assert len(result) == 1024


# ---------------------------------------------------------------------------
# _load_model — PyTorch path
# ---------------------------------------------------------------------------


class TestLoadModelDoubleCheck:
    def test_double_checked_locking_returns_early(self):
        """Cover line 26: _model set between outer check and lock acquisition."""
        import whisper_dictation.vad as vad_mod

        original_model = vad_mod._model
        original_lock = vad_mod._model_lock
        sentinel = MagicMock(name="already_loaded_vad")

        class SetModelOnEnterLock:
            def __enter__(self_lock):
                original_lock.__enter__()
                vad_mod._model = sentinel
                return self_lock

            def __exit__(self_lock, *args):
                return original_lock.__exit__(*args)

        try:
            vad_mod._model = None
            vad_mod._model_lock = SetModelOnEnterLock()
            vad_mod._load_model()
            assert vad_mod._model is sentinel
        finally:
            vad_mod._model = original_model
            vad_mod._model_lock = original_lock


class TestLoadModelTorch:
    def test_load_model_with_torch(self):
        """Test _load_model when torch is available."""
        import whisper_dictation.vad as vad_mod

        original_model = vad_mod._model
        vad_mod._model = None

        try:
            mock_model = MagicMock()
            mock_torch = MagicMock()
            mock_torch.hub.load.return_value = (mock_model, [MagicMock(), None, None])

            with patch.dict("sys.modules", {"torch": mock_torch}):
                with patch(
                    "builtins.__import__",
                    side_effect=lambda name, *a, **kw: (
                        mock_torch if name == "torch" else __import__(name, *a, **kw)
                    ),
                ):
                    vad_mod._load_model()

            assert vad_mod._model is mock_model

        finally:
            vad_mod._model = original_model

    def test_load_model_already_loaded(self):
        """Test _load_model short-circuits when model is already loaded."""
        import whisper_dictation.vad as vad_mod

        original_model = vad_mod._model
        try:
            vad_mod._model = MagicMock()  # pretend already loaded
            with patch("whisper_dictation.vad._load_onnx_model") as mock_onnx:
                vad_mod._load_model()
                mock_onnx.assert_not_called()
        finally:
            vad_mod._model = original_model

    def test_load_model_torch_import_error_falls_back_to_onnx(self):
        """Test _load_model falls back to ONNX when torch import fails."""
        import whisper_dictation.vad as vad_mod

        original_model = vad_mod._model

        vad_mod._model = None

        try:
            with patch("whisper_dictation.vad._load_onnx_model") as mock_onnx:
                # Make torch import raise ImportError
                real_import = (
                    __builtins__.__import__ if hasattr(__builtins__, "__import__") else __import__
                )

                def fake_import(name, *args, **kwargs):
                    if name == "torch":
                        raise ImportError("no torch")
                    return real_import(name, *args, **kwargs)

                with patch("builtins.__import__", side_effect=fake_import):
                    vad_mod._load_model()

                mock_onnx.assert_called_once()
        finally:
            vad_mod._model = original_model


# ---------------------------------------------------------------------------
# _load_onnx_model
# ---------------------------------------------------------------------------


class TestLoadOnnxModel:
    def test_load_onnx_model_custom_url_skips_hash(self, tmp_path):
        """Test custom VAD model URL skips SHA-256 verification."""
        import whisper_dictation.vad as vad_mod

        original_model = vad_mod._model
        vad_mod._model = None

        mock_response = MagicMock()
        mock_response.read.return_value = b"custom model"
        mock_response.__enter__ = MagicMock(return_value=mock_response)
        mock_response.__exit__ = MagicMock(return_value=False)

        try:
            with (
                patch("whisper_dictation.vad.OnnxVAD"),
                patch("platformdirs.user_cache_dir", return_value=str(tmp_path)),
                patch("urllib.request.urlopen", return_value=mock_response),
                patch("whisper_dictation.vad._verify_model_hash") as mock_verify,
                patch.dict(
                    "os.environ", {"DICTATION_VAD_MODEL_URL": "https://example.com/model.onnx"}
                ),
                patch.dict("sys.modules", {"onnxruntime": MagicMock()}),
            ):
                vad_mod._load_onnx_model()
                mock_verify.assert_not_called()
        finally:
            vad_mod._model = original_model

    def test_load_onnx_model_downloads_when_missing(self, tmp_path):
        """Test _load_onnx_model downloads model when cache doesn't exist."""
        import whisper_dictation.vad as vad_mod

        original_model = vad_mod._model
        vad_mod._model = None

        mock_response = MagicMock()
        mock_response.read.return_value = b"fake model data"
        mock_response.__enter__ = MagicMock(return_value=mock_response)
        mock_response.__exit__ = MagicMock(return_value=False)

        try:
            mock_ort = MagicMock()

            with (
                patch("whisper_dictation.vad.OnnxVAD") as mock_onnx_vad,
                patch("platformdirs.user_cache_dir", return_value=str(tmp_path)),
                patch("urllib.request.urlopen", return_value=mock_response) as mock_download,
                patch("whisper_dictation.vad._verify_model_hash"),
                patch.dict("sys.modules", {"onnxruntime": mock_ort}),
            ):
                vad_mod._load_onnx_model()
                mock_download.assert_called_once()
                mock_onnx_vad.assert_called_once()
        finally:
            vad_mod._model = original_model

    def test_load_onnx_model_cleans_up_on_hash_failure(self, tmp_path):
        """Test _load_onnx_model removes temp file when hash verification fails."""
        import whisper_dictation.vad as vad_mod

        original_model = vad_mod._model
        vad_mod._model = None

        mock_response = MagicMock()
        mock_response.read.return_value = b"bad model data"
        mock_response.__enter__ = MagicMock(return_value=mock_response)
        mock_response.__exit__ = MagicMock(return_value=False)

        try:
            with (
                patch("platformdirs.user_cache_dir", return_value=str(tmp_path)),
                patch("urllib.request.urlopen", return_value=mock_response),
                patch(
                    "whisper_dictation.vad._verify_model_hash",
                    side_effect=RuntimeError("hash mismatch"),
                ),
            ):
                with pytest.raises(RuntimeError, match="hash mismatch"):
                    vad_mod._load_onnx_model()

            # Temp file should be cleaned up
            assert not (tmp_path / "silero_vad.tmp").exists()
            # Cache file should not exist either
            assert not (tmp_path / "silero_vad.onnx").exists()
        finally:
            vad_mod._model = original_model

    def test_load_onnx_model_cleans_up_on_download_failure(self, tmp_path):
        """Test _load_onnx_model removes .tmp file when download itself fails."""
        import whisper_dictation.vad as vad_mod

        original_model = vad_mod._model
        vad_mod._model = None

        tmp_file = tmp_path / "silero_vad.tmp"

        try:
            with (
                patch("platformdirs.user_cache_dir", return_value=str(tmp_path)),
                patch(
                    "urllib.request.urlopen",
                    side_effect=ConnectionError("download interrupted"),
                ),
            ):
                with pytest.raises(ConnectionError, match="download interrupted"):
                    vad_mod._load_onnx_model()

            # .tmp file should be cleaned up
            assert not tmp_file.exists()
            # Cache file should not exist
            assert not (tmp_path / "silero_vad.onnx").exists()
        finally:
            vad_mod._model = original_model

    def test_load_onnx_model_uses_cached(self, tmp_path):
        """Test _load_onnx_model skips download when cache exists."""
        import whisper_dictation.vad as vad_mod

        original_model = vad_mod._model
        vad_mod._model = None

        try:
            # Create a fake cached model file
            cache_dir = tmp_path / "whisper-dictation"
            cache_dir.mkdir()
            (cache_dir / "silero_vad.onnx").write_text("fake")

            mock_ort = MagicMock()

            with (
                patch("whisper_dictation.vad.OnnxVAD"),
                patch("platformdirs.user_cache_dir", return_value=str(cache_dir)),
                patch("urllib.request.urlopen") as mock_download,
                patch.dict("sys.modules", {"onnxruntime": mock_ort}),
            ):
                vad_mod._load_onnx_model()
                mock_download.assert_not_called()
        finally:
            vad_mod._model = original_model


# ---------------------------------------------------------------------------
# _verify_model_hash
# ---------------------------------------------------------------------------


class TestVadModelUrlValidation:
    def test_invalid_scheme_raises_at_import(self):
        """Test that file:// scheme in DICTATION_VAD_MODEL_URL is rejected."""
        import importlib
        import sys

        saved = sys.modules.pop("whisper_dictation.vad", None)
        try:
            with patch.dict("os.environ", {"DICTATION_VAD_MODEL_URL": "file:///etc/passwd"}):
                with pytest.raises(ValueError, match="must use http or https"):
                    import whisper_dictation.vad

                    importlib.reload(whisper_dictation.vad)
        finally:
            if saved is not None:
                sys.modules["whisper_dictation.vad"] = saved

    def test_valid_https_custom_url_accepted(self):
        """Test that https:// custom URL is accepted."""
        import importlib
        import sys

        saved = sys.modules.pop("whisper_dictation.vad", None)
        try:
            with patch.dict(
                "os.environ", {"DICTATION_VAD_MODEL_URL": "https://example.com/model.onnx"}
            ):
                import whisper_dictation.vad

                importlib.reload(whisper_dictation.vad)
                # Should not raise
        finally:
            if saved is not None:
                sys.modules["whisper_dictation.vad"] = saved


class TestVerifyModelHash:
    def test_matching_hash_passes(self, tmp_path):
        """Test _verify_model_hash succeeds when hash matches."""
        from whisper_dictation.vad import _verify_model_hash

        model_file = tmp_path / "model.onnx"
        model_file.write_bytes(b"test data")

        import hashlib

        expected = hashlib.sha256(b"test data").hexdigest()
        # Should not raise
        _verify_model_hash(model_file, expected)

    def test_mismatched_hash_raises_and_deletes(self, tmp_path):
        """Test _verify_model_hash raises RuntimeError and deletes file on mismatch."""
        from whisper_dictation.vad import _verify_model_hash

        model_file = tmp_path / "model.onnx"
        model_file.write_bytes(b"test data")

        with pytest.raises(RuntimeError, match="Model integrity check failed"):
            _verify_model_hash(model_file, "0" * 64)

        assert not model_file.exists()


# ---------------------------------------------------------------------------
# OnnxVAD class
# ---------------------------------------------------------------------------


class TestOnnxVAD:
    def test_init(self):
        """Test OnnxVAD initialization."""
        mock_ort = MagicMock()
        mock_session = MagicMock()
        mock_ort.SessionOptions.return_value = MagicMock()
        mock_ort.InferenceSession.return_value = mock_session

        with patch.dict("sys.modules", {"onnxruntime": mock_ort}):
            from whisper_dictation.vad import OnnxVAD

            vad = OnnxVAD.__new__(OnnxVAD)
            # Manually call __init__ with mocked ort
            with patch(
                "builtins.__import__",
                side_effect=lambda name, *a, **kw: (
                    mock_ort if name == "onnxruntime" else __import__(name, *a, **kw)
                ),
            ):
                vad.__init__("fake_model.onnx")

            assert vad.session is mock_session
            assert vad._h.shape == (2, 1, 64)
            assert vad._c.shape == (2, 1, 64)

    def test_reset_states(self):
        """Test OnnxVAD.reset_states zeros out hidden states."""
        from whisper_dictation.vad import OnnxVAD

        vad = OnnxVAD.__new__(OnnxVAD)
        vad._h = np.ones((2, 1, 64), dtype=np.float32)
        vad._c = np.ones((2, 1, 64), dtype=np.float32)
        vad.reset_states()
        np.testing.assert_array_equal(vad._h, np.zeros((2, 1, 64), dtype=np.float32))
        np.testing.assert_array_equal(vad._c, np.zeros((2, 1, 64), dtype=np.float32))

    def test_call_float32_input(self):
        """Test OnnxVAD __call__ with float32 input."""
        from whisper_dictation.vad import OnnxVAD

        vad = OnnxVAD.__new__(OnnxVAD)
        vad._h = np.zeros((2, 1, 64), dtype=np.float32)
        vad._c = np.zeros((2, 1, 64), dtype=np.float32)
        vad._sr = np.array(16000, dtype=np.int64)

        mock_session = MagicMock()
        new_h = np.zeros((2, 1, 64), dtype=np.float32)
        new_c = np.zeros((2, 1, 64), dtype=np.float32)
        mock_session.run.return_value = [np.array([[0.85]]), new_h, new_c]
        vad.session = mock_session

        audio = np.zeros(512, dtype=np.float32)
        result = vad(audio, 16000)

        assert result == pytest.approx(0.85)
        mock_session.run.assert_called_once()
        # Verify input was reshaped to 2D
        call_inputs = mock_session.run.call_args[0][1]
        assert call_inputs["input"].ndim == 2

    def test_call_int16_input_converted(self):
        """Test OnnxVAD __call__ converts int16 to float32."""
        from whisper_dictation.vad import OnnxVAD

        vad = OnnxVAD.__new__(OnnxVAD)
        vad._h = np.zeros((2, 1, 64), dtype=np.float32)
        vad._c = np.zeros((2, 1, 64), dtype=np.float32)
        vad._sr = np.array(16000, dtype=np.int64)

        mock_session = MagicMock()
        new_h = np.zeros((2, 1, 64), dtype=np.float32)
        new_c = np.zeros((2, 1, 64), dtype=np.float32)
        mock_session.run.return_value = [np.array([[0.5]]), new_h, new_c]
        vad.session = mock_session

        audio = np.zeros(512, dtype=np.int16)
        result = vad(audio, 16000)

        assert result == pytest.approx(0.5)
        call_inputs = mock_session.run.call_args[0][1]
        assert call_inputs["input"].dtype == np.float32

    def test_call_updates_hidden_states(self):
        """Test OnnxVAD __call__ updates _h and _c from session output."""
        from whisper_dictation.vad import OnnxVAD

        vad = OnnxVAD.__new__(OnnxVAD)
        vad._h = np.zeros((2, 1, 64), dtype=np.float32)
        vad._c = np.zeros((2, 1, 64), dtype=np.float32)
        vad._sr = np.array(16000, dtype=np.int64)

        mock_session = MagicMock()
        new_h = np.ones((2, 1, 64), dtype=np.float32)
        new_c = np.ones((2, 1, 64), dtype=np.float32) * 2
        mock_session.run.return_value = [np.array([[0.7]]), new_h, new_c]
        vad.session = mock_session

        vad(np.zeros(512, dtype=np.float32), 16000)

        np.testing.assert_array_equal(vad._h, new_h)
        np.testing.assert_array_equal(vad._c, new_c)


class TestLoadOnnxModelCustomUrl:
    def test_custom_url_skips_hash_verification(self, tmp_path):
        """Test that a custom VAD model URL skips SHA-256 verification."""
        import whisper_dictation.vad as vad_mod

        original_model = vad_mod._model
        vad_mod._model = None

        mock_response = MagicMock()
        mock_response.read.return_value = b"fake"
        mock_response.__enter__ = MagicMock(return_value=mock_response)
        mock_response.__exit__ = MagicMock(return_value=False)

        try:
            with (
                patch("whisper_dictation.vad.OnnxVAD"),
                patch("platformdirs.user_cache_dir", return_value=str(tmp_path)),
                patch("urllib.request.urlopen", return_value=mock_response),
                patch("whisper_dictation.vad._verify_model_hash") as mock_verify,
                patch.dict("os.environ", {"DICTATION_VAD_MODEL_URL": "http://custom/model.onnx"}),
                patch.dict("sys.modules", {"onnxruntime": MagicMock()}),
            ):
                vad_mod._load_onnx_model()
                mock_verify.assert_not_called()
        finally:
            vad_mod._model = original_model
