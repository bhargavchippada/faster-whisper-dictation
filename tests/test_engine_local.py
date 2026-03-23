"""Tests for whisper_dictation.engine.local — LocalEngine."""

from __future__ import annotations

import sys
from types import ModuleType
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from whisper_dictation.config import EngineConfig, ServerConfig
from whisper_dictation.engine.local import LocalEngine


@pytest.fixture
def server_config():
    return ServerConfig(model="tiny", language="en")


@pytest.fixture
def engine_config():
    return EngineConfig(type="local", compute_type="auto", device="auto")


@pytest.fixture
def engine(server_config, engine_config):
    return LocalEngine(server_config, engine_config)


# ---------------------------------------------------------------------------
# Init
# ---------------------------------------------------------------------------


class TestLocalEngineInit:
    def test_stores_config(self, server_config, engine_config):
        eng = LocalEngine(server_config, engine_config)
        assert eng._model_name == "tiny"
        assert eng._language == "en"
        assert eng._compute_type == "auto"
        assert eng._device == "auto"
        assert eng._model is None


# ---------------------------------------------------------------------------
# _ensure_model
# ---------------------------------------------------------------------------


class TestEnsureModel:
    def test_import_error_raises_runtime_error(self, engine):
        with patch.dict(sys.modules, {"faster_whisper": None}):
            with pytest.raises(RuntimeError, match="faster-whisper not installed"):
                engine._ensure_model()

    def test_loads_model_with_cpu_int8(self, engine):
        mock_whisper = MagicMock()
        mock_model_instance = MagicMock()
        mock_whisper.WhisperModel.return_value = mock_model_instance

        with (
            patch.dict(sys.modules, {"faster_whisper": mock_whisper}),
            patch.dict(sys.modules, {"torch": None}),
        ):
            engine._ensure_model()

        mock_whisper.WhisperModel.assert_called_once_with(
            "tiny",
            device="cpu",
            compute_type="int8",
        )
        assert engine._model is mock_model_instance

    def test_loads_model_with_cuda_float16(self, server_config):
        eng = LocalEngine(server_config, EngineConfig(type="local", compute_type="auto", device="auto"))

        mock_whisper = MagicMock()
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = True

        with (
            patch.dict(sys.modules, {"faster_whisper": mock_whisper}),
            patch.dict(sys.modules, {"torch": mock_torch}),
        ):
            eng._ensure_model()

        mock_whisper.WhisperModel.assert_called_once_with(
            "tiny",
            device="cuda",
            compute_type="float16",
        )

    def test_explicit_device_and_compute(self, server_config):
        eng = LocalEngine(
            server_config,
            EngineConfig(type="local", compute_type="float32", device="cpu"),
        )
        mock_whisper = MagicMock()
        with patch.dict(sys.modules, {"faster_whisper": mock_whisper}):
            eng._ensure_model()

        mock_whisper.WhisperModel.assert_called_once_with(
            "tiny",
            device="cpu",
            compute_type="float32",
        )

    def test_only_loads_once(self, engine):
        mock_whisper = MagicMock()
        with (
            patch.dict(sys.modules, {"faster_whisper": mock_whisper}),
            patch.dict(sys.modules, {"torch": None}),
        ):
            engine._ensure_model()
            engine._ensure_model()

        assert mock_whisper.WhisperModel.call_count == 1

    def test_double_checked_locking_returns_early(self, engine):
        """Cover line 33: _model set between outer check and lock acquisition."""
        import threading

        original_lock = engine._model_lock
        sentinel = MagicMock(name="already_loaded_model")

        class SetModelOnEnterLock:
            """A lock that sets _model when acquired, simulating a race."""

            def __enter__(self_lock):
                original_lock.__enter__()
                engine._model = sentinel
                return self_lock

            def __exit__(self_lock, *args):
                return original_lock.__exit__(*args)

        engine._model = None
        engine._model_lock = SetModelOnEnterLock()
        engine._ensure_model()
        # Model should be the sentinel set inside the lock, not a new one
        assert engine._model is sentinel


# ---------------------------------------------------------------------------
# transcribe
# ---------------------------------------------------------------------------


class TestTranscribe:
    def test_transcribe_returns_text(self, engine):
        mock_model = MagicMock()
        seg1 = MagicMock()
        seg1.text = " Hello "
        seg2 = MagicMock()
        seg2.text = " world "
        mock_model.transcribe.return_value = ([seg1, seg2], None)
        engine._model = mock_model

        audio = np.zeros(16000, dtype=np.float32)
        text = engine.transcribe(audio, 16000)
        assert text == "Hello world"

    def test_transcribe_empty_segments(self, engine):
        mock_model = MagicMock()
        mock_model.transcribe.return_value = ([], None)
        engine._model = mock_model

        text = engine.transcribe(np.zeros(16000, dtype=np.float32))
        assert text == ""

    def test_transcribe_converts_int16(self, engine):
        mock_model = MagicMock()
        mock_model.transcribe.return_value = ([], None)
        engine._model = mock_model

        audio = np.array([16384, -16384], dtype=np.int16)
        engine.transcribe(audio, 16000)

        # Verify the audio passed to model is float32
        passed_audio = mock_model.transcribe.call_args[0][0]
        assert passed_audio.dtype == np.float32

    def test_transcribe_calls_with_language_and_vad(self, engine):
        mock_model = MagicMock()
        mock_model.transcribe.return_value = ([], None)
        engine._model = mock_model

        engine.transcribe(np.zeros(100, dtype=np.float32), 16000)
        call_kwargs = mock_model.transcribe.call_args[1]
        assert call_kwargs["language"] == "en"
        assert call_kwargs["vad_filter"] is True


# ---------------------------------------------------------------------------
# is_available
# ---------------------------------------------------------------------------


class TestIsAvailable:
    def test_available_when_model_loads(self, engine):
        mock_whisper = MagicMock()
        with (
            patch.dict(sys.modules, {"faster_whisper": mock_whisper}),
            patch.dict(sys.modules, {"torch": None}),
        ):
            assert engine.is_available() is True

    def test_not_available_on_import_error(self, engine):
        with patch.dict(sys.modules, {"faster_whisper": None}):
            assert engine.is_available() is False

    def test_not_available_on_generic_error(self, engine):
        mock_whisper = MagicMock()
        mock_whisper.WhisperModel.side_effect = Exception("GPU error")
        with (
            patch.dict(sys.modules, {"faster_whisper": mock_whisper}),
            patch.dict(sys.modules, {"torch": None}),
        ):
            assert engine.is_available() is False


# ---------------------------------------------------------------------------
# close
# ---------------------------------------------------------------------------


class TestClose:
    def test_close_clears_model(self, engine):
        engine._model = MagicMock()
        engine.close()
        assert engine._model is None

    def test_close_when_no_model(self, engine):
        engine.close()
        assert engine._model is None
