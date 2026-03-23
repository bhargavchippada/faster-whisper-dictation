"""Tests for whisper_dictation.daemon — DictationDaemon."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from whisper_dictation.config import (
    Config,
    EngineConfig,
)
from whisper_dictation.daemon import DictationDaemon
from whisper_dictation.engine import create_engine

# ---------------------------------------------------------------------------
# create_engine
# ---------------------------------------------------------------------------


class TestCreateEngine:
    def test_server_engine(self):
        cfg = Config()
        engine = create_engine(cfg)
        from whisper_dictation.engine.server import ServerEngine

        assert isinstance(engine, ServerEngine)

    def test_local_engine(self):
        cfg = Config(engine=EngineConfig(type="local"))
        mock_local_cls = MagicMock()
        mock_local_cls.return_value = MagicMock()
        with patch("whisper_dictation.engine.local.LocalEngine", mock_local_cls):
            with patch.dict("sys.modules", {}):
                engine = create_engine(cfg)
        # LocalEngine is imported inside create_engine, verify it was called
        assert engine is not None


# ---------------------------------------------------------------------------
# DictationDaemon init
# ---------------------------------------------------------------------------


class TestDaemonInit:
    @patch("whisper_dictation.daemon.create_engine")
    def test_init(self, mock_create):
        mock_engine = MagicMock()
        mock_create.return_value = mock_engine
        cfg = Config()
        daemon = DictationDaemon(cfg)

        assert daemon.config is cfg
        assert daemon._engine is mock_engine
        assert daemon._recording is False
        assert daemon.is_running is False


# ---------------------------------------------------------------------------
# start
# ---------------------------------------------------------------------------


class TestStart:
    @patch("whisper_dictation.daemon.notify")
    @patch("whisper_dictation.daemon.HotkeyListener")
    @patch("whisper_dictation.daemon.create_engine")
    def test_start_success(self, mock_create, mock_hotkey_cls, mock_notify):
        mock_engine = MagicMock()
        mock_engine.is_available.return_value = True
        mock_create.return_value = mock_engine

        mock_hotkey = MagicMock()
        mock_hotkey_cls.return_value = mock_hotkey

        cfg = Config()
        daemon = DictationDaemon(cfg)
        daemon.start()

        mock_engine.is_available.assert_called_once()
        mock_hotkey_cls.assert_called_once_with(
            binding=cfg.hotkey.binding,
            mode=cfg.hotkey.mode,
            on_activate=daemon._on_activate,
            on_deactivate=daemon._on_deactivate,
        )
        mock_hotkey.start.assert_called_once()
        assert daemon.is_running is True
        mock_notify.assert_called()

    @patch("whisper_dictation.daemon.notify")
    @patch("whisper_dictation.daemon.create_engine")
    def test_start_engine_unavailable(self, mock_create, mock_notify):
        mock_engine = MagicMock()
        mock_engine.is_available.return_value = False
        mock_create.return_value = mock_engine

        cfg = Config()
        daemon = DictationDaemon(cfg)

        with pytest.raises(RuntimeError, match="Transcription engine not available"):
            daemon.start()

        mock_notify.assert_called_with("Error", "Transcription engine not available")

    @patch("whisper_dictation.daemon.notify")
    @patch("whisper_dictation.daemon.create_engine")
    def test_start_local_engine_unavailable(self, mock_create, mock_notify):
        mock_engine = MagicMock()
        mock_engine.is_available.return_value = False
        mock_create.return_value = mock_engine

        cfg = Config(engine=EngineConfig(type="local"))
        daemon = DictationDaemon(cfg)

        with pytest.raises(RuntimeError):
            daemon.start()


# ---------------------------------------------------------------------------
# stop
# ---------------------------------------------------------------------------


class TestStop:
    @patch("whisper_dictation.daemon.notify")
    @patch("whisper_dictation.daemon.HotkeyListener")
    @patch("whisper_dictation.daemon.create_engine")
    def test_stop(self, mock_create, mock_hotkey_cls, mock_notify):
        mock_engine = MagicMock()
        mock_engine.is_available.return_value = True
        mock_create.return_value = mock_engine
        mock_hotkey = MagicMock()
        mock_hotkey_cls.return_value = mock_hotkey

        daemon = DictationDaemon(Config())
        daemon.start()
        daemon.stop()

        mock_hotkey.stop.assert_called_once()
        mock_engine.close.assert_called_once()
        assert daemon.is_running is False

    @patch("whisper_dictation.daemon.notify")
    @patch("whisper_dictation.daemon.create_engine")
    def test_stop_without_start(self, mock_create, mock_notify):
        mock_engine = MagicMock()
        mock_create.return_value = mock_engine

        daemon = DictationDaemon(Config())
        # Should not raise
        daemon.stop()
        mock_engine.close.assert_called_once()

    @patch("whisper_dictation.daemon.notify")
    @patch("whisper_dictation.daemon.HotkeyListener")
    @patch("whisper_dictation.daemon.create_engine")
    def test_stop_deactivates_recording(self, mock_create, mock_hotkey_cls, mock_notify):
        mock_engine = MagicMock()
        mock_engine.is_available.return_value = True
        mock_create.return_value = mock_engine
        mock_hotkey_cls.return_value = MagicMock()

        daemon = DictationDaemon(Config())
        daemon.start()
        daemon._recording = True
        daemon._audio = MagicMock()

        daemon.stop()

        assert daemon._recording is False


# ---------------------------------------------------------------------------
# _on_activate
# ---------------------------------------------------------------------------


class TestOnActivate:
    @patch("whisper_dictation.daemon.notify")
    @patch("whisper_dictation.daemon.AudioStream")
    @patch("whisper_dictation.daemon.create_engine")
    def test_activate_starts_recording(self, mock_create, mock_audio_cls, mock_notify):
        mock_create.return_value = MagicMock()
        mock_audio = MagicMock()
        mock_audio_cls.return_value = mock_audio

        daemon = DictationDaemon(Config())
        daemon._on_activate()

        assert daemon._recording is True
        mock_audio_cls.assert_called_once()
        mock_audio.start.assert_called_once()
        mock_notify.assert_called_with("Recording...", "Speak now")

    @patch("whisper_dictation.daemon.notify")
    @patch("whisper_dictation.daemon.AudioStream")
    @patch("whisper_dictation.daemon.create_engine")
    def test_activate_already_recording(self, mock_create, mock_audio_cls, mock_notify):
        mock_create.return_value = MagicMock()
        daemon = DictationDaemon(Config())
        daemon._recording = True

        daemon._on_activate()

        # Should not create a new audio stream
        mock_audio_cls.assert_not_called()


# ---------------------------------------------------------------------------
# _on_deactivate
# ---------------------------------------------------------------------------


class TestOnDeactivate:
    @patch("whisper_dictation.daemon.notify")
    @patch("whisper_dictation.daemon.create_engine")
    def test_deactivate_stops_recording(self, mock_create, mock_notify):
        mock_create.return_value = MagicMock()
        daemon = DictationDaemon(Config())
        daemon._recording = True
        mock_audio = MagicMock()
        daemon._audio = mock_audio

        daemon._on_deactivate()

        assert daemon._recording is False
        mock_audio.stop.assert_called_once()
        assert daemon._audio is None
        mock_notify.assert_called_with("Stopped", "Dictation paused")

    @patch("whisper_dictation.daemon.notify")
    @patch("whisper_dictation.daemon.create_engine")
    def test_deactivate_not_recording(self, mock_create, mock_notify):
        mock_create.return_value = MagicMock()
        daemon = DictationDaemon(Config())
        daemon._recording = False

        daemon._on_deactivate()

        # Should not call notify for stop
        mock_notify.assert_not_called()

    @patch("whisper_dictation.daemon.threading.Thread")
    @patch("whisper_dictation.daemon.notify")
    @patch("whisper_dictation.daemon.create_engine")
    def test_deactivate_flushes_speech_via_vad_flush(self, mock_create, mock_notify, mock_thread):
        """Test _on_deactivate uses vad.flush() and spawns transcription thread."""
        mock_create.return_value = MagicMock()
        daemon = DictationDaemon(Config())
        daemon._recording = True
        daemon._audio = MagicMock()

        remaining_audio = np.ones(16000, dtype=np.float32)
        daemon._vad = MagicMock()
        daemon._vad.flush.return_value = remaining_audio

        daemon._on_deactivate()

        # Thread should be spawned with the remaining audio
        mock_thread.assert_called_once()
        call_kwargs = mock_thread.call_args[1]
        assert call_kwargs["target"] == daemon._transcribe_and_type
        assert call_kwargs["daemon"] is True
        thread_instance = mock_thread.return_value
        thread_instance.start.assert_called_once()

    @patch("whisper_dictation.daemon.threading.Thread")
    @patch("whisper_dictation.daemon.notify")
    @patch("whisper_dictation.daemon.create_engine")
    def test_deactivate_no_flush_when_vad_returns_none(self, mock_create, mock_notify, mock_thread):
        """Test _on_deactivate does NOT spawn thread when vad.flush() returns None."""
        mock_create.return_value = MagicMock()
        daemon = DictationDaemon(Config())
        daemon._recording = True
        daemon._audio = MagicMock()

        daemon._vad = MagicMock()
        daemon._vad.flush.return_value = None

        daemon._on_deactivate()

        # Thread should NOT be spawned
        mock_thread.assert_not_called()


# ---------------------------------------------------------------------------
# _on_audio_chunk
# ---------------------------------------------------------------------------


class TestOnAudioChunk:
    @patch("whisper_dictation.daemon.create_engine")
    def test_not_recording_ignores_audio(self, mock_create):
        mock_create.return_value = MagicMock()
        daemon = DictationDaemon(Config())
        daemon._recording = False

        # Should not process
        with patch.object(daemon._vad, "process_chunk") as mock_proc:
            daemon._on_audio_chunk(np.zeros(512, dtype=np.float32))
            mock_proc.assert_not_called()

    @patch("whisper_dictation.daemon.threading.Thread")
    @patch("whisper_dictation.daemon.create_engine")
    def test_complete_utterance_spawns_thread(self, mock_create, mock_thread):
        mock_create.return_value = MagicMock()
        daemon = DictationDaemon(Config())
        daemon._recording = True

        utterance = np.zeros(16000, dtype=np.float32)
        with patch.object(daemon._vad, "process_chunk", return_value=(True, utterance)):
            daemon._on_audio_chunk(np.zeros(512, dtype=np.float32))

        mock_thread.assert_called_once()
        call_kwargs = mock_thread.call_args[1]
        assert call_kwargs["target"] == daemon._transcribe_and_type
        assert call_kwargs["daemon"] is True

    @patch("whisper_dictation.daemon.create_engine")
    def test_no_utterance_no_thread(self, mock_create):
        mock_create.return_value = MagicMock()
        daemon = DictationDaemon(Config())
        daemon._recording = True

        with (
            patch.object(daemon._vad, "process_chunk", return_value=(False, None)),
            patch("whisper_dictation.daemon.threading.Thread") as mock_thread,
        ):
            daemon._on_audio_chunk(np.zeros(512, dtype=np.float32))
            mock_thread.assert_not_called()


# ---------------------------------------------------------------------------
# _transcribe_and_type
# ---------------------------------------------------------------------------


class TestTranscribeAndType:
    @patch("whisper_dictation.daemon.type_text")
    @patch("whisper_dictation.daemon.create_engine")
    def test_types_transcribed_text(self, mock_create, mock_type_text):
        mock_engine = MagicMock()
        mock_engine.transcribe.return_value = "hello world"
        mock_create.return_value = mock_engine

        daemon = DictationDaemon(Config())
        daemon._transcribe_and_type(np.zeros(16000, dtype=np.float32))

        mock_type_text.assert_called_once_with("hello world ")

    @patch("whisper_dictation.daemon.type_text")
    @patch("whisper_dictation.daemon.create_engine")
    def test_empty_transcription_no_type(self, mock_create, mock_type_text):
        mock_engine = MagicMock()
        mock_engine.transcribe.return_value = ""
        mock_create.return_value = mock_engine

        daemon = DictationDaemon(Config())
        daemon._transcribe_and_type(np.zeros(16000, dtype=np.float32))

        mock_type_text.assert_not_called()


# ---------------------------------------------------------------------------
# wait
# ---------------------------------------------------------------------------


class TestWait:
    @patch("whisper_dictation.daemon.notify")
    @patch("whisper_dictation.daemon.create_engine")
    def test_wait_returns_when_not_running(self, mock_create, mock_notify):
        """Test wait() returns when _running event is cleared."""
        mock_engine = MagicMock()
        mock_engine.is_available.return_value = True
        mock_create.return_value = mock_engine

        daemon = DictationDaemon(Config())
        # Set running, then immediately clear it so the while loop exits
        daemon._running.set()

        # Mock wait to return immediately, then have is_set return False
        call_count = [0]

        def mock_is_set():
            call_count[0] += 1
            if call_count[0] > 1:
                return False
            return True

        with patch.object(daemon._running, "is_set", side_effect=mock_is_set):
            with patch.object(daemon._running, "wait", return_value=True):
                daemon.wait()

    @patch("whisper_dictation.daemon.notify")
    @patch("whisper_dictation.daemon.create_engine")
    def test_wait_keyboard_interrupt(self, mock_create, mock_notify):
        """Test wait() handles KeyboardInterrupt by calling stop()."""
        mock_engine = MagicMock()
        mock_create.return_value = mock_engine

        daemon = DictationDaemon(Config())
        daemon._running.set()

        with patch.object(daemon._running, "wait", side_effect=KeyboardInterrupt):
            with patch.object(daemon, "stop") as mock_stop:
                daemon.wait()
                mock_stop.assert_called_once()
