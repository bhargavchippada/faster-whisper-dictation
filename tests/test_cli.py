"""Tests for whisper_dictation.cli — argument parsing, subcommands."""

from __future__ import annotations

import json
import os
import signal
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from whisper_dictation.cli import (
    _cleanup_pid,
    _read_pid,
    _setup_logging,
    _write_pid,
    cmd_devices,
    cmd_start,
    cmd_status,
    cmd_stop,
    cmd_transcribe,
    main,
)


# ---------------------------------------------------------------------------
# _setup_logging
# ---------------------------------------------------------------------------


class TestSetupLogging:
    @patch("whisper_dictation.cli.logging.basicConfig")
    def test_default_info_level(self, mock_config):
        import logging
        _setup_logging(verbose=False)
        mock_config.assert_called_once()
        assert mock_config.call_args[1]["level"] == logging.INFO

    @patch("whisper_dictation.cli.logging.basicConfig")
    def test_verbose_debug_level(self, mock_config):
        import logging
        _setup_logging(verbose=True)
        assert mock_config.call_args[1]["level"] == logging.DEBUG


# ---------------------------------------------------------------------------
# PID management
# ---------------------------------------------------------------------------


class TestPidManagement:
    def test_write_and_read_pid(self, tmp_path):
        pid_file = tmp_path / "daemon.pid"
        with (
            patch("whisper_dictation.cli.CONFIG_DIR", tmp_path),
            patch("whisper_dictation.cli.PID_FILE", pid_file),
        ):
            _write_pid()
            assert pid_file.exists()
            assert int(pid_file.read_text()) == os.getpid()

            pid = _read_pid()
            assert pid == os.getpid()

    def test_read_pid_no_file(self, tmp_path):
        pid_file = tmp_path / "daemon.pid"
        with patch("whisper_dictation.cli.PID_FILE", pid_file):
            assert _read_pid() is None

    def test_read_pid_stale_process(self, tmp_path):
        pid_file = tmp_path / "daemon.pid"
        pid_file.write_text("999999999")  # nonexistent PID
        with patch("whisper_dictation.cli.PID_FILE", pid_file):
            result = _read_pid()
        assert result is None
        assert not pid_file.exists()

    def test_read_pid_invalid_content(self, tmp_path):
        pid_file = tmp_path / "daemon.pid"
        pid_file.write_text("not_a_number")
        with patch("whisper_dictation.cli.PID_FILE", pid_file):
            result = _read_pid()
        assert result is None

    def test_cleanup_pid(self, tmp_path):
        pid_file = tmp_path / "daemon.pid"
        state_file = tmp_path / "state.json"
        pid_file.write_text("123")
        state_file.write_text("{}")

        with (
            patch("whisper_dictation.cli.PID_FILE", pid_file),
            patch("whisper_dictation.cli.STATE_FILE", state_file),
        ):
            _cleanup_pid()

        assert not pid_file.exists()
        assert not state_file.exists()

    def test_cleanup_pid_missing_files(self, tmp_path):
        pid_file = tmp_path / "daemon.pid"
        state_file = tmp_path / "state.json"
        with (
            patch("whisper_dictation.cli.PID_FILE", pid_file),
            patch("whisper_dictation.cli.STATE_FILE", state_file),
        ):
            # Should not raise
            _cleanup_pid()


# ---------------------------------------------------------------------------
# cmd_stop
# ---------------------------------------------------------------------------


class TestCmdStop:
    def test_no_daemon_running(self, tmp_path, capsys):
        pid_file = tmp_path / "daemon.pid"
        with patch("whisper_dictation.cli.PID_FILE", pid_file):
            cmd_stop(MagicMock())
        captured = capsys.readouterr()
        assert "No daemon running" in captured.out

    def test_stop_running_daemon(self, tmp_path, capsys):
        pid_file = tmp_path / "daemon.pid"
        state_file = tmp_path / "state.json"
        pid_file.write_text(str(os.getpid()))

        with (
            patch("whisper_dictation.cli.PID_FILE", pid_file),
            patch("whisper_dictation.cli.STATE_FILE", state_file),
            patch("whisper_dictation.cli._read_pid", return_value=os.getpid()),
            patch("os.kill") as mock_kill,
        ):
            cmd_stop(MagicMock())
            mock_kill.assert_called_once_with(os.getpid(), signal.SIGTERM)

    def test_stop_dead_process(self, tmp_path, capsys):
        pid_file = tmp_path / "daemon.pid"
        state_file = tmp_path / "state.json"

        with (
            patch("whisper_dictation.cli.PID_FILE", pid_file),
            patch("whisper_dictation.cli.STATE_FILE", state_file),
            patch("whisper_dictation.cli._read_pid", return_value=99999),
            patch("os.kill", side_effect=ProcessLookupError),
        ):
            cmd_stop(MagicMock())
        captured = capsys.readouterr()
        assert "not found" in captured.out


# ---------------------------------------------------------------------------
# cmd_status
# ---------------------------------------------------------------------------


class TestCmdStatus:
    def test_status_stopped(self, tmp_path, capsys):
        pid_file = tmp_path / "daemon.pid"
        with patch("whisper_dictation.cli.PID_FILE", pid_file):
            cmd_status(MagicMock())
        captured = capsys.readouterr()
        assert "stopped" in captured.out

    def test_status_running(self, tmp_path, capsys):
        pid_file = tmp_path / "daemon.pid"
        state_file = tmp_path / "state.json"
        state_file.write_text(json.dumps({
            "mode": "toggle",
            "hotkey": "alt+v",
            "engine": "server",
            "server_url": "http://localhost:10300",
        }))

        with (
            patch("whisper_dictation.cli.PID_FILE", pid_file),
            patch("whisper_dictation.cli.STATE_FILE", state_file),
            patch("whisper_dictation.cli._read_pid", return_value=12345),
        ):
            cmd_status(MagicMock())

        captured = capsys.readouterr()
        assert "running" in captured.out
        assert "12345" in captured.out
        assert "toggle" in captured.out
        assert "alt+v" in captured.out

    def test_status_running_no_state_file(self, tmp_path, capsys):
        pid_file = tmp_path / "daemon.pid"
        state_file = tmp_path / "state.json"

        with (
            patch("whisper_dictation.cli.PID_FILE", pid_file),
            patch("whisper_dictation.cli.STATE_FILE", state_file),
            patch("whisper_dictation.cli._read_pid", return_value=12345),
        ):
            cmd_status(MagicMock())

        captured = capsys.readouterr()
        assert "running" in captured.out


# ---------------------------------------------------------------------------
# cmd_devices
# ---------------------------------------------------------------------------


class TestCmdDevices:
    @patch("whisper_dictation.audio.list_devices")
    def test_lists_devices(self, mock_list, capsys):
        mock_list.return_value = [
            {"index": 0, "name": "Built-in Mic", "channels": 2, "sample_rate": 44100.0},
            {"index": 3, "name": "USB Mic", "channels": 1, "sample_rate": 16000.0},
        ]
        cmd_devices(MagicMock())

        captured = capsys.readouterr()
        assert "Built-in Mic" in captured.out
        assert "USB Mic" in captured.out

    @patch("whisper_dictation.audio.list_devices")
    def test_no_devices(self, mock_list, capsys):
        mock_list.return_value = []
        cmd_devices(MagicMock())
        captured = capsys.readouterr()
        assert "No audio input devices found" in captured.out


# ---------------------------------------------------------------------------
# cmd_start
# ---------------------------------------------------------------------------


class TestCmdStart:
    def test_start_already_running(self, tmp_path):
        with patch("whisper_dictation.cli._read_pid", return_value=12345):
            args = MagicMock()
            args.config = None
            args.mode = None
            args.hotkey = None
            args.engine = None
            args.server_url = None

            with pytest.raises(SystemExit):
                cmd_start(args)

    def test_start_runtime_error(self, tmp_path):
        from whisper_dictation.config import Config

        mock_daemon = MagicMock()
        mock_daemon.start.side_effect = RuntimeError("engine down")

        args = MagicMock()
        args.config = None
        args.mode = None
        args.hotkey = None
        args.engine = None
        args.server_url = None

        with (
            patch("whisper_dictation.cli._read_pid", return_value=None),
            patch("whisper_dictation.cli.load_config", return_value=Config()),
            patch("whisper_dictation.daemon.DictationDaemon", return_value=mock_daemon),
            patch("whisper_dictation.cli._write_pid"),
            patch("whisper_dictation.cli._cleanup_pid"),
            patch("whisper_dictation.cli.CONFIG_DIR", tmp_path),
            patch("whisper_dictation.cli.STATE_FILE", tmp_path / "state.json"),
            patch("signal.signal"),
            pytest.raises(SystemExit),
        ):
            cmd_start(args)


# ---------------------------------------------------------------------------
# cmd_transcribe
# ---------------------------------------------------------------------------


class TestCmdTranscribe:
    def test_no_file_or_record(self, tmp_path):
        args = MagicMock()
        args.config = None
        args.server_url = None
        args.engine = None
        args.file = None
        args.record = None

        with pytest.raises(SystemExit):
            cmd_transcribe(args)

    def test_transcribe_file(self, tmp_path):
        import wave
        import numpy as np

        from whisper_dictation.config import Config

        # Create a test WAV file
        wav_path = tmp_path / "test.wav"
        audio = np.zeros(16000, dtype=np.int16)
        with wave.open(str(wav_path), "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(16000)
            w.writeframes(audio.tobytes())

        mock_engine = MagicMock()
        mock_engine.transcribe.return_value = "hello"

        args = MagicMock()
        args.config = None
        args.server_url = None
        args.engine = None
        args.file = str(wav_path)
        args.record = None

        with (
            patch("whisper_dictation.cli.load_config", return_value=Config()),
            patch("whisper_dictation.engine.server.ServerEngine", return_value=mock_engine),
        ):
            cmd_transcribe(args)
        mock_engine.transcribe.assert_called_once()

    def test_transcribe_file_no_speech(self, tmp_path):
        import wave
        import numpy as np

        from whisper_dictation.config import Config

        wav_path = tmp_path / "test.wav"
        audio = np.zeros(16000, dtype=np.int16)
        with wave.open(str(wav_path), "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(16000)
            w.writeframes(audio.tobytes())

        mock_engine = MagicMock()
        mock_engine.transcribe.return_value = ""

        args = MagicMock()
        args.config = None
        args.server_url = None
        args.engine = None
        args.file = str(wav_path)
        args.record = None

        with (
            patch("whisper_dictation.cli.load_config", return_value=Config()),
            patch("whisper_dictation.engine.server.ServerEngine", return_value=mock_engine),
            pytest.raises(SystemExit),
        ):
            cmd_transcribe(args)


# ---------------------------------------------------------------------------
# main() arg parsing
# ---------------------------------------------------------------------------


class TestMain:
    def test_no_command_shows_help(self, capsys):
        with patch("sys.argv", ["prog"]):
            with pytest.raises(SystemExit) as exc_info:
                main()
            assert exc_info.value.code == 0

    def test_verbose_flag(self):
        with (
            patch("sys.argv", ["prog", "-v", "status"]),
            patch("whisper_dictation.cli.cmd_status") as mock_status,
            patch("whisper_dictation.cli._setup_logging") as mock_logging,
        ):
            main()
            mock_logging.assert_called_once_with(True)

    def test_start_subcommand_with_args(self):
        with (
            patch("sys.argv", [
                "prog", "start",
                "--mode", "hold",
                "--hotkey", "ctrl+d",
                "--engine", "local",
                "--server-url", "http://x:9000",
            ]),
            patch("whisper_dictation.cli.cmd_start") as mock_start,
            patch("whisper_dictation.cli._setup_logging"),
        ):
            main()
            mock_start.assert_called_once()
            args = mock_start.call_args[0][0]
            assert args.mode == "hold"
            assert args.hotkey == "ctrl+d"
            assert args.engine == "local"
            assert args.server_url == "http://x:9000"

    def test_stop_subcommand(self):
        with (
            patch("sys.argv", ["prog", "stop"]),
            patch("whisper_dictation.cli.cmd_stop") as mock_stop,
            patch("whisper_dictation.cli._setup_logging"),
        ):
            main()
            mock_stop.assert_called_once()

    def test_devices_subcommand(self):
        with (
            patch("sys.argv", ["prog", "devices"]),
            patch("whisper_dictation.cli.cmd_devices") as mock_devices,
            patch("whisper_dictation.cli._setup_logging"),
        ):
            main()
            mock_devices.assert_called_once()

    def test_transcribe_with_file(self):
        with (
            patch("sys.argv", ["prog", "transcribe", "audio.wav"]),
            patch("whisper_dictation.cli.cmd_transcribe") as mock_trans,
            patch("whisper_dictation.cli._setup_logging"),
        ):
            main()
            args = mock_trans.call_args[0][0]
            assert args.file == "audio.wav"

    def test_transcribe_with_record(self):
        with (
            patch("sys.argv", ["prog", "transcribe", "--record", "5.0"]),
            patch("whisper_dictation.cli.cmd_transcribe") as mock_trans,
            patch("whisper_dictation.cli._setup_logging"),
        ):
            main()
            args = mock_trans.call_args[0][0]
            assert args.record == 5.0
