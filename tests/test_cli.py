"""Tests for whisper_dictation.cli — argument parsing, subcommands."""

from __future__ import annotations

import json
import os
import signal
from unittest.mock import MagicMock, patch

import pytest

from whisper_dictation.cli import (
    _cleanup_pid,
    _daemonize,
    _read_pid,
    _setup_logging,
    _write_pid,
    cmd_config,
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
        import whisper_dictation.cli as cli_mod

        pid_file = tmp_path / "daemon.pid"
        try:
            with (
                patch("whisper_dictation.cli.CONFIG_DIR", tmp_path),
                patch("whisper_dictation.cli.PID_FILE", pid_file),
            ):
                _write_pid()
                assert pid_file.exists()
                assert int(pid_file.read_text()) == os.getpid()

                pid = _read_pid()
                assert pid == os.getpid()
        finally:
            if cli_mod._pid_lock_fd is not None:
                os.close(cli_mod._pid_lock_fd)
                cli_mod._pid_lock_fd = None

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

    def test_read_pid_permission_error_means_alive(self, tmp_path):
        """PermissionError from os.kill means the process IS alive (different user)."""
        pid_file = tmp_path / "daemon.pid"
        pid_file.write_text("12345")
        with (
            patch("whisper_dictation.cli.PID_FILE", pid_file),
            patch("os.kill", side_effect=PermissionError("not owner")),
        ):
            result = _read_pid()
        assert result == 12345
        assert pid_file.exists()  # PID file should NOT be deleted

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

    def test_write_pid_lock_already_held(self, tmp_path):
        """Test _write_pid exits when flock raises OSError (lock held by another)."""
        pid_file = tmp_path / "daemon.pid"
        mock_fcntl = MagicMock()
        mock_fcntl.LOCK_EX = 2
        mock_fcntl.LOCK_NB = 4
        mock_fcntl.flock.side_effect = OSError("lock held")

        with (
            patch("whisper_dictation.cli.CONFIG_DIR", tmp_path),
            patch("whisper_dictation.cli.PID_FILE", pid_file),
            patch.dict("sys.modules", {"fcntl": mock_fcntl}),
            patch("os.open", return_value=42),
            patch("os.close") as mock_close,
            pytest.raises(SystemExit) as exc_info,
        ):
            _write_pid()

        assert exc_info.value.code == 1
        mock_close.assert_called_once_with(42)

    def test_write_pid_no_fcntl_fallback(self, tmp_path):
        """Test _write_pid falls back to simple write when fcntl is unavailable."""
        pid_file = tmp_path / "daemon.pid"

        # Make fcntl import raise ImportError inside _write_pid
        import builtins

        real_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name == "fcntl":
                raise ImportError("no fcntl")
            return real_import(name, *args, **kwargs)

        with (
            patch("whisper_dictation.cli.CONFIG_DIR", tmp_path),
            patch("whisper_dictation.cli.PID_FILE", pid_file),
            patch("builtins.__import__", side_effect=fake_import),
        ):
            _write_pid()

        assert pid_file.exists()
        assert int(pid_file.read_text()) == os.getpid()

    def test_cleanup_pid_closes_lock_fd(self, tmp_path):
        """Test _cleanup_pid closes the lock fd when _pid_lock_fd is set."""
        import whisper_dictation.cli as cli_mod

        pid_file = tmp_path / "daemon.pid"
        state_file = tmp_path / "state.json"
        pid_file.write_text("123")

        cli_mod._pid_lock_fd = 42
        try:
            with (
                patch("whisper_dictation.cli.PID_FILE", pid_file),
                patch("whisper_dictation.cli.STATE_FILE", state_file),
                patch("os.close") as mock_close,
            ):
                _cleanup_pid()

            mock_close.assert_called_once_with(42)
            assert cli_mod._pid_lock_fd is None
        finally:
            cli_mod._pid_lock_fd = None

    def test_cleanup_pid_close_oserror_suppressed(self, tmp_path):
        """Test _cleanup_pid suppresses OSError when closing lock fd."""
        import whisper_dictation.cli as cli_mod

        pid_file = tmp_path / "daemon.pid"
        state_file = tmp_path / "state.json"
        pid_file.write_text("123")

        cli_mod._pid_lock_fd = 42
        try:
            with (
                patch("whisper_dictation.cli.PID_FILE", pid_file),
                patch("whisper_dictation.cli.STATE_FILE", state_file),
                patch("os.close", side_effect=OSError("bad fd")),
            ):
                # Should not raise
                _cleanup_pid()

            assert cli_mod._pid_lock_fd is None
        finally:
            cli_mod._pid_lock_fd = None


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

        # First call: SIGTERM, second call: os.kill(pid, 0) → ProcessLookupError (exited)
        kill_calls = [None, ProcessLookupError]

        def mock_kill_side_effect(pid, sig):
            effect = kill_calls.pop(0)
            if effect is not None:
                raise effect

        with (
            patch("whisper_dictation.cli.PID_FILE", pid_file),
            patch("whisper_dictation.cli.STATE_FILE", state_file),
            patch("whisper_dictation.cli._read_pid", return_value=os.getpid()),
            patch("os.kill", side_effect=mock_kill_side_effect) as mock_kill,
            patch("time.sleep"),
        ):
            cmd_stop(MagicMock())
            # First call is SIGTERM, second is liveness check
            assert mock_kill.call_args_list[0] == ((os.getpid(), signal.SIGTERM),)
            assert mock_kill.call_args_list[1] == ((os.getpid(), 0),)

    def test_stop_waits_then_sigkill(self, tmp_path, capsys):
        """Process doesn't exit after SIGTERM — falls back to SIGKILL."""
        pid_file = tmp_path / "daemon.pid"
        state_file = tmp_path / "state.json"

        call_count = [0]

        def mock_kill_side_effect(pid, sig):
            call_count[0] += 1
            if call_count[0] == 1:
                return  # SIGTERM succeeds
            if sig == signal.SIGKILL:
                return  # SIGKILL succeeds
            # All liveness checks (sig=0) succeed — process won't die
            return

        with (
            patch("whisper_dictation.cli.PID_FILE", pid_file),
            patch("whisper_dictation.cli.STATE_FILE", state_file),
            patch("whisper_dictation.cli._read_pid", return_value=99999),
            patch("os.kill", side_effect=mock_kill_side_effect),
            patch("time.sleep"),
        ):
            cmd_stop(MagicMock())
        captured = capsys.readouterr()
        assert "Stopped" in captured.out

    def test_stop_sigkill_race_process_already_dead(self, tmp_path, capsys):
        """Process dies between last liveness check and SIGKILL."""
        pid_file = tmp_path / "daemon.pid"
        state_file = tmp_path / "state.json"

        call_count = [0]

        def mock_kill_side_effect(pid, sig):
            call_count[0] += 1
            if call_count[0] == 1:
                return  # SIGTERM succeeds
            if sig == signal.SIGKILL:
                raise ProcessLookupError  # died just before SIGKILL
            # All liveness checks (sig=0) succeed — process appears alive
            return

        with (
            patch("whisper_dictation.cli.PID_FILE", pid_file),
            patch("whisper_dictation.cli.STATE_FILE", state_file),
            patch("whisper_dictation.cli._read_pid", return_value=99999),
            patch("os.kill", side_effect=mock_kill_side_effect),
            patch("time.sleep"),
        ):
            cmd_stop(MagicMock())
        captured = capsys.readouterr()
        assert "Stopped" in captured.out

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
        state_file.write_text(
            json.dumps(
                {
                    "mode": "toggle",
                    "hotkey": "alt+v",
                    "engine": "server",
                    "server_url": "http://localhost:9090",
                }
            )
        )

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

    def test_status_running_corrupt_state_file(self, tmp_path, capsys):
        """Test status command when state file has corrupt JSON (covers except Exception: pass)."""
        pid_file = tmp_path / "daemon.pid"
        state_file = tmp_path / "state.json"
        state_file.write_text("not valid json{{{")

        with (
            patch("whisper_dictation.cli.PID_FILE", pid_file),
            patch("whisper_dictation.cli.STATE_FILE", state_file),
            patch("whisper_dictation.cli._read_pid", return_value=12345),
        ):
            cmd_status(MagicMock())

        captured = capsys.readouterr()
        assert "running" in captured.out
        assert "12345" in captured.out

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
            args.background = False

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
        args.background = False

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

    def test_start_with_mode_override(self, tmp_path):
        from whisper_dictation.config import Config

        mock_daemon = MagicMock()

        args = MagicMock()
        args.config = None
        args.mode = "hold"
        args.hotkey = None
        args.engine = None
        args.server_url = None
        args.background = False

        with (
            patch("whisper_dictation.cli._read_pid", return_value=None),
            patch("whisper_dictation.cli.load_config", return_value=Config()),
            patch("whisper_dictation.daemon.DictationDaemon", return_value=mock_daemon) as mock_cls,
            patch("whisper_dictation.cli._write_pid"),
            patch("whisper_dictation.cli._cleanup_pid"),
            patch("whisper_dictation.cli.CONFIG_DIR", tmp_path),
            patch("whisper_dictation.cli.STATE_FILE", tmp_path / "state.json"),
            patch("signal.signal"),
        ):
            cmd_start(args)

        # Verify mode was overridden
        config_passed = mock_cls.call_args[0][0]
        assert config_passed.hotkey.mode == "hold"

    def test_start_with_hotkey_override(self, tmp_path):
        from whisper_dictation.config import Config

        mock_daemon = MagicMock()

        args = MagicMock()
        args.config = None
        args.mode = None
        args.hotkey = "ctrl+d"
        args.engine = None
        args.server_url = None
        args.background = False

        with (
            patch("whisper_dictation.cli._read_pid", return_value=None),
            patch("whisper_dictation.cli.load_config", return_value=Config()),
            patch("whisper_dictation.daemon.DictationDaemon", return_value=mock_daemon) as mock_cls,
            patch("whisper_dictation.cli._write_pid"),
            patch("whisper_dictation.cli._cleanup_pid"),
            patch("whisper_dictation.cli.CONFIG_DIR", tmp_path),
            patch("whisper_dictation.cli.STATE_FILE", tmp_path / "state.json"),
            patch("signal.signal"),
        ):
            cmd_start(args)

        config_passed = mock_cls.call_args[0][0]
        assert config_passed.hotkey.binding == "ctrl+d"

    def test_start_with_engine_override(self, tmp_path):
        from whisper_dictation.config import Config

        mock_daemon = MagicMock()

        args = MagicMock()
        args.config = None
        args.mode = None
        args.hotkey = None
        args.engine = "local"
        args.server_url = None
        args.background = False

        with (
            patch("whisper_dictation.cli._read_pid", return_value=None),
            patch("whisper_dictation.cli.load_config", return_value=Config()),
            patch("whisper_dictation.daemon.DictationDaemon", return_value=mock_daemon) as mock_cls,
            patch("whisper_dictation.cli._write_pid"),
            patch("whisper_dictation.cli._cleanup_pid"),
            patch("whisper_dictation.cli.CONFIG_DIR", tmp_path),
            patch("whisper_dictation.cli.STATE_FILE", tmp_path / "state.json"),
            patch("signal.signal"),
        ):
            cmd_start(args)

        config_passed = mock_cls.call_args[0][0]
        assert config_passed.engine.type == "local"

    def test_start_with_server_url_override(self, tmp_path):
        from whisper_dictation.config import Config

        mock_daemon = MagicMock()

        args = MagicMock()
        args.config = None
        args.mode = None
        args.hotkey = None
        args.engine = None
        args.server_url = "http://custom:9000"
        args.background = False

        with (
            patch("whisper_dictation.cli._read_pid", return_value=None),
            patch("whisper_dictation.cli.load_config", return_value=Config()),
            patch("whisper_dictation.daemon.DictationDaemon", return_value=mock_daemon) as mock_cls,
            patch("whisper_dictation.cli._write_pid"),
            patch("whisper_dictation.cli._cleanup_pid"),
            patch("whisper_dictation.cli.CONFIG_DIR", tmp_path),
            patch("whisper_dictation.cli.STATE_FILE", tmp_path / "state.json"),
            patch("signal.signal"),
        ):
            cmd_start(args)

        config_passed = mock_cls.call_args[0][0]
        assert config_passed.server.url == "http://custom:9000"

    def test_start_successful_daemon_wait(self, tmp_path):
        """Test cmd_start successful path where daemon.wait() returns normally."""
        from whisper_dictation.config import Config

        mock_daemon = MagicMock()

        args = MagicMock()
        args.config = None
        args.mode = None
        args.hotkey = None
        args.engine = None
        args.server_url = None
        args.background = False

        with (
            patch("whisper_dictation.cli._read_pid", return_value=None),
            patch("whisper_dictation.cli.load_config", return_value=Config()),
            patch("whisper_dictation.daemon.DictationDaemon", return_value=mock_daemon),
            patch("whisper_dictation.cli._write_pid"),
            patch("whisper_dictation.cli._cleanup_pid") as mock_cleanup,
            patch("whisper_dictation.cli.CONFIG_DIR", tmp_path),
            patch("whisper_dictation.cli.STATE_FILE", tmp_path / "state.json"),
            patch("signal.signal"),
        ):
            cmd_start(args)

        mock_daemon.start.assert_called_once()
        mock_daemon.wait.assert_called_once()
        mock_cleanup.assert_called()

    def test_start_signal_handler_calls_shutdown(self, tmp_path):
        """Test that the registered signal handler calls daemon.stop and sys.exit."""
        from whisper_dictation.config import Config

        mock_daemon = MagicMock()
        registered_handlers = {}

        def capture_signal(sig, handler):
            registered_handlers[sig] = handler

        args = MagicMock()
        args.config = None
        args.mode = None
        args.hotkey = None
        args.engine = None
        args.server_url = None
        args.background = False

        with (
            patch("whisper_dictation.cli._read_pid", return_value=None),
            patch("whisper_dictation.cli.load_config", return_value=Config()),
            patch("whisper_dictation.daemon.DictationDaemon", return_value=mock_daemon),
            patch("whisper_dictation.cli._write_pid"),
            patch("whisper_dictation.cli._cleanup_pid"),
            patch("whisper_dictation.cli.CONFIG_DIR", tmp_path),
            patch("whisper_dictation.cli.STATE_FILE", tmp_path / "state.json"),
            patch("signal.signal", side_effect=capture_signal),
        ):
            cmd_start(args)

        # Verify SIGTERM handler was registered
        assert signal.SIGTERM in registered_handlers
        handler = registered_handlers[signal.SIGTERM]

        # Call the handler — should call request_stop() (signal-safe)
        mock_daemon.reset_mock()
        handler(signal.SIGTERM, None)
        mock_daemon.request_stop.assert_called_once()

    def test_start_cleans_up_pid_when_state_file_write_fails(self, tmp_path):
        from whisper_dictation.config import Config

        args = MagicMock()
        args.config = None
        args.mode = None
        args.hotkey = None
        args.engine = None
        args.server_url = None
        args.streaming = False
        args.background = False

        with (
            patch("whisper_dictation.cli._read_pid", return_value=None),
            patch("whisper_dictation.cli.load_config", return_value=Config()),
            patch("whisper_dictation.cli._write_pid"),
            patch("whisper_dictation.cli._cleanup_pid") as mock_cleanup,
            patch("whisper_dictation.cli.CONFIG_DIR", tmp_path),
            patch("whisper_dictation.cli.STATE_FILE", tmp_path / "state.json"),
            patch("os.open", side_effect=OSError("disk full")),
            pytest.raises(OSError, match="disk full"),
        ):
            cmd_start(args)

        mock_cleanup.assert_called_once()

    def test_start_background_no_fork_exits(self, tmp_path, capsys):
        """--background on platforms without os.fork should print error and exit."""
        from whisper_dictation.config import Config

        args = MagicMock()
        args.config = None
        args.mode = None
        args.hotkey = None
        args.engine = None
        args.server_url = None
        args.streaming = False
        args.background = True
        args.verbose = False

        # Temporarily remove os.fork to simulate Windows/WASI
        original_fork = os.fork
        try:
            del os.fork
            with (
                patch("whisper_dictation.cli._read_pid", return_value=None),
                patch("whisper_dictation.cli.load_config", return_value=Config()),
                pytest.raises(SystemExit),
            ):
                cmd_start(args)
        finally:
            os.fork = original_fork

        captured = capsys.readouterr()
        assert "--background is not supported" in captured.err

    def test_start_background_calls_daemonize(self, tmp_path):
        """--background on Unix should call _daemonize and re-init logging."""
        from whisper_dictation.config import Config

        mock_daemon = MagicMock()

        args = MagicMock()
        args.config = None
        args.mode = None
        args.hotkey = None
        args.engine = None
        args.server_url = None
        args.streaming = False
        args.background = True
        args.verbose = False

        with (
            patch("whisper_dictation.cli._read_pid", return_value=None),
            patch("whisper_dictation.cli.load_config", return_value=Config()),
            patch("whisper_dictation.daemon.DictationDaemon", return_value=mock_daemon),
            patch("whisper_dictation.cli._write_pid"),
            patch("whisper_dictation.cli._cleanup_pid"),
            patch("whisper_dictation.cli.CONFIG_DIR", tmp_path),
            patch("whisper_dictation.cli.STATE_FILE", tmp_path / "state.json"),
            patch("signal.signal"),
            patch("whisper_dictation.cli._daemonize") as mock_daemonize,
        ):
            cmd_start(args)

        mock_daemonize.assert_called_once()
        mock_daemon.start.assert_called_once()

    def test_start_foreground_does_not_daemonize(self, tmp_path):
        """Without --background, _daemonize should not be called."""
        from whisper_dictation.config import Config

        mock_daemon = MagicMock()

        args = MagicMock()
        args.config = None
        args.mode = None
        args.hotkey = None
        args.engine = None
        args.server_url = None
        args.streaming = False
        args.background = False
        args.verbose = False

        with (
            patch("whisper_dictation.cli._read_pid", return_value=None),
            patch("whisper_dictation.cli.load_config", return_value=Config()),
            patch("whisper_dictation.daemon.DictationDaemon", return_value=mock_daemon),
            patch("whisper_dictation.cli._write_pid"),
            patch("whisper_dictation.cli._cleanup_pid"),
            patch("whisper_dictation.cli.CONFIG_DIR", tmp_path),
            patch("whisper_dictation.cli.STATE_FILE", tmp_path / "state.json"),
            patch("signal.signal"),
            patch("whisper_dictation.cli._daemonize") as mock_daemonize,
        ):
            cmd_start(args)

        mock_daemonize.assert_not_called()


# ---------------------------------------------------------------------------
# _daemonize
# ---------------------------------------------------------------------------


class TestDaemonize:
    def test_first_fork_parent_exits(self):
        """Parent process (fork returns child pid) should call os._exit(0)."""
        with (
            patch("os.fork", return_value=42),
            patch("os._exit", side_effect=SystemExit(0)) as mock_exit,
            patch("whisper_dictation.cli.LOG_FILE", "/tmp/test.log"),
            pytest.raises(SystemExit),
        ):
            _daemonize()
        mock_exit.assert_called_once_with(0)

    def test_second_fork_parent_exits(self):
        """After setsid, second fork parent should exit."""
        fork_calls = [0]

        def mock_fork():
            fork_calls[0] += 1
            if fork_calls[0] == 1:
                return 0  # first fork: child continues
            return 42  # second fork: parent exits

        with (
            patch("os.fork", side_effect=mock_fork),
            patch("os.setsid"),
            patch("os._exit", side_effect=SystemExit(0)) as mock_exit,
            pytest.raises(SystemExit),
        ):
            _daemonize()
        mock_exit.assert_called_once_with(0)

    def test_child_redirects_stdio(self, tmp_path):
        """Final child should redirect stdin/stdout/stderr."""
        log_file = tmp_path / "daemon.log"
        fd_sequence = [10, 20]  # devnull=10, log=20

        def mock_fork():
            return 0  # always child

        with (
            patch("os.fork", side_effect=mock_fork),
            patch("os.setsid"),
            patch("os.chdir"),
            patch("os.closerange"),
            patch("os.open", side_effect=fd_sequence),
            patch("os.dup2") as mock_dup2,
            patch("os.close"),
            patch("whisper_dictation.cli.CONFIG_DIR", tmp_path),
            patch("whisper_dictation.cli.LOG_FILE", log_file),
        ):
            _daemonize()

        # stdin redirected to /dev/null (fd 10)
        mock_dup2.assert_any_call(10, 0)
        # stdout and stderr redirected to log file (fd 20)
        mock_dup2.assert_any_call(20, 1)
        mock_dup2.assert_any_call(20, 2)

    def test_child_calls_chdir_and_closerange(self, tmp_path):
        """Final child should chdir('/') and close inherited fds."""
        log_file = tmp_path / "daemon.log"

        with (
            patch("os.fork", return_value=0),
            patch("os.setsid"),
            patch("os.chdir") as mock_chdir,
            patch("os.closerange") as mock_closerange,
            patch("os.sysconf", return_value=256),
            patch("os.open", return_value=99),
            patch("os.dup2"),
            patch("os.close"),
            patch("whisper_dictation.cli.CONFIG_DIR", tmp_path),
            patch("whisper_dictation.cli.LOG_FILE", log_file),
        ):
            _daemonize()

        mock_chdir.assert_called_once_with("/")
        mock_closerange.assert_called_once_with(3, 256)

    def test_closerange_fallback_when_sysconf_unavailable(self, tmp_path):
        """Falls back to 1024 when os.sysconf is unavailable."""
        log_file = tmp_path / "daemon.log"

        with (
            patch("os.fork", return_value=0),
            patch("os.setsid"),
            patch("os.chdir"),
            patch("os.closerange") as mock_closerange,
            patch("os.sysconf", side_effect=AttributeError),
            patch("os.open", return_value=99),
            patch("os.dup2"),
            patch("os.close"),
            patch("whisper_dictation.cli.CONFIG_DIR", tmp_path),
            patch("whisper_dictation.cli.LOG_FILE", log_file),
        ):
            _daemonize()

        mock_closerange.assert_called_once_with(3, 1024)

    def test_no_fork_raises_not_implemented(self):
        """On platforms without os.fork, raise NotImplementedError."""
        # Temporarily remove os.fork to simulate Windows
        original_fork = os.fork
        try:
            del os.fork
            with pytest.raises(NotImplementedError, match="os.fork"):
                _daemonize()
        finally:
            os.fork = original_fork


# ---------------------------------------------------------------------------
# cmd_status — log file display
# ---------------------------------------------------------------------------


class TestCmdStatusLogFile:
    def test_status_shows_log_file_when_exists(self, tmp_path, capsys):
        log_file = tmp_path / "daemon.log"
        log_file.write_text("some log output")
        state_file = tmp_path / "state.json"

        with (
            patch("whisper_dictation.cli._read_pid", return_value=12345),
            patch("whisper_dictation.cli.STATE_FILE", state_file),
            patch("whisper_dictation.cli.LOG_FILE", log_file),
            patch("whisper_dictation.cli.PID_FILE", tmp_path / "daemon.pid"),
        ):
            cmd_status(MagicMock())

        captured = capsys.readouterr()
        assert "Log:" in captured.out
        assert "daemon.log" in captured.out

    def test_status_no_log_file(self, tmp_path, capsys):
        log_file = tmp_path / "daemon.log"  # does not exist
        state_file = tmp_path / "state.json"

        with (
            patch("whisper_dictation.cli._read_pid", return_value=12345),
            patch("whisper_dictation.cli.STATE_FILE", state_file),
            patch("whisper_dictation.cli.LOG_FILE", log_file),
            patch("whisper_dictation.cli.PID_FILE", tmp_path / "daemon.pid"),
        ):
            cmd_status(MagicMock())

        captured = capsys.readouterr()
        assert "Log:" not in captured.out


# ---------------------------------------------------------------------------
# cmd_transcribe
# ---------------------------------------------------------------------------


def _make_silent_wav(tmp_path):
    """Create a silent WAV file for testing, return its path."""
    import wave

    import numpy as np

    wav_path = tmp_path / "test.wav"
    audio = np.zeros(16000, dtype=np.int16)
    with wave.open(str(wav_path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes(audio.tobytes())
    return wav_path


def _make_transcribe_args(*, file=None, record=None, engine=None, server_url=None):
    """Build a mock args namespace for cmd_transcribe."""
    args = MagicMock()
    args.config = None
    args.server_url = server_url
    args.engine = engine
    args.file = file
    args.record = record
    return args


class TestCmdTranscribe:
    def test_no_file_or_record(self):
        with pytest.raises(SystemExit):
            cmd_transcribe(_make_transcribe_args())

    def test_transcribe_file_not_found(self, tmp_path):
        args = _make_transcribe_args(file=str(tmp_path / "nonexistent.wav"))
        with pytest.raises(SystemExit):
            cmd_transcribe(args)

    def test_transcribe_record_invalid_duration(self):
        args = _make_transcribe_args(record=-1.0)
        with pytest.raises(SystemExit):
            cmd_transcribe(args)

    def test_transcribe_record_duration_too_long(self):
        args = _make_transcribe_args(record=91.0)
        with pytest.raises(SystemExit):
            cmd_transcribe(args)

    def test_transcribe_invalid_wav_file(self, tmp_path):
        """Test transcribe rejects non-WAV files with clear error."""
        bad_file = tmp_path / "notawav.wav"
        bad_file.write_bytes(b"this is not a WAV file")

        from whisper_dictation.config import Config

        mock_engine = MagicMock()
        with (
            patch("whisper_dictation.cli.load_config", return_value=Config()),
            patch("whisper_dictation.engine.create_engine", return_value=mock_engine),
            pytest.raises(SystemExit),
        ):
            cmd_transcribe(_make_transcribe_args(file=str(bad_file)))

    def test_transcribe_file(self, tmp_path):
        from whisper_dictation.config import Config

        wav_path = _make_silent_wav(tmp_path)
        mock_engine = MagicMock()
        mock_engine.transcribe.return_value = "hello"

        with (
            patch("whisper_dictation.cli.load_config", return_value=Config()),
            patch("whisper_dictation.engine.create_engine", return_value=mock_engine),
        ):
            cmd_transcribe(_make_transcribe_args(file=str(wav_path)))
        mock_engine.transcribe.assert_called_once()
        mock_engine.close.assert_called_once()

    def test_transcribe_with_record(self, capsys):
        import numpy as np

        from whisper_dictation.config import Config

        mock_engine = MagicMock()
        mock_engine.transcribe.return_value = "recorded speech"

        mock_sd = MagicMock()
        mock_sd.rec.return_value = np.zeros((16000, 1), dtype=np.float32)

        with (
            patch("whisper_dictation.cli.load_config", return_value=Config()),
            patch("whisper_dictation.engine.create_engine", return_value=mock_engine),
            patch.dict("sys.modules", {"sounddevice": mock_sd}),
        ):
            cmd_transcribe(_make_transcribe_args(record=1.0))

        captured = capsys.readouterr()
        assert "recorded speech" in captured.out
        mock_engine.close.assert_called_once()

    def test_transcribe_record_no_speech(self):
        import numpy as np

        from whisper_dictation.config import Config

        mock_engine = MagicMock()
        mock_engine.transcribe.return_value = ""

        mock_sd = MagicMock()
        mock_sd.rec.return_value = np.zeros((16000, 1), dtype=np.float32)

        with (
            patch("whisper_dictation.cli.load_config", return_value=Config()),
            patch("whisper_dictation.engine.create_engine", return_value=mock_engine),
            patch.dict("sys.modules", {"sounddevice": mock_sd}),
            pytest.raises(SystemExit),
        ):
            cmd_transcribe(_make_transcribe_args(record=1.0))

    def test_transcribe_with_server_url_override(self, tmp_path):
        from whisper_dictation.config import Config

        wav_path = _make_silent_wav(tmp_path)
        mock_engine = MagicMock()
        mock_engine.transcribe.return_value = "hello"

        with (
            patch("whisper_dictation.cli.load_config", return_value=Config()),
            patch(
                "whisper_dictation.engine.create_engine", return_value=mock_engine
            ) as mock_create,
        ):
            cmd_transcribe(
                _make_transcribe_args(file=str(wav_path), server_url="http://custom:9000")
            )
        call_config = mock_create.call_args[0][0]
        assert call_config.server.url == "http://custom:9000"

    def test_transcribe_with_engine_override(self, tmp_path):
        from whisper_dictation.config import Config

        wav_path = _make_silent_wav(tmp_path)
        mock_engine = MagicMock()
        mock_engine.transcribe.return_value = "hello"

        with (
            patch("whisper_dictation.cli.load_config", return_value=Config()),
            patch("whisper_dictation.engine.create_engine", return_value=mock_engine),
        ):
            cmd_transcribe(_make_transcribe_args(file=str(wav_path), engine="local"))
        mock_engine.transcribe.assert_called_once()

    def test_transcribe_file_no_speech(self, tmp_path):
        from whisper_dictation.config import Config

        wav_path = _make_silent_wav(tmp_path)
        mock_engine = MagicMock()
        mock_engine.transcribe.return_value = ""

        with (
            patch("whisper_dictation.cli.load_config", return_value=Config()),
            patch("whisper_dictation.engine.create_engine", return_value=mock_engine),
            pytest.raises(SystemExit),
        ):
            cmd_transcribe(_make_transcribe_args(file=str(wav_path)))
        mock_engine.close.assert_called_once()


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
            patch("whisper_dictation.cli.cmd_status"),
            patch("whisper_dictation.cli._setup_logging") as mock_logging,
        ):
            main()
            mock_logging.assert_called_once_with(True)

    def test_start_subcommand_with_args(self):
        with (
            patch(
                "sys.argv",
                [
                    "prog",
                    "start",
                    "--mode",
                    "hold",
                    "--hotkey",
                    "ctrl+d",
                    "--engine",
                    "local",
                    "--server-url",
                    "http://x:9000",
                ],
            ),
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
            assert args.background is False

    def test_start_background_flag(self):
        with (
            patch("sys.argv", ["prog", "start", "-b"]),
            patch("whisper_dictation.cli.cmd_start") as mock_start,
            patch("whisper_dictation.cli._setup_logging"),
        ):
            main()
            args = mock_start.call_args[0][0]
            assert args.background is True

    def test_start_background_long_flag(self):
        with (
            patch("sys.argv", ["prog", "start", "--background"]),
            patch("whisper_dictation.cli.cmd_start") as mock_start,
            patch("whisper_dictation.cli._setup_logging"),
        ):
            main()
            args = mock_start.call_args[0][0]
            assert args.background is True

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

    def test_config_subcommand(self):
        with (
            patch("sys.argv", ["prog", "config"]),
            patch("whisper_dictation.cli.cmd_config") as mock_config,
            patch("whisper_dictation.cli._setup_logging"),
        ):
            main()
            mock_config.assert_called_once()


# ---------------------------------------------------------------------------
# cmd_config
# ---------------------------------------------------------------------------


class TestCmdConfig:
    def test_generate_creates_file(self, tmp_path, capsys):
        config_dir = tmp_path / "config"
        config_file = config_dir / "config.toml"

        args = MagicMock()
        args.generate = True
        args.force = False
        args.path = False

        with (
            patch("whisper_dictation.cli.CONFIG_DIR", config_dir),
            patch("whisper_dictation.cli.CONFIG_FILE", config_file),
        ):
            cmd_config(args)

        assert config_file.exists()
        content = config_file.read_text()
        assert "[server]" in content
        assert "[hotkey]" in content
        captured = capsys.readouterr()
        assert "Config written to" in captured.out

    def test_generate_refuses_overwrite(self, tmp_path):
        config_file = tmp_path / "config.toml"
        config_file.write_text("existing")

        args = MagicMock()
        args.generate = True
        args.force = False
        args.path = False

        with (
            patch("whisper_dictation.cli.CONFIG_DIR", tmp_path),
            patch("whisper_dictation.cli.CONFIG_FILE", config_file),
            pytest.raises(SystemExit),
        ):
            cmd_config(args)

    def test_generate_force_overwrites(self, tmp_path, capsys):
        config_file = tmp_path / "config.toml"
        config_file.write_text("old")

        args = MagicMock()
        args.generate = True
        args.force = True
        args.path = False

        with (
            patch("whisper_dictation.cli.CONFIG_DIR", tmp_path),
            patch("whisper_dictation.cli.CONFIG_FILE", config_file),
        ):
            cmd_config(args)

        assert "[server]" in config_file.read_text()

    def test_path_prints_config_path(self, tmp_path, capsys):
        config_file = tmp_path / "config.toml"

        args = MagicMock()
        args.generate = False
        args.force = False
        args.path = True

        with patch("whisper_dictation.cli.CONFIG_FILE", config_file):
            cmd_config(args)

        captured = capsys.readouterr()
        assert str(config_file) in captured.out

    def test_show_displays_config(self, tmp_path, capsys):
        config_file = tmp_path / "config.toml"

        args = MagicMock()
        args.generate = False
        args.force = False
        args.path = False

        from whisper_dictation.config import Config

        with (
            patch("whisper_dictation.cli.CONFIG_FILE", config_file),
            patch("whisper_dictation.cli.load_config", return_value=Config()),
        ):
            cmd_config(args)

        captured = capsys.readouterr()
        assert "[server]" in captured.out
        assert "[hotkey]" in captured.out
        assert "[vad]" in captured.out
        assert "[audio]" in captured.out
        assert "[engine]" in captured.out
        assert "not found" in captured.out

    def test_show_with_existing_config(self, tmp_path, capsys):
        config_file = tmp_path / "config.toml"
        config_file.write_text("[server]\nurl = 'http://x'\n")

        args = MagicMock()
        args.generate = False
        args.force = False
        args.path = False

        from whisper_dictation.config import Config

        with (
            patch("whisper_dictation.cli.CONFIG_FILE", config_file),
            patch("whisper_dictation.cli.load_config", return_value=Config()),
        ):
            cmd_config(args)

        captured = capsys.readouterr()
        assert "exists" in captured.out

    def test_show_displays_prompt_and_hotwords(self, tmp_path, capsys):
        config_file = tmp_path / "config.toml"

        args = MagicMock()
        args.generate = False
        args.force = False
        args.path = False

        from whisper_dictation.config import Config, ServerConfig

        cfg = Config(server=ServerConfig(prompt="dictation mode", hotwords="Claude,Whisper"))
        with (
            patch("whisper_dictation.cli.CONFIG_FILE", config_file),
            patch("whisper_dictation.cli.load_config", return_value=cfg),
        ):
            cmd_config(args)

        captured = capsys.readouterr()
        assert "dictation mode" in captured.out
        assert "Claude,Whisper" in captured.out
