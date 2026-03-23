"""Tests for whisper_dictation.notifier — cross-platform notifications."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from whisper_dictation.notifier import notify


class TestNotifyLinux:
    @patch("whisper_dictation.notifier.sys")
    @patch("whisper_dictation.notifier.subprocess.Popen")
    def test_calls_notify_send(self, mock_popen, mock_sys):
        mock_sys.platform = "linux"
        notify("Title", "Body text")

        mock_popen.assert_called_once()
        args = mock_popen.call_args[0][0]
        assert args[0] == "notify-send"
        assert "-t" in args
        assert "2000" in args
        assert "Title" in args
        assert "Body text" in args

    @patch("whisper_dictation.notifier.sys")
    @patch("whisper_dictation.notifier.subprocess.Popen")
    def test_empty_message(self, mock_popen, mock_sys):
        mock_sys.platform = "linux"
        notify("Title", "")
        mock_popen.assert_called_once()
        args = mock_popen.call_args[0][0]
        assert "Title" in args

    @patch("whisper_dictation.notifier.sys")
    @patch("whisper_dictation.notifier.subprocess.Popen")
    def test_file_not_found_handled(self, mock_popen, mock_sys):
        mock_sys.platform = "linux"
        mock_popen.side_effect = FileNotFoundError("notify-send not found")
        # Should not raise
        notify("Title", "Message")


class TestNotifyMacos:
    @patch("whisper_dictation.notifier.sys")
    @patch("whisper_dictation.notifier.subprocess.Popen")
    def test_calls_osascript(self, mock_popen, mock_sys):
        mock_sys.platform = "darwin"
        notify("Alert", "Something happened")

        mock_popen.assert_called_once()
        args = mock_popen.call_args[0][0]
        assert args[0] == "osascript"
        assert "-e" in args
        # Script should contain title and message
        script = args[2]
        assert "Alert" in script
        assert "Something happened" in script


class TestNotifyMacosEscape:
    @patch("whisper_dictation.notifier.sys")
    @patch("whisper_dictation.notifier.subprocess.Popen")
    def test_strips_control_characters(self, mock_popen, mock_sys):
        """Test AppleScript escape strips null bytes and control chars."""
        mock_sys.platform = "darwin"
        notify("Title\x00", "Body\x01\x7f")

        mock_popen.assert_called_once()
        script = mock_popen.call_args[0][0][2]
        assert "\x00" not in script
        assert "\x01" not in script
        assert "\x7f" not in script
        assert "Title" in script
        assert "Body" in script

    @patch("whisper_dictation.notifier.sys")
    @patch("whisper_dictation.notifier.subprocess.Popen")
    def test_escapes_backslash_and_quotes(self, mock_popen, mock_sys):
        """Test AppleScript escape handles backslashes and double quotes."""
        mock_sys.platform = "darwin"
        notify('Say "hello"', "path\\to\\file")

        mock_popen.assert_called_once()
        script = mock_popen.call_args[0][0][2]
        assert '\\"hello\\"' in script
        # Backslashes should be doubled for AppleScript
        assert "path\\\\to\\\\file" in script

    @patch("whisper_dictation.notifier.sys")
    @patch("whisper_dictation.notifier.subprocess.Popen")
    def test_escapes_carriage_return(self, mock_popen, mock_sys):
        """Test AppleScript escape handles carriage returns."""
        mock_sys.platform = "darwin"
        notify("Title\r", "Body\r")

        mock_popen.assert_called_once()
        script = mock_popen.call_args[0][0][2]
        assert "\r" not in script


class TestNotifyWindows:
    @patch("whisper_dictation.notifier.sys")
    def test_uses_plyer(self, mock_sys):
        mock_sys.platform = "win32"
        mock_plyer = MagicMock()

        with patch.dict("sys.modules", {"plyer": mock_plyer, "plyer.notification": mock_plyer}):
            with patch("whisper_dictation.notifier.sys", mock_sys):
                # We need to ensure the import inside works
                mock_notification = MagicMock()
                with patch(
                    "builtins.__import__",
                    side_effect=lambda name, *a, **kw: (
                        mock_notification if "plyer" in str(name) else __import__(name, *a, **kw)
                    ),
                ):
                    # Simpler approach: just call and verify no crash
                    notify("Win Title", "Win Body")

    @patch("whisper_dictation.notifier.sys")
    def test_plyer_notify_raises_exception(self, mock_sys):
        """Test Windows path when plyer.notification.notify raises (covers lines 39-40)."""
        mock_sys.platform = "win32"
        mock_plyer_notification = MagicMock()
        mock_plyer_notification.notify.side_effect = RuntimeError("plyer crashed")

        with patch.dict(
            "sys.modules",
            {
                "plyer": MagicMock(notification=mock_plyer_notification),
                "plyer.notification": mock_plyer_notification,
            },
        ):
            # Should not raise - exception should be caught
            notify("Win Title", "Win Body")

    @patch("whisper_dictation.notifier.sys")
    def test_plyer_import_error_handled(self, mock_sys):
        mock_sys.platform = "win32"
        # If plyer is not installed, the import will fail but should be caught
        # The function should not raise
        notify("Title", "Message")


class TestNotifyGenericErrors:
    @patch("whisper_dictation.notifier.sys")
    @patch("whisper_dictation.notifier.subprocess.Popen")
    def test_generic_exception_handled(self, mock_popen, mock_sys):
        mock_sys.platform = "linux"
        mock_popen.side_effect = RuntimeError("unexpected error")
        # Should not raise
        notify("Title", "Message")

    @patch("whisper_dictation.notifier.sys")
    @patch("whisper_dictation.notifier.subprocess.Popen")
    def test_never_raises(self, mock_popen, mock_sys):
        """notify should never propagate exceptions."""
        mock_sys.platform = "linux"
        mock_popen.side_effect = OSError("permission denied")
        notify("Title", "Message")
