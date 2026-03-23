"""Tests for whisper_dictation.typer — type_text and platform dispatch."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from whisper_dictation import typer

# ---------------------------------------------------------------------------
# type_text dispatch
# ---------------------------------------------------------------------------


class TestTypeText:
    def test_empty_text_returns_immediately(self):
        with patch("whisper_dictation.typer.subprocess") as mock_sp:
            typer.type_text("")
            mock_sp.run.assert_not_called()

    @patch("whisper_dictation.typer._type_linux_x11")
    @patch("whisper_dictation.typer._detect_display", return_value="x11")
    @patch("whisper_dictation.typer.sys")
    def test_linux_x11_dispatch(self, mock_sys, mock_detect, mock_type):
        mock_sys.platform = "linux"
        typer.type_text("hello")
        mock_type.assert_called_once_with("hello")

    @patch("whisper_dictation.typer._type_linux_wayland")
    @patch("whisper_dictation.typer._detect_display", return_value="wayland")
    @patch("whisper_dictation.typer.sys")
    def test_linux_wayland_dispatch(self, mock_sys, mock_detect, mock_type):
        mock_sys.platform = "linux"
        typer.type_text("world")
        mock_type.assert_called_once_with("world")

    @patch("whisper_dictation.typer._type_macos")
    @patch("whisper_dictation.typer.sys")
    def test_macos_dispatch(self, mock_sys, mock_type):
        mock_sys.platform = "darwin"
        typer.type_text("text")
        mock_type.assert_called_once_with("text")

    @patch("whisper_dictation.typer._type_windows")
    @patch("whisper_dictation.typer.sys")
    def test_windows_dispatch(self, mock_sys, mock_type):
        mock_sys.platform = "win32"
        typer.type_text("text")
        mock_type.assert_called_once_with("text")

    @patch("whisper_dictation.typer.sys")
    def test_unsupported_platform(self, mock_sys):
        mock_sys.platform = "freebsd"
        # Should not raise
        typer.type_text("text")


# ---------------------------------------------------------------------------
# _detect_display
# ---------------------------------------------------------------------------


class TestDetectDisplay:
    def test_returns_env_var(self):
        with patch.dict("os.environ", {"XDG_SESSION_TYPE": "wayland"}):
            assert typer._detect_display() == "wayland"

    def test_defaults_to_x11(self):
        with patch.dict("os.environ", {}, clear=True):
            assert typer._detect_display() == "x11"


# ---------------------------------------------------------------------------
# _type_linux_x11
# ---------------------------------------------------------------------------


class TestTypeLinuxX11:
    @patch("whisper_dictation.typer.time.sleep")
    @patch("whisper_dictation.typer.subprocess.run")
    def test_clipboard_save_paste_restore(self, mock_run, mock_sleep):
        # First call returns previous clipboard content
        mock_result = MagicMock()
        mock_result.stdout = "old_clip"
        mock_run.return_value = mock_result

        typer._type_linux_x11("new text")

        assert mock_run.call_count == 4

        # 1. Save clipboard
        first_call = mock_run.call_args_list[0]
        assert "xclip" in first_call[0][0]
        assert "-o" in first_call[0][0]

        # 2. Set clipboard
        second_call = mock_run.call_args_list[1]
        assert second_call[1]["input"] == b"new text"

        # 3. Paste (ctrl+v)
        third_call = mock_run.call_args_list[2]
        assert "xdotool" in third_call[0][0]
        assert "ctrl+v" in third_call[0][0]

        # 4. Restore clipboard
        fourth_call = mock_run.call_args_list[3]
        assert fourth_call[1]["input"] == b"old_clip"

        mock_sleep.assert_called_once()


# ---------------------------------------------------------------------------
# _type_linux_wayland
# ---------------------------------------------------------------------------


class TestTypeLinuxWayland:
    @patch("whisper_dictation.typer.time.sleep")
    @patch("whisper_dictation.typer.subprocess.run")
    def test_clipboard_save_paste_restore(self, mock_run, mock_sleep):
        mock_result = MagicMock()
        mock_result.stdout = "old_wayland"
        mock_run.return_value = mock_result

        typer._type_linux_wayland("wayland text")

        assert mock_run.call_count == 4

        # 1. Save clipboard (wl-paste)
        assert "wl-paste" in mock_run.call_args_list[0][0][0]

        # 2. Set clipboard (wl-copy)
        assert "wl-copy" in mock_run.call_args_list[1][0][0]
        assert "wayland text" in mock_run.call_args_list[1][0][0]

        # 3. Paste (ydotool Ctrl+V keycodes)
        assert "ydotool" in mock_run.call_args_list[2][0][0]

        # 4. Restore (wl-copy)
        assert "wl-copy" in mock_run.call_args_list[3][0][0]
        assert "old_wayland" in mock_run.call_args_list[3][0][0]

        mock_sleep.assert_called_once()


# ---------------------------------------------------------------------------
# _type_macos
# ---------------------------------------------------------------------------


class TestTypeMacos:
    @patch("whisper_dictation.typer.time.sleep")
    @patch("whisper_dictation.typer.subprocess.run")
    def test_clipboard_save_paste_restore(self, mock_run, mock_sleep):
        mock_result = MagicMock()
        mock_result.stdout = "old_mac"
        mock_run.return_value = mock_result

        typer._type_macos("mac text")

        assert mock_run.call_count == 4

        # 1. Save clipboard (pbpaste)
        assert "pbpaste" in mock_run.call_args_list[0][0][0]

        # 2. Set clipboard (pbcopy)
        assert "pbcopy" in mock_run.call_args_list[1][0][0]
        assert mock_run.call_args_list[1][1]["input"] == b"mac text"

        # 3. Paste via osascript
        assert "osascript" in mock_run.call_args_list[2][0][0]

        # 4. Restore clipboard (pbcopy)
        assert "pbcopy" in mock_run.call_args_list[3][0][0]
        assert mock_run.call_args_list[3][1]["input"] == b"old_mac"

        mock_sleep.assert_called_once()


# ---------------------------------------------------------------------------
# _type_windows
# ---------------------------------------------------------------------------


class TestTypeWindows:
    @patch("whisper_dictation.typer.time.sleep")
    @patch("whisper_dictation.typer._win_clipboard_set")
    @patch("whisper_dictation.typer._win_clipboard_get", return_value="old_win")
    @patch("whisper_dictation.typer._send_ctrl_v")
    def test_clipboard_flow(self, mock_ctrl_v, mock_get, mock_set, mock_sleep):
        typer._type_windows("win text")

        mock_get.assert_called_once()
        assert mock_set.call_count == 2
        mock_set.assert_any_call("win text")
        mock_set.assert_any_call("old_win")
        mock_ctrl_v.assert_called_once()
        mock_sleep.assert_called_once()

    @patch("whisper_dictation.typer.time.sleep")
    @patch("whisper_dictation.typer._win_clipboard_set", side_effect=Exception("boom"))
    @patch("whisper_dictation.typer._win_clipboard_get", return_value="")
    @patch("whisper_dictation.typer._send_ctrl_v")
    def test_handles_exception_gracefully(self, mock_ctrl_v, mock_get, mock_set, mock_sleep):
        # Should not raise
        typer._type_windows("text")


# ---------------------------------------------------------------------------
# _send_ctrl_v
# ---------------------------------------------------------------------------


class TestWinClipboardGet:
    def test_reads_clipboard_text(self):
        mock_ctypes = MagicMock()
        mock_user32 = MagicMock()
        mock_kernel32 = MagicMock()
        mock_ctypes.windll.user32 = mock_user32
        mock_ctypes.windll.kernel32 = mock_kernel32

        mock_user32.GetClipboardData.return_value = 12345
        mock_kernel32.GlobalLock.return_value = 67890
        mock_ctypes.wstring_at.return_value = "clipboard text"

        with patch.dict("sys.modules", {"ctypes": mock_ctypes, "ctypes.wintypes": MagicMock()}):
            result = typer._win_clipboard_get()

        assert result == "clipboard text"
        mock_user32.OpenClipboard.assert_called_once_with(0)
        mock_user32.CloseClipboard.assert_called_once()
        mock_kernel32.GlobalUnlock.assert_called_once()

    def test_returns_empty_on_no_handle(self):
        mock_ctypes = MagicMock()
        mock_user32 = MagicMock()
        mock_ctypes.windll.user32 = mock_user32
        mock_ctypes.windll.kernel32 = MagicMock()

        mock_user32.GetClipboardData.return_value = 0  # no handle

        with patch.dict("sys.modules", {"ctypes": mock_ctypes, "ctypes.wintypes": MagicMock()}):
            result = typer._win_clipboard_get()

        assert result == ""

    def test_returns_empty_on_null_ptr(self):
        mock_ctypes = MagicMock()
        mock_user32 = MagicMock()
        mock_kernel32 = MagicMock()
        mock_ctypes.windll.user32 = mock_user32
        mock_ctypes.windll.kernel32 = mock_kernel32

        mock_user32.GetClipboardData.return_value = 12345
        mock_kernel32.GlobalLock.return_value = 0  # null ptr

        with patch.dict("sys.modules", {"ctypes": mock_ctypes, "ctypes.wintypes": MagicMock()}):
            result = typer._win_clipboard_get()

        assert result == ""

    def test_returns_empty_when_clipboard_open_fails(self):
        """Test _win_clipboard_get returns empty when OpenClipboard fails."""
        mock_ctypes = MagicMock()
        mock_user32 = MagicMock()
        mock_ctypes.windll.user32 = mock_user32
        mock_ctypes.windll.kernel32 = MagicMock()

        mock_user32.OpenClipboard.return_value = 0  # failure

        with patch.dict("sys.modules", {"ctypes": mock_ctypes, "ctypes.wintypes": MagicMock()}):
            result = typer._win_clipboard_get()

        assert result == ""
        mock_user32.CloseClipboard.assert_not_called()


class TestWinClipboardSet:
    def test_sets_clipboard_text(self):
        mock_ctypes = MagicMock()
        mock_user32 = MagicMock()
        mock_kernel32 = MagicMock()
        mock_ctypes.windll.user32 = mock_user32
        mock_ctypes.windll.kernel32 = mock_kernel32
        mock_kernel32.GlobalAlloc.return_value = 99
        mock_kernel32.GlobalLock.return_value = 100

        with patch.dict("sys.modules", {"ctypes": mock_ctypes, "ctypes.wintypes": MagicMock()}):
            typer._win_clipboard_set("hello")

        mock_user32.OpenClipboard.assert_called_once_with(0)
        mock_user32.EmptyClipboard.assert_called_once()
        mock_user32.SetClipboardData.assert_called_once()
        mock_user32.CloseClipboard.assert_called_once()
        mock_kernel32.GlobalUnlock.assert_called_once()
        mock_ctypes.memmove.assert_called_once()

    def test_frees_memory_when_clipboard_open_fails(self):
        """Test _win_clipboard_set frees allocated memory when OpenClipboard fails."""
        mock_ctypes = MagicMock()
        mock_user32 = MagicMock()
        mock_kernel32 = MagicMock()
        mock_ctypes.windll.user32 = mock_user32
        mock_ctypes.windll.kernel32 = mock_kernel32
        mock_kernel32.GlobalAlloc.return_value = 99
        mock_kernel32.GlobalLock.return_value = 100

        mock_user32.OpenClipboard.return_value = 0  # failure

        with patch.dict("sys.modules", {"ctypes": mock_ctypes, "ctypes.wintypes": MagicMock()}):
            typer._win_clipboard_set("hello")

        mock_kernel32.GlobalFree.assert_called_once_with(99)
        mock_user32.EmptyClipboard.assert_not_called()
        mock_user32.CloseClipboard.assert_not_called()

    def test_global_alloc_failure_raises(self):
        mock_ctypes = MagicMock()
        mock_kernel32 = MagicMock()
        mock_ctypes.windll.user32 = MagicMock()
        mock_ctypes.windll.kernel32 = mock_kernel32
        mock_kernel32.GlobalAlloc.return_value = 0  # failure

        with patch.dict("sys.modules", {"ctypes": mock_ctypes, "ctypes.wintypes": MagicMock()}):
            with pytest.raises(RuntimeError, match="GlobalAlloc failed"):
                typer._win_clipboard_set("hello")

    def test_global_lock_failure_frees_and_raises(self):
        mock_ctypes = MagicMock()
        mock_kernel32 = MagicMock()
        mock_ctypes.windll.user32 = MagicMock()
        mock_ctypes.windll.kernel32 = mock_kernel32
        mock_kernel32.GlobalAlloc.return_value = 99
        mock_kernel32.GlobalLock.return_value = 0  # failure

        with patch.dict("sys.modules", {"ctypes": mock_ctypes, "ctypes.wintypes": MagicMock()}):
            with pytest.raises(RuntimeError, match="GlobalLock failed"):
                typer._win_clipboard_set("hello")

        mock_kernel32.GlobalFree.assert_called_once_with(99)


class TestSendCtrlV:
    def test_send_ctrl_v_calls_keybd_event(self):
        """Test _send_ctrl_v calls ctypes.windll.user32.keybd_event."""
        mock_ctypes = MagicMock()
        mock_user32 = MagicMock()
        mock_ctypes.windll.user32 = mock_user32

        with patch.dict("sys.modules", {"ctypes": mock_ctypes, "ctypes.wintypes": MagicMock()}):
            # We need to reload to pick up the mock ctypes
            typer._send_ctrl_v()

        # keybd_event should be called 4 times:
        # ctrl down, v down, v up, ctrl up
        assert mock_user32.keybd_event.call_count == 4


# ---------------------------------------------------------------------------
# PASTE_DELAY
# ---------------------------------------------------------------------------


class TestPasteDelay:
    def test_default_paste_delay(self):
        with patch.dict("os.environ", {}, clear=True):
            # Reimport to get default
            import importlib

            importlib.reload(typer)
            assert typer.PASTE_DELAY == 0.15

    def test_custom_paste_delay(self):
        with patch.dict("os.environ", {"DICTATION_PASTE_DELAY": "0.5"}):
            import importlib

            importlib.reload(typer)
            assert typer.PASTE_DELAY == 0.5
        # Reset
        with patch.dict("os.environ", {}, clear=True):
            import importlib

            importlib.reload(typer)


class TestParsePasteDelay:
    def test_valid_value(self):
        assert typer._parse_paste_delay("0.15") == 0.15

    def test_non_numeric_raises(self):
        with pytest.raises(ValueError, match="must be a number"):
            typer._parse_paste_delay("abc")

    def test_nan_raises(self):
        with pytest.raises(ValueError, match="must be 0.0-10.0"):
            typer._parse_paste_delay("nan")

    def test_negative_raises(self):
        with pytest.raises(ValueError, match="must be 0.0-10.0"):
            typer._parse_paste_delay("-1.0")

    def test_too_large_raises(self):
        with pytest.raises(ValueError, match="must be 0.0-10.0"):
            typer._parse_paste_delay("99.0")

    def test_inf_raises(self):
        with pytest.raises(ValueError, match="must be 0.0-10.0"):
            typer._parse_paste_delay("inf")
