"""Tests for whisper_dictation.typer — type_text and platform dispatch."""

from __future__ import annotations

from unittest.mock import MagicMock, call, patch

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
    @patch("whisper_dictation.typer.subprocess.run")
    @patch("whisper_dictation.typer._send_ctrl_v")
    def test_clipboard_flow(self, mock_ctrl_v, mock_run, mock_sleep):
        mock_result = MagicMock()
        mock_result.stdout = "old_win\n"
        mock_run.return_value = mock_result

        typer._type_windows("win text")

        # Should call powershell Get-Clipboard, Set-Clipboard, restore
        assert mock_run.call_count >= 2
        mock_ctrl_v.assert_called_once()
        mock_sleep.assert_called_once()

    @patch("whisper_dictation.typer.time.sleep")
    @patch("whisper_dictation.typer.subprocess.run")
    @patch("whisper_dictation.typer._send_ctrl_v")
    def test_handles_exception_gracefully(self, mock_ctrl_v, mock_run, mock_sleep):
        mock_run.side_effect = Exception("boom")
        # Should not raise
        typer._type_windows("text")


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
