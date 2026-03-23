"""Tests for whisper_dictation.hotkey — HotkeyListener and _parse_hotkey."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from whisper_dictation.hotkey.listener import HotkeyListener, _parse_hotkey


# ---------------------------------------------------------------------------
# _parse_hotkey
# ---------------------------------------------------------------------------


class TestParseHotkey:
    def test_simple_binding(self):
        mods, key = _parse_hotkey("alt+v")
        assert mods == {"alt"}
        assert key == "v"

    def test_multiple_modifiers(self):
        mods, key = _parse_hotkey("ctrl+shift+d")
        assert mods == {"ctrl", "shift"}
        assert key == "d"

    def test_single_key(self):
        mods, key = _parse_hotkey("f1")
        assert mods == set()
        assert key == "f1"

    def test_case_insensitive(self):
        mods, key = _parse_hotkey("Alt+Shift+V")
        assert mods == {"alt", "shift"}
        assert key == "v"

    def test_whitespace_stripped(self):
        mods, key = _parse_hotkey(" alt + v ")
        assert mods == {"alt"}
        assert key == "v"

    def test_super_key(self):
        mods, key = _parse_hotkey("super+a")
        assert mods == {"super"}
        assert key == "a"


# ---------------------------------------------------------------------------
# HotkeyListener init
# ---------------------------------------------------------------------------


class TestHotkeyListenerInit:
    def test_init_toggle_mode(self):
        activate = MagicMock()
        deactivate = MagicMock()
        listener = HotkeyListener("alt+v", "toggle", activate, deactivate)

        assert listener.binding == "alt+v"
        assert listener.mode == "toggle"
        assert listener._modifiers == {"alt"}
        assert listener._key == "v"
        assert listener._active is False

    def test_init_hold_mode(self):
        listener = HotkeyListener("ctrl+d", "hold", MagicMock(), MagicMock())
        assert listener.mode == "hold"
        assert listener._modifiers == {"ctrl"}
        assert listener._key == "d"


# ---------------------------------------------------------------------------
# _handle_press / _handle_release — toggle mode
# ---------------------------------------------------------------------------


class TestToggleMode:
    @pytest.fixture
    def toggle_listener(self):
        activate = MagicMock()
        deactivate = MagicMock()
        listener = HotkeyListener("alt+v", "toggle", activate, deactivate)
        return listener, activate, deactivate

    def test_first_press_activates(self, toggle_listener):
        listener, activate, deactivate = toggle_listener
        listener._handle_press()
        assert listener._active is True
        activate.assert_called_once()
        deactivate.assert_not_called()

    def test_second_press_deactivates(self, toggle_listener):
        listener, activate, deactivate = toggle_listener
        listener._handle_press()
        listener._handle_press()
        assert listener._active is False
        activate.assert_called_once()
        deactivate.assert_called_once()

    def test_toggle_cycle(self, toggle_listener):
        listener, activate, deactivate = toggle_listener
        listener._handle_press()  # ON
        listener._handle_press()  # OFF
        listener._handle_press()  # ON
        assert listener._active is True
        assert activate.call_count == 2
        assert deactivate.call_count == 1

    def test_release_ignored_in_toggle_mode(self, toggle_listener):
        listener, activate, deactivate = toggle_listener
        listener._handle_press()
        listener._handle_release()
        # In toggle mode, release deactivates if active
        assert listener._active is False
        deactivate.assert_called_once()


# ---------------------------------------------------------------------------
# _handle_press / _handle_release — hold mode
# ---------------------------------------------------------------------------


class TestHoldMode:
    @pytest.fixture
    def hold_listener(self):
        activate = MagicMock()
        deactivate = MagicMock()
        listener = HotkeyListener("alt+v", "hold", activate, deactivate)
        return listener, activate, deactivate

    def test_press_activates(self, hold_listener):
        listener, activate, deactivate = hold_listener
        listener._handle_press()
        assert listener._active is True
        activate.assert_called_once()
        deactivate.assert_not_called()

    def test_release_deactivates(self, hold_listener):
        listener, activate, deactivate = hold_listener
        listener._handle_press()
        listener._handle_release()
        assert listener._active is False
        deactivate.assert_called_once()

    def test_repeated_press_no_double_activate(self, hold_listener):
        listener, activate, deactivate = hold_listener
        listener._handle_press()
        listener._handle_press()
        # Should only activate once
        assert activate.call_count == 1

    def test_release_without_press_no_deactivate(self, hold_listener):
        listener, activate, deactivate = hold_listener
        listener._handle_release()
        deactivate.assert_not_called()


# ---------------------------------------------------------------------------
# _use_evdev
# ---------------------------------------------------------------------------


class TestUseEvdev:
    def test_not_linux(self):
        listener = HotkeyListener("alt+v", "toggle", MagicMock(), MagicMock())
        with patch("whisper_dictation.hotkey.listener.sys") as mock_sys:
            mock_sys.platform = "darwin"
            assert listener._use_evdev() is False

    def test_linux_x11(self):
        listener = HotkeyListener("alt+v", "toggle", MagicMock(), MagicMock())
        with (
            patch("whisper_dictation.hotkey.listener.sys") as mock_sys,
            patch.dict("os.environ", {"XDG_SESSION_TYPE": "x11"}),
        ):
            mock_sys.platform = "linux"
            assert listener._use_evdev() is False

    def test_linux_wayland_with_evdev(self):
        listener = HotkeyListener("alt+v", "toggle", MagicMock(), MagicMock())
        mock_evdev = MagicMock()
        with (
            patch("whisper_dictation.hotkey.listener.sys") as mock_sys,
            patch.dict("os.environ", {"XDG_SESSION_TYPE": "wayland"}),
            patch.dict("sys.modules", {"evdev": mock_evdev}),
            patch("builtins.__import__", side_effect=lambda name, *a, **kw: (
                mock_evdev if name == "evdev" else __import__(name, *a, **kw)
            )),
        ):
            mock_sys.platform = "linux"
            assert listener._use_evdev() is True

    def test_linux_wayland_without_evdev(self):
        listener = HotkeyListener("alt+v", "toggle", MagicMock(), MagicMock())
        with (
            patch("whisper_dictation.hotkey.listener.sys") as mock_sys,
            patch.dict("os.environ", {"XDG_SESSION_TYPE": "wayland"}),
        ):
            mock_sys.platform = "linux"
            # Remove evdev if it exists
            with patch.dict("sys.modules", {"evdev": None}):
                # The import inside _use_evdev will fail
                result = listener._use_evdev()
                # Falls back to False when evdev import fails
                assert result is False


# ---------------------------------------------------------------------------
# start / stop
# ---------------------------------------------------------------------------


class TestStartStop:
    @patch("whisper_dictation.hotkey.listener.HotkeyListener._start_pynput")
    @patch("whisper_dictation.hotkey.listener.HotkeyListener._use_evdev", return_value=False)
    def test_start_pynput(self, mock_evdev, mock_start_pynput):
        listener = HotkeyListener("alt+v", "toggle", MagicMock(), MagicMock())
        listener.start()
        mock_start_pynput.assert_called_once()

    @patch("whisper_dictation.hotkey.listener.HotkeyListener._start_evdev")
    @patch("whisper_dictation.hotkey.listener.HotkeyListener._use_evdev", return_value=True)
    def test_start_evdev(self, mock_use_evdev, mock_start_evdev):
        listener = HotkeyListener("alt+v", "toggle", MagicMock(), MagicMock())
        listener.start()
        mock_start_evdev.assert_called_once()

    def test_stop_without_listener(self):
        listener = HotkeyListener("alt+v", "toggle", MagicMock(), MagicMock())
        # Should not raise
        listener.stop()
        assert listener._listener is None

    def test_stop_with_listener(self):
        listener = HotkeyListener("alt+v", "toggle", MagicMock(), MagicMock())
        mock_listener = MagicMock()
        listener._listener = mock_listener
        listener.stop()
        mock_listener.stop.assert_called_once()
        assert listener._listener is None

    def test_stop_exception_in_listener_stop(self):
        listener = HotkeyListener("alt+v", "toggle", MagicMock(), MagicMock())
        mock_listener = MagicMock()
        mock_listener.stop.side_effect = RuntimeError("stop failed")
        listener._listener = mock_listener
        # Should not raise
        listener.stop()
        assert listener._listener is None
