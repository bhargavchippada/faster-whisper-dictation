"""Platform-aware text input — type text into the focused application."""

from __future__ import annotations

import logging
import os
import subprocess
import sys
import threading
import time

log = logging.getLogger(__name__)


def _parse_paste_delay(value: str) -> float:
    """Parse and validate PASTE_DELAY from environment."""
    try:
        delay = float(value)
    except ValueError:
        raise ValueError(f"DICTATION_PASTE_DELAY must be a number, got {value!r}") from None
    if not (0.0 <= delay <= 10.0) or delay != delay:  # NaN check
        raise ValueError(f"DICTATION_PASTE_DELAY must be 0.0-10.0, got {delay}")
    return delay


PASTE_DELAY = _parse_paste_delay(os.environ.get("DICTATION_PASTE_DELAY", "0.15"))
_clipboard_lock = threading.Lock()


def _detect_display() -> str:
    """Detect display server on Linux."""
    return os.environ.get("XDG_SESSION_TYPE", "x11")


def _type_linux_x11(text: str) -> None:
    """Type via clipboard paste on X11.

    Uses xclip to set clipboard, then xdotool to send Ctrl+Shift+V
    (works in both terminals and editors). Preserves previous clipboard.
    """
    prev = subprocess.run(
        ["xclip", "-selection", "clipboard", "-o"],
        capture_output=True,
        text=True,
        check=False,
    ).stdout

    subprocess.run(
        ["xclip", "-selection", "clipboard"],
        input=text.encode(),
        check=False,
    )
    try:
        # Ctrl+Shift+V works in terminals AND editors
        subprocess.run(
            ["xdotool", "key", "--clearmodifiers", "ctrl+shift+v"],
            check=False,
        )
        time.sleep(PASTE_DELAY)
    finally:
        subprocess.run(
            ["xclip", "-selection", "clipboard"],
            input=prev.encode(),
            check=False,
        )


def _type_linux_wayland(text: str) -> None:
    """Type via ydotool + wl-clipboard on Wayland."""
    prev = subprocess.run(
        ["wl-paste"],
        capture_output=True,
        text=True,
        check=False,
    ).stdout

    subprocess.run(["wl-copy", "--", text], check=False)
    try:
        subprocess.run(
            ["ydotool", "key", "29:1", "47:1", "47:0", "29:0"],
            check=False,
        )
        time.sleep(PASTE_DELAY)
    finally:
        subprocess.run(["wl-copy", "--", prev], check=False)


def _type_macos(text: str) -> None:
    """Type via pbcopy + AppleScript on macOS."""
    prev = subprocess.run(
        ["pbpaste"],
        capture_output=True,
        text=True,
        check=False,
    ).stdout

    subprocess.run(["pbcopy"], input=text.encode(), check=False)
    try:
        script = 'tell application "System Events" to keystroke "v" using command down'
        subprocess.run(["osascript", "-e", script], check=False)
        time.sleep(PASTE_DELAY)
    finally:
        subprocess.run(["pbcopy"], input=prev.encode(), check=False)


def _type_windows(text: str) -> None:
    """Type via clipboard + keyboard simulation on Windows."""
    prev = _win_clipboard_get()
    try:
        _win_clipboard_set(text)
        _send_ctrl_v()
        time.sleep(PASTE_DELAY)
    except Exception:
        log.error("Windows typing failed", exc_info=True)
    finally:
        try:
            _win_clipboard_set(prev)
        except Exception:
            log.debug("Failed to restore clipboard", exc_info=True)


_CF_UNICODETEXT = 13
_GMEM_MOVEABLE = 0x0002
_VK_CONTROL = 0x11
_VK_V = 0x56
_KEYEVENTF_KEYUP = 0x0002


def _win_clipboard_get() -> str:
    """Read clipboard text on Windows via Win32 API."""
    import ctypes

    user32 = ctypes.windll.user32
    kernel32 = ctypes.windll.kernel32

    if not user32.OpenClipboard(0):
        log.debug("Failed to open clipboard for reading")
        return ""
    try:
        handle = user32.GetClipboardData(_CF_UNICODETEXT)
        if not handle:
            return ""
        ptr = kernel32.GlobalLock(handle)
        try:
            return ctypes.wstring_at(ptr) if ptr else ""
        finally:
            kernel32.GlobalUnlock(handle)
    finally:
        user32.CloseClipboard()


def _win_clipboard_set(text: str) -> None:
    """Write text to clipboard on Windows via Win32 API."""
    import ctypes

    user32 = ctypes.windll.user32
    kernel32 = ctypes.windll.kernel32

    encoded = text.encode("utf-16-le") + b"\x00\x00"
    handle = kernel32.GlobalAlloc(_GMEM_MOVEABLE, len(encoded))
    if not handle:
        raise RuntimeError("GlobalAlloc failed")
    ptr = kernel32.GlobalLock(handle)
    if not ptr:
        kernel32.GlobalFree(handle)
        raise RuntimeError("GlobalLock failed")
    ctypes.memmove(ptr, encoded, len(encoded))
    kernel32.GlobalUnlock(handle)

    if not user32.OpenClipboard(0):
        log.debug("Failed to open clipboard for writing")
        kernel32.GlobalFree(handle)
        return
    try:
        user32.EmptyClipboard()
        user32.SetClipboardData(_CF_UNICODETEXT, handle)
    finally:
        user32.CloseClipboard()


def _send_ctrl_v() -> None:
    """Send Ctrl+V on Windows via ctypes."""
    import ctypes

    user32 = ctypes.windll.user32
    user32.keybd_event(_VK_CONTROL, 0, 0, 0)
    user32.keybd_event(_VK_V, 0, 0, 0)
    user32.keybd_event(_VK_V, 0, _KEYEVENTF_KEYUP, 0)
    user32.keybd_event(_VK_CONTROL, 0, _KEYEVENTF_KEYUP, 0)


def type_text(text: str) -> None:
    """Type text into the currently focused application.

    Thread-safe: acquires a lock to prevent concurrent clipboard corruption.
    """
    if not text:
        return

    log.debug("Typing %d chars", len(text))

    with _clipboard_lock:
        _type_text_impl(text)


def _type_text_impl(text: str) -> None:
    """Internal implementation — must be called under _clipboard_lock."""
    if sys.platform == "linux":
        display = _detect_display()
        if display == "wayland":
            _type_linux_wayland(text)
        else:
            _type_linux_x11(text)
    elif sys.platform == "darwin":
        _type_macos(text)
    elif sys.platform == "win32":
        _type_windows(text)
    else:
        log.warning("Unsupported platform: %s", sys.platform)
