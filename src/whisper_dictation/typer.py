"""Platform-aware text input — type text into the focused application."""

from __future__ import annotations

import logging
import os
import subprocess
import sys
import time

log = logging.getLogger(__name__)

PASTE_DELAY = float(os.environ.get("DICTATION_PASTE_DELAY", "0.15"))


def _detect_display() -> str:
    """Detect display server on Linux."""
    return os.environ.get("XDG_SESSION_TYPE", "x11")


def _type_linux_x11(text: str) -> None:
    """Type via xdotool + xclip on X11."""
    prev = subprocess.run(
        ["xclip", "-selection", "clipboard", "-o"],
        capture_output=True, text=True, check=False,
    ).stdout

    subprocess.run(
        ["xclip", "-selection", "clipboard"],
        input=text.encode(), check=False,
    )
    subprocess.run(
        ["xdotool", "key", "--clearmodifiers", "ctrl+v"],
        check=False,
    )
    time.sleep(PASTE_DELAY)

    subprocess.run(
        ["xclip", "-selection", "clipboard"],
        input=prev.encode(), check=False,
    )


def _type_linux_wayland(text: str) -> None:
    """Type via ydotool + wl-clipboard on Wayland."""
    prev = subprocess.run(
        ["wl-paste"], capture_output=True, text=True, check=False,
    ).stdout

    subprocess.run(["wl-copy", text], check=False)
    subprocess.run(
        ["ydotool", "key", "29:1", "47:1", "47:0", "29:0"],
        check=False,
    )
    time.sleep(PASTE_DELAY)

    subprocess.run(["wl-copy", prev], check=False)


def _type_macos(text: str) -> None:
    """Type via pbcopy + AppleScript on macOS."""
    prev = subprocess.run(
        ["pbpaste"], capture_output=True, text=True, check=False,
    ).stdout

    subprocess.run(["pbcopy"], input=text.encode(), check=False)
    subprocess.run(
        ["osascript", "-e", 'tell application "System Events" to keystroke "v" using command down'],
        check=False,
    )
    time.sleep(PASTE_DELAY)

    subprocess.run(["pbcopy"], input=prev.encode(), check=False)


def _type_windows(text: str) -> None:
    """Type via pyperclip + keyboard simulation on Windows."""
    try:
        import ctypes
        import ctypes.wintypes

        # Save clipboard
        prev = ""
        try:
            import subprocess as sp
            r = sp.run(["powershell", "-command", "Get-Clipboard"], capture_output=True, text=True, check=False)
            prev = r.stdout.rstrip("\n")
        except Exception:
            pass

        # Set clipboard
        sp.run(
            ["powershell", "-command", f"Set-Clipboard -Value '{text}'"],
            check=False,
        )

        # Ctrl+V via SendInput
        _send_ctrl_v()
        time.sleep(PASTE_DELAY)

        # Restore
        sp.run(
            ["powershell", "-command", f"Set-Clipboard -Value '{prev}'"],
            check=False,
        )
    except Exception:
        log.error("Windows typing failed", exc_info=True)


def _send_ctrl_v() -> None:
    """Send Ctrl+V on Windows via ctypes."""
    import ctypes

    VK_CONTROL = 0x11
    VK_V = 0x56
    KEYEVENTF_KEYUP = 0x0002

    user32 = ctypes.windll.user32
    user32.keybd_event(VK_CONTROL, 0, 0, 0)
    user32.keybd_event(VK_V, 0, 0, 0)
    user32.keybd_event(VK_V, 0, KEYEVENTF_KEYUP, 0)
    user32.keybd_event(VK_CONTROL, 0, KEYEVENTF_KEYUP, 0)


def type_text(text: str) -> None:
    """Type text into the currently focused application."""
    if not text:
        return

    log.debug("Typing %d chars", len(text))

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
