"""Cross-platform desktop notifications."""

from __future__ import annotations

import logging
import re
import subprocess
import sys

log = logging.getLogger(__name__)


def notify(title: str, message: str = "") -> None:
    """Show a desktop notification. Best-effort, non-blocking, never raises."""
    try:
        if sys.platform == "linux":
            subprocess.Popen(
                ["notify-send", "-t", "2000", "-a", "Dictation", title, message],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        elif sys.platform == "darwin":
            # Escape characters that break AppleScript string literals
            def _applescript_escape(s: str) -> str:
                s = s.replace("\\", "\\\\").replace('"', '\\"')
                s = s.replace("\n", "\\n").replace("\r", "\\r")
                return re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", s)

            safe_title = _applescript_escape(title)
            safe_message = _applescript_escape(message)
            script = f'display notification "{safe_message}" with title "{safe_title}"'
            subprocess.Popen(
                ["osascript", "-e", script],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        else:
            # Windows — use plyer as fallback
            try:
                from plyer import notification as plyer_notify

                plyer_notify.notify(
                    title=title,
                    message=message or title,
                    app_name="Dictation",
                    timeout=2,
                )
            except Exception:
                log.debug("plyer notification failed")
    except FileNotFoundError:
        log.debug("notification command not found")
    except Exception:
        log.debug("notification failed", exc_info=True)
