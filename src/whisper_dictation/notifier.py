"""Cross-platform desktop notifications."""

from __future__ import annotations

import logging
import subprocess
import sys

log = logging.getLogger(__name__)


def notify(title: str, message: str = "") -> None:
    """Show a desktop notification. Best-effort, never raises."""
    try:
        if sys.platform == "linux":
            subprocess.run(
                ["notify-send", "-t", "2000", "-a", "Dictation", title, message],
                capture_output=True,
                check=False,
            )
        elif sys.platform == "darwin":
            safe_title = title.replace('\\', '\\\\').replace('"', '\\"')
            safe_message = message.replace('\\', '\\\\').replace('"', '\\"')
            script = f'display notification "{safe_message}" with title "{safe_title}"'
            subprocess.run(
                ["osascript", "-e", script],
                capture_output=True,
                check=False,
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
