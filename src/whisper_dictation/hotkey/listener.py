"""Cross-platform hotkey listener supporting hold and toggle modes.

Uses pynput for macOS/Windows/Linux-X11, with evdev fallback for Wayland.
"""

from __future__ import annotations

import logging
import os
import sys
import threading
import time
from collections.abc import Callable

log = logging.getLogger(__name__)

_DEFAULT_THROTTLE_MS = 30


def _parse_poll_ms(value: str) -> int:
    try:
        return max(10, int(value))
    except ValueError:
        return _DEFAULT_THROTTLE_MS


_PYNPUT_POLL_MS = _parse_poll_ms(os.environ.get("DICTATION_HOTKEY_POLL_MS", "30"))
_PYNPUT_SELECT_THROTTLE_S = _PYNPUT_POLL_MS / 1000.0
_throttle_applied = False


def _parse_hold_debounce_ms(value: str) -> float:
    """Parse DICTATION_HOLD_DEBOUNCE_MS to seconds, clamped to [50, 500]."""
    try:
        ms = int(value)
    except ValueError:
        log.warning("Invalid DICTATION_HOLD_DEBOUNCE_MS %r; using default 250ms", value)
        return 0.25
    return max(0.05, min(0.5, ms / 1000.0))


_HOLD_DEBOUNCE_S = _parse_hold_debounce_ms(
    os.environ.get("DICTATION_HOLD_DEBOUNCE_MS", "250"),
)


def _throttle_pynput_xrecord() -> None:
    """Patch Xlib Display.send_and_recv to prevent pynput CPU busy-loop."""
    global _throttle_applied

    if _throttle_applied:
        return

    try:
        from Xlib.protocol.display import Display
    except ImportError:
        return
    except Exception:
        log.debug("Could not import Xlib for throttling", exc_info=True)
        return

    original = Display.send_and_recv
    warmup_calls = 20
    call_count = 0

    def throttled(self, *args, **kwargs):
        nonlocal call_count
        result = original(self, *args, **kwargs)
        call_count += 1
        if call_count > warmup_calls:
            time.sleep(_PYNPUT_SELECT_THROTTLE_S)
        return result

    throttled._throttled = True  # type: ignore[attr-defined]
    Display.send_and_recv = throttled
    _throttle_applied = True
    log.debug("Throttled Xlib send_and_recv to %dms", _PYNPUT_POLL_MS)


def _parse_hotkey(binding: str) -> tuple[set[str], str]:
    """Parse a hotkey string like 'alt+v' into (modifiers, key).

    Returns:
        (modifier_set, key_name) — modifiers are normalized to lowercase.
    """
    parts = [p.strip().lower() for p in binding.split("+")]
    key = parts[-1]
    modifiers = set(parts[:-1])
    return modifiers, key


class HotkeyListener:
    """Listens for a hotkey and fires callbacks for press/release events.

    Supports two modes:
    - hold: on_activate fires on key down, on_deactivate on key up
    - toggle: on_activate on first press, on_deactivate on second press
    """

    def __init__(
        self,
        binding: str,
        mode: str,
        on_activate: Callable[[], None],
        on_deactivate: Callable[[], None],
    ):
        self.binding = binding
        self.mode = mode
        self.on_activate = on_activate
        self.on_deactivate = on_deactivate

        self._modifiers, self._key = _parse_hotkey(binding)
        self._active = False
        self._release_stamp: float = 0.0  # monotonic time of last release
        self._release_watcher: threading.Thread | None = None
        self._key_released = True  # True = key was released since last toggle
        self._listener = None
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._lock = threading.Lock()
        self._last_toggle_time = 0.0

    def _use_evdev(self) -> bool:
        """Check if we should use evdev (Linux).

        Prefer evdev on all Linux — pynput's XRecord backend busy-loops
        on X11, consuming 100% CPU even when idle.  evdev uses select()
        on /dev/input devices, which blocks properly.

        Falls back to pynput if evdev is not installed or input devices
        are not readable (user not in 'input' group).
        """
        if sys.platform != "linux":
            return False
        try:
            import evdev

            # Verify we can actually read input devices
            devices = evdev.list_devices()
            if not devices:
                log.info("evdev: no input devices found, falling back to pynput")
                return False
            # Try opening the first device to check permissions
            try:
                dev = evdev.InputDevice(devices[0])
                dev.close()
            except PermissionError:
                log.warning(
                    "evdev: no permission to read input devices. "
                    "Falling back to pynput (may use high CPU). "
                    "Fix: sudo usermod -aG input $USER && logout"
                )
                return False
            return True
        except ImportError:
            session_type = os.environ.get("XDG_SESSION_TYPE", "")
            if session_type == "wayland":
                log.warning(
                    "Wayland detected but python-evdev not installed. "
                    "Falling back to pynput (may not work). "
                    "Install with: pip install evdev"
                )
            return False

    def start(self) -> None:
        """Start listening for the hotkey in a background thread."""
        self._stop_event.clear()
        if self._use_evdev():
            self._start_evdev()
        else:
            self._start_pynput()

    def stop(self) -> None:
        """Stop listening."""
        self._stop_event.set()
        with self._lock:
            self._release_stamp = 0.0
        if self._listener is not None:
            try:
                self._listener.stop()
            except Exception:
                log.debug("Error stopping listener", exc_info=True)
            self._listener = None
        if self._thread is not None:
            self._thread.join(timeout=1.5)
            self._thread = None
        if self._release_watcher is not None:
            self._release_watcher.join(timeout=_HOLD_DEBOUNCE_S + 0.1)
            self._release_watcher = None

    @staticmethod
    def _throttle_xlib() -> None:
        """Backward-compatible wrapper around the module-level pynput throttle."""
        _throttle_pynput_xrecord()

    def _start_pynput(self) -> None:
        """Start hotkey listener using pynput."""
        if sys.platform == "linux" and os.environ.get("XDG_SESSION_TYPE", "x11") == "x11":
            _throttle_pynput_xrecord()

        from pynput import keyboard

        modifier_map = {
            "alt": {keyboard.Key.alt, keyboard.Key.alt_l, keyboard.Key.alt_r, keyboard.Key.alt_gr},
            "ctrl": {keyboard.Key.ctrl, keyboard.Key.ctrl_l, keyboard.Key.ctrl_r},
            "control": {keyboard.Key.ctrl, keyboard.Key.ctrl_l, keyboard.Key.ctrl_r},
            "shift": {keyboard.Key.shift, keyboard.Key.shift_l, keyboard.Key.shift_r},
            "cmd": {keyboard.Key.cmd, keyboard.Key.cmd_l, keyboard.Key.cmd_r},
            "super": {keyboard.Key.cmd, keyboard.Key.cmd_l, keyboard.Key.cmd_r},
            "meta": {keyboard.Key.cmd, keyboard.Key.cmd_l, keyboard.Key.cmd_r},
        }

        required_modifier_keys: set[keyboard.Key] = set()
        for mod in self._modifiers:
            keys = modifier_map.get(mod, set())
            required_modifier_keys.update(keys)

        pressed_modifiers: set = set()

        def _key_name(key) -> str:
            if hasattr(key, "char") and key.char:
                return key.char.lower()
            if hasattr(key, "name"):
                return key.name.lower()
            return str(key).lower()

        def _is_modifier_held() -> bool:
            # Check that at least one key from each required modifier group is pressed
            for mod in self._modifiers:
                mod_keys = modifier_map.get(mod, set())
                if not pressed_modifiers & mod_keys:
                    return False
            return True

        def on_press(key):
            if key in required_modifier_keys:
                pressed_modifiers.add(key)

            name = _key_name(key)
            if name == self._key and _is_modifier_held():
                self._handle_press()

        def on_release(key):
            if key in required_modifier_keys:
                pressed_modifiers.discard(key)
                if self.mode == "hold" and self._active and not _is_modifier_held():
                    self._handle_release()

            name = _key_name(key)
            if name == self._key:
                with self._lock:
                    self._key_released = True
                if self.mode == "hold" and self._active:
                    self._handle_release()

        self._listener = keyboard.Listener(on_press=on_press, on_release=on_release)
        self._listener.daemon = True
        self._listener.start()
        log.info("Hotkey listener started (pynput): %s [%s mode]", self.binding, self.mode)

    def _start_evdev(self) -> None:
        """Start hotkey listener using evdev (Linux, works on Wayland)."""
        thread = threading.Thread(target=self._evdev_loop, daemon=True)
        thread.start()
        self._thread = thread
        log.info("Hotkey listener started (evdev): %s [%s mode]", self.binding, self.mode)

    def _evdev_loop(self) -> None:
        """Main loop for evdev-based hotkey detection."""
        import evdev
        from evdev import ecodes

        key_map = {
            "alt": {ecodes.KEY_LEFTALT, ecodes.KEY_RIGHTALT},
            "ctrl": {ecodes.KEY_LEFTCTRL, ecodes.KEY_RIGHTCTRL},
            "control": {ecodes.KEY_LEFTCTRL, ecodes.KEY_RIGHTCTRL},
            "shift": {ecodes.KEY_LEFTSHIFT, ecodes.KEY_RIGHTSHIFT},
            "cmd": {ecodes.KEY_LEFTMETA, ecodes.KEY_RIGHTMETA},
            "super": {ecodes.KEY_LEFTMETA, ecodes.KEY_RIGHTMETA},
            "meta": {ecodes.KEY_LEFTMETA, ecodes.KEY_RIGHTMETA},
        }

        letter_map = {
            chr(c): getattr(ecodes, f"KEY_{chr(c).upper()}", None)
            for c in range(ord("a"), ord("z") + 1)
        }

        target_key = letter_map.get(self._key)
        if target_key is None:
            log.error("Unsupported key for evdev: %s", self._key)
            return

        required_codes: set[int] = set()
        for mod in self._modifiers:
            codes = key_map.get(mod, set())
            required_codes.update(codes)

        # Find keyboard devices
        devices = []
        for path in evdev.list_devices():
            dev = evdev.InputDevice(path)
            caps = dev.capabilities()
            if ecodes.EV_KEY in caps and target_key in caps[ecodes.EV_KEY]:
                devices.append(dev)
                log.debug("Using input device: %s (%s)", dev.name, dev.path)
            else:
                dev.close()

        if not devices:
            log.error("No suitable input devices found for evdev")
            return

        pressed: set[int] = set()

        import select

        try:
            while not self._stop_event.is_set():
                r, _, _ = select.select(devices, [], [], 1.0)
                for dev in r:
                    try:
                        for event in dev.read():
                            if event.type != ecodes.EV_KEY:
                                continue

                            if event.value == 1:  # key down
                                pressed.add(event.code)
                            elif event.value == 0:  # key up
                                pressed.discard(event.code)

                            # Check if our hotkey combo is active
                            mods_held = all(
                                any(c in pressed for c in key_map.get(mod, set()))
                                for mod in self._modifiers
                            )

                            if event.code == target_key:
                                if event.value == 1 and mods_held:
                                    self._handle_press()
                                elif event.value == 0:
                                    with self._lock:
                                        self._key_released = True
                                    if self.mode == "hold" and self._active:
                                        self._handle_release()

                            # Hold mode: modifier release deactivates
                            if (
                                self.mode == "hold"
                                and self._active
                                and event.value == 0
                                and event.code in required_codes
                            ):
                                if not mods_held:
                                    self._handle_release()

                    except OSError:
                        log.warning("evdev device removed: %s", dev.path)
                        devices.remove(dev)
                        try:
                            dev.close()
                        except Exception:
                            pass
                        if not devices:
                            log.error("All input devices lost")
                            return
                    except Exception:
                        log.debug("evdev read error", exc_info=True)
        finally:
            for dev in devices:
                try:
                    dev.close()
                except Exception:
                    pass

    def _handle_press(self) -> None:
        import time

        callback = None
        with self._lock:
            if self.mode == "toggle":
                now = time.monotonic()
                if now - self._last_toggle_time < 0.15:
                    log.debug("Toggle debounce — ignoring duplicate press")
                    return
                # Require a physical key release before accepting toggle-off.
                # X11 auto-repeat sends continuous press events while held —
                # without this guard, one leaks past the time debounce and
                # triggers an unwanted toggle-off.
                if not self._key_released:
                    log.debug("Toggle debounce — key not released yet")
                    return
                self._last_toggle_time = now
                self._key_released = False
                if self._active:
                    self._active = False
                    log.debug("Toggle OFF")
                    callback = self.on_deactivate
                else:
                    self._active = True
                    log.debug("Toggle ON")
                    callback = self.on_activate
            elif self.mode == "hold":
                # Cancel any pending release (watcher checks stamp)
                self._release_stamp = 0.0
                if not self._active:
                    self._active = True
                    log.debug("Hold ON")
                    callback = self.on_activate
        if callback is not None:
            callback()

    def _handle_release(self) -> None:
        """Release handler — only meaningful in hold mode.

        Sets a monotonic timestamp and ensures a single watcher thread is
        running.  The watcher sleeps for the debounce interval and then
        checks whether the stamp is still current (no re-press cleared it).
        This eliminates the Timer-accumulation bug where many Timer threads
        compete and one fires in the gap between an auto-repeat release and
        the next press.
        """
        if self.mode != "hold":
            return
        with self._lock:
            if not self._active:
                return
            self._release_stamp = time.monotonic()
            if self._release_watcher is None or not self._release_watcher.is_alive():
                self._release_watcher = threading.Thread(
                    target=self._release_watcher_loop, daemon=True,
                )
                self._release_watcher.start()

    def _release_watcher_loop(self) -> None:
        """Single watcher thread: sleep → check stamp → fire or exit.

        At most one instance runs at a time (checked under lock before
        spawning).  Each re-press clears the stamp, which tells this
        thread to exit or re-wait.
        """
        debounce = _HOLD_DEBOUNCE_S
        sleep_time = debounce
        while not self._stop_event.is_set():
            time.sleep(sleep_time)
            callback = None
            with self._lock:
                if self._stop_event.is_set():
                    return
                if self._release_stamp == 0.0:
                    return  # cancelled by re-press — exit thread
                elapsed = time.monotonic() - self._release_stamp
                if elapsed >= debounce:
                    # No re-press arrived within window — real release
                    self._active = False
                    self._release_stamp = 0.0
                    log.debug("Hold OFF")
                    callback = self.on_deactivate
                else:
                    # Stamp refreshed by new release — sleep only remaining time
                    sleep_time = max(0.01, debounce - elapsed)
            if callback is not None:
                if not self._stop_event.is_set():
                    callback()
                return
