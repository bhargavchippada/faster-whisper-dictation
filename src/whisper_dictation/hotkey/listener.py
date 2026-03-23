"""Cross-platform hotkey listener supporting hold and toggle modes.

Uses pynput for macOS/Windows/Linux-X11, with evdev fallback for Wayland.
"""

from __future__ import annotations

import logging
import os
import sys
import threading
from collections.abc import Callable

log = logging.getLogger(__name__)


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
        self._listener = None
        self._stop_event = threading.Event()
        self._lock = threading.Lock()

    def _use_evdev(self) -> bool:
        """Check if we should use evdev (Linux Wayland)."""
        if sys.platform != "linux":
            return False
        session_type = os.environ.get("XDG_SESSION_TYPE", "")
        if session_type == "wayland":
            try:
                import evdev  # noqa: F401

                return True
            except ImportError:
                log.warning(
                    "Wayland detected but python-evdev not installed. "
                    "Falling back to pynput (may not work). "
                    "Install with: pip install evdev"
                )
        return False

    def start(self) -> None:
        """Start listening for the hotkey in a background thread."""
        if self._use_evdev():
            self._start_evdev()
        else:
            self._start_pynput()

    def stop(self) -> None:
        """Stop listening."""
        self._stop_event.set()
        if self._listener is not None:
            try:
                self._listener.stop()
            except Exception:
                pass
            self._listener = None

    def _start_pynput(self) -> None:
        """Start hotkey listener using pynput."""
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
                # In hold mode, if modifier released while active, deactivate
                if self.mode == "hold" and self._active and not _is_modifier_held():
                    self._handle_release()

            name = _key_name(key)
            if name == self._key:
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
            if ecodes.EV_KEY in caps:
                keys = caps[ecodes.EV_KEY]
                if target_key in keys:
                    devices.append(dev)
                    log.debug("Using input device: %s (%s)", dev.name, dev.path)

        if not devices:
            log.error("No suitable input devices found for evdev")
            return

        pressed: set[int] = set()

        import select

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
                            elif event.value == 0 and self.mode == "hold" and self._active:
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

                except Exception:
                    log.debug("evdev read error", exc_info=True)

    def _handle_press(self) -> None:
        with self._lock:
            if self.mode == "toggle":
                if self._active:
                    self._active = False
                    log.debug("Toggle OFF")
                    self.on_deactivate()
                else:
                    self._active = True
                    log.debug("Toggle ON")
                    self.on_activate()
            elif self.mode == "hold":
                if not self._active:
                    self._active = True
                    log.debug("Hold ON")
                    self.on_activate()

    def _handle_release(self) -> None:
        with self._lock:
            if self._active:
                self._active = False
                log.debug("Hold OFF")
                self.on_deactivate()
