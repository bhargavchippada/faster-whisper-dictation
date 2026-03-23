# CLAUDE.md — faster-whisper-dictation

## Project Overview

Real-time speech-to-text dictation tool. Captures microphone audio, detects speech via Silero VAD, transcribes via faster-whisper (REST API or local), and types text into the focused application. Cross-platform (Linux X11/Wayland, macOS, Windows).

## Architecture

```
src/whisper_dictation/
├── cli.py              # CLI: start, stop, status, devices, transcribe
├── config.py           # TOML config + env vars + CLI flags (frozen dataclasses)
├── daemon.py           # Main daemon: hotkey → audio → VAD → engine → typer
├── audio.py            # Audio capture via sounddevice, WAV conversion
├── vad.py              # Silero VAD speech detection (ONNX, no PyTorch required)
├── typer.py            # Platform-aware text input (clipboard + paste)
├── notifier.py         # Cross-platform desktop notifications
├── engine/
│   ├── __init__.py     # Package exports + create_engine() factory
│   ├── base.py         # TranscriptionEngine ABC
│   ├── server.py       # REST API engine (Speaches, OpenAI-compatible)
│   └── local.py        # Local faster-whisper engine (optional dependency)
└── hotkey/
    ├── __init__.py     # Public HotkeyListener export
    └── listener.py     # pynput (macOS/Windows/X11) + evdev (Wayland)
```

## Key Design Decisions

- **Immutable config**: All config dataclasses are `frozen=True`. Build new instances, never mutate.
- **Engine factory**: `create_engine()` in `engine/__init__.py` centralizes engine instantiation. `TranscriptionEngine` ABC allows swapping server/local backends transparently.
- **Silero VAD over RMS energy**: Silero is ML-based, far more accurate for speech detection.
- **ONNX by default**: VAD uses ONNX Runtime, not PyTorch, to keep the dependency footprint small.
- **pynput + evdev**: pynput handles macOS/Windows/X11; evdev handles Linux Wayland where pynput fails.
- **No shell scripts**: Everything is Python. The legacy `scripts/` directory is from the pre-rewrite era.

## Security Considerations

- **No command injection**: All subprocess calls use list arguments, never shell=True or f-strings.
  Windows clipboard uses Win32 API directly instead of PowerShell to avoid injection.
- **Clipboard hygiene**: Previous clipboard contents are saved before paste and restored after
  via `finally` blocks to ensure restoration even on exceptions.
  Wayland `wl-copy` uses `--` to prevent argument injection from text starting with `-`.
- **PID file locking**: PID file uses `fcntl.flock` for exclusive access on Unix, with `os.kill(pid, 0)` for liveness checks. Lock fd stored in module-level `_pid_lock_fd`.
- **No network exposure**: Docker server binds to `127.0.0.1` only. Audio never leaves localhost.
- **VAD model integrity**: ONNX model is fetched from GitHub over HTTPS with a 60s timeout and cached locally. SHA-256 hash verification is opt-in via `DICTATION_VAD_VERIFY_HASH=true`. Custom model URLs (`DICTATION_VAD_MODEL_URL`) are validated to use http/https scheme at import time.
- **Windows clipboard safety**: `OpenClipboard` return values are checked; allocated memory is freed on failure.
- **Server URL validation**: `validate()` checks that `server.url` uses http/https scheme and has a valid hostname. Prevents SSRF via config injection.
- **Input validation**: Config values are type-checked via frozen dataclasses with explicit validation. Environment variable overrides are coerced with clear error messages on type mismatch.
- **Paste delay validation**: `DICTATION_PASTE_DELAY` is validated at import time (must be 0.0-10.0, rejects NaN/Inf).
- **Error resilience**: Transcription and typing exceptions are caught and logged without crashing the daemon. Audio stream start failures reset recording state and notify the user.
- **AppleScript sanitization**: Notification messages strip null bytes and control characters before interpolation into AppleScript strings.

## Development

```bash
# Install in dev mode
uv sync --dev

# Run tests
uv run pytest -v

# Run tests with coverage
uv run pytest tests/ --cov=whisper_dictation --cov-report=term-missing

# Run with verbose logging
uv run faster-whisper-dictation -v start

# Lint
uv run ruff check src/ tests/
```

## Testing

- Tests use pytest with mocking for hardware dependencies (audio, keyboard, clipboard).
- No tests should require a running Whisper server, microphone, or display server.
- Use `unittest.mock.patch` for all external subprocess calls and hardware interfaces.
- Target: 100% test coverage. All new code must include tests.

## CI

GitHub Actions runs on every push/PR to main:
- Python 3.10, 3.11, 3.12, 3.13, 3.14 matrix
- Lint with ruff
- Tests with coverage gate (minimum 80%)
- 306 tests, 100% coverage

## Config Priority (highest to lowest)

1. CLI flags (`--hotkey`, `--mode`, etc.)
2. Environment variables (`DICTATION_HOTKEY`, etc.)
3. Config file (`~/.config/faster-whisper-dictation/config.toml`)
4. Defaults in dataclass definitions

## Platform Support Matrix

| Feature | Linux X11 | Linux Wayland | macOS | Windows |
|---------|-----------|---------------|-------|---------|
| Hotkey | pynput | evdev | pynput | pynput |
| Typing | xdotool+xclip | ydotool+wl-clipboard | pbcopy+osascript | ctypes (Win32 API) |
| Notify | notify-send | notify-send | osascript | plyer |
| Audio | sounddevice | sounddevice | sounddevice | sounddevice |

## Conventions

- Package name: `faster-whisper-dictation` (PyPI), `whisper_dictation` (import)
- CLI command: `faster-whisper-dictation`
- Python 3.10+ required
- No shell scripts — everything is Python
- Type hints on all public functions
- Logging via `logging` module, not print statements
- Immutable data patterns — frozen dataclasses, `dataclasses.replace()` for overrides
- All errors handled explicitly — no silent swallowing
- Use `collections.abc.Callable` (not `typing.Callable`)
- Config overrides via `dataclasses.replace()`, not manual field reconstruction
