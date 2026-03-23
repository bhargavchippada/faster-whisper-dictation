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
│   ├── base.py         # TranscriptionEngine ABC
│   ├── server.py       # REST API engine (Speaches, OpenAI-compatible)
│   └── local.py        # Local faster-whisper engine (optional dependency)
└── hotkey/
    ├── __init__.py     # Public HotkeyListener export
    └── listener.py     # pynput (macOS/Windows/X11) + evdev (Wayland)
```

## Key Design Decisions

- **Immutable config**: All config dataclasses are `frozen=True`. Build new instances, never mutate.
- **Engine abstraction**: `TranscriptionEngine` ABC allows swapping server/local backends transparently.
- **Silero VAD over RMS energy**: Silero is ML-based, far more accurate for speech detection.
- **ONNX by default**: VAD uses ONNX Runtime, not PyTorch, to keep the dependency footprint small.
- **pynput + evdev**: pynput handles macOS/Windows/X11; evdev handles Linux Wayland where pynput fails.
- **No shell scripts**: Everything is Python. The legacy `scripts/` directory is from the pre-rewrite era.

## Security Considerations

- **No command injection**: All subprocess calls use list arguments, never shell=True or f-strings.
  Windows clipboard uses Win32 API directly instead of PowerShell to avoid injection.
- **Clipboard hygiene**: Previous clipboard contents are saved before paste and restored after.
  Wayland `wl-copy` uses `--` to prevent argument injection from text starting with `-`.
- **PID file race conditions**: PID file uses `os.kill(pid, 0)` to verify process liveness before reuse.
- **No network exposure**: Docker server binds to `127.0.0.1` only. Audio never leaves localhost.
- **VAD model download**: ONNX model is fetched from GitHub over HTTPS and cached locally.
- **Input validation**: Config values are type-checked via frozen dataclasses.

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
- Python 3.10, 3.11, 3.12, 3.13 matrix
- Lint with ruff
- Tests with coverage gate (minimum 80%)

## Config Priority (highest to lowest)

1. CLI flags (`--hotkey`, `--mode`, etc.)
2. Environment variables (`DICTATION_HOTKEY`, etc.)
3. Config file (`~/.config/faster-whisper-dictation/config.toml`)
4. Defaults in dataclass definitions

## Platform Support Matrix

| Feature | Linux X11 | Linux Wayland | macOS | Windows |
|---------|-----------|---------------|-------|---------|
| Hotkey | pynput | evdev | pynput | pynput |
| Typing | xdotool+xclip | ydotool+wl-clipboard | pbcopy+osascript | ctypes+powershell |
| Notify | notify-send | notify-send | osascript | plyer |
| Audio | sounddevice | sounddevice | sounddevice | sounddevice |

## Conventions

- Package name: `faster-whisper-dictation` (PyPI), `whisper_dictation` (import)
- CLI command: `faster-whisper-dictation`
- Python 3.10+ required
- No shell scripts — everything is Python
- Type hints on all public functions
- Logging via `logging` module, not print statements
- Immutable data patterns — frozen dataclasses, no mutation
- All errors handled explicitly — no silent swallowing
