# CLAUDE.md — faster-whisper-dictation

## Project Overview

Real-time speech-to-text dictation tool. Captures microphone audio, detects speech via Silero VAD, transcribes via faster-whisper (REST API or local), and types text into the focused application.

## Architecture

```
src/whisper_dictation/
├── cli.py          # CLI entry points (start, stop, status, devices, transcribe)
├── config.py       # Config loading: TOML file → env vars → CLI flags
├── daemon.py       # Main daemon: ties hotkey → audio → VAD → engine → typer
├── audio.py        # Audio capture via sounddevice
├── vad.py          # Silero VAD speech detection (ONNX, no PyTorch required)
├── typer.py        # Platform-aware text input (xdotool/xclip, ydotool/wl-clipboard, macOS, Windows)
├── notifier.py     # Cross-platform desktop notifications
├── engine/
│   ├── base.py     # TranscriptionEngine ABC
│   ├── server.py   # REST API engine (Speaches, OpenAI-compatible)
│   └── local.py    # Local faster-whisper engine (optional dependency)
└── hotkey/
    ├── listener.py # HotkeyListener: pynput (macOS/Windows/X11) + evdev (Wayland)
    └── platform.py # Platform detection
```

## Key Design Decisions

- **Immutable config**: All config dataclasses are `frozen=True`. Build new instances, never mutate.
- **Engine abstraction**: `TranscriptionEngine` ABC allows swapping server/local backends transparently.
- **Silero VAD over RMS energy**: Silero is ML-based, far more accurate for speech detection.
- **ONNX by default**: VAD uses ONNX Runtime, not PyTorch, to keep the dependency footprint small.
- **pynput + evdev**: pynput handles macOS/Windows/X11; evdev handles Linux Wayland where pynput fails.

## Development

```bash
# Install in dev mode
uv sync --dev

# Run tests
uv run pytest -v

# Run with verbose logging
uv run faster-whisper-dictation -v start

# Lint
uv run ruff check src/ tests/
```

## Testing

- Tests use pytest with mocking for hardware dependencies (audio, keyboard, clipboard).
- No tests should require a running Whisper server, microphone, or display server.
- Use `unittest.mock.patch` for all external subprocess calls and hardware interfaces.

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
