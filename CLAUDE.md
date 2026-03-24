# CLAUDE.md — faster-whisper-dictation

## Project Overview

Speech-to-text dictation tool. Captures microphone audio, detects speech boundaries via Silero VAD, transcribes via faster-whisper (WebSocket, REST API, or local), and types text into the focused application. Supports batch mode (default, highest accuracy) and streaming mode (experimental, real-time). Server mode uses WhisperLiveKit via WebSocket for both batch and streaming. Cross-platform (Linux X11/Wayland, macOS, Windows).

## Architecture

```
src/whisper_dictation/
├── cli.py              # CLI: start, stop, status, config, devices, transcribe
├── config.py           # TOML config + env vars + CLI flags (frozen dataclasses)
├── daemon.py           # Main daemon: hotkey → audio → VAD → engine → typer (batch + streaming)
├── audio.py            # Audio capture via sounddevice, WAV conversion
├── vad.py              # Silero VAD speech detection (ONNX, no PyTorch required)
├── typer.py            # Platform-aware text input (clipboard + paste)
├── notifier.py         # Cross-platform desktop notifications
├── engine/
│   ├── __init__.py     # Package exports + create_engine() factory
│   ├── base.py         # TranscriptionEngine ABC
│   ├── server.py       # REST API engine (OpenAI-compatible, fallback)
│   ├── local.py        # Local faster-whisper engine (optional dependency)
│   └── whisperlivekit.py # WhisperLiveKit WebSocket engine (streaming + batch)
└── hotkey/
    ├── __init__.py     # Public HotkeyListener export
    └── listener.py     # evdev (Linux preferred) + pynput (macOS/Windows/X11 fallback)
```

## Key Design Decisions

- **Immutable config**: All config dataclasses are `frozen=True`. Build new instances, never mutate.
- **Engine factory**: `create_engine()` in `engine/__init__.py` centralizes REST/local engine instantiation. `create_ws_engine()` creates `WhisperLiveKitEngine` for WebSocket streaming/batch. `TranscriptionEngine` ABC allows swapping server/local backends transparently.
- **Silero VAD over RMS energy**: Silero is ML-based, far more accurate for speech detection.
- **ONNX by default**: VAD uses ONNX Runtime, not PyTorch, to keep the dependency footprint small.
- **pynput + evdev**: evdev is preferred on all Linux (properly blocks via select, distinguishes real key-up from auto-repeat). pynput is fallback for macOS/Windows/X11 without input-device permissions. Callbacks fire outside the hotkey lock to prevent deadlocks.
- **Hold mode debounce (pynput)**: X11 auto-repeat generates synthetic release+press pairs that break hold-to-talk. A single watcher thread uses monotonic timestamps — each release sets a stamp, each re-press clears it. The watcher sleeps for `_HOLD_DEBOUNCE_S` (250ms default, env `DICTATION_HOLD_DEBOUNCE_MS`), then checks if the stamp survived. At most one watcher thread runs at a time. evdev doesn't need this (value=2 events are auto-repeat, value=0 is real release).
- **Toggle mode key-release guard**: `_key_released` flag prevents X11 auto-repeat from toggling off. A toggle-off press is only accepted after a physical key release event. This prevents the pattern: Toggle ON → auto-repeat leaks past time debounce → Toggle OFF (0.2s).
- **Non-blocking notifications**: `subprocess.Popen` (not `.run`) for Linux/macOS so notifications never block the daemon.
- **Persistent HTTP session in server mode**: `ServerEngine` reuses a `requests.Session` to reduce per-request overhead when talking to the local STT server.
- **WhisperLiveKit WebSocket engine**: `WhisperLiveKitEngine` handles both streaming (real-time callbacks) and batch (synchronous) transcription via WhisperLiveKit's protocol (no client handshake, int16 PCM audio, `ready_to_stop` completion signal). Does NOT extend `TranscriptionEngine` ABC (async push model vs sync request/response).
- **Thread safety in WS engine**: `_seg_lock` protects `_latest_full_text`, `_streamed_stable_text`, and `_eoa_sent` shared between caller and receiver threads. `_END_SENTINEL = object()` uses identity check (`is`) not equality. Sender/receiver thread death sets `_stop_event` and `_flush_done` to unblock callers.
- **Word-boundary streaming emission**: `_process_response` emits only complete words (up to the last space via `rfind(" ")`). Buffer text (`buffer_transcription`) is excluded from streaming emission (unstable/partial). `startswith()` detects append-only growth vs in-place revision. Revision suppresses emission and resyncs the cursor. Trailing partial word emitted on deactivation via `get_pending_text()`.
- **Batch over streaming**: Batch mode sends complete utterances for transcription (highest accuracy). Streaming mode (`--streaming`) is experimental — sends partial audio chunks for real-time output but with lower quality. In server mode, both paths use WebSocket (WhisperLiveKit).
- **Background daemon**: `start -b` uses Unix double-fork (`_daemonize()`) to detach from terminal. Follows Stevens APUE: setsid, chdir("/"), closerange, O_APPEND log, O_CLOEXEC. Logs to `~/.config/faster-whisper-dictation/daemon.log`.

## Server Setup

WhisperLiveKit is pip-installable and serves as the WebSocket backend (default URL: `http://localhost:8000`):

```bash
uv tool install whisperlivekit

# Batch mode (default) — standard config:
LD_LIBRARY_PATH=/usr/local/lib/ollama/cuda_v12:$LD_LIBRARY_PATH \
  wlk serve --model large-v3 --language en --pcm-input

# Streaming mode — optimized for dictation quality:
LD_LIBRARY_PATH=/usr/local/lib/ollama/cuda_v12:$LD_LIBRARY_PATH \
  wlk serve --model large-v3 --language en --pcm-input \
  --min-chunk-size 1.5 --confidence-validation
```

### Streaming server flags

| Flag | Default | Recommended | Effect |
|------|---------|-------------|--------|
| `--min-chunk-size` | 0.1s | **1.5s** | Audio accumulated before processing. 0.1s = garbled slow speech; 1.5s = enough context |
| `--confidence-validation` | off | **on** | Commits high-confidence tokens immediately, reduces text flip-flopping |
| `--buffer_trimming` | segment | sentence | Sentence-based trimming for cleaner output (optional) |
| `--buffer_trimming_sec` | 15 | 25 | Longer audio context window (optional, uses more VRAM) |

**Do NOT use `--no-vac`**: VAC (server-side Voice Activity Controller) prevents silence from reaching Whisper. Without it, silence triggers hallucination loops ("Thank you" repeated endlessly). Keep VAC enabled (the default). The client-side Silero VAD is a separate layer that detects speech boundaries for segmentation — it does not replace server-side VAC.

**Note:** WhisperLiveKit requires CUDA 12 (`libcublas.so.12`). If your system has CUDA 13, set `LD_LIBRARY_PATH` to a directory containing CUDA 12 libs (e.g. from Ollama). Without this, the model silently produces empty transcriptions.

## Security Considerations

- **No command injection**: All subprocess calls use list arguments, never shell=True or f-strings.
  Windows clipboard uses Win32 API directly instead of PowerShell to avoid injection.
- **Clipboard hygiene**: Previous clipboard contents are saved before paste and restored after
  via `finally` blocks to ensure restoration even on exceptions.
  Wayland `wl-copy` uses `--` to prevent argument injection from text starting with `-`.
- **PID file locking**: PID file uses `fcntl.flock` for exclusive access on Unix, with `os.kill(pid, 0)` for liveness checks. Lock fd stored in module-level `_pid_lock_fd`.
- **VAD model integrity**: ONNX model is fetched from GitHub over HTTPS with a 60s timeout and cached locally. SHA-256 hash verification is opt-in via `DICTATION_VAD_VERIFY_HASH=true`. Custom model URLs (`DICTATION_VAD_MODEL_URL`) are validated to use http/https scheme at import time.
- **Windows clipboard safety**: `OpenClipboard` return values are checked; allocated memory is freed on failure.
- **Server URL validation**: `validate()` checks that `server.url` uses http/https scheme and has a valid hostname. Prevents SSRF via config injection.
- **Input validation**: Config values are type-checked via frozen dataclasses with explicit validation. Environment variable overrides are coerced with clear error messages on type mismatch.
- **Paste delay validation**: `DICTATION_PASTE_DELAY` is validated at import time (must be 0.0-10.0, rejects NaN/Inf).
- **Error resilience**: Transcription and typing exceptions are caught and logged without crashing the daemon. Audio stream start failures reset recording state and notify the user.
- **AppleScript sanitization**: Notification messages strip null bytes and control characters before interpolation into AppleScript strings.
- **WebSocket URL validation**: `_http_to_ws_url()` rejects non-http/https schemes with ValueError. `_is_loopback()` uses `ipaddress.ip_address().is_loopback` for full IPv4/v6 range. Unencrypted `ws://` to non-loopback triggers a warning.
- **WS message size cap**: `_MAX_MESSAGE_BYTES = 1MB` enforced at both websockets library level (`max_size`) and application level. Prevents memory exhaustion from malicious server responses.
- **WS batch lines cap**: `_MAX_BATCH_LINES = 1000` limits lines processed per message to prevent unbounded memory growth.
- **WS close() unblocks waiters**: `close()` sets `_flush_done` to prevent `wait_for_completion()` from hanging indefinitely.
- **WS engine snapshot pattern**: `_on_audio_chunk` snapshots `ws_engine = self._ws_engine` before use to prevent race with `_on_deactivate` nulling the reference.
- **Batch audio buffer cap**: `_max_batch_chunks` (derived from `max_speech_s / 0.032`) prevents unbounded memory growth from held hotkeys. Chunks beyond the cap are silently dropped.
- **VAD model download size limit**: `_MAX_MODEL_BYTES = 50MB` enforced via streaming download. Prevents memory exhaustion from malicious `DICTATION_VAD_MODEL_URL` responses.
- **VAD cache directory permissions**: Created with `mode=0o700` to prevent other users from replacing the cached model.
- **Evdev device removal handling**: `OSError` during `dev.read()` removes the device from the poll list and closes it, preventing CPU-saturating busy-loops when USB keyboards are unplugged.
- **Streaming repetition filter**: `_on_ws_text` suppresses hallucination loops (e.g. Whisper repeating "Thank you" during silence). Normalized comparison (lowercase, trailing punctuation stripped) allows 2 identical emissions, suppresses from 3rd. Resets on different text or new recording session.

## Daemon Management

Before starting the daemon, ALWAYS clean up existing processes:

```bash
# 1. Graceful stop first
faster-whisper-dictation stop

# 2. Verify — kill any orphans left by crashes or background launches
pgrep -f 'faster-whisper-dictation start' && pkill -f 'faster-whisper-dictation' || true

# 3. Then start fresh
faster-whisper-dictation start
```

Always try `faster-whisper-dictation stop` before resorting to `pkill`. Background `&` launches and crashes can leave orphan processes that `stop` won't find.

## Development

```bash
# Install in dev mode
uv sync --extra dev

# Run tests
uv run pytest -v

# Run tests with coverage
uv run pytest tests/ --cov=whisper_dictation --cov-report=term-missing

# Run with verbose logging
uv run faster-whisper-dictation -v start

# Lint
uv run ruff check src/ tests/

# Build clean artifacts without cache
uv build --clear --no-cache

# Install globally (editable — picks up code changes automatically)
uv tool install -e . --force
```

## Testing

- Tests use pytest with mocking for hardware dependencies (audio, keyboard, clipboard).
- No tests should require a running Whisper server, microphone, or display server.
- Use `unittest.mock.patch` for all external subprocess calls and hardware interfaces.
- Target: 100% test coverage. All new code must include tests.
- Tests must never hang — mock `wait_for_completion()` in batch tests with `patch.object(engine, 'wait_for_completion', return_value=True)`. See `tests/test_engine_whisperlivekit.py` for established patterns.
- Current status: 519 tests, 100% line coverage, runs in ~5s.

## CI

GitHub Actions runs on every push/PR to main:
- Python 3.10, 3.11, 3.12, 3.13, 3.14 matrix
- Lint with ruff
- Tests with coverage gate (minimum 80%)
- 519 tests, 100% coverage

## Performance Notes

- In server mode, idle daemon CPU should be effectively 0%. Recent live `pidstat` samples showed `0.00%` average CPU while idle.
- Server-mode transcription is dominated by localhost HTTP round-trips and backend execution, not local Python CPU.
- If CPU appears hot while idle on Linux, check whether the daemon fell back to `pynput`; `evdev` is preferred when input-device permissions are available.

## Config Priority (highest to lowest)

1. CLI flags (`--hotkey`, `--mode`, etc.)
2. Environment variables (`DICTATION_HOTKEY`, etc.)
3. Config file (`~/.config/faster-whisper-dictation/config.toml`)
4. Defaults in dataclass definitions

## Platform Support Matrix

| Feature | Linux X11 | Linux Wayland | macOS | Windows |
|---------|-----------|---------------|-------|---------|
| Hotkey | evdev (preferred) / pynput | evdev | pynput | pynput |
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
