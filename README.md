# faster-whisper-dictation

[![CI](https://github.com/bhargavchippada/faster-whisper-dictation/actions/workflows/ci.yml/badge.svg)](https://github.com/bhargavchippada/faster-whisper-dictation/actions/workflows/ci.yml)
[![Python 3.10--3.14](https://img.shields.io/badge/python-3.10--3.14-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Real-time speech-to-text dictation powered by [faster-whisper](https://github.com/SYSTRAN/faster-whisper). Speak and watch text appear instantly in any application -- fully offline, no cloud APIs, no data leaves your machine.

<p align="center">
  <img src="https://raw.githubusercontent.com/bhargavchippada/faster-whisper-dictation/main/assets/demo-server.gif" alt="Demo: server mode with hold-to-talk" width="740">
</p>

## How it works

```
Microphone --> Silero VAD --> WhisperLiveKit Server --> Type into focused app
(sounddevice)  (local)       (WebSocket / REST)        (platform-native)
```

Audio is captured from your microphone, speech boundaries are detected locally using [Silero VAD](https://github.com/snakers4/silero-vad), each complete utterance is sent to a [WhisperLiveKit](https://github.com/QuentinFuxa/WhisperLiveKit) server via WebSocket for transcription, and the result is typed into whatever application has focus.

## Why local Whisper?

Cloud dictation services (Google, Apple, Microsoft) send your audio to remote servers. Every word you speak is processed, stored, and potentially used for training -- even sensitive conversations, passwords spoken aloud, or private thoughts.

**faster-whisper-dictation keeps everything on your machine:**

- **Zero network dependency** -- audio never leaves your computer
- **No accounts or API keys** -- install and run, no sign-up required
- **No telemetry** -- the tool collects nothing about your usage
- **Full model control** -- you choose which Whisper model to run and where
- **Audit-friendly** -- open source, read every line of what handles your audio

Even in server mode, the default configuration binds to `localhost` -- your audio stays on your machine.
Recent live benchmarks on the current build showed the daemon averaging `0.00%` CPU while idle in server mode.

## Features

- **Batch transcription** -- speak a full utterance, release the hotkey, and the complete text is typed at once (default, most accurate)
- **Hold-to-talk** -- hold the hotkey to dictate, release to stop
- **Toggle mode** -- press hotkey to start, press again to stop
- **Configurable hotkey** -- default `Alt+V`, fully customizable
- **Background daemon** -- `start -b` detaches from terminal, logs to file
- **Cross-platform** -- Linux (X11 + Wayland), macOS, Windows
- **WhisperLiveKit backend** -- server mode uses WhisperLiveKit via WebSocket (int16 PCM); also exposes an OpenAI-compatible REST API
- **Local engine fallback** -- optional built-in faster-whisper engine, no server needed
- **Fully offline** -- all processing happens on your machine
- **Privacy-first** -- no cloud, no accounts, no telemetry
- **Streaming mode** *(experimental)* -- `--streaming` sends partial audio for real-time text, but quality is lower than batch mode

## Install

Requires Python 3.10+.

```bash
# Install with uv (recommended)
uv tool install faster-whisper-dictation

# Or with pipx
pipx install faster-whisper-dictation

# Or with pip
pip install faster-whisper-dictation

# Build release artifacts from a checkout
uv build --clear --no-cache
```

### Optional: local engine (no server needed)

```bash
# CPU only
uv tool install "faster-whisper-dictation[local]"

# With NVIDIA GPU acceleration
uv tool install "faster-whisper-dictation[local-gpu]"
```

### Platform dependencies

<details>
<summary><b>Linux (X11)</b></summary>

```bash
sudo apt install -y xdotool xclip libportaudio2 libnotify-bin
```
</details>

<details>
<summary><b>Linux (Wayland)</b></summary>

```bash
sudo apt install -y wl-clipboard ydotool libportaudio2 libnotify-bin
sudo systemctl enable --now ydotool
sudo usermod -aG input $USER   # then re-login
```
</details>

<details>
<summary><b>macOS / Windows</b></summary>

No additional system dependencies needed.
</details>

## Quick start

### Option A: WhisperLiveKit server (recommended)

WhisperLiveKit is a pip-installable Whisper server. No Docker required.

```bash
# 1. Install WhisperLiveKit (requires Python 3.11+)
pip install whisperlivekit          # CPU
pip install whisperlivekit[gpu]     # GPU (requires CUDA)

# 2. Start the server
wlk --model large-v3 --language en

# 3. Install and start dictation (in another terminal)
pip install faster-whisper-dictation
faster-whisper-dictation start

# 4. Press Alt+V to start/stop dictation
```

### Option B: Local engine (no server needed)

```bash
# Install with built-in faster-whisper engine
uv tool install "faster-whisper-dictation[local]"

# Start (downloads model on first run, ~3GB)
faster-whisper-dictation start --engine local
```

### Generate a config file (optional)

```bash
# Create a commented config file with all defaults
faster-whisper-dictation config --generate

# View current settings
faster-whisper-dictation config
```

## Usage

```bash
# Start the dictation daemon (toggle mode, default)
faster-whisper-dictation start

# Start in hold-to-talk mode
faster-whisper-dictation start --mode hold

# Use a custom hotkey
faster-whisper-dictation start --hotkey "ctrl+shift+d"

# Use a different server
faster-whisper-dictation start --server-url http://my-server:8000

# Use local engine instead of server
faster-whisper-dictation start --engine local

# Experimental: real-time streaming (lower accuracy, WIP)
faster-whisper-dictation start --streaming

# Run as a background daemon (Unix only, no need for &)
faster-whisper-dictation start -b
faster-whisper-dictation start --background --mode hold

# Check status
faster-whisper-dictation status

# Stop the daemon
faster-whisper-dictation stop

# List audio devices
faster-whisper-dictation devices

# Transcribe a file
faster-whisper-dictation transcribe recording.wav

# Record and transcribe
faster-whisper-dictation transcribe --record 5

# Show current config
faster-whisper-dictation config

# Generate default config file
faster-whisper-dictation config --generate
```

## Configuration

Settings can be configured via CLI flags, environment variables, or config file. Priority: CLI flags > env vars > config file > defaults.

Config file location: `~/.config/faster-whisper-dictation/config.toml`

```toml
[server]
url = "http://localhost:8000"
model = "Systran/faster-whisper-large-v3"
language = "en"
timeout = 10            # request timeout in seconds
# prompt = ""           # bias transcription (e.g. domain vocabulary)
# temperature = 0.0     # 0.0 = accurate, higher = creative
# hotwords = ""         # comma-separated words to boost recognition

[hotkey]
binding = "alt+v"       # any key combo supported by your platform
mode = "toggle"         # "toggle" or "hold"

[vad]
threshold = 0.5         # Silero VAD confidence threshold (0.0-1.0)
silence_ms = 200        # silence duration to end an utterance
min_speech_ms = 250     # minimum speech duration to accept
max_speech_s = 90.0     # max single utterance duration (seconds)

[audio]
sample_rate = 16000
channels = 1
device = null           # null = system default, or device name/index

[engine]
type = "server"         # "server" or "local"
compute_type = "float16" # "float16" (GPU), "int8" (CPU), "auto"
device = "auto"          # "auto", "cuda", "cpu"

[websocket]
reconnect_attempts = 3  # retries on connection failure
reconnect_delay = 1.0   # seconds between retries
```

### Environment variables

| Variable | Default | Description |
|----------|---------|-------------|
| `WHISPER_SERVER_URL` | `http://localhost:8000` | Whisper server URL |
| `WHISPER_MODEL` | `Systran/faster-whisper-large-v3` | Model name |
| `WHISPER_LANG` | `en` | Language code |
| `WHISPER_TIMEOUT` | `10` | Request timeout (seconds) |
| `WHISPER_PROMPT` | (empty) | Bias transcription (e.g. domain vocabulary) |
| `WHISPER_TEMPERATURE` | `0.0` | Transcription temperature (0.0 = accurate) |
| `WHISPER_HOTWORDS` | (empty) | Comma-separated words to boost recognition |
| `DICTATION_HOTKEY` | `alt+v` | Hotkey binding |
| `DICTATION_MODE` | `toggle` | `toggle` or `hold` |
| `DICTATION_ENGINE` | `server` | `server` or `local` |
| `DICTATION_ENGINE_COMPUTE` | `auto` | Compute type: `float16`, `int8`, `auto` |
| `DICTATION_ENGINE_DEVICE` | `auto` | Device: `cuda`, `cpu`, `auto` |
| `DICTATION_AUDIO_DEVICE` | (system default) | Audio input device name |
| `DICTATION_SAMPLE_RATE` | `16000` | Audio sample rate (Hz) |
| `DICTATION_VAD_THRESHOLD` | `0.5` | VAD confidence threshold (0.0-1.0) |
| `DICTATION_VAD_SILENCE_MS` | `200` | Silence duration to end utterance (ms) |
| `DICTATION_VAD_MIN_SPEECH_MS` | `250` | Minimum speech duration to accept (ms) |
| `DICTATION_VAD_MAX_SPEECH_S` | `90.0` | Maximum single utterance duration (s) |
| `DICTATION_VAD_MODEL_URL` | (pinned release) | Custom Silero VAD ONNX model URL |
| `DICTATION_VAD_VERIFY_HASH` | `false` | Enable SHA-256 hash verification on model download |
| `DICTATION_WS_RECONNECT_ATTEMPTS` | `3` | WebSocket reconnection attempts |
| `DICTATION_WS_RECONNECT_DELAY` | `1.0` | Delay between reconnection attempts (s) |

## Architecture

```
faster-whisper-dictation/
├── src/whisper_dictation/
│   ├── cli.py              # CLI: start, stop, status, config, devices, transcribe
│   ├── config.py           # TOML config + env vars + CLI flags + validation
│   ├── daemon.py           # Main daemon: hotkey -> audio -> VAD -> engine -> typer
│   ├── engine/
│   │   ├── __init__.py     # create_engine() factory
│   │   ├── base.py         # TranscriptionEngine ABC
│   │   ├── server.py       # REST API engine (OpenAI-compatible, fallback)
│   │   ├── whisperlivekit.py # WhisperLiveKit WebSocket engine (batch + streaming)
│   │   └── local.py        # Local faster-whisper engine
│   ├── hotkey/
│   │   └── listener.py     # pynput + evdev hotkey detection
│   ├── audio.py            # Audio capture via sounddevice
│   ├── vad.py              # Silero VAD (ONNX, SHA-256 verified)
│   ├── typer.py            # Platform-aware text input (clipboard + paste)
│   └── notifier.py         # Cross-platform desktop notifications
├── tests/                  # 478 tests, 100% coverage
├── .github/workflows/      # CI: lint + test on Python 3.10-3.14
└── pyproject.toml          # Package config (uv/pip installable)
```

### Engine modes

| Mode | Backend | Setup | Best for |
|------|---------|-------|----------|
| **Server** (default) | [WhisperLiveKit](https://github.com/QuentinFuxa/WhisperLiveKit) via WebSocket | `pip install whisperlivekit && wlk --model large-v3` | GPU users, streaming + batch, shared servers |
| **Local** | Built-in faster-whisper | `pip install "faster-whisper-dictation[local]"` | Simple setup, single-user, offline |

Server mode uses WebSocket for both batch and streaming transcription (shared GPU model, lower latency). REST API is available as a fallback.

### Platform support

| Feature | Linux X11 | Linux Wayland | macOS | Windows |
|---------|-----------|---------------|-------|---------|
| Hotkey | pynput | evdev | pynput | pynput |
| Text input | xdotool + xclip | ydotool + wl-clipboard | pbcopy + osascript | ctypes |
| Notifications | notify-send | notify-send | osascript | plyer |
| Audio capture | sounddevice | sounddevice | sounddevice | sounddevice |

## WhisperLiveKit server

[WhisperLiveKit](https://github.com/QuentinFuxa/WhisperLiveKit) is a pip-installable Whisper transcription server that exposes both a WebSocket endpoint (`/asr`) for streaming int16 PCM audio and an OpenAI-compatible REST endpoint (`/v1/audio/transcriptions`) for batch transcription.

### Installation

```bash
# CPU mode
pip install whisperlivekit

# GPU mode (requires CUDA)
pip install whisperlivekit[gpu]
```

**Note:** WhisperLiveKit requires Python 3.11+. The dictation client itself works with Python 3.10+, so you may need separate environments if running both on the same machine.

### Running the server

```bash
# Basic usage
wlk --model large-v3 --language en

# With PCM input (used by the dictation client)
wlk --model large-v3 --language en --pcm-input

# Specify host and port
wlk --model large-v3 --language en --host 0.0.0.0 --port 8000
```

### Server capabilities

| Feature | Description |
|---------|-------------|
| WebSocket streaming | `/asr` endpoint, int16 PCM audio |
| REST API | `/v1/audio/transcriptions` (OpenAI-compatible) |
| GPU acceleration | CUDA via `whisperlivekit[gpu]` |
| CPU mode | Works without GPU, slower |
| Diarization | Speaker identification support |
| Translation | Translate speech to English |
| Multiple models | Any faster-whisper compatible model |

### Resource usage

| Setting | GPU mode | CPU mode |
|---------|----------|----------|
| Compute | NVIDIA CUDA (float16) | CPU (int8) |
| Memory | ~2GB VRAM | ~2GB RAM |
| Default port | `8000` | `8000` |

## API compatibility

The server exposes an OpenAI-compatible transcription endpoint. You can point `faster-whisper-dictation` at any compatible server:

```bash
# Use with a remote server
faster-whisper-dictation start --server-url https://my-whisper.example.com

# Use with Groq
faster-whisper-dictation start --server-url https://api.groq.com/openai
```

## Security

- **No command injection** -- all subprocess calls use list arguments, never `shell=True`. Windows clipboard uses Win32 API directly (no PowerShell). Wayland uses `--` separator to prevent flag injection.
- **Clipboard hygiene** -- previous clipboard is saved before paste and restored after via `finally` blocks, under a thread lock to prevent concurrent corruption.
- **PID file locking** -- exclusive `fcntl.flock` prevents duplicate daemon instances (falls back to simple PID on Windows).
- **Model integrity** -- ONNX VAD model downloads use a 60s timeout. SHA-256 verification is opt-in (`DICTATION_VAD_VERIFY_HASH=true`). Partial downloads are atomically cleaned up. Custom model URLs validated to use http/https.
- **Config validation** -- all values validated with clear error messages. Server URLs checked for http/https scheme. Invalid env vars rejected at startup.
- **Localhost by default** -- the dictation client connects to `localhost` by default. To restrict server network exposure, run `wlk --host 127.0.0.1`.
- **No telemetry** -- zero data collection, no phone-home, no analytics.
- **WebSocket safety** -- message size capped at 1MB, batch segment count capped at 1000 to prevent memory exhaustion. Non-loopback unencrypted WebSocket connections trigger a warning.

## Troubleshooting

| Problem | Solution |
|---------|----------|
| Hotkey not responding | Check `faster-whisper-dictation status`. On Wayland, ensure your user is in the `input` group. |
| "Server not reachable" | Start the WhisperLiveKit server: `wlk --model large-v3 --language en`. Or use `--engine local`. |
| No text appears | Verify your mic: `faster-whisper-dictation transcribe --record 5` |
| Wrong microphone | List devices with `faster-whisper-dictation devices` and set `audio.device` in config. |
| Text in wrong window | Text is typed into the focused window when transcription completes. Keep focus on target app. |
| Whisper hallucinations | Increase VAD threshold: `vad.threshold = 0.7` in config. |
| Wrong words (e.g. "passed" instead of "fast") | Set `server.prompt` or `server.hotwords` in config to bias transcription. |
| ydotool not working | Run `sudo systemctl start ydotool` and add user to `input` group. |

## Development

```bash
# Clone and install dev dependencies
git clone https://github.com/bhargavchippada/faster-whisper-dictation.git
cd faster-whisper-dictation
uv sync --dev

# Run tests
uv run pytest -v

# Run tests with coverage
uv run pytest tests/ --cov=whisper_dictation --cov-report=term-missing

# Build fresh artifacts without cache
uv build --clear --no-cache

# Lint
uv run ruff check src/ tests/
```

## Contributing

Contributions are welcome. Please open an issue first to discuss what you'd like to change.

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/my-change`)
3. Install dev dependencies: `uv sync --dev`
4. Write tests first, then implement
5. Ensure tests pass and coverage is maintained
6. Open a pull request

## License

[MIT](LICENSE)
