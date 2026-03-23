# faster-whisper-dictation

Real-time speech-to-text dictation powered by [faster-whisper](https://github.com/SYSTRAN/faster-whisper). Speak and watch text appear instantly in any application — fully offline, no cloud APIs, no data leaves your machine.

## How it works

```
Microphone ──▶ Silero VAD ──▶ Whisper Server ──▶ Type into focused app
(sounddevice)  (local)        (REST API)         (platform-native)
```

Audio is captured from your microphone, speech boundaries are detected locally using [Silero VAD](https://github.com/snakers4/silero-vad), each utterance is sent to a Whisper server for transcription, and the result is typed into whatever application has focus — in real-time, as you speak.

## Features

- **Real-time streaming** — text appears as you speak, not after you stop
- **Hold-to-talk** — hold the hotkey to dictate, release to stop
- **Toggle mode** — press hotkey to start, press again to stop
- **Configurable hotkey** — default `Alt+V`, fully customizable
- **Cross-platform** — Linux (X11 + Wayland), macOS, Windows
- **Flexible backend** — works with any OpenAI-compatible STT server (local Docker, remote, Groq, etc.)
- **Local engine fallback** — optional built-in faster-whisper engine, no server needed
- **Fully offline** — all processing happens on your machine

## Install

Requires Python 3.10+.

```bash
# Install with uv (recommended)
uv tool install faster-whisper-dictation

# Or with pipx
pipx install faster-whisper-dictation

# Or with pip
pip install faster-whisper-dictation
```

### Optional: local engine (no Docker server needed)

```bash
# CPU only
uv tool install "faster-whisper-dictation[local]"

# With NVIDIA GPU acceleration
uv tool install "faster-whisper-dictation[local-gpu]"
```

### Platform dependencies

**Linux (X11):**
```bash
sudo apt install -y xdotool xclip libportaudio2 libnotify-bin
```

**Linux (Wayland):**
```bash
sudo apt install -y wl-clipboard ydotool libportaudio2 libnotify-bin
sudo systemctl enable --now ydotool
sudo usermod -aG input $USER   # then re-login
```

**macOS / Windows:** No additional system dependencies needed.

## Quick start

### Option A: With Docker server (recommended for GPU users)

```bash
# 1. Clone and start the whisper server
git clone https://github.com/yourusername/faster-whisper-dictation.git
cd faster-whisper-dictation

docker compose up -d          # GPU (NVIDIA CUDA)
# docker compose -f docker-compose.cpu.yml up -d   # CPU fallback

# 2. Start dictation (toggle mode)
faster-whisper-dictation start

# 3. Press Alt+V to start/stop dictation
```

### Option B: Local engine (no Docker needed)

```bash
# Install with local engine support
uv tool install "faster-whisper-dictation[local]"

# Start with local engine (downloads model on first run, ~3GB)
faster-whisper-dictation start --engine local
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
faster-whisper-dictation start --server-url http://my-server:10300

# Use local engine instead of server
faster-whisper-dictation start --engine local

# Check status
faster-whisper-dictation status

# Stop the daemon
faster-whisper-dictation stop

# Transcribe a file
faster-whisper-dictation transcribe recording.wav
```

## Configuration

Settings can be configured via CLI flags, environment variables, or config file.

Config file location: `~/.config/faster-whisper-dictation/config.toml`

```toml
[server]
url = "http://localhost:10300"
model = "Systran/faster-whisper-large-v3"
language = "en"

[hotkey]
binding = "alt+v"       # any key combo supported by your platform
mode = "toggle"         # "toggle" or "hold"

[vad]
threshold = 0.5         # Silero VAD confidence threshold (0.0-1.0)
silence_ms = 800        # silence duration to end an utterance
min_speech_ms = 250     # minimum speech duration to accept

[audio]
sample_rate = 16000
channels = 1
device = null           # null = system default, or device name/index

[engine]
type = "server"         # "server" or "local"
compute_type = "float16" # "float16" (GPU), "int8" (CPU), "auto"
device = "auto"          # "auto", "cuda", "cpu"
```

### Environment variables

All config options can be overridden via environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `WHISPER_SERVER_URL` | `http://localhost:10300` | Whisper server URL |
| `WHISPER_MODEL` | `Systran/faster-whisper-large-v3` | Model name |
| `WHISPER_LANG` | `en` | Language code |
| `DICTATION_HOTKEY` | `alt+v` | Hotkey binding |
| `DICTATION_MODE` | `toggle` | `toggle` or `hold` |
| `DICTATION_ENGINE` | `server` | `server` or `local` |

## Architecture

```
faster-whisper-dictation/
├── src/whisper_dictation/
│   ├── __init__.py
│   ├── cli.py              # CLI entry points
│   ├── config.py            # Configuration management
│   ├── daemon.py            # Background daemon
│   ├── engine/
│   │   ├── __init__.py
│   │   ├── base.py          # Engine interface
│   │   ├── server.py        # REST API engine (Option B)
│   │   └── local.py         # Local faster-whisper engine (Option A)
│   ├── hotkey/
│   │   ├── __init__.py
│   │   ├── listener.py      # Platform-aware hotkey detection
│   │   └── platform.py      # Platform detection utilities
│   ├── audio.py             # Audio capture and VAD
│   ├── vad.py               # Silero VAD wrapper
│   ├── typer.py             # Platform-aware text input
│   └── notifier.py          # Cross-platform notifications
├── docker-compose.yml       # GPU server
├── docker-compose.cpu.yml   # CPU server
├── pyproject.toml
└── README.md
```

### Engine modes

| Mode | Backend | Setup | Best for |
|------|---------|-------|----------|
| **Server** (default) | Docker container with Speaches | `docker compose up -d` | GPU users, shared servers, flexibility |
| **Local** | Built-in faster-whisper | `pip install faster-whisper-dictation[local]` | Simple setup, single-user, offline |

Both engines expose the same interface — the dictation daemon doesn't care where transcription happens.

## Docker server

The server component runs [Speaches](https://github.com/speaches-ai/speaches), which provides an OpenAI-compatible transcription API.

| Setting | GPU mode | CPU mode |
|---------|----------|----------|
| Compose file | `docker-compose.yml` | `docker-compose.cpu.yml` |
| Image | `speaches:0.9.0-rc.3-cuda` | `speaches:0.9.0-rc.3-cpu` |
| Compute | NVIDIA CUDA (float16) | CPU (int8) |
| Memory | ~600MB VRAM | ~2GB RAM |
| Port | `10300` (localhost) | `10300` (localhost) |

```bash
docker compose up -d      # start
docker compose logs -f    # view logs
docker compose down       # stop
```

## API compatibility

The server exposes an OpenAI-compatible transcription endpoint. You can point `faster-whisper-dictation` at any compatible server:

```bash
# Use with a remote server
faster-whisper-dictation start --server-url https://my-whisper.example.com

# Use with Groq
faster-whisper-dictation start --server-url https://api.groq.com/openai
```

## Troubleshooting

- **Hotkey not responding** — Check `faster-whisper-dictation status`. If not running, start with `faster-whisper-dictation start`. On Wayland, ensure your user is in the `input` group.
- **"Server not reachable"** — Start the Docker server: `docker compose up -d`. Or use local engine: `--engine local`.
- **No text appears** — Verify your microphone works: `faster-whisper-dictation transcribe --record 5` to record and transcribe 5 seconds.
- **Wrong microphone** — List devices with `faster-whisper-dictation devices` and set in config: `audio.device = "fifine Microphone"`.
- **Text appears in wrong window** — Text is typed into the focused window at the moment transcription completes. Keep focus on the target application.
- **Whisper hallucinations ("Thank you")** — The Silero VAD filters silence, but you can increase the threshold: `vad.threshold = 0.7`.
- **ydotool not working (Wayland)** — Ensure the daemon is running (`sudo systemctl start ydotool`) and your user is in the `input` group.

## Contributing

Contributions are welcome. Please open an issue first to discuss what you'd like to change.

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/my-change`)
3. Install dev dependencies: `uv sync --dev`
4. Run tests: `uv run pytest`
5. Open a pull request

## License

[MIT](LICENSE)
