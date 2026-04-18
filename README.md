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
- **Streaming mode** -- `--streaming` sends audio in real-time for live text output; best with fast, continuous speech (see [Streaming mode](#streaming-mode) below)

## Install

Requires Python 3.10+ and [uv](https://docs.astral.sh/uv/) (recommended) or pip.

```bash
# Install with uv (recommended — isolated env, globally available)
uv tool install faster-whisper-dictation

# Or with pip
pip install faster-whisper-dictation
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
sudo apt install -y xdotool xclip libportaudio2 libnotify-bin python3-evdev

# Recommended: enable evdev for reliable hold-to-talk mode
sudo usermod -aG input $USER   # then re-login
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

WhisperLiveKit is a separate Whisper transcription server that must be installed and running before starting the dictation client. No Docker required — it's pip/uv installable.

```bash
# 1. Install WhisperLiveKit (separate from the dictation client)
uv tool install whisperlivekit

# 2. Start the server (Terminal 1) — must be running before the client
wlk serve --model large-v3 --language en --pcm-input \
  --min-chunk-size 1.5 --confidence-validation

# If your system has CUDA 13+ but needs CUDA 12 libs (e.g. from Ollama):
LD_LIBRARY_PATH=/usr/local/lib/ollama/cuda_v12:$LD_LIBRARY_PATH \
  wlk serve --model large-v3 --language en --pcm-input \
  --min-chunk-size 1.5 --confidence-validation

# 3. Install and start dictation (Terminal 2)
uv tool install faster-whisper-dictation
faster-whisper-dictation start              # batch mode (most accurate)
faster-whisper-dictation start --streaming  # real-time streaming mode

# 4. Press Alt+V to start/stop dictation
```

> **Note:** The WhisperLiveKit server (`wlk`) and the dictation client (`faster-whisper-dictation`) are installed separately. The server must be running before starting the client. If you see "Server not reachable", make sure `wlk serve` is running in another terminal.

> **After login or USB keyboard reconnect:** The daemon must be restarted — the hotkey listener loses device handles. Run `faster-whisper-dictation stop && faster-whisper-dictation start -b`.

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

# Real-time streaming (requires server tuning, see Streaming mode section)
faster-whisper-dictation start --streaming
faster-whisper-dictation start --streaming --mode hold

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
# prompt = ""           # domain vocabulary or style example (not instructions)
# temperature = 0.0     # 0.0 = accurate, higher = creative
# hotwords = ""         # comma-separated words to boost recognition

[hotkey]
binding = "alt+v"       # modifiers + single letter, e.g. "alt+v" or "ctrl+shift+d"
mode = "toggle"         # "toggle" or "hold"

[vad]
threshold = 0.6         # Silero VAD confidence threshold (0.0-1.0)
silence_ms = 200        # silence duration to end an utterance
min_speech_ms = 250     # minimum speech duration to accept
max_speech_s = 90.0     # max single utterance duration (seconds)

[audio]
sample_rate = 16000
channels = 1
device = null           # null = system default, or device name/index

[engine]
type = "server"         # "server" or "local"
compute_type = "auto"    # "auto", "float16" (GPU), "int8" (CPU)
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
| `WHISPER_PROMPT` | (empty) | Domain vocabulary or style example for Whisper |
| `WHISPER_TEMPERATURE` | `0.0` | Transcription temperature (0.0 = accurate) |
| `WHISPER_HOTWORDS` | (empty) | Comma-separated words to boost recognition |
| `DICTATION_HOTKEY` | `alt+v` | Hotkey binding |
| `DICTATION_MODE` | `toggle` | `toggle` or `hold` |
| `DICTATION_ENGINE` | `server` | `server` or `local` |
| `DICTATION_ENGINE_COMPUTE` | `auto` | Compute type: `float16`, `int8`, `auto` |
| `DICTATION_ENGINE_DEVICE` | `auto` | Device: `cuda`, `cpu`, `auto` |
| `DICTATION_AUDIO_DEVICE` | (system default) | Audio input device name |
| `DICTATION_SAMPLE_RATE` | `16000` | Audio sample rate (Hz) |
| `DICTATION_VAD_THRESHOLD` | `0.6` | VAD confidence threshold (0.0-1.0) |
| `DICTATION_VAD_SILENCE_MS` | `200` | Silence duration to end utterance (ms) |
| `DICTATION_VAD_MIN_SPEECH_MS` | `250` | Minimum speech duration to accept (ms) |
| `DICTATION_VAD_MAX_SPEECH_S` | `90.0` | Maximum single utterance duration (s) |
| `DICTATION_VAD_MODEL_URL` | (pinned release) | Custom Silero VAD ONNX model URL |
| `DICTATION_VAD_VERIFY_HASH` | `false` | Enable SHA-256 hash verification on model download |
| `DICTATION_PASTE_DELAY` | `0.15` | Clipboard paste delay in seconds (0.0-10.0) |
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
├── tests/                  # 519 tests, 100% coverage
├── .github/workflows/      # CI: lint + test on Python 3.10-3.14
└── pyproject.toml          # Package config (uv/pip installable)
```

### Engine modes

| Mode | Backend | Setup | Best for |
|------|---------|-------|----------|
| **Server** (default) | [WhisperLiveKit](https://github.com/QuentinFuxa/WhisperLiveKit) via WebSocket | `uv tool install whisperlivekit && wlk serve --model large-v3 --pcm-input` | GPU users, streaming + batch, shared servers |
| **Local** | Built-in faster-whisper | `uv tool install "faster-whisper-dictation[local]"` | Simple setup, single-user, offline |

Server mode uses WebSocket for both batch and streaming transcription (shared GPU model, lower latency). REST API is available as a fallback.

### Platform support

| Feature | Linux X11 | Linux Wayland | macOS | Windows |
|---------|-----------|---------------|-------|---------|
| Hotkey | evdev (preferred) / pynput | evdev | pynput | pynput |
| Text input | xdotool + xclip | ydotool + wl-clipboard | pbcopy + osascript | ctypes |
| Notifications | notify-send | notify-send | osascript | plyer |
| Audio capture | sounddevice | sounddevice | sounddevice | sounddevice |

## WhisperLiveKit server

[WhisperLiveKit](https://github.com/QuentinFuxa/WhisperLiveKit) is a pip-installable Whisper transcription server that exposes both a WebSocket endpoint (`/asr`) for streaming int16 PCM audio and an OpenAI-compatible REST endpoint (`/v1/audio/transcriptions`) for batch transcription.

### Installation

```bash
# Recommended: install as a uv tool (isolated env, globally available)
uv tool install whisperlivekit

# Or with pip
pip install whisperlivekit
```

### Running the server

```bash
# Recommended (works for both batch and streaming):
wlk serve --model large-v3 --language en --pcm-input \
  --min-chunk-size 1.5 --confidence-validation

# If CUDA 12 libs are not in default path (e.g. system has CUDA 13):
LD_LIBRARY_PATH=/usr/local/lib/ollama/cuda_v12:$LD_LIBRARY_PATH \
  wlk serve --model large-v3 --language en --pcm-input \
  --min-chunk-size 1.5 --confidence-validation

# Specify host and port
wlk serve --model large-v3 --language en --pcm-input --host 0.0.0.0 --port 8000
```

> **CUDA 12 required:** WhisperLiveKit's faster-whisper backend needs `libcublas.so.12`. If your system has CUDA 13+, set `LD_LIBRARY_PATH` to include CUDA 12 libraries. Without this, the model loads but silently produces empty transcriptions.

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
- **Localhost by default** -- the dictation client connects to `localhost` by default. To restrict server network exposure, run `wlk serve --host 127.0.0.1`.
- **No telemetry** -- zero data collection, no phone-home, no analytics.
- **WebSocket safety** -- message size capped at 1MB, lines per server message capped at 1000 to prevent memory exhaustion. Non-loopback unencrypted WebSocket connections trigger a warning.

## Transcription quality

The default settings are tuned for accurate dictation out of the box. Whisper `large-v3` handles punctuation and capitalization well without any prompt.

### Batch vs streaming

| | Batch (default) | Streaming (`--streaming`) |
|---|---|---|
| **How it works** | Record full utterance → send all audio → type result | Send audio in real-time → type words as they arrive |
| **Accuracy** | Excellent — full audio context | Good for fast speech, weaker for slow/paused speech |
| **Latency** | Wait until you stop speaking | ~1.5s behind real-time |
| **Best for** | Careful dictation, slow speech, accuracy-first | Fast continuous dictation, real-time feedback |

**Recommendation:** Start with batch mode. Switch to streaming only if you need real-time feedback and speak at a natural-to-fast pace.

### Tuning tips

- **`server.hotwords`** — Comma-separated list of words to boost recognition. Useful for proper nouns, technical terms, or words Whisper frequently gets wrong. Example: `"FastAPI,PyTorch,Kubernetes,streaming,toggle"`.
- **`server.prompt`** — Empty by default. Whisper treats this as **text to emulate** (not instructions). Use it for domain vocabulary, e.g. `"We deployed the Kubernetes cluster and updated the Docker containers."` — this helps the model recognize specific terms. Do **not** write instructions like "Use proper punctuation" — Whisper will misinterpret them and produce worse output.
- **`server.temperature`** — Defaults to `0.0` (most deterministic). Higher values (0.2-0.5) produce more varied output but less accurate transcription.
- **`vad.threshold`** — Defaults to `0.6`. Controls how aggressively Silero VAD detects speech. Higher values (0.7-0.8) reduce false triggers from background noise but may clip quiet speech. Lower values (0.3-0.5) are more sensitive.
- **`vad.silence_ms`** — Defaults to `200`. How long to wait after speech stops before considering the utterance complete. Increase to `500-800` if your speech has natural pauses that get cut off.

### Model selection

| Model | Size | Speed | Accuracy | VRAM |
|-------|------|-------|----------|------|
| `large-v3` | 3GB | Slower | Best | ~3GB |
| `large-v3-turbo` | 1.6GB | Fast | Very good | ~2GB |
| `medium` | 1.5GB | Fast | Good | ~2GB |
| `small` | 500MB | Very fast | Acceptable | ~1GB |

Use `large-v3` for best quality (default). Use `large-v3-turbo` for a good speed/quality balance. Smaller models are faster but less accurate, especially for accented speech or technical vocabulary.

```bash
# Example: use turbo model for faster processing
wlk serve --model large-v3-turbo --language en --pcm-input \
  --min-chunk-size 1.5 --confidence-validation
```

### Dictating code (programming vocabulary)

Whisper transcribes programming terms like `git push` as "get push", `useState` as "use state", or `async` as "a sink" because they're under-represented in its training data. Fix this with Whisper's `initial_prompt` — a short text sample Whisper conditions on to bias decoding toward the vocabulary you actually use.

Two things to know, straight from the [official OpenAI Whisper prompting guide](https://developers.openai.com/cookbook/examples/whisper_prompting_guide):

- **Use natural prose, not a keyword list.** Whisper emulates the *style* of the prompt, so sentences that use the terms in context work much better than a comma-separated glossary.
- **Don't write instructions.** "Transcribe code correctly" gets typed verbatim. Write example sentences; Whisper copies the style.

Pass it on the server side via WhisperLiveKit's `--static-init-prompt` flag (persists across segments). Here's a stack-neutral general-purpose prompt for programmers:

```bash
LD_LIBRARY_PATH=/usr/local/lib/ollama/cuda_v12:$LD_LIBRARY_PATH \
  wlk serve --model large-v3 --language en --pcm-input \
  --min-chunk-size 1.5 --confidence-validation \
  --static-init-prompt "Earlier I ran git status, git pull, and git push origin main, then opened a pull request and resolved the merge conflicts. I committed the fix, rebased onto main, and cherry-picked a patch onto the release branch. In the code I wrote a Python function, a JavaScript handler, and a small shell script. The service speaks JSON over HTTPS to a REST API, reads from a SQL database, and writes to a queue. I grep the logs, tail the output, curl the endpoint, ssh into the server, and kill the stuck process. Async and await, promises, try and catch, null and undefined, string and integer, boolean and array — all standard. I install dependencies with npm and pip, build with docker, and debug with a stack trace."
```

This covers the common homophone traps (git verbs, async/await, null/undefined), universal tooling (grep, curl, ssh, kill, npm, pip, docker), and core protocols (JSON, HTTPS, REST, SQL) without locking in a specific framework.

**To extend for your stack:** append one or two sentences in natural prose (not a list) that use *your* terms in context — e.g. `"Our backend runs on FastAPI with PostgreSQL and Redis, and the frontend uses Next.js with Tailwind."` Keep the whole prompt under ~200 tokens; Whisper truncates at 224 and keeps only the last 224.

## Streaming mode

Streaming mode (`--streaming`) sends audio to the server in real-time and types text as it arrives, instead of waiting for the full utterance. This trades some accuracy for lower latency.

### Server setup for streaming

The default WhisperLiveKit config processes audio every 100ms, which produces garbled output for slow or paused speech. For dictation, increase the processing window:

```bash
# Optimized for streaming dictation:
wlk serve --model large-v3 --language en --pcm-input \
  --min-chunk-size 1.5 \
  --confidence-validation

# If CUDA 12 libs are not in default path (e.g. system has CUDA 13):
LD_LIBRARY_PATH=/usr/local/lib/ollama/cuda_v12:$LD_LIBRARY_PATH \
  wlk serve --model large-v3 --language en --pcm-input \
  --min-chunk-size 1.5 --confidence-validation

# Start the client in streaming mode (in another terminal):
faster-whisper-dictation start --streaming

# Or with hold-to-talk:
faster-whisper-dictation start --streaming --mode hold
```

### Server tuning flags

| Flag | Default | Recommended | Why |
|------|---------|-------------|-----|
| `--min-chunk-size` | 0.1s | **1.5** | Accumulates 1.5s of audio before running inference. Gives Whisper enough context for accurate decoding, especially with slow speech. |
| `--confidence-validation` | off | **on** | Commits high-confidence tokens immediately without waiting for LocalAgreement confirmation. Reduces text flip-flopping. |
| `--buffer_trimming` | segment | sentence | *(optional)* Sentence-based buffer trimming for cleaner output. |
| `--buffer_trimming_sec` | 15 | 25-30 | *(optional)* Keeps more audio context. Tradeoff: higher VRAM usage. |

> **Do NOT use `--no-vac`**: VAC (server-side Voice Activity Controller) prevents silence from reaching Whisper. Disabling it causes hallucination loops where Whisper repeats phrases like "Thank you" during silence. Keep VAC enabled (the default). This is a [known open issue](https://github.com/QuentinFuxa/WhisperLiveKit/issues/338) in WhisperLiveKit.

### Streaming quality notes

| Speaking style | Quality | Notes |
|---------------|---------|-------|
| Fast, continuous | Good | Enough audio context per processing window |
| Normal pace | Good with tuning | `--min-chunk-size 1.5` is key |
| Slow with pauses | Acceptable | Some words may be delayed; batch mode is better for this |

**Recommendation:** Use batch mode (default, no `--streaming`) for highest accuracy. Streaming is best for fast, continuous dictation where real-time feedback matters.

### Tips for best results

- **Speak at a natural pace** — streaming quality improves significantly with continuous speech vs. slow, fragmented speech with long pauses
- **Use a good microphone** — a headset or close-range mic reduces background noise and improves recognition
- **Set hotwords** for domain-specific vocabulary you use frequently (`server.hotwords` in config)
- **Restart the WLK server** if quality degrades after extended use — each new WebSocket session gets fresh state, but the server process benefits from a periodic restart
- **Use batch mode for important text** — switch between streaming (fast drafts) and batch (accurate final text) depending on the task

### Linux: hold mode requires evdev

On Linux, hold-to-talk in streaming mode works best with **evdev** (not pynput). evdev natively distinguishes real key releases from X11 auto-repeat, so hold mode works indefinitely.

```bash
# Add your user to the input group for evdev access:
sudo usermod -aG input $USER
# Then log out and log back in
```

Without evdev, pynput is used as a fallback. Hold mode with pynput has a 250ms debounce to handle X11 auto-repeat, but may still release prematurely on some systems. Toggle mode works with both backends.

## Troubleshooting

| Problem | Solution |
|---------|----------|
| Hotkey not responding | Check `faster-whisper-dictation status`. On Linux, ensure your user is in the `input` group (`sudo usermod -aG input $USER` then re-login) for evdev support. |
| Hotkey stops working after login/USB reconnect | The evdev listener loses device handles on logout/login or USB keyboard reconnect. Restart the daemon: `faster-whisper-dictation stop && faster-whisper-dictation start -b` |
| Hold mode releases early | On Linux, install evdev access (see above). Without it, pynput's X11 backend has auto-repeat issues. Tune with `DICTATION_HOLD_DEBOUNCE_MS=300`. |
| Streaming garbled/slow speech | Increase server `--min-chunk-size` (default 0.1s, try 1.5). See [Streaming mode](#streaming-mode). |
| "Server not reachable" | Start the WhisperLiveKit server: `wlk serve --model large-v3 --language en --pcm-input`. Or use `--engine local`. |
| No text appears | Verify your mic: `faster-whisper-dictation transcribe --record 5`. If it records silence, the OS is routing audio from the wrong device (see below). |
| "No speech detected" in logs | The mic is either muted or not the active input. Open your OS sound settings and confirm the correct microphone is selected as the default input device, and that its level is non-zero while you speak. On Linux check `pavucontrol` → Input Devices; on macOS check System Settings → Sound → Input; on Windows check Settings → System → Sound → Input. |
| Wrong microphone | Pick it at the OS level (default input device) or pin it explicitly: run `faster-whisper-dictation devices` and set `audio.device` in the config (or `DICTATION_AUDIO_DEVICE`) to the exact device name. |
| Text in wrong window | Text is typed into the focused window when transcription completes. Keep focus on target app. |
| Whisper hallucinations | Increase VAD threshold: `vad.threshold = 0.7` in config. In streaming mode, repeated phrases (e.g. "Thank you") during silence are auto-suppressed after 2 occurrences. |
| Wrong words (e.g. "passed" instead of "fast") | Set `server.prompt` or `server.hotwords` in config to bias transcription. |
| ydotool not working | Run `sudo systemctl start ydotool` and add user to `input` group. |

## Development

```bash
# Clone and install dev dependencies
git clone https://github.com/bhargavchippada/faster-whisper-dictation.git
cd faster-whisper-dictation
uv sync --extra dev

# Run tests
uv run pytest -v

# Run tests with coverage
uv run pytest tests/ --cov=whisper_dictation --cov-report=term-missing

# Build and install globally (editable — picks up code changes automatically)
uv build --clear --no-cache
uv tool install -e . --force

# Lint
uv run ruff check src/ tests/
```

## Contributing

Contributions are welcome. Please open an issue first to discuss what you'd like to change.

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/my-change`)
3. Install dev dependencies: `uv sync --extra dev`
4. Write tests first, then implement
5. Ensure tests pass and coverage is maintained
6. Open a pull request

## License

[MIT](LICENSE)
