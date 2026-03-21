# faster-whisper-dictation

Local speech-to-text dictation powered by [Speaches](https://github.com/speaches-ai/speaches) (formerly faster-whisper-server). Speak, transcribe, and type — fully offline, no cloud APIs, no data leaves your machine.

## How it works

```
 Microphone ──▶ pw-record ──▶ Whisper server ──▶ type into focused app
 (PipeWire)     (16kHz WAV)   (GPU or CPU, Docker)  (xdotool/xclip or wl-copy/ydotool)
```

Audio is recorded through PipeWire, sent to a local Dockerized Whisper server for transcription, and the result is typed into whatever window has focus. Supports both X11 and Wayland.

## Requirements

- **Docker** (with [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) for GPU mode)
- **Linux** with X11 or Wayland
- **PipeWire** for audio capture

## Quick start

```bash
# 1. Clone
git clone https://github.com/bhargavchippada/faster-whisper-dictation.git
cd faster-whisper-dictation

# 2. Install dependencies
# X11
sudo apt install -y pipewire curl xdotool xclip libnotify-bin python3
# Wayland (instead of xdotool/xclip)
# sudo apt install -y wl-clipboard ydotool
# Streaming mode + hold-to-talk hotkey mode (optional)
# sudo apt install -y libportaudio2 gnome-terminal
# pip install sounddevice

# 3. Start the whisper server (first run downloads ~3GB model)
docker compose up -d          # GPU (NVIDIA CUDA)
# docker compose -f docker-compose.cpu.yml up -d   # CPU fallback (no GPU needed)

# 4. Verify the server is running
curl http://localhost:10300/health

# 5. Test with a recording
pw-record --rate 16000 --channels 1 --format s16 /tmp/test.wav &
sleep 5 && kill $!
./scripts/transcribe.sh /tmp/test.wav
```

## Setting up a keyboard shortcut

**GNOME Settings** → Keyboard → Custom Shortcuts:

| Field | Value |
|-------|-------|
| Name | Dictation |
| Command | `/full/path/to/scripts/dictate.sh` |
| Shortcut | `Alt+V` (or your preference) |

Or via CLI:

```bash
KEYBINDING_PATH="/org/gnome/settings-daemon/plugins/media-keys/custom-keybindings/custom0/"
gsettings set org.gnome.settings-daemon.plugins.media-keys custom-keybindings "['$KEYBINDING_PATH']"
gsettings set org.gnome.settings-daemon.plugins.media-keys.custom-keybinding:$KEYBINDING_PATH name "Dictation"
gsettings set org.gnome.settings-daemon.plugins.media-keys.custom-keybinding:$KEYBINDING_PATH command "$PWD/scripts/dictate.sh"
gsettings set org.gnome.settings-daemon.plugins.media-keys.custom-keybinding:$KEYBINDING_PATH binding "<Alt>v"
```

## Scripts

| Script | Mode | Description |
|--------|------|-------------|
| `dictate.sh` | Toggle (hotkey) | Press shortcut to start, press again to stop and transcribe. Best for keyboard shortcuts. |
| `dictate-hold-hotkey.sh` | Hold-to-talk (hotkey) | Opens a terminal with real-time streaming transcription. Enter to finish and type, Ctrl+C to cancel. Requires `sounddevice`. |
| `dictate-hold.sh` | Push-to-talk (terminal) | Run in a terminal, speak, press Enter to stop and transcribe. |
| `dictate-stream.py` | Streaming (continuous) | Local VAD detects speech, sends each utterance to the REST API. Transcribes in real-time as you speak. |
| `transcribe.sh` | CLI | `./scripts/transcribe.sh <file> [lang]` — transcribes an audio file to stdout. |

### Streaming mode

The streaming script uses local voice activity detection to find speech boundaries, then sends each utterance to the server's REST API for transcription. It continuously listens and types each utterance as soon as you finish speaking.

```bash
# Install dependencies
sudo apt install -y libportaudio2
pip install sounddevice

# Stream continuously — types each utterance as you speak
./scripts/dictate-stream.py

# Transcribe a single utterance then exit
./scripts/dictate-stream.py --once

# Print to stdout only (no typing) — used by dictate-hold-hotkey.sh
./scripts/dictate-stream.py --no-type
```

## Configuration

Override defaults via environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `WHISPER_BASE_URL` | `http://localhost:10300` | Whisper server URL |
| `WHISPER_MODEL` | `Systran/faster-whisper-large-v3` | Model name ([options](https://huggingface.co/Systran)) |
| `WHISPER_LANG` | `en` | Language code |
| `DICTATION_TMP_DIR` | `/tmp/dictation-<uid>` | Temp directory for recordings |
| `DICTATION_MIN_DURATION` | `0.5` | Minimum recording duration in seconds |
| `DICTATION_MIN_ENERGY` | `500` | Minimum RMS energy (silence filter threshold) |
| `DICTATION_PASTE_DELAY` | `0.3` | Seconds to wait after Ctrl+V before restoring clipboard |
| `DICTATION_VAD_ENERGY` | `500` | Streaming mode: RMS energy threshold for speech detection |
| `DICTATION_VAD_SILENCE_MS` | `800` | Streaming mode: silence duration (ms) to end an utterance |
| `DICTATION_VAD_MIN_SPEECH_MS` | `300` | Streaming mode: minimum speech duration (ms) to accept |

```bash
# Example: transcribe Spanish audio
WHISPER_LANG=es ./scripts/transcribe.sh audio.wav
```

## CPU mode

No NVIDIA GPU? Use the CPU compose file instead:

```bash
docker compose -f docker-compose.cpu.yml up -d
```

This uses `int8` quantization for lower memory usage. Transcription will be slower but works on any machine. The default `docker-compose.yml` uses NVIDIA CUDA with `float16` for real-time speed.

## API

The server exposes an OpenAI-compatible transcription endpoint:

```bash
curl http://localhost:10300/v1/audio/transcriptions \
  -F "file=@audio.wav" \
  -F "model=Systran/faster-whisper-large-v3" \
  -F "language=en"
# {"text": "transcribed text here"}
```

## Wayland support

Scripts auto-detect X11 vs Wayland via `$XDG_SESSION_TYPE` and use the appropriate tools:

| | X11 | Wayland |
|--|-----|---------|
| Clipboard | `xclip` | `wl-copy` / `wl-paste` |
| Key simulation | `xdotool` | `ydotool` |

For Wayland, install `wl-clipboard` and `ydotool`:

```bash
sudo apt install wl-clipboard ydotool
sudo systemctl enable --now ydotool   # ydotool needs its daemon running
sudo usermod -aG input $USER          # then re-login
```

## Docker details

| Setting | GPU mode | CPU mode |
|---------|----------|----------|
| Compose file | `docker-compose.yml` | `docker-compose.cpu.yml` |
| Image | `speaches:0.9.0-rc.3-cuda` | `speaches:0.9.0-rc.3-cpu` |
| Compute | NVIDIA CUDA (float16) | CPU (int8) |
| VRAM/RAM | ~600MB VRAM | ~2GB RAM |
| Port | `10300` (localhost) | `10300` (localhost) |
| Model cache | Docker volume `faster-whisper-models` | Docker volume `faster-whisper-models` |

```bash
docker compose up -d      # start
docker compose logs -f    # view logs
docker compose down       # stop
```

## Troubleshooting

- **"Whisper server is not running"** — Run `docker compose up -d` and wait for the model to load. Check `docker compose logs`.
- **Transcribes as "Thank you"** — Whisper hallucination from silence. The silence filter should catch this, but verify your mic works: `pw-record --rate 16000 --channels 1 --format s16 /tmp/test.wav` then `pw-play /tmp/test.wav`.
- **"Too short" or "Audio too quiet"** — The silence filter rejected the recording. Adjust thresholds: `DICTATION_MIN_DURATION=0.3` or `DICTATION_MIN_ENERGY=200`.
- **Text appears in wrong window** — Text is pasted into the currently focused window. Make sure focus is correct before transcription finishes.
- **ydotool not working** — Ensure the daemon is running (`sudo systemctl start ydotool`) and your user is in the `input` group.
- **Hold-hotkey terminal opens and closes immediately** — Usually `sounddevice` not installed for the python3 in your PATH. Run `python3 -c "import sounddevice"` to check. Install with `pip install sounddevice`. If using a keyboard shortcut, the PATH may differ from your terminal — see `common.sh` PATH augmentation.
- **Hold-hotkey does nothing (no terminal)** — A stale lock from a previous run. Check with `fuser /tmp/dictation-$(id -u)/hold-hotkey.lock` and kill the listed process, or `rm /tmp/dictation-$(id -u)/hold-hotkey.lock`.

## Contributing

Contributions are welcome. Please open an issue first to discuss what you'd like to change.

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/my-change`)
3. Commit your changes
4. Open a pull request

## License

[MIT](LICENSE)
