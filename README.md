# faster-whisper-dictation

Local speech-to-text dictation using [faster-whisper-server](https://github.com/fedirz/faster-whisper-server) with CUDA acceleration. Record audio, transcribe it locally, and type the result into the focused window — fully offline, no API keys needed.

## Prerequisites

- **NVIDIA GPU** with CUDA support
- **Docker** with [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)
- **Linux** with X11 (uses `xdotool`/`xclip` — Wayland not supported)

## Setup

### 1. Clone

```bash
git clone https://github.com/bhargavchippada/faster-whisper-dictation.git
cd faster-whisper-dictation
```

### 2. Install system dependencies

```bash
sudo apt install -y pipewire curl xdotool xclip libnotify-bin python3
```

### 3. Start the whisper server

```bash
docker compose up -d
```

First run downloads the large-v3 model (~3GB) — takes a few minutes.

### 4. Verify

```bash
curl http://localhost:10300/health
```

### 5. Bind a keyboard shortcut

GNOME Settings → Keyboard → Custom Shortcuts:

- **Name:** Dictation Hold-to-talk
- **Command:** `/path/to/faster-whisper-dictation/scripts/dictate-hold-hotkey.sh`
- **Shortcut:** `Alt+V` (or your preference)

Or via command line:

```bash
KEYBINDING_PATH="/org/gnome/settings-daemon/plugins/media-keys/custom-keybindings/custom0/"
gsettings set org.gnome.settings-daemon.plugins.media-keys custom-keybindings "['$KEYBINDING_PATH']"
gsettings set org.gnome.settings-daemon.plugins.media-keys.custom-keybinding:$KEYBINDING_PATH name "Dictation Hold-to-talk"
gsettings set org.gnome.settings-daemon.plugins.media-keys.custom-keybinding:$KEYBINDING_PATH command "/path/to/faster-whisper-dictation/scripts/dictate-hold-hotkey.sh"
gsettings set org.gnome.settings-daemon.plugins.media-keys.custom-keybinding:$KEYBINDING_PATH binding "<Alt>v"
```

### 6. Test

```bash
# Record 5 seconds and transcribe
pw-record --rate 16000 --channels 1 --format s16 /tmp/test.wav &
sleep 5 && kill $!
./scripts/transcribe.sh /tmp/test.wav
```

## Usage

| Script | Mode | How it works |
|--------|------|--------------|
| `scripts/dictate-hold-hotkey.sh` | Hold-to-talk (hotkey) | Opens a terminal to record, press Enter to stop, transcribes and types into the original window |
| `scripts/dictate.sh` | Toggle (hotkey) | Press shortcut to start recording, press again to stop, transcribe, and type |
| `scripts/dictate-hold.sh` | Push-to-talk (terminal) | Run in terminal, speak, press Enter to transcribe and type |
| `scripts/transcribe.sh` | CLI | `./scripts/transcribe.sh <file> [lang]` — outputs text to stdout |

## Configuration

All scripts read defaults from `scripts/common.sh`. Override via environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `WHISPER_BASE_URL` | `http://localhost:10300` | Server base URL |
| `WHISPER_MODEL` | `large-v3` | Whisper model name |
| `WHISPER_LANG` | `en` | Language code |

Example:

```bash
WHISPER_LANG=es ./scripts/transcribe.sh audio.wav
```

## API

OpenAI-compatible endpoint:

```bash
curl http://localhost:10300/v1/audio/transcriptions \
  -F "file=@audio.wav" \
  -F "model=large-v3" \
  -F "language=en"
# Returns: {"text": "transcribed text here"}
```

## Docker details

- **Port:** 10300 (localhost only)
- **GPU:** NVIDIA CUDA (float16)
- **Model cache:** Docker volume `faster-whisper-models` (persists across restarts)
- **VRAM:** ~1GB when loaded, released when idle

## License

[MIT](LICENSE)
