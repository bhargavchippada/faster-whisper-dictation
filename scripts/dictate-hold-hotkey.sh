#!/usr/bin/env bash
# Hold-to-talk dictation via keyboard shortcut
# Opens a terminal to record, press Enter to stop, transcribes and types into the original window
#
# Dependencies: pw-record (pipewire), curl, notify-send (libnotify), gnome-terminal, python3
#   X11: xdotool, xclip | Wayland: wl-clipboard, ydotool
# Server: http://localhost:10300 (faster-whisper-server)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$SCRIPT_DIR/common.sh"

LOCKFILE="$TMP_DIR/hold-hotkey.lock"
AUDIO_FILE="$TMP_DIR/hotkey_recording.wav"

# Prevent concurrent invocations
exec 8>"$LOCKFILE"
flock -n 8 || exit 0
trap 'rm -f "$AUDIO_FILE"' EXIT

rm -f "$AUDIO_FILE"

# Remember which window has focus before the terminal steals it
ORIG_WINDOW=$(get_active_window)

check_health

# Record in a small terminal — closes when user presses Enter
export DICTATION_AUDIO_FILE="$AUDIO_FILE"
gnome-terminal --wait --title="🎤 Recording... Enter to stop" --geometry=40x1 -- bash -c '
  cleanup() { kill "$PID" 2>/dev/null; wait "$PID" 2>/dev/null; }
  trap cleanup EXIT
  pw-record --rate 16000 --channels 1 --format s16 "$DICTATION_AUDIO_FILE" &
  PID=$!
  read -r -s
'

if [[ ! -s "$AUDIO_FILE" ]]; then
  notify "❌ Error" "No audio recorded"
  exit 1
fi

check_audio "$AUDIO_FILE" || exit 1
notify "🎤 Processing..." "Transcribing audio"
transcribe "$AUDIO_FILE"

# Refocus the original window and type
activate_window "$ORIG_WINDOW"
type_text "$text"
notify "✅ Dictated" "${text:0:80}"
