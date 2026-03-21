#!/usr/bin/env bash
# Hold-to-talk dictation via keyboard shortcut
# Opens a terminal to record, press Enter to stop, transcribes and types into the original window
#
# Dependencies: pw-record (pipewire), curl, xdotool, xclip, notify-send, gnome-terminal, python3
# Server: http://localhost:10300 (faster-whisper-server)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$SCRIPT_DIR/common.sh"

LOCKFILE="$TMP_DIR/hold.lock"
if [[ -f "$LOCKFILE" ]] && kill -0 "$(cat "$LOCKFILE" 2>/dev/null)" 2>/dev/null; then
  exit 0
fi
AUDIO_FILE="$TMP_DIR/hold_recording.wav"

echo "$$" > "$LOCKFILE"
trap 'rm -f "$LOCKFILE" "$AUDIO_FILE"' EXIT

rm -f "$AUDIO_FILE"

# Remember which window has focus before the terminal steals it
ORIG_WINDOW=$(xdotool getactivewindow)

check_health

# Record in a small terminal — closes when user presses Enter
gnome-terminal --wait --title="🎤 Recording... Enter to stop" --geometry=40x1 -- bash -c '
  cleanup() { kill "$PID" 2>/dev/null; wait "$PID" 2>/dev/null; }
  trap cleanup EXIT
  pw-record --rate 16000 --channels 1 --format s16 "'"$AUDIO_FILE"'" &
  PID=$!
  read -r -s
'

if [[ ! -s "$AUDIO_FILE" ]]; then
  notify "❌ Error" "No audio recorded"
  exit 1
fi

notify "🎤 Processing..." "Transcribing audio"
transcribe "$AUDIO_FILE"

# Refocus the original window and type
xdotool windowactivate --sync "$ORIG_WINDOW"
sleep 0.1
type_text "$text"
notify "✅ Dictated" "${text:0:80}"
