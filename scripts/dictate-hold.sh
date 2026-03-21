#!/usr/bin/env bash
# Push-to-talk dictation — run in a terminal, press Enter to stop and transcribe
# Usage: ./dictate-hold.sh
#
# Dependencies: pw-record (pipewire), curl, notify-send (libnotify), python3
#   X11: xdotool, xclip | Wayland: wl-clipboard, ydotool
# Server: http://localhost:10300 (faster-whisper-server)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$SCRIPT_DIR/common.sh"

AUDIO_FILE="$TMP_DIR/hold_recording.wav"

cleanup() {
  if [[ -n "${REC_PID:-}" ]]; then
    kill "$REC_PID" 2>/dev/null || true
  fi
  rm -f "$AUDIO_FILE"
}
trap cleanup EXIT

check_health

notify "🎤 Recording..." "Press Enter to stop"

start_recording "$AUDIO_FILE"

# Wait for Enter key (or the process gets killed by key release)
read -r -s 2>/dev/null || true
stop_recording

if [[ ! -s "$AUDIO_FILE" ]]; then
  notify "❌ Error" "No audio recorded"
  exit 1
fi

check_audio "$AUDIO_FILE" || exit 1
notify "🎤 Processing..." "Transcribing audio"

transcribe "$AUDIO_FILE"
type_text "$text"
notify "✅ Dictated" "${text:0:80}"
