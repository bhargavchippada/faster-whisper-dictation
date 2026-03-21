#!/usr/bin/env bash
# Push-to-talk dictation — hold a key to record, release to transcribe
# Usage: bind to a keyboard shortcut as a "hold" action, or run manually:
#   ./dictate-hold.sh        # starts recording, Ctrl+C or Enter to stop & transcribe
#
# Dependencies: pw-record (pipewire), curl, xdotool, xclip, notify-send (libnotify)
# Server: http://localhost:10300 (faster-whisper-server)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$SCRIPT_DIR/common.sh"

AUDIO_FILE="$TMP_DIR/hold_recording.wav"

cleanup() {
  [[ -n "${REC_PID:-}" ]] && kill "$REC_PID" 2>/dev/null || true
  rm -f "$AUDIO_FILE"
}
trap cleanup EXIT

check_health

notify "🎤 Recording..." "Release key or press Enter to stop"

start_recording "$AUDIO_FILE"

# Wait for Enter key (or the process gets killed by key release)
read -r -s 2>/dev/null || true
stop_recording

if [[ ! -s "$AUDIO_FILE" ]]; then
  notify "❌ Error" "No audio recorded"
  exit 1
fi

notify "🎤 Processing..." "Transcribing audio"

transcribe "$AUDIO_FILE"
type_text "$text"
notify "✅ Dictated" "${text:0:80}"
