#!/usr/bin/env bash
# Dictation script — record audio, transcribe via local faster-whisper, type into focused window
# Usage: bind to a keyboard shortcut (e.g., Alt+V)
#   Press shortcut → recording starts (notification shown)
#   Press shortcut again → recording stops, transcribes, types result
#
# Dependencies: pw-record (pipewire), curl, xdotool, xclip, notify-send (libnotify)
# Server: http://localhost:10300 (faster-whisper-server)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$SCRIPT_DIR/common.sh"

PIDFILE="$TMP_DIR/rec.pid"
AUDIO_FILE="$TMP_DIR/recording.wav"

cleanup() { rm -f "$AUDIO_FILE" "$PIDFILE"; }

# Read PID safely — empty string if file missing/empty
stored_pid=$(cat "$PIDFILE" 2>/dev/null) || stored_pid=""

# Toggle: if recording is running, stop it and transcribe. Otherwise start recording.
if [[ -n "$stored_pid" ]] && kill -0 "$stored_pid" 2>/dev/null && \
   [[ "$(cat /proc/"$stored_pid"/comm 2>/dev/null)" == "pw-record" ]]; then
  # ── STOP recording ──
  kill "$stored_pid" 2>/dev/null || true
  wait "$stored_pid" 2>/dev/null || true
  rm -f "$PIDFILE"
  trap cleanup EXIT
  notify "🎤 Processing..." "Transcribing audio"

  if [[ ! -s "$AUDIO_FILE" ]]; then
    notify "❌ Error" "No audio recorded"
    exit 1
  fi

  check_health
  transcribe "$AUDIO_FILE"
  type_text "$text"
  notify "✅ Dictated" "${text:0:80}"
else
  # ── START recording ──
  # Clean up stale PID file if it exists
  rm -f "$PIDFILE" "$AUDIO_FILE"

  check_health
  notify "🎤 Recording..." "Press shortcut again to stop"

  # Record in background: 16kHz mono (optimal for Whisper)
  start_recording "$AUDIO_FILE"
  echo "$REC_PID" > "$PIDFILE"
fi
