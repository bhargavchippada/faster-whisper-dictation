#!/usr/bin/env bash
# Dictation script — record audio, transcribe via local faster-whisper, type into focused window
# Usage: bind to a keyboard shortcut (e.g., Alt+V)
#   Press shortcut → recording starts (notification shown)
#   Press shortcut again → recording stops, transcribes, types result
#
# Dependencies: pw-record (pipewire), curl, notify-send (libnotify), python3
#   X11: xdotool, xclip | Wayland: wl-clipboard, ydotool
# Server: http://localhost:10300 (faster-whisper-server)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$SCRIPT_DIR/common.sh"

PIDFILE="$TMP_DIR/rec.pid"
LOCKFILE="$TMP_DIR/rec.lock"
AUDIO_FILE="$TMP_DIR/recording.wav"

cleanup() { rm -f "$AUDIO_FILE" "$PIDFILE"; }

# Prevent race conditions from rapid hotkey presses
exec 9>"$LOCKFILE"
flock -n 9 || exit 0

# Read PID safely — empty string if file missing/empty
stored_pid=$(cat "$PIDFILE" 2>/dev/null) || stored_pid=""

# Toggle: if recording is running, stop it and transcribe. Otherwise start recording.
if [[ -n "$stored_pid" ]] && kill -0 "$stored_pid" 2>/dev/null && \
   [[ "$(cat /proc/"$stored_pid"/comm 2>/dev/null)" == "pw-record" ]]; then
  # ── STOP recording ──
  kill "$stored_pid" 2>/dev/null || true
  # Wait for pw-record to flush and exit (not a child, so wait(1) won't work)
  for _ in $(seq 1 60); do
    kill -0 "$stored_pid" 2>/dev/null || break
    sleep 0.05
  done
  kill -9 "$stored_pid" 2>/dev/null || true
  rm -f "$PIDFILE"
  trap cleanup EXIT

  if [[ ! -s "$AUDIO_FILE" ]]; then
    notify "❌ Error" "No audio recorded"
    exit 1
  fi

  check_audio "$AUDIO_FILE" || exit 1
  notify "🎤 Processing..." "Transcribing audio"
  check_health
  transcribe "$AUDIO_FILE"
  # Release lock before typing (prevents xclip from inheriting the flock fd)
  exec 9>&-
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
