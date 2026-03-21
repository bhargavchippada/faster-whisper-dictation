#!/usr/bin/env bash
# Shared config and helpers for faster-whisper scripts

WHISPER_BASE_URL="${WHISPER_BASE_URL:-http://localhost:10300}"
WHISPER_URL="$WHISPER_BASE_URL/v1/audio/transcriptions"
WHISPER_MODEL="${WHISPER_MODEL:-large-v3}"
WHISPER_LANG="${WHISPER_LANG:-en}"
TMP_DIR="${DICTATION_TMP_DIR:-/tmp/dictation}"

mkdir -p "$TMP_DIR"

# Start recording to a file in the background — sets $REC_PID
# Uses pw-record (PipeWire) to respect the system default audio source
# Usage: start_recording <output_file>
start_recording() {
  pw-record --rate 16000 --channels 1 --format s16 "$1" &
  REC_PID=$!
}

# Stop a recording started by start_recording
stop_recording() {
  kill "$REC_PID" 2>/dev/null || true
  wait "$REC_PID" 2>/dev/null || true
}

notify() {
  notify-send -t 2000 -a "Dictation" "$1" "$2" 2>/dev/null || true
}

# Check if the whisper server is reachable
check_health() {
  if ! curl -sf --max-time 3 "$WHISPER_BASE_URL/health" >/dev/null 2>&1; then
    notify "❌ Error" "Whisper server is not running"
    echo "Error: whisper server not reachable at $WHISPER_BASE_URL" >&2
    exit 1
  fi
}

# Transcribe an audio file → sets $text, exits on failure
transcribe() {
  local audio_file="$1"
  local lang="${2:-$WHISPER_LANG}"

  local http_code response_body
  response_body=$(curl -s -w '\n%{http_code}' "$WHISPER_URL" \
    -F "file=@$audio_file" \
    -F "model=$WHISPER_MODEL" \
    -F "language=$lang" \
    --max-time 60)

  http_code=$(echo "$response_body" | tail -n1)
  response_body=$(echo "$response_body" | sed '$d')

  if [[ "$http_code" -ne 200 ]]; then
    notify "❌ Error" "Server returned HTTP $http_code"
    echo "Error: server returned HTTP $http_code: $response_body" >&2
    exit 1
  fi

  text=$(echo "$response_body" | python3 -c "import sys,json; print(json.load(sys.stdin).get('text','').strip())" 2>/dev/null) || {
    notify "❌ Error" "Failed to parse server response"
    echo "Error: failed to parse response: $response_body" >&2
    exit 1
  }

  if [[ -z "$text" ]]; then
    notify "❌ Error" "No speech detected"
    exit 1
  fi
}

# Type text into the focused window via clipboard (avoids xdotool special char issues)
type_text() {
  local prev_clip
  prev_clip=$(xclip -selection clipboard -o 2>/dev/null) || prev_clip=""
  printf '%s' "$1" | xclip -selection clipboard
  xdotool key --clearmodifiers ctrl+v
  sleep 0.3
  printf '%s' "$prev_clip" | xclip -selection clipboard
}
