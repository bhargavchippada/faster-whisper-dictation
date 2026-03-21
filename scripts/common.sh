#!/usr/bin/env bash
# Shared config and helpers for faster-whisper scripts

WHISPER_BASE_URL="${WHISPER_BASE_URL:-http://localhost:10300}"
WHISPER_URL="$WHISPER_BASE_URL/v1/audio/transcriptions"
WHISPER_MODEL="${WHISPER_MODEL:-Systran/faster-whisper-large-v3}"
WHISPER_LANG="${WHISPER_LANG:-en}"
TMP_DIR="${DICTATION_TMP_DIR:-/tmp/dictation-$(id -u)}"

# Detect display server once at source time
if [[ "${XDG_SESSION_TYPE:-}" == "wayland" ]]; then
  DICTATION_DISPLAY="wayland"
else
  DICTATION_DISPLAY="x11"
fi

mkdir -p "$TMP_DIR"
chmod 700 "$TMP_DIR" 2>/dev/null
# Verify we own the directory (defense against pre-creation by another user)
if [[ "$(stat -c %u "$TMP_DIR")" != "$(id -u)" ]]; then
  echo "Error: $TMP_DIR is not owned by current user — refusing to use it" >&2
  exit 1
fi

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
  notify-send -t 2000 -a "Dictation" "$1" "${2:-}" 2>/dev/null || true
}

# Check if audio file has enough speech content to be worth transcribing
# Prevents Whisper hallucinations ("Thank you", etc.) from silence/noise
check_audio() {
  local audio_file="$1"
  local min_duration="${DICTATION_MIN_DURATION:-0.5}"
  local min_energy="${DICTATION_MIN_ENERGY:-500}"

  # Single Python call: analyze WAV and check thresholds
  # Exit codes: 0=OK, 1=too short, 2=too quiet, 3=error
  local result
  result=$(python3 -c "
import wave, struct, math, sys
audio_file, min_dur, min_nrg = sys.argv[1], float(sys.argv[2]), float(sys.argv[3])
with wave.open(audio_file, 'rb') as w:
    nframes = w.getnframes()
    duration = nframes / w.getframerate()
    if duration < min_dur:
        print(f'{duration:.2f}')
        sys.exit(1)
    frames = w.readframes(nframes)
    samples = struct.unpack(f'<{len(frames)//2}h', frames)
    rms = math.sqrt(sum(s*s for s in samples) / len(samples)) if samples else 0
    if rms < min_nrg:
        print(f'{rms:.0f}')
        sys.exit(2)
" "$audio_file" "$min_duration" "$min_energy" 2>/dev/null)
  local rc=$?

  case $rc in
    0) return 0 ;;
    1) notify "⚠️ Too short" "Recording was ${result}s (min ${min_duration}s)"; return 1 ;;
    2) notify "⚠️ No speech detected" "Audio too quiet (energy: ${result})"; return 1 ;;
    *) notify "❌ Error" "Audio check failed (is python3 installed?)"; return 1 ;;
  esac
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

  local http_code response_body curl_exit=0
  response_body=$(curl -s -w '\n%{http_code}' "$WHISPER_URL" \
    -F "file=@$audio_file" \
    -F "model=$WHISPER_MODEL" \
    -F "language=$lang" \
    --max-time 60) || curl_exit=$?

  if [[ "$curl_exit" -ne 0 ]]; then
    if [[ "$curl_exit" -eq 28 ]]; then
      notify "❌ Error" "Transcription timed out"
      echo "Error: server timed out (curl exit 28)" >&2
    else
      notify "❌ Error" "Server request failed"
      echo "Error: curl failed with exit code $curl_exit" >&2
    fi
    exit 1
  fi

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

# Type text into the focused window via clipboard
type_text() {
  local prev_clip
  local paste_delay="${DICTATION_PASTE_DELAY:-0.3}"
  if [[ "$DICTATION_DISPLAY" == "wayland" ]]; then
    prev_clip=$(wl-paste 2>/dev/null) || prev_clip=""
    printf '%s' "$1" | wl-copy
    ydotool key 29:1 47:1 47:0 29:0  # Ctrl+V (keycode 29=Ctrl, 47=V)
    sleep "$paste_delay"
    printf '%s' "$prev_clip" | wl-copy
  else
    prev_clip=$(xclip -selection clipboard -o 2>/dev/null) || prev_clip=""
    printf '%s' "$1" | xclip -selection clipboard
    xdotool key --clearmodifiers ctrl+v
    sleep "$paste_delay"
    printf '%s' "$prev_clip" | xclip -selection clipboard
  fi
}

# Get the currently focused window ID (X11 only, returns empty on Wayland)
get_active_window() {
  if [[ "$DICTATION_DISPLAY" == "x11" ]]; then
    xdotool getactivewindow
  fi
}

# Refocus a window by ID (X11 only, no-op on Wayland)
activate_window() {
  if [[ "$DICTATION_DISPLAY" == "x11" ]] && [[ -n "${1:-}" ]]; then
    xdotool windowactivate --sync "$1"
    sleep 0.1
  fi
}
