#!/usr/bin/env bash
# Hold-to-talk dictation via keyboard shortcut (streaming mode)
# Opens a terminal with real-time streaming transcription.
# Enter to finish and type text. Ctrl+C to cancel.
#
# Dependencies: curl, notify-send (libnotify), gnome-terminal, python3, sounddevice (pip)
#   X11: xdotool, xclip | Wayland: wl-clipboard, ydotool
# Server: http://localhost:10300 (Speaches)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$SCRIPT_DIR/common.sh"

LOCKFILE="$TMP_DIR/hold-hotkey.lock"
OUTPUT_FILE="$TMP_DIR/hotkey_stream_output.txt"

# Prevent concurrent invocations
exec 8>"$LOCKFILE"
flock -n 8 || exit 0
trap 'rm -f "$OUTPUT_FILE"' EXIT

rm -f "$OUTPUT_FILE"

# Remember which window has focus before the terminal steals it
ORIG_WINDOW=$(get_active_window)

check_health

# Stream in a terminal — real-time transcription
# Use a wrapper script to avoid env-var inheritance issues with gnome-terminal
WRAPPER="$TMP_DIR/hotkey_wrapper.sh"
cat > "$WRAPPER" <<WRAPPER_EOF
#!/usr/bin/env bash
python3 "$SCRIPT_DIR/dictate-stream.py" --no-type > "$OUTPUT_FILE" &
PID=\$!
trap "kill \$PID 2>/dev/null; wait \$PID 2>/dev/null; rm -f '$OUTPUT_FILE'; exit 1" INT
read -r -s -p "Press Enter when done, Ctrl+C to cancel..."
kill \$PID 2>/dev/null; wait \$PID 2>/dev/null
WRAPPER_EOF
chmod +x "$WRAPPER"
gnome-terminal --wait --title="Dictation - Enter to finish, Ctrl+C to cancel" --geometry=60x5 -- "$WRAPPER"

# Read accumulated transcripts
if [[ ! -s "$OUTPUT_FILE" ]]; then
  notify "❌ Error" "No speech detected"
  exit 1
fi

# Join utterance lines into a single string
text=$(tr "\n" " " < "$OUTPUT_FILE" | sed "s/  */ /g; s/^ //; s/ $//")

if [[ -z "$text" ]]; then
  notify "❌ Error" "No speech detected"
  exit 1
fi

# Release lock before typing (prevents xclip from inheriting the flock fd)
exec 8>&-

# Refocus the original window and type
activate_window "$ORIG_WINDOW"
type_text "$text"
notify "✅ Dictated" "${text:0:80}"
