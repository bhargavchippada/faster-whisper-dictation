#!/usr/bin/env bash
# Transcribe an audio file via local faster-whisper server
# Usage: ./transcribe.sh <audio_file> [language]
# Output: plain text to stdout

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$SCRIPT_DIR/common.sh"

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <audio_file> [language]" >&2
  exit 1
fi

AUDIO_FILE="$1"
TRANS_LANG="${2:-$WHISPER_LANG}"

if [[ ! -f "$AUDIO_FILE" ]]; then
  echo "Error: file not found: $AUDIO_FILE" >&2
  exit 1
fi

check_health
transcribe "$AUDIO_FILE" "$TRANS_LANG"
echo "$text"
