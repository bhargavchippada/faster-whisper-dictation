#!/usr/bin/env bash
# Self-contained demo — server mode with WhisperLiveKit (batch + streaming)
set -e
c=$'\033'

title() { echo -e "\n  ${c}[1;36m$1${c}[0m\n"; sleep 0.3; }
cmd() { echo -e "  ${c}[0m\$ ${c}[1;37m$1${c}[0m"; sleep 0.6; }
out() { echo -e "  $@"; }
pause() { sleep "${1:-1}"; }
clr() { clear; }

# --- Install ---
clr
title "Install"
cmd "uv tool install whisperlivekit"
out "${c}[32m✓${c}[0m Installed: wlk"
cmd "uv tool install faster-whisper-dictation"
out "${c}[32m✓${c}[0m Installed: faster-whisper-dictation"
pause 2

# --- Start WLK server ---
clr
title "Start WhisperLiveKit server"
cmd "wlk serve --model large-v3 --language en --pcm-input \\"
out "    --min-chunk-size 1.5 --confidence-validation"
pause 0.5
out ""
out "  ${c}[32m✓${c}[0m WhisperLiveKit (CUDA · RTX 5090)"
out "    WebSocket  ws://localhost:8000/asr"
out "    REST API   http://localhost:8000/v1/audio/transcriptions"
pause 2

# --- Start daemon (batch, hold) ---
clr
title "Batch mode: hold Alt+V to dictate"
cmd "faster-whisper-dictation start --mode hold -b"
out "${c}[32m✓${c}[0m Dictation Ready — hold Alt+V to speak"
pause 1.5
out ""
out "${c}[33m♫${c}[0m Recording ██░░░░░░░░  listening..."
pause 0.7
out "${c}[33m♫${c}[0m Recording ██████░░░░  speech detected"
pause 0.7
out "${c}[33m♫${c}[0m Recording ██████████  3s of audio"
pause 0.5
out "${c}[90m  [ release Alt+V → transcribe ]${c}[0m"
pause 0.8
echo
out "${c}[32m✓${c}[0m Hello, this is a test of the dictation system."
out "${c}[90m  → typed into the focused application${c}[0m"
pause 3

# --- Streaming mode ---
clr
title "Streaming mode: text appears as you speak"
cmd "faster-whisper-dictation start --streaming"
out "${c}[32m✓${c}[0m Streaming Ready — press Alt+V to toggle"
pause 1.5
out ""
out "  ${c}[33m♫${c}[0m  ${c}[32mHello,${c}[0m"
pause 0.5
out "  ${c}[33m♫${c}[0m  ${c}[32mHello, this is${c}[0m"
pause 0.4
out "  ${c}[33m♫${c}[0m  ${c}[32mHello, this is a test of${c}[0m"
pause 0.4
out "  ${c}[33m♫${c}[0m  ${c}[32mHello, this is a test of the dictation${c}[0m"
pause 0.4
out "  ${c}[33m♫${c}[0m  ${c}[32mHello, this is a test of the dictation system.${c}[0m"
out "${c}[90m  → words appear in real-time as you speak${c}[0m"
pause 3

# --- Stop ---
clr
title "Stop"
cmd "faster-whisper-dictation stop"
out "Stopped daemon"
pause 2
