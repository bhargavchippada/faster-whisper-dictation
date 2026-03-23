#!/usr/bin/env bash
# Self-contained demo — server mode with batch transcription
set -e
c=$'\033'

title() { echo -e "\n  ${c}[1;36m$1${c}[0m\n"; sleep 0.3; }
cmd() { echo -e "  ${c}[0m\$ ${c}[1;37m$1${c}[0m"; sleep 0.6; }
out() { echo -e "  $@"; }
pause() { sleep "${1:-1}"; }
clr() { clear; }

# --- Docker up ---
clr
title "Start the Whisper server"
cmd "docker compose up -d"
out "${c}[32m✓${c}[0m Container whisper-server Started"
out "  → http://localhost:10300"
pause 2

# --- Start daemon ---
clr
title "Start dictation daemon (hold-to-talk)"
cmd "faster-whisper-dictation start --mode hold -b"
out "Daemon starting in background"
out "  log: ~/.config/faster-whisper-dictation/daemon.log"
pause 1.8

# --- Status ---
clr
title "Check daemon status"
cmd "faster-whisper-dictation status"
out "Status: running (PID 48291)"
out "  Mode:    hold"
out "  Hotkey:  alt+v"
out "  Engine:  server"
out "  Server:  http://localhost:10300"
pause 2.5

# --- Hold Alt+V: recording + batch transcription ---
clr
title "Hold Alt+V and speak"
out "${c}[33m♫${c}[0m Recording ██░░░░░░░░  listening..."
pause 0.7
out "${c}[33m♫${c}[0m Recording ██████░░░░  speech detected"
pause 0.7
out "${c}[33m♫${c}[0m Recording ██████████  processing..."
pause 0.5
out "${c}[90m[ Release Alt+V → transcribe ]${c}[0m"
pause 0.5
echo
out "${c}[32m✓${c}[0m Hello, this is a test of the dictation system"
out "${c}[90m  → typed into focused app${c}[0m"
pause 3

# --- Stop ---
clr
title "Stop the daemon"
cmd "faster-whisper-dictation stop"
out "Stopped daemon (PID 48291)"
pause 2
