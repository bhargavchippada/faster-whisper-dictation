#!/usr/bin/env python3
"""Streaming dictation using local VAD and the REST transcription API.

Records from the default microphone, detects speech boundaries locally using
energy-based voice activity detection, and sends each utterance to the server's
REST API for transcription. Results are typed into the focused window in real-time.

Usage:
    ./dictate-stream.py              # stream until Ctrl+C
    ./dictate-stream.py --once       # transcribe one utterance then exit
    ./dictate-stream.py --no-type    # print to stdout only (for use by other scripts)

Dependencies: pip install sounddevice
Server: http://localhost:10300 (Speaches / faster-whisper-server)
"""

import argparse
import io
import json
import math
import os
import struct
import subprocess
import sys
import threading
import time
import wave

try:
    import sounddevice as sd
except ImportError:
    print("Error: sounddevice not installed. Run: pip install sounddevice", file=sys.stderr)
    sys.exit(1)

SAMPLE_RATE = 16000  # 16kHz mono — optimal for Whisper
CHANNELS = 1
DTYPE = "int16"
BLOCK_MS = 30  # VAD frame size in ms
BLOCK_SIZE = SAMPLE_RATE * BLOCK_MS // 1000  # samples per block

BASE_URL = os.environ.get("WHISPER_BASE_URL", "http://localhost:10300")
TRANSCRIBE_URL = f"{BASE_URL}/v1/audio/transcriptions"
MODEL = os.environ.get("WHISPER_MODEL", "Systran/faster-whisper-large-v3")
LANG = os.environ.get("WHISPER_LANG", "en")
PASTE_DELAY = float(os.environ.get("DICTATION_PASTE_DELAY", "0.3"))

# VAD thresholds (configurable via env)
VAD_ENERGY_THRESHOLD = float(os.environ.get("DICTATION_VAD_ENERGY", "500"))
VAD_SILENCE_MS = int(os.environ.get("DICTATION_VAD_SILENCE_MS", "800"))
VAD_MIN_SPEECH_MS = int(os.environ.get("DICTATION_VAD_MIN_SPEECH_MS", "300"))


def detect_display():
    """Detect X11 vs Wayland."""
    return os.environ.get("XDG_SESSION_TYPE", "x11")


def type_text(text):
    """Type text into the focused window via clipboard."""
    display = detect_display()
    if display == "wayland":
        prev = subprocess.run(["wl-paste"], capture_output=True, text=True, check=False).stdout
        subprocess.run(["wl-copy", text], check=False)
        subprocess.run(["ydotool", "key", "29:1", "47:1", "47:0", "29:0"], check=False)
        time.sleep(PASTE_DELAY)
        subprocess.run(["wl-copy", prev], check=False)
    else:
        prev = subprocess.run(["xclip", "-selection", "clipboard", "-o"],
                              capture_output=True, text=True, check=False).stdout
        subprocess.run(["xclip", "-selection", "clipboard"], input=text.encode(), check=False)
        subprocess.run(["xdotool", "key", "--clearmodifiers", "ctrl+v"], check=False)
        time.sleep(PASTE_DELAY)
        subprocess.run(["xclip", "-selection", "clipboard"], input=prev.encode(), check=False)


def notify(title, body=""):
    """Show desktop notification."""
    subprocess.run(["notify-send", "-t", "2000", "-a", "Dictation", title, body],
                   capture_output=True, check=False)


def audio_to_wav(frames, sample_rate=SAMPLE_RATE):
    """Convert raw int16 audio frames to an in-memory WAV file."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(CHANNELS)
        w.setsampwidth(2)  # 16-bit
        w.setframerate(sample_rate)
        w.writeframes(b"".join(frames))
    return buf.getvalue()


def transcribe_audio(wav_data):
    """Send WAV audio to the REST API and return the transcribed text."""
    try:
        result = subprocess.run(
            ["curl", "-s", "--max-time", "30",
             TRANSCRIBE_URL,
             "-F", f"file=@-;filename=audio.wav",
             "-F", f"model={MODEL}",
             "-F", f"language={LANG}"],
            input=wav_data,
            capture_output=True,
            check=False,
        )
        if result.returncode != 0:
            return ""
        resp = json.loads(result.stdout)
        return resp.get("text", "").strip()
    except Exception:
        return ""


def rms_energy(data):
    """Calculate RMS energy of int16 audio data."""
    n_samples = len(data) // 2
    if n_samples == 0:
        return 0.0
    samples = struct.unpack(f"<{n_samples}h", data)
    return math.sqrt(sum(s * s for s in samples) / n_samples)


def stream_dictation(once=False, no_type=False):
    """Main streaming loop: record → VAD → transcribe → type."""
    # Verify server is reachable
    health = subprocess.run(
        ["curl", "-sf", "--max-time", "3", f"{BASE_URL}/health"],
        capture_output=True, check=False)
    if health.returncode != 0:
        notify("❌ Error", "Whisper server is not running")
        print(f"Error: whisper server not reachable at {BASE_URL}", file=sys.stderr)
        sys.exit(1)

    if not no_type:
        notify("🎤 Streaming...", "Speak now (Ctrl+C to stop)")

    if not no_type:
        print("Listening... (Ctrl+C to stop)", file=sys.stderr)

    speech_frames = []
    is_speaking = False
    silence_blocks = 0
    speech_blocks = 0
    silence_threshold = VAD_SILENCE_MS // BLOCK_MS  # blocks of silence to end utterance
    min_speech_blocks = VAD_MIN_SPEECH_MS // BLOCK_MS  # minimum speech to accept

    stop = threading.Event()

    def audio_callback(indata, frames, time_info, status):
        nonlocal is_speaking, silence_blocks, speech_blocks

        if status:
            print(f"Audio: {status}", file=sys.stderr)

        data = bytes(indata)
        energy = rms_energy(data)

        if energy >= VAD_ENERGY_THRESHOLD:
            # Speech detected
            if not is_speaking:
                is_speaking = True
                silence_blocks = 0
                speech_blocks = 0
            speech_blocks += 1
            silence_blocks = 0
            speech_frames.append(data)
        elif is_speaking:
            # Silence after speech
            silence_blocks += 1
            speech_frames.append(data)  # include trailing silence for context

            if silence_blocks >= silence_threshold:
                # End of utterance — transcribe in background
                if speech_blocks >= min_speech_blocks:
                    frames_copy = list(speech_frames)
                    threading.Thread(
                        target=process_utterance,
                        args=(frames_copy, once, no_type),
                        daemon=True,
                    ).start()
                speech_frames.clear()
                is_speaking = False
                silence_blocks = 0
                speech_blocks = 0

    def process_utterance(frames, once_mode, no_type_mode):
        """Transcribe an utterance and type/print the result."""
        wav_data = audio_to_wav(frames)
        text = transcribe_audio(wav_data)
        if text:
            print(text, flush=True)
            if not no_type_mode:
                type_text(text + " ")
        if once_mode:
            stop.set()

    try:
        with sd.RawInputStream(
            samplerate=SAMPLE_RATE,
            channels=CHANNELS,
            dtype=DTYPE,
            blocksize=BLOCK_SIZE,
            callback=audio_callback,
        ):
            if once:
                stop.wait()
            else:
                while not stop.is_set():
                    stop.wait(timeout=0.5)
    except Exception as e:
        notify("❌ Error", str(e)[:80])
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    finally:
        if not no_type:
            notify("🎤 Stopped", "Streaming dictation ended")


def main():
    parser = argparse.ArgumentParser(description="Streaming dictation via local VAD + REST API")
    parser.add_argument("--once", action="store_true", help="Transcribe one utterance then exit")
    parser.add_argument("--no-type", action="store_true",
                        help="Print transcripts to stdout without typing into focused window")
    args = parser.parse_args()

    try:
        stream_dictation(once=args.once, no_type=args.no_type)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
