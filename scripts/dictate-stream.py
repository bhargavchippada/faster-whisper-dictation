#!/usr/bin/env python3
"""Streaming dictation via the Speaches WebSocket Realtime API.

Records from the default microphone and streams audio to the server.
The server uses VAD to detect speech boundaries and returns transcriptions
for each utterance. Results are typed into the focused window in real-time.

Usage:
    ./dictate-stream.py              # stream until Ctrl+C
    ./dictate-stream.py --once       # transcribe one utterance then exit

Dependencies: pip install sounddevice websockets
Server: ws://localhost:10300/v1/realtime (Speaches)
"""

import argparse
import asyncio
import base64
import json
import os
import subprocess
import sys
import time

try:
    import sounddevice as sd
except ImportError:
    print("Error: sounddevice not installed. Run: pip install sounddevice", file=sys.stderr)
    sys.exit(1)

try:
    import websockets
except ImportError:
    print("Error: websockets not installed. Run: pip install websockets", file=sys.stderr)
    sys.exit(1)

SAMPLE_RATE = 24000  # Server expects 24kHz for realtime API
CHANNELS = 1
DTYPE = "int16"
CHUNK_SIZE = 2400  # 100ms of audio at 24kHz

BASE_URL = os.environ.get("WHISPER_BASE_URL", "http://localhost:10300")
WS_URL = BASE_URL.replace("http://", "ws://").replace("https://", "wss://") + "/v1/realtime"
MODEL = os.environ.get("WHISPER_MODEL", "Systran/faster-whisper-large-v3")
LANG = os.environ.get("WHISPER_LANG", "en")
PASTE_DELAY = float(os.environ.get("DICTATION_PASTE_DELAY", "0.3"))


def detect_display():
    """Detect X11 vs Wayland."""
    return os.environ.get("XDG_SESSION_TYPE", "x11")


def type_text(text):
    """Type text into the focused window via clipboard."""
    display = detect_display()
    if display == "wayland":
        # Save clipboard, set new content, paste, restore
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


async def stream_dictation(once=False):
    """Connect to the realtime API and stream microphone audio."""
    ws_url = f"{WS_URL}?model={MODEL}&intent=transcription&language={LANG}"

    notify("🎤 Streaming...", "Speak now (Ctrl+C to stop)")

    try:
        async with websockets.connect(ws_url, open_timeout=10) as ws:
            # Configure session for transcription-only mode
            await ws.send(json.dumps({
                "type": "session.update",
                "session": {
                    "input_audio_transcription": {"model": MODEL},
                    "turn_detection": {
                        "type": "server_vad",
                        "threshold": 0.5,
                        "silence_duration_ms": 1000,
                        "create_response": False,
                    }
                }
            }))

            # Consume handshake events (session.created, session.updated)
            for _ in range(3):
                msg = json.loads(await asyncio.wait_for(ws.recv(), timeout=10))
                if msg.get("type") in ("session.created", "session.updated"):
                    continue
                elif msg.get("type") == "error":
                    print(f"Server error during setup: {msg}", file=sys.stderr)
                    sys.exit(1)
                else:
                    break

            stop_event = asyncio.Event()

            async def receive_events():
                """Listen for transcription results."""
                loop = asyncio.get_running_loop()
                try:
                    async for raw in ws:
                        event = json.loads(raw)
                        etype = event.get("type", "")
                        if etype == "conversation.item.input_audio_transcription.completed":
                            transcript = event.get("transcript", "").strip()
                            if transcript:
                                print(transcript)
                                await loop.run_in_executor(None, type_text, transcript + " ")
                                if once:
                                    stop_event.set()
                                    return
                        elif etype == "error":
                            print(f"Server error: {event}", file=sys.stderr)
                except websockets.exceptions.ConnectionClosed:
                    pass
                finally:
                    stop_event.set()

            async def send_audio():
                """Stream microphone audio to the server."""
                loop = asyncio.get_running_loop()
                queue = asyncio.Queue()

                def audio_callback(indata, frames, time_info, status):
                    if status:
                        print(f"Audio: {status}", file=sys.stderr)
                    loop.call_soon_threadsafe(queue.put_nowait, bytes(indata))

                stream = sd.RawInputStream(
                    samplerate=SAMPLE_RATE, channels=CHANNELS,
                    dtype=DTYPE, blocksize=CHUNK_SIZE,
                    callback=audio_callback,
                )
                with stream:
                    while not stop_event.is_set():
                        try:
                            data = await asyncio.wait_for(queue.get(), timeout=0.5)
                            encoded = base64.b64encode(data).decode()
                            await ws.send(json.dumps({
                                "type": "input_audio_buffer.append",
                                "audio": encoded,
                            }))
                        except asyncio.TimeoutError:
                            continue
                        except websockets.exceptions.ConnectionClosed:
                            break

            recv_task = asyncio.create_task(receive_events())
            send_task = asyncio.create_task(send_audio())

            try:
                if once:
                    await stop_event.wait()
                else:
                    await asyncio.gather(recv_task, send_task)
            finally:
                for task in (recv_task, send_task):
                    if not task.done():
                        task.cancel()
                        try:
                            await task
                        except asyncio.CancelledError:
                            pass

    except ConnectionRefusedError:
        notify("❌ Error", "Whisper server is not running")
        print(f"Error: cannot connect to {ws_url}", file=sys.stderr)
        sys.exit(1)
    except websockets.exceptions.InvalidStatus as e:
        notify("❌ Error", f"Server returned HTTP {e.response.status_code}")
        if e.response.status_code in (403, 404):
            print(f"Error: {ws_url} returned {e.response.status_code}. "
                  "Streaming requires the speaches image (ghcr.io/speaches-ai/speaches). "
                  "See README for setup.", file=sys.stderr)
        else:
            print(f"Error: WebSocket connection rejected: {e}", file=sys.stderr)
        sys.exit(1)
    except (asyncio.TimeoutError, websockets.exceptions.WebSocketException, OSError) as e:
        notify("❌ Error", "Connection lost")
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        notify("❌ Error", str(e)[:80])
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    finally:
        notify("🎤 Stopped", "Streaming dictation ended")


def main():
    parser = argparse.ArgumentParser(description="Streaming dictation via faster-whisper-server")
    parser.add_argument("--once", action="store_true", help="Transcribe one utterance then exit")
    args = parser.parse_args()

    try:
        asyncio.run(stream_dictation(once=args.once))
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
