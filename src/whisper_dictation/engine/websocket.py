"""WebSocket streaming engine — real-time transcription via WhisperLive server."""

from __future__ import annotations

import json
import logging
import queue
import threading
import time
import uuid
from collections.abc import Callable
from urllib.parse import urlparse, urlunparse

import numpy as np
import websockets.sync.client as ws_sync

log = logging.getLogger(__name__)

# Max queued audio frames before dropping (~3s at 32ms chunks)
_MAX_SEND_QUEUE = 100


def _http_to_ws_url(http_url: str) -> str:
    """Convert http(s):// URL to ws(s):// URL."""
    parsed = urlparse(http_url)
    scheme = "wss" if parsed.scheme == "https" else "ws"
    return urlunparse(parsed._replace(scheme=scheme))


def _build_config_message(
    *,
    uid: str,
    language: str = "en",
    model: str = "small",
    use_vad: bool = True,
) -> dict:
    """Build the WhisperLive handshake config message."""
    return {
        "uid": uid,
        "language": language,
        "task": "transcribe",
        "model": model,
        "use_vad": use_vad,
    }


def _audio_to_float32_bytes(audio: np.ndarray) -> bytes:
    """Convert audio to float32 PCM bytes for WhisperLive.

    WhisperLive expects raw float32 PCM at 16kHz mono as binary WS frames.
    """
    if audio.dtype in (np.float32, np.float64):
        return audio.astype(np.float32).tobytes()
    # int16 → float32
    return (audio.astype(np.float32) / 32768.0).tobytes()


class WebSocketEngine:
    """Real-time streaming transcription via WhisperLive WebSocket.

    Connects to a WhisperLive server. Audio chunks are streamed as binary
    float32 PCM frames; transcription results arrive as JSON with completed
    and partial segments.

    Protocol:
    1. Connect to ws://<host>:<port>
    2. Send JSON config message (uid, language, model, use_vad)
    3. Wait for SERVER_READY
    4. Stream audio as binary float32 frames
    5. Receive JSON segments with completed/partial flags
    6. Send END_OF_AUDIO text frame when done

    This does NOT implement TranscriptionEngine ABC — async callbacks
    vs synchronous request/response.
    """

    def __init__(
        self,
        *,
        server_url: str,
        model: str,
        language: str,
        reconnect_attempts: int = 3,
        reconnect_delay: float = 1.0,
        use_vad: bool = True,
        on_text: Callable[[str], None],
    ):
        self._server_url = server_url.rstrip("/")
        self._model = model
        self._language = language
        self._reconnect_attempts = reconnect_attempts
        self._reconnect_delay = reconnect_delay
        self._use_vad = use_vad
        self._on_text = on_text
        self._uid = str(uuid.uuid4())

        self._ws = None
        self._send_queue: queue.Queue[bytes | None] = queue.Queue(maxsize=_MAX_SEND_QUEUE)
        self._sender_thread: threading.Thread | None = None
        self._receiver_thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._connected = threading.Event()
        self._flush_done = threading.Event()

        # Track last emitted text to only type new content
        self._last_emitted = ""

    @property
    def ws_url(self) -> str:
        """Construct the WebSocket URL."""
        return _http_to_ws_url(self._server_url)

    def connect(self) -> None:
        """Open WebSocket connection with retry, send config, start threads."""
        self._stop_event.clear()
        self._connected.clear()
        self._last_emitted = ""
        self._uid = str(uuid.uuid4())

        last_error: Exception | None = None
        attempts = self._reconnect_attempts + 1

        for attempt in range(attempts):
            try:
                self._ws = ws_sync.connect(self.ws_url)
                break
            except Exception as e:
                last_error = e
                if attempt < attempts - 1:
                    log.warning(
                        "WebSocket connect attempt %d/%d failed: %s, retrying in %.1fs",
                        attempt + 1,
                        attempts,
                        e,
                        self._reconnect_delay,
                    )
                    time.sleep(self._reconnect_delay)

        if self._ws is None:
            log.error("WebSocket connection failed after %d attempts: %s", attempts, self.ws_url)
            raise last_error  # type: ignore[misc]

        # Send WhisperLive config handshake
        config_msg = _build_config_message(
            uid=self._uid,
            language=self._language,
            model=self._model,
            use_vad=self._use_vad,
        )
        self._ws.send(json.dumps(config_msg))

        # Start receiver first (to catch SERVER_READY)
        self._receiver_thread = threading.Thread(target=self._receiver_loop, daemon=True)
        self._receiver_thread.start()

        # Start sender thread
        self._sender_thread = threading.Thread(target=self._sender_loop, daemon=True)
        self._sender_thread.start()

        self._connected.set()
        log.info("WebSocket connected: %s (uid=%s)", self.ws_url, self._uid[:8])

    def send_audio(self, audio: np.ndarray) -> None:
        """Queue audio for sending. Thread-safe, non-blocking."""
        if not self._connected.is_set():
            return
        pcm_bytes = _audio_to_float32_bytes(audio)
        try:
            self._send_queue.put_nowait(pcm_bytes)
        except queue.Full:
            log.debug("Send queue full, dropping audio chunk")

    def flush(self) -> None:
        """Send END_OF_AUDIO to signal end of stream."""
        if not self._connected.is_set():
            return
        self._flush_done.clear()
        # Queue a sentinel bytes value that sender recognizes as END_OF_AUDIO
        try:
            self._send_queue.put_nowait(b"__END__")
        except queue.Full:
            pass

    def wait_for_completion(self, timeout: float = 5.0) -> bool:
        """Wait for server to finish processing after flush."""
        return self._flush_done.wait(timeout=timeout)

    def close(self) -> None:
        """Close WebSocket connection and stop threads."""
        self._stop_event.set()
        self._connected.clear()

        # Signal sender to exit
        try:
            self._send_queue.put_nowait(None)
        except queue.Full:
            pass

        ws = self._ws
        self._ws = None
        if ws is not None:
            try:
                ws.close()
            except Exception:
                log.debug("WebSocket close error", exc_info=True)

        if self._sender_thread is not None:
            self._sender_thread.join(timeout=2.0)
            if self._sender_thread.is_alive():
                log.warning("WebSocket sender thread did not exit within timeout")
            self._sender_thread = None
        if self._receiver_thread is not None:
            self._receiver_thread.join(timeout=2.0)
            if self._receiver_thread.is_alive():
                log.warning("WebSocket receiver thread did not exit within timeout")
            self._receiver_thread = None

        while not self._send_queue.empty():
            try:
                self._send_queue.get_nowait()
            except queue.Empty:
                break

        log.info("WebSocket closed")

    def is_available(self) -> bool:
        """Check if the WhisperLive server is reachable."""
        try:
            ws = ws_sync.connect(self.ws_url, close_timeout=3, open_timeout=3)
            ws.close()
            return True
        except Exception:
            return False

    def _sender_loop(self) -> None:
        """Send queued audio frames to WebSocket."""
        while not self._stop_event.is_set():
            try:
                data = self._send_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            if data is None:
                break  # shutdown signal

            ws = self._ws
            if ws is None:
                break

            try:
                if data == b"__END__":
                    ws.send("END_OF_AUDIO")
                else:
                    ws.send(data)
            except Exception:
                log.debug("WebSocket send failed", exc_info=True)
                break

    def _receiver_loop(self) -> None:
        """Receive JSON messages from WhisperLive and dispatch."""
        while not self._stop_event.is_set():
            ws = self._ws
            if ws is None:
                break

            try:
                raw = ws.recv(timeout=0.5)
            except TimeoutError:
                continue
            except Exception:
                if not self._stop_event.is_set():
                    log.debug("WebSocket recv error", exc_info=True)
                break

            # WhisperLive sends JSON text frames
            if isinstance(raw, bytes):
                log.debug("WS binary frame ignored (%d bytes)", len(raw))
                continue

            try:
                msg = json.loads(raw)
            except (json.JSONDecodeError, TypeError):
                log.debug("Invalid JSON from WebSocket: %r", raw[:200] if raw else raw)
                continue

            self._handle_message(msg)

    def _handle_message(self, msg: dict) -> None:
        """Process a WhisperLive JSON message."""
        # SERVER_READY handshake response
        if msg.get("message") == "SERVER_READY":
            log.debug("WS server ready (backend=%s)", msg.get("backend", "?"))
            return

        # WAIT message (server busy)
        if msg.get("status") == "WAIT":
            log.warning("WS server busy, wait %s minutes", msg.get("message", "?"))
            return

        # Transcription segments
        segments = msg.get("segments")
        if segments is not None:
            self._process_segments(segments)
            return

        # Language detection
        if "language" in msg and "language_prob" in msg:
            log.debug("WS detected language: %s (%.0f%%)", msg["language"], msg["language_prob"] * 100)
            return

        log.debug("WS unhandled message: %s", str(msg)[:200])

    def _process_segments(self, segments: list[dict]) -> None:
        """Process transcription segments and emit new text.

        WhisperLive segments have:
        - text: the transcribed text
        - completed: True if finalized, False if partial/interim

        We concatenate all segment text and emit only the new portion
        (diff against what was previously emitted).
        """
        if not segments:
            return

        # Build full text from all segments
        full_text = " ".join(seg.get("text", "").strip() for seg in segments).strip()
        if not full_text:
            return

        # Check if the last segment is completed (final)
        last_completed = segments[-1].get("completed", False)

        # Emit only new text (diff from last emitted)
        if len(full_text) > len(self._last_emitted):
            new_text = full_text[len(self._last_emitted):].strip()
            if new_text:
                log.debug("WS text (%s): %s", "final" if last_completed else "partial", new_text[:80])
                self._on_text(new_text)
                self._last_emitted = full_text

        # Signal completion when all segments are finalized
        if last_completed:
            self._flush_done.set()
