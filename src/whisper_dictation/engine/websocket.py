"""WebSocket streaming engine — real-time transcription via WhisperLive server."""

from __future__ import annotations

import ipaddress
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

# Sentinel for END_OF_AUDIO signal in the send queue
_END_SENTINEL = object()

# Chunk size for batch mode audio splitting (32ms at 16kHz = 512 samples)
_BATCH_CHUNK_SAMPLES = 512

# Max segments collected in batch mode to prevent unbounded memory
_MAX_BATCH_SEGMENTS = 1000

# Max incoming WS message size before discarding (1 MB)
_MAX_MESSAGE_BYTES = 1 * 1024 * 1024


def _http_to_ws_url(http_url: str) -> str:
    """Convert http(s):// URL to ws(s):// URL.

    Raises ValueError if scheme is not http or https.
    """
    parsed = urlparse(http_url)
    if parsed.scheme not in ("http", "https"):
        raise ValueError(
            f"server_url must use http or https, got {parsed.scheme!r}"
        )
    scheme = "wss" if parsed.scheme == "https" else "ws"
    return urlunparse(parsed._replace(scheme=scheme))


def _is_loopback(url: str) -> bool:
    """Check if URL points to a loopback address."""
    parsed = urlparse(url)
    host = (parsed.hostname or "").lower()
    if host in ("localhost", ""):
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def _build_config_message(
    *,
    uid: str,
    language: str = "en",
    model: str = "small",
    use_vad: bool = True,
) -> dict[str, str | bool]:
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
    return (audio.astype(np.float32) / 32768.0).tobytes()


class WebSocketEngine:
    """Real-time streaming and batch transcription via WhisperLive WebSocket.

    Supports two modes:
    - Streaming: connect() → send_audio() per chunk → flush() → close()
      Text arrives incrementally via on_text callback.
    - Batch: transcribe_batch(audio) → returns full text synchronously.
      Connects, sends all audio, waits for completion, returns text.

    Protocol (WhisperLive):
    1. Connect to ws://<host>:<port>
    2. Send JSON config message (uid, language, model, use_vad)
    3. Wait for SERVER_READY
    4. Stream audio as binary float32 frames
    5. Receive JSON segments with completed/partial flags
    6. Send END_OF_AUDIO text frame when done

    Thread safety: public methods (connect, send_audio, flush, close) are safe
    to call from one caller thread. The receiver and sender threads run
    internally. Do NOT call connect/transcribe_batch concurrently from
    multiple threads — the engine is single-caller.
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
        on_text: Callable[[str], None] | None = None,
    ):
        self._server_url = server_url.rstrip("/")
        self._model = model
        self._language = language
        self._reconnect_attempts = reconnect_attempts
        self._reconnect_delay = reconnect_delay
        self._use_vad = use_vad
        self._on_text = on_text or (lambda t: None)
        self._uid = str(uuid.uuid4())

        self._ws = None
        self._send_queue: queue.Queue[bytes | object | None] = queue.Queue(
            maxsize=_MAX_SEND_QUEUE,
        )
        self._sender_thread: threading.Thread | None = None
        self._receiver_thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._connected = threading.Event()
        self._server_ready = threading.Event()
        self._flush_done = threading.Event()

        # Lock protects _emitted_count and _batch_collected which are
        # shared between the caller thread and the receiver thread.
        self._seg_lock = threading.Lock()
        self._emitted_count = 0
        self._batch_collected: list[str] | None = None

    @property
    def ws_url(self) -> str:
        """Construct the WebSocket URL."""
        return _http_to_ws_url(self._server_url)

    def connect(self) -> None:
        """Open WebSocket connection with retry, send config, start threads."""
        self._stop_event.clear()
        self._connected.clear()
        self._server_ready.clear()
        self._flush_done.clear()
        with self._seg_lock:
            self._emitted_count = 0
            self._batch_collected = None
        self._uid = str(uuid.uuid4())

        if not _is_loopback(self._server_url) and self.ws_url.startswith("ws://"):
            log.warning(
                "WebSocket to non-localhost %s uses unencrypted ws://. "
                "Consider https:// (wss://) for remote servers.",
                self._server_url,
            )

        last_error: Exception | None = None
        attempts = self._reconnect_attempts + 1

        for attempt in range(attempts):
            try:
                self._ws = ws_sync.connect(self.ws_url, max_size=_MAX_MESSAGE_BYTES)
                break
            except Exception as e:
                last_error = e
                if attempt < attempts - 1:
                    log.warning(
                        "WebSocket connect attempt %d/%d failed: %s, retrying in %.1fs",
                        attempt + 1, attempts, e, self._reconnect_delay,
                    )
                    time.sleep(self._reconnect_delay)

        if self._ws is None:
            log.error("WebSocket connection failed after %d attempts: %s", attempts, self.ws_url)
            raise last_error  # type: ignore[misc]

        # Start threads BEFORE sending config so the receiver is ready
        # to catch SERVER_READY immediately.
        self._receiver_thread = threading.Thread(target=self._receiver_loop, daemon=True)
        self._receiver_thread.start()
        self._sender_thread = threading.Thread(target=self._sender_loop, daemon=True)
        self._sender_thread.start()

        config_msg = _build_config_message(
            uid=self._uid, language=self._language,
            model=self._model, use_vad=self._use_vad,
        )
        self._ws.send(json.dumps(config_msg))

        if not self._server_ready.wait(timeout=10.0):
            log.warning("WhisperLive did not send SERVER_READY within 10s, proceeding anyway")

        self._connected.set()
        log.info("WebSocket connected: %s (uid=%s)", self.ws_url, self._uid[:8])

    def send_audio(self, audio: np.ndarray) -> None:
        """Queue audio for sending. Thread-safe, non-blocking."""
        if not self._connected.is_set():
            return
        try:
            self._send_queue.put_nowait(_audio_to_float32_bytes(audio))
        except queue.Full:
            log.debug("Send queue full, dropping audio chunk")

    def flush(self) -> None:
        """Send END_OF_AUDIO to signal end of stream."""
        if not self._connected.is_set():
            return
        self._flush_done.clear()
        with self._seg_lock:
            self._emitted_count = 0  # reset for post-flush segment processing
        try:
            self._send_queue.put_nowait(_END_SENTINEL)
        except queue.Full:
            log.warning("Send queue full, END_OF_AUDIO dropped")

    def wait_for_completion(self, timeout: float = 5.0) -> bool:
        """Wait for server to finish processing after flush."""
        return self._flush_done.wait(timeout=timeout)

    def transcribe_batch(self, audio: np.ndarray, sample_rate: int = 16000) -> str:
        """Transcribe audio synchronously via WebSocket (batch mode).

        Connects, sends all audio in chunks, sends END_OF_AUDIO, waits for
        completed segments, returns concatenated text. Uses use_vad=False
        since the audio is already a complete utterance.
        """
        saved_vad = self._use_vad
        self._use_vad = False

        try:
            self.connect()
            # Set AFTER connect() which resets _batch_collected to None
            with self._seg_lock:
                self._batch_collected = []

            # Send audio in chunks
            for i in range(0, len(audio), _BATCH_CHUNK_SAMPLES):
                chunk = audio[i: i + _BATCH_CHUNK_SAMPLES]
                self.send_audio(chunk)

            self.flush()

            duration = len(audio) / sample_rate
            timeout = max(15.0, duration * 0.5)
            if not self.wait_for_completion(timeout=timeout):
                log.warning("Batch transcription timed out after %.0fs", timeout)

            self.close()
            with self._seg_lock:
                collected = list(self._batch_collected or [])
            return " ".join(collected).strip()
        except Exception:
            log.error("Batch WS transcription failed", exc_info=True)
            self.close()
            return ""
        finally:
            self._use_vad = saved_vad
            with self._seg_lock:
                self._batch_collected = None

    def close(self) -> None:
        """Close WebSocket connection and stop threads."""
        self._stop_event.set()
        self._connected.clear()

        # Send None sentinel to unblock the sender thread
        try:
            self._send_queue.put_nowait(None)
        except queue.Full:
            pass

        # Join threads BEFORE closing socket so sender can finish
        # any pending END_OF_AUDIO frame.
        if self._sender_thread is not None:
            self._sender_thread.join(timeout=2.0)
            if self._sender_thread.is_alive():
                log.warning("WebSocket sender thread did not exit within timeout")
            self._sender_thread = None

        ws = self._ws
        self._ws = None
        if ws is not None:
            try:
                ws.close()
            except Exception:
                log.debug("WebSocket close error", exc_info=True)

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
            with ws_sync.connect(self.ws_url, close_timeout=3, open_timeout=3):
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
                break

            ws = self._ws
            if ws is None:
                break

            try:
                if data is _END_SENTINEL:
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

            if isinstance(raw, bytes):
                log.debug("WS binary frame ignored (%d bytes)", len(raw))
                continue

            if isinstance(raw, str) and len(raw) > _MAX_MESSAGE_BYTES:
                log.warning("WS message too large (%d bytes), skipping", len(raw))
                continue

            try:
                msg = json.loads(raw)
            except (json.JSONDecodeError, TypeError):
                log.debug("Invalid JSON: %r", raw[:200] if raw else raw)
                continue

            try:
                self._handle_message(msg)
            except Exception:
                log.warning("Error handling WS message", exc_info=True)

    def _handle_message(self, msg: dict) -> None:
        """Process a WhisperLive JSON message."""
        if not isinstance(msg, dict):
            return

        if msg.get("message") == "SERVER_READY":
            log.debug("WS server ready (backend=%s)", msg.get("backend", "?"))
            self._server_ready.set()
            return

        if msg.get("status") == "WAIT":
            log.warning("WS server busy, wait %s minutes", msg.get("message", "?"))
            return

        segments = msg.get("segments")
        if isinstance(segments, list):
            self._process_segments(segments)
            return

        if "language" in msg and "language_prob" in msg:
            try:
                prob = float(msg["language_prob"]) * 100
                log.debug("WS detected language: %s (%.0f%%)", msg["language"], prob)
            except (TypeError, ValueError):
                log.debug("WS language detection: %s", msg.get("language"))
            return

        log.debug("WS unhandled message: %s", str(msg)[:200])

    def _process_segments(self, segments: list) -> None:
        """Process transcription segments — emit only completed (finalized) text.

        In streaming mode, emits via on_text callback.
        In batch mode (_batch_collected is not None), collects into list.
        """
        valid = [s for s in segments if isinstance(s, dict)]
        if not valid:
            return

        completed = [s for s in valid if s.get("completed", False)]

        with self._seg_lock:
            new_completed = completed[self._emitted_count:]
            self._emitted_count = len(completed)
            batch = self._batch_collected

        for seg in new_completed:
            text = seg.get("text", "").strip()
            if text:
                log.debug("WS text (final): %s", text[:80])
                if batch is not None:
                    if len(batch) < _MAX_BATCH_SEGMENTS:
                        batch.append(text)
                else:
                    self._on_text(text)

        last_seg = valid[-1]
        if last_seg.get("completed", False):
            self._flush_done.set()
