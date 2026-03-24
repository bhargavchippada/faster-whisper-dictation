"""WhisperLiveKit streaming engine — real-time transcription via WhisperLiveKit server."""

from __future__ import annotations

import ipaddress
import json
import logging
import queue
import threading
import time
import uuid
from collections.abc import Callable
from urllib.parse import urlencode, urlparse, urlunparse

import numpy as np
import websockets.sync.client as ws_sync

log = logging.getLogger(__name__)

# Max queued audio frames before dropping (~3s at 32ms chunks)
_MAX_SEND_QUEUE = 100

# Sentinel for end-of-audio signal in the send queue
_END_SENTINEL = object()

# Chunk size for batch mode audio splitting (32ms at 16kHz = 512 samples)
_BATCH_CHUNK_SAMPLES = 512

# Max lines collected in batch mode to prevent unbounded memory
_MAX_BATCH_LINES = 1000

# Max incoming WS message size before discarding (1 MB)
_MAX_MESSAGE_BYTES = 1 * 1024 * 1024


def _http_to_ws_url(http_url: str, *, path: str = "/asr", language: str = "en") -> str:
    """Convert http(s):// URL to ws(s):// URL with /asr path and language query param.

    Raises ValueError if scheme is not http or https.
    """
    parsed = urlparse(http_url)
    if parsed.scheme not in ("http", "https"):
        raise ValueError(
            f"server_url must use http or https, got {parsed.scheme!r}"
        )
    scheme = "wss" if parsed.scheme == "https" else "ws"
    # Append /asr path and language query param
    new_path = parsed.path.rstrip("/") + path
    query = urlencode({"language": language})
    return urlunparse(parsed._replace(scheme=scheme, path=new_path, query=query))


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


def _audio_to_int16_bytes(audio: np.ndarray) -> bytes:
    """Convert audio to int16 (s16le) PCM bytes for WhisperLiveKit.

    WhisperLiveKit expects raw int16 PCM at 16kHz mono as binary WS frames.
    """
    if audio.dtype == np.int16:
        return audio.tobytes()
    if audio.dtype in (np.float32, np.float64):
        return np.clip(audio * 32767.0, -32768, 32767).astype(np.int16).tobytes()
    return (audio.astype(np.float32) / 32768.0 * 32767.0).astype(np.int16).tobytes()


class WhisperLiveKitEngine:
    """Real-time streaming and batch transcription via WhisperLiveKit WebSocket.

    Supports two modes:
    - Streaming: connect() → send_audio() per chunk → flush() → close()
      Text arrives incrementally via on_text callback.
    - Batch: transcribe_batch(audio) → returns full text synchronously.
      Connects, sends all audio, waits for completion, returns text.

    Protocol (WhisperLiveKit):
    1. Connect to ws://<host>:<port>/asr?language=<lang>
    2. Server sends config message first (no client handshake needed)
    3. Stream audio as binary int16 (s16le) frames
    4. Receive JSON with "lines" (completed) and "buffer_transcription" (partial)
    5. Send empty frame b"" to signal end of audio
    6. Server sends {"type": "ready_to_stop"} when done

    Thread safety: public methods (connect, send_audio, flush, close) are safe
    to call from one caller thread. The receiver and sender threads run
    internally. Do NOT call connect/transcribe_batch concurrently from
    multiple threads — the engine is single-caller.
    """

    def __init__(
        self,
        *,
        server_url: str,
        language: str,
        reconnect_attempts: int = 3,
        reconnect_delay: float = 1.0,
        on_text: Callable[[str], None] | None = None,
    ):
        self._server_url = server_url.rstrip("/")
        self._language = language
        if reconnect_attempts < 0:
            raise ValueError("reconnect_attempts must be >= 0")
        if reconnect_delay <= 0:
            raise ValueError("reconnect_delay must be > 0")
        self._reconnect_attempts = reconnect_attempts
        self._reconnect_delay = reconnect_delay
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
        self._flush_done = threading.Event()
        self._eoa_sent = False

        # Lock protects _latest_full_text shared between caller and receiver.
        self._seg_lock = threading.Lock()
        self._latest_full_text: str = ""

    @property
    def ws_url(self) -> str:
        """Construct the WebSocket URL with /asr path and language query param."""
        return _http_to_ws_url(
            self._server_url, path="/asr", language=self._language,
        )

    def connect(self) -> None:
        """Open WebSocket connection with retry, receive server config, start threads."""
        self._stop_event.clear()
        self._connected.clear()
        self._flush_done.clear()
        # Drain stale data from previous session
        while not self._send_queue.empty():
            try:
                self._send_queue.get_nowait()
            except queue.Empty:
                break
        with self._seg_lock:
            self._latest_full_text = ""
            self._eoa_sent = False
        self._uid = str(uuid.uuid4())

        ws_url = self.ws_url
        if not _is_loopback(self._server_url) and ws_url.startswith("ws://"):
            log.warning(
                "WebSocket to non-localhost %s uses unencrypted ws://. "
                "Consider https:// (wss://) for remote servers.",
                self._server_url,
            )

        last_error: Exception | None = None
        attempts = self._reconnect_attempts + 1

        for attempt in range(attempts):
            try:
                self._ws = ws_sync.connect(ws_url, max_size=_MAX_MESSAGE_BYTES)
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
            log.error("WebSocket connection failed after %d attempts: %s", attempts, ws_url)
            raise last_error  # type: ignore[misc]

        try:
            # Start threads — receiver will pick up the server's initial config message
            self._receiver_thread = threading.Thread(target=self._receiver_loop, daemon=True)
            self._receiver_thread.start()
            self._sender_thread = threading.Thread(target=self._sender_loop, daemon=True)
            self._sender_thread.start()

            # No client handshake — server sends config first (handled by receiver).
            # WhisperLiveKit is ready to accept audio immediately after connect.
            self._connected.set()
            log.info("WhisperLiveKit connected: %s (uid=%s)", ws_url, self._uid[:8])
        except Exception:
            self.close()
            raise

    def send_audio(self, audio: np.ndarray) -> None:
        """Queue audio for sending. Thread-safe, non-blocking."""
        if not self._connected.is_set():
            return
        try:
            self._send_queue.put_nowait(_audio_to_int16_bytes(audio))
        except queue.Full:
            log.debug("Send queue full, dropping audio chunk")

    def _send_audio_blocking(self, audio: np.ndarray) -> None:
        """Queue audio with blocking put. Used by batch mode to avoid drops.

        Checks _stop_event between retries to avoid hanging if the sender
        thread has died (e.g. connection drop mid-batch).
        """
        if not self._connected.is_set():
            return
        data = _audio_to_int16_bytes(audio)
        while not self._stop_event.is_set():
            try:
                self._send_queue.put(data, timeout=0.5)
                return
            except queue.Full:
                continue

    def flush(self, *, send_eoa: bool = True) -> None:
        """Signal end of audio stream.

        Args:
            send_eoa: If True, send empty frame to signal end of audio.
                WhisperLiveKit uses b"" as the end-of-audio marker and
                responds with ready_to_stop when done. For streaming mode,
                set False to keep the connection alive for pending results;
                wait_for_completion() will time out and get_pending_text()
                provides the fallback.
        """
        if not self._connected.is_set():
            return
        self._flush_done.clear()
        if send_eoa:
            with self._seg_lock:
                self._eoa_sent = True
            try:
                self._send_queue.put_nowait(_END_SENTINEL)
            except queue.Full:
                with self._seg_lock:
                    self._eoa_sent = False
                log.warning("Send queue full, end-of-audio dropped")

    def wait_for_completion(self, timeout: float = 5.0) -> bool:
        """Wait for server to finish processing after flush."""
        return self._flush_done.wait(timeout=timeout)

    def transcribe_batch(self, audio: np.ndarray, sample_rate: int = 16000) -> str:
        """Transcribe audio synchronously via WebSocket (batch mode).

        Connects, sends all audio in chunks, sends end-of-audio, waits for
        ready_to_stop, returns concatenated text.
        """
        try:
            self.connect()

            # Send audio in chunks (blocking put to avoid drops)
            for i in range(0, len(audio), _BATCH_CHUNK_SAMPLES):
                chunk = audio[i: i + _BATCH_CHUNK_SAMPLES]
                self._send_audio_blocking(chunk)

            # Send end-of-audio and wait for ready_to_stop
            self._flush_done.clear()
            with self._seg_lock:
                self._eoa_sent = True
            try:
                self._send_queue.put(_END_SENTINEL, timeout=5.0)
            except queue.Full:
                with self._seg_lock:
                    self._eoa_sent = False
                log.warning("Send queue full, end-of-audio not sent")

            duration = len(audio) / sample_rate
            timeout = max(30.0, duration * 2.0)
            if not self.wait_for_completion(timeout=timeout):
                log.warning("Batch transcription timed out after %.0fs", timeout)

            return self.get_pending_text()
        except Exception:
            log.error("Batch WLK transcription failed", exc_info=True)
            raise
        finally:
            self.close()

    def close(self) -> None:
        """Close WebSocket connection and stop threads."""
        self._stop_event.set()
        self._connected.clear()
        self._flush_done.set()  # unblock any wait_for_completion() callers

        # Send None sentinel to unblock the sender thread
        try:
            self._send_queue.put_nowait(None)
        except queue.Full:
            pass

        # Join threads BEFORE closing socket so sender can finish
        self._join_thread(self._sender_thread, "sender")
        self._sender_thread = None

        ws = self._ws
        self._ws = None
        if ws is not None:
            try:
                ws.close()
            except Exception:
                log.debug("WebSocket close error", exc_info=True)

        self._join_thread(self._receiver_thread, "receiver")
        self._receiver_thread = None

        while not self._send_queue.empty():
            try:
                self._send_queue.get_nowait()
            except queue.Empty:
                break

        log.info("WhisperLiveKit closed")

    def _join_thread(self, thread: threading.Thread | None, name: str) -> None:
        """Join a started thread without raising on synthetic/unstarted test threads."""
        if thread is None:
            return
        try:
            if thread.ident is not None or thread.is_alive():
                thread.join(timeout=2.0)
                if thread.is_alive():
                    log.warning("WLK %s thread did not exit within timeout", name)
        except RuntimeError:
            log.debug("WLK %s thread was never started", name)

    def get_pending_text(self) -> str:
        """Return the latest accumulated transcription text.

        WhisperLiveKit updates lines in-place, so this always returns
        the most recent full text. Text is emitted once on deactivation.
        """
        with self._seg_lock:
            return self._latest_full_text

    def is_available(self) -> bool:
        """Check if the WhisperLiveKit server is reachable."""
        try:
            with ws_sync.connect(self.ws_url, close_timeout=3, open_timeout=3) as ws:
                # Consume the server's initial config message before closing
                # to avoid leaving the server handler in a confused state.
                try:
                    ws.recv(timeout=2.0)
                except TimeoutError:
                    pass
                return True
        except Exception:
            return False

    def _sender_loop(self) -> None:
        """Send queued audio frames to WebSocket."""
        try:
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
                        ws.send(b"")  # WhisperLiveKit EOA = empty frame
                    else:
                        ws.send(data)
                except Exception:
                    log.debug("WebSocket send failed", exc_info=True)
                    break
        finally:
            # Signal callers so they don't hang waiting for a dead connection
            self._stop_event.set()
            self._flush_done.set()

    def _receiver_loop(self) -> None:
        """Receive JSON messages from WhisperLiveKit and dispatch."""
        try:
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
                    log.debug("WLK binary frame ignored (%d bytes)", len(raw))
                    continue

                if isinstance(raw, str) and len(raw) > _MAX_MESSAGE_BYTES:
                    log.warning("WLK message too large, skipping")
                    continue

                try:
                    msg = json.loads(raw)
                except (json.JSONDecodeError, TypeError):
                    log.debug("Invalid JSON: %r", raw[:200] if raw else raw)
                    continue

                try:
                    self._handle_message(msg)
                except Exception:
                    log.warning("Error handling WLK message", exc_info=True)
        finally:
            # Signal callers so they don't hang waiting for a dead connection
            self._stop_event.set()
            self._flush_done.set()

    def _handle_message(self, msg: dict[str, object]) -> None:
        """Process a WhisperLiveKit JSON message."""
        if not isinstance(msg, dict):
            return

        # Server initial config message (informational)
        if msg.get("type") == "config":
            log.debug("WLK server config: %s", str(msg)[:200])
            return

        # Server signals transcription complete
        if msg.get("type") == "ready_to_stop":
            log.debug("WLK server ready to stop")
            with self._seg_lock:
                should_signal = self._eoa_sent
            if should_signal:
                self._flush_done.set()
            return

        # Transcription response: lines (completed) + buffer_transcription (partial)
        lines = msg.get("lines")
        buffer_text = msg.get("buffer_transcription")
        if isinstance(lines, list) or isinstance(buffer_text, str):
            log.debug(
                "WLK response: lines=%d, buffer=%r",
                len(lines) if isinstance(lines, list) else 0,
                (buffer_text[:80] if isinstance(buffer_text, str) else ""),
            )
            self._process_response(
                lines=lines if isinstance(lines, list) else [],
                buffer_text=buffer_text if isinstance(buffer_text, str) else "",
            )
            return

        log.debug("WLK unhandled message: %s", str(msg)[:200])

    def _process_response(
        self, lines: list[dict[str, object]], buffer_text: str,
    ) -> None:
        """Process WhisperLiveKit transcription response.

        WhisperLiveKit updates lines in-place (the same line grows as
        transcription progresses) rather than appending new completed
        segments. We track the full text and emit it on flush/close.

        Args:
            lines: List of line dicts, each with "text" key. Typically
                one line that keeps growing.
            buffer_text: Current partial/in-progress transcription text.
        """
        valid_lines = [ln for ln in lines if isinstance(ln, dict)]

        # Build full text from all lines + buffer
        line_texts = [ln.get("text", "").strip() for ln in valid_lines]
        all_text_parts = [t for t in line_texts if t]
        if buffer_text.strip():
            all_text_parts.append(buffer_text.strip())
        full_text = " ".join(all_text_parts)

        with self._seg_lock:
            self._latest_full_text = full_text
