"""WebSocket streaming engine — real-time transcription via Speaches Realtime API."""

from __future__ import annotations

import base64
import json
import logging
import queue
import threading
from collections.abc import Callable
from urllib.parse import urlparse, urlunparse

import numpy as np
import websockets.sync.client as ws_sync

log = logging.getLogger(__name__)


def _http_to_ws_url(http_url: str) -> str:
    """Convert http(s):// URL to ws(s):// URL."""
    parsed = urlparse(http_url)
    scheme = "wss" if parsed.scheme == "https" else "ws"
    return urlunparse(parsed._replace(scheme=scheme))


def _build_session_update(
    *,
    silence_duration_ms: int = 500,
    vad_threshold: float = 0.5,
) -> dict:
    """Build a session.update message for the Realtime API."""
    return {
        "type": "session.update",
        "session": {
            "input_audio_transcription": {},
            "turn_detection": {
                "type": "server_vad",
                "silence_duration_ms": silence_duration_ms,
                "threshold": vad_threshold,
                "create_response": False,
            },
        },
    }


def _build_audio_append(audio_b64: str) -> dict:
    """Build an input_audio_buffer.append message."""
    return {
        "type": "input_audio_buffer.append",
        "audio": audio_b64,
    }


def _build_audio_commit() -> dict:
    """Build an input_audio_buffer.commit message."""
    return {"type": "input_audio_buffer.commit"}


def _encode_audio_b64(audio: np.ndarray) -> str:
    """Encode float32 audio as base64 PCM int16."""
    if audio.dtype == np.float32:
        pcm = (audio * 32768).clip(-32768, 32767).astype(np.int16)
    else:
        pcm = audio.astype(np.int16)
    return base64.b64encode(pcm.tobytes()).decode("ascii")


class WebSocketEngine:
    """Real-time streaming transcription via WebSocket.

    Connects to a Speaches-compatible Realtime API endpoint. Audio chunks
    are streamed continuously; transcription results arrive asynchronously
    via the on_text callback.

    This does NOT implement the TranscriptionEngine ABC because the interface
    is fundamentally different: async callbacks vs synchronous request/response.
    """

    def __init__(
        self,
        *,
        server_url: str,
        model: str,
        language: str,
        reconnect_attempts: int = 3,
        reconnect_delay: float = 1.0,
        server_vad_silence_ms: int = 500,
        server_vad_threshold: float = 0.5,
        on_text: Callable[[str], None],
    ):
        self._server_url = server_url.rstrip("/")
        self._model = model
        self._language = language
        self._reconnect_attempts = reconnect_attempts
        self._reconnect_delay = reconnect_delay
        self._server_vad_silence_ms = server_vad_silence_ms
        self._server_vad_threshold = server_vad_threshold
        self._on_text = on_text

        self._ws = None
        self._send_queue: queue.Queue[str | None] = queue.Queue()
        self._sender_thread: threading.Thread | None = None
        self._receiver_thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._connected = threading.Event()

    @property
    def ws_url(self) -> str:
        """Construct the WebSocket URL with model and language query params."""
        base = _http_to_ws_url(self._server_url)
        url = f"{base}/v1/realtime?intent=transcription&model={self._model}"
        if self._language:
            url += f"&language={self._language}"
        return url

    def connect(self) -> None:
        """Open WebSocket connection and start sender/receiver threads."""
        self._stop_event.clear()
        self._connected.clear()

        try:
            self._ws = ws_sync.connect(self.ws_url)
        except Exception:
            log.error("WebSocket connection failed: %s", self.ws_url, exc_info=True)
            raise

        # Send session configuration
        session_msg = _build_session_update(
            silence_duration_ms=self._server_vad_silence_ms,
            vad_threshold=self._server_vad_threshold,
        )
        self._ws.send(json.dumps(session_msg))
        self._connected.set()

        # Start sender thread (reads from queue, writes to WS)
        self._sender_thread = threading.Thread(target=self._sender_loop, daemon=True)
        self._sender_thread.start()

        # Start receiver thread (reads from WS, calls on_text)
        self._receiver_thread = threading.Thread(target=self._receiver_loop, daemon=True)
        self._receiver_thread.start()

        log.info("WebSocket connected: %s", self.ws_url)

    def send_audio(self, audio: np.ndarray) -> None:
        """Queue audio for sending. Thread-safe, non-blocking."""
        if not self._connected.is_set():
            return
        b64 = _encode_audio_b64(audio)
        msg = json.dumps(_build_audio_append(b64))
        try:
            self._send_queue.put_nowait(msg)
        except queue.Full:
            log.debug("Send queue full, dropping audio chunk")

    def flush(self) -> None:
        """Send commit message to trigger server-side processing of remaining audio."""
        if not self._connected.is_set():
            return
        msg = json.dumps(_build_audio_commit())
        try:
            self._send_queue.put_nowait(msg)
        except queue.Full:
            pass

    def close(self) -> None:
        """Close WebSocket connection and stop threads."""
        self._stop_event.set()
        self._connected.clear()

        # Signal sender to exit
        try:
            self._send_queue.put_nowait(None)
        except queue.Full:
            pass

        if self._ws is not None:
            try:
                self._ws.close()
            except Exception:
                log.debug("WebSocket close error", exc_info=True)
            self._ws = None

        if self._sender_thread is not None:
            self._sender_thread.join(timeout=2.0)
            self._sender_thread = None
        if self._receiver_thread is not None:
            self._receiver_thread.join(timeout=2.0)
            self._receiver_thread = None

        # Drain the queue
        while not self._send_queue.empty():
            try:
                self._send_queue.get_nowait()
            except queue.Empty:
                break

        log.info("WebSocket closed")

    def is_available(self) -> bool:
        """Check if the Realtime API endpoint is reachable."""
        try:
            ws = ws_sync.connect(self.ws_url, close_timeout=3, open_timeout=3)
            ws.close()
            return True
        except Exception:
            return False

    def _sender_loop(self) -> None:
        """Send queued messages to WebSocket."""
        while not self._stop_event.is_set():
            try:
                msg = self._send_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            if msg is None:
                break  # shutdown signal

            if self._ws is None:
                break

            try:
                self._ws.send(msg)
            except Exception:
                log.debug("WebSocket send failed", exc_info=True)
                break

    def _receiver_loop(self) -> None:
        """Receive messages from WebSocket and dispatch."""
        while not self._stop_event.is_set():
            if self._ws is None:
                break

            try:
                raw = self._ws.recv(timeout=0.5)
            except TimeoutError:
                continue
            except Exception:
                if not self._stop_event.is_set():
                    log.debug("WebSocket recv error", exc_info=True)
                break

            try:
                msg = json.loads(raw)
            except (json.JSONDecodeError, TypeError):
                log.debug("Invalid JSON from WebSocket: %r", raw[:200] if raw else raw)
                continue

            msg_type = msg.get("type", "")
            self._handle_message(msg_type, msg)

    def _handle_message(self, msg_type: str, msg: dict) -> None:
        """Dispatch a received WebSocket message by type."""
        if msg_type == "conversation.item.input_audio_transcription.completed":
            transcript = msg.get("transcript", "").strip()
            if transcript:
                log.debug("WS transcription: %s", transcript[:80])
                self._on_text(transcript)

        elif msg_type == "conversation.item.input_audio_transcription.delta":
            # Partial/delta text — could be used for live preview in future
            delta = msg.get("delta", "")
            log.debug("WS delta: %s", delta[:80] if delta else "")

        elif msg_type == "error":
            error = msg.get("error", {})
            log.error("WebSocket server error: %s", error.get("message", str(error)))

        elif msg_type in ("session.created", "session.updated"):
            log.debug("WS session: %s", msg_type)

        else:
            log.debug("WS unhandled message type: %s", msg_type)
