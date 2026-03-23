"""Tests for whisper_dictation.engine.websocket — WebSocketEngine."""

from __future__ import annotations

import base64
import json
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from whisper_dictation.engine.websocket import (
    WebSocketEngine,
    _build_audio_append,
    _build_audio_commit,
    _build_session_update,
    _encode_audio_b64,
    _http_to_ws_url,
)

# ---------------------------------------------------------------------------
# URL conversion
# ---------------------------------------------------------------------------


class TestHttpToWsUrl:
    def test_http_to_ws(self):
        assert _http_to_ws_url("http://localhost:10300") == "ws://localhost:10300"

    def test_https_to_wss(self):
        assert _http_to_ws_url("https://api.example.com") == "wss://api.example.com"

    def test_preserves_path(self):
        assert _http_to_ws_url("http://host:8080/api") == "ws://host:8080/api"

    def test_trailing_slash(self):
        assert _http_to_ws_url("http://host:8080/") == "ws://host:8080/"


# ---------------------------------------------------------------------------
# Protocol message builders
# ---------------------------------------------------------------------------


class TestBuildSessionUpdate:
    def test_default_values(self):
        msg = _build_session_update()
        assert msg["type"] == "session.update"
        td = msg["session"]["turn_detection"]
        assert td["type"] == "server_vad"
        assert td["silence_duration_ms"] == 500
        assert td["threshold"] == 0.5
        assert td["create_response"] is False

    def test_custom_values(self):
        msg = _build_session_update(silence_duration_ms=300, vad_threshold=0.8)
        td = msg["session"]["turn_detection"]
        assert td["silence_duration_ms"] == 300
        assert td["threshold"] == 0.8


class TestBuildAudioAppend:
    def test_structure(self):
        msg = _build_audio_append("AQID")
        assert msg["type"] == "input_audio_buffer.append"
        assert msg["audio"] == "AQID"


class TestBuildAudioCommit:
    def test_structure(self):
        msg = _build_audio_commit()
        assert msg["type"] == "input_audio_buffer.commit"


# ---------------------------------------------------------------------------
# Audio encoding
# ---------------------------------------------------------------------------


class TestEncodeAudioB64:
    def test_float32_encoding(self):
        audio = np.array([0.5, -0.5], dtype=np.float32)
        b64 = _encode_audio_b64(audio)
        decoded = base64.b64decode(b64)
        pcm = np.frombuffer(decoded, dtype=np.int16)
        assert pcm[0] == 16384  # 0.5 * 32768
        assert pcm[1] == -16384

    def test_int16_passthrough(self):
        audio = np.array([100, -100], dtype=np.int16)
        b64 = _encode_audio_b64(audio)
        decoded = base64.b64decode(b64)
        pcm = np.frombuffer(decoded, dtype=np.int16)
        assert pcm[0] == 100
        assert pcm[1] == -100

    def test_float64_encoding(self):
        audio = np.array([0.5, -0.5], dtype=np.float64)
        b64 = _encode_audio_b64(audio)
        decoded = base64.b64decode(b64)
        pcm = np.frombuffer(decoded, dtype=np.int16)
        assert pcm[0] == 16384
        assert pcm[1] == -16384

    def test_clipping(self):
        audio = np.array([1.5, -1.5], dtype=np.float32)
        b64 = _encode_audio_b64(audio)
        decoded = base64.b64decode(b64)
        pcm = np.frombuffer(decoded, dtype=np.int16)
        assert pcm[0] == 32767
        assert pcm[1] == -32768


# ---------------------------------------------------------------------------
# WebSocketEngine
# ---------------------------------------------------------------------------


class TestWebSocketEngineInit:
    def test_ws_url_construction(self):
        engine = WebSocketEngine(
            server_url="http://localhost:10300",
            model="tiny",
            language="en",
            on_text=MagicMock(),
        )
        assert engine.ws_url == "ws://localhost:10300/v1/realtime?intent=transcription&model=tiny&language=en"

    def test_ws_url_https(self):
        engine = WebSocketEngine(
            server_url="https://api.example.com",
            model="large-v3",
            language="",
            on_text=MagicMock(),
        )
        url = engine.ws_url
        assert url.startswith("wss://")
        assert "language=" not in url

    def test_ws_url_trailing_slash_stripped(self):
        engine = WebSocketEngine(
            server_url="http://host:8080/",
            model="tiny",
            language="en",
            on_text=MagicMock(),
        )
        assert "host:8080/v1/" in engine.ws_url


class TestWebSocketEngineConnect:
    def test_connect_sends_session_update(self):
        mock_ws = MagicMock()
        on_text = MagicMock()
        engine = WebSocketEngine(
            server_url="http://localhost:10300",
            model="tiny",
            language="en",
            on_text=on_text,
        )

        with patch("whisper_dictation.engine.websocket.ws_sync") as mock_ws_mod:
            mock_ws_mod.connect.return_value = mock_ws
            engine.connect()

        # Verify session.update was sent
        sent = mock_ws.send.call_args[0][0]
        msg = json.loads(sent)
        assert msg["type"] == "session.update"
        assert engine._connected.is_set()

        engine.close()

    def test_connect_failure_raises(self):
        engine = WebSocketEngine(
            server_url="http://localhost:99999",
            model="tiny",
            language="en",
            reconnect_attempts=0,
            on_text=MagicMock(),
        )

        with patch("whisper_dictation.engine.websocket.ws_sync") as mock_ws_mod:
            mock_ws_mod.connect.side_effect = ConnectionRefusedError("refused")
            with pytest.raises(ConnectionRefusedError):
                engine.connect()

        assert not engine._connected.is_set()

    def test_connect_retries_on_failure(self):
        """Connect retries up to reconnect_attempts before raising."""
        engine = WebSocketEngine(
            server_url="http://localhost:10300",
            model="tiny",
            language="en",
            reconnect_attempts=2,
            reconnect_delay=0.01,
            on_text=MagicMock(),
        )

        with patch("whisper_dictation.engine.websocket.ws_sync") as mock_ws_mod:
            mock_ws_mod.connect.side_effect = ConnectionRefusedError("refused")
            with pytest.raises(ConnectionRefusedError):
                engine.connect()

        # 1 initial + 2 retries = 3 total attempts
        assert mock_ws_mod.connect.call_count == 3

    def test_connect_succeeds_on_retry(self):
        """Connect succeeds after initial failure."""
        mock_ws = MagicMock()
        engine = WebSocketEngine(
            server_url="http://localhost:10300",
            model="tiny",
            language="en",
            reconnect_attempts=2,
            reconnect_delay=0.01,
            on_text=MagicMock(),
        )

        with patch("whisper_dictation.engine.websocket.ws_sync") as mock_ws_mod:
            mock_ws_mod.connect.side_effect = [ConnectionRefusedError("fail"), mock_ws]
            engine.connect()

        assert engine._connected.is_set()
        assert mock_ws_mod.connect.call_count == 2
        engine.close()


class TestWebSocketEngineSendAudio:
    def test_send_audio_queues_message(self):
        engine = WebSocketEngine(
            server_url="http://localhost:10300",
            model="tiny",
            language="en",
            on_text=MagicMock(),
        )
        engine._connected.set()

        audio = np.zeros(512, dtype=np.float32)
        engine.send_audio(audio)

        msg = engine._send_queue.get_nowait()
        parsed = json.loads(msg)
        assert parsed["type"] == "input_audio_buffer.append"
        assert "audio" in parsed

    def test_send_audio_drops_when_queue_full(self):
        """Audio is silently dropped when queue is full."""
        engine = WebSocketEngine(
            server_url="http://localhost:10300",
            model="tiny",
            language="en",
            on_text=MagicMock(),
        )
        engine._connected.set()

        # Fill the queue
        for _ in range(engine._send_queue.maxsize):
            engine.send_audio(np.zeros(16, dtype=np.float32))

        # This should not raise — just drops
        engine.send_audio(np.zeros(16, dtype=np.float32))
        assert engine._send_queue.full()

    def test_send_audio_when_disconnected_is_noop(self):
        engine = WebSocketEngine(
            server_url="http://localhost:10300",
            model="tiny",
            language="en",
            on_text=MagicMock(),
        )
        # Not connected
        engine.send_audio(np.zeros(512, dtype=np.float32))
        assert engine._send_queue.empty()


class TestWebSocketEngineFlush:
    def test_flush_queues_commit(self):
        engine = WebSocketEngine(
            server_url="http://localhost:10300",
            model="tiny",
            language="en",
            on_text=MagicMock(),
        )
        engine._connected.set()

        engine.flush()

        msg = engine._send_queue.get_nowait()
        parsed = json.loads(msg)
        assert parsed["type"] == "input_audio_buffer.commit"

    def test_flush_when_disconnected_is_noop(self):
        engine = WebSocketEngine(
            server_url="http://localhost:10300",
            model="tiny",
            language="en",
            on_text=MagicMock(),
        )
        engine.flush()
        assert engine._send_queue.empty()


class TestWebSocketEngineClose:
    def test_close_cleans_up(self):
        mock_ws = MagicMock()
        engine = WebSocketEngine(
            server_url="http://localhost:10300",
            model="tiny",
            language="en",
            on_text=MagicMock(),
        )

        with patch("whisper_dictation.engine.websocket.ws_sync") as mock_ws_mod:
            mock_ws_mod.connect.return_value = mock_ws
            engine.connect()

        engine.close()

        assert engine._ws is None
        assert not engine._connected.is_set()
        mock_ws.close.assert_called_once()

    def test_close_without_connect_is_safe(self):
        engine = WebSocketEngine(
            server_url="http://localhost:10300",
            model="tiny",
            language="en",
            on_text=MagicMock(),
        )
        engine.close()  # should not raise

    def test_close_handles_ws_error(self):
        mock_ws = MagicMock()
        mock_ws.close.side_effect = RuntimeError("already closed")
        engine = WebSocketEngine(
            server_url="http://localhost:10300",
            model="tiny",
            language="en",
            on_text=MagicMock(),
        )
        engine._ws = mock_ws
        engine._connected.set()

        engine.close()  # should not raise
        assert engine._ws is None


class TestWebSocketEngineIsAvailable:
    def test_available(self):
        engine = WebSocketEngine(
            server_url="http://localhost:10300",
            model="tiny",
            language="en",
            on_text=MagicMock(),
        )

        with patch("whisper_dictation.engine.websocket.ws_sync") as mock_ws_mod:
            mock_ws_mod.connect.return_value = MagicMock()
            assert engine.is_available() is True

    def test_unavailable(self):
        engine = WebSocketEngine(
            server_url="http://localhost:99999",
            model="tiny",
            language="en",
            on_text=MagicMock(),
        )

        with patch("whisper_dictation.engine.websocket.ws_sync") as mock_ws_mod:
            mock_ws_mod.connect.side_effect = ConnectionRefusedError
            assert engine.is_available() is False


class TestWebSocketEngineHandleMessage:
    def _make_engine(self):
        on_text = MagicMock()
        engine = WebSocketEngine(
            server_url="http://localhost:10300",
            model="tiny",
            language="en",
            on_text=on_text,
        )
        return engine, on_text

    def test_transcription_completed_calls_on_text(self):
        engine, on_text = self._make_engine()
        engine._handle_message(
            "conversation.item.input_audio_transcription.completed",
            {"transcript": "hello world"},
        )
        on_text.assert_called_once_with("hello world")

    def test_transcription_completed_empty_ignored(self):
        engine, on_text = self._make_engine()
        engine._handle_message(
            "conversation.item.input_audio_transcription.completed",
            {"transcript": "  "},
        )
        on_text.assert_not_called()

    def test_delta_message_logged(self):
        engine, on_text = self._make_engine()
        engine._handle_message(
            "conversation.item.input_audio_transcription.delta",
            {"delta": "hel"},
        )
        on_text.assert_not_called()

    def test_error_message_logged(self):
        engine, on_text = self._make_engine()
        engine._handle_message(
            "error",
            {"error": {"message": "model not found"}},
        )
        on_text.assert_not_called()

    def test_session_created(self):
        engine, on_text = self._make_engine()
        engine._handle_message("session.created", {})
        on_text.assert_not_called()

    def test_unknown_message_type(self):
        engine, on_text = self._make_engine()
        engine._handle_message("some.unknown.type", {"data": 123})
        on_text.assert_not_called()


class TestWebSocketEngineSenderLoop:
    def test_sender_sends_queued_messages(self):
        mock_ws = MagicMock()
        engine = WebSocketEngine(
            server_url="http://localhost:10300",
            model="tiny",
            language="en",
            on_text=MagicMock(),
        )
        engine._ws = mock_ws

        # Queue a message then shutdown signal
        engine._send_queue.put('{"type": "test"}')
        engine._send_queue.put(None)

        engine._sender_loop()

        mock_ws.send.assert_called_once_with('{"type": "test"}')

    def test_sender_exits_on_send_error(self):
        mock_ws = MagicMock()
        mock_ws.send.side_effect = RuntimeError("broken pipe")
        engine = WebSocketEngine(
            server_url="http://localhost:10300",
            model="tiny",
            language="en",
            on_text=MagicMock(),
        )
        engine._ws = mock_ws
        engine._send_queue.put('{"type": "test"}')

        engine._sender_loop()  # should not raise


class TestWebSocketEngineReceiverLoop:
    def test_receiver_dispatches_transcription(self):
        on_text = MagicMock()
        mock_ws = MagicMock()

        msg = json.dumps({
            "type": "conversation.item.input_audio_transcription.completed",
            "transcript": "hello",
        })
        # First recv returns message, second raises to exit loop
        mock_ws.recv.side_effect = [msg, RuntimeError("closed")]

        engine = WebSocketEngine(
            server_url="http://localhost:10300",
            model="tiny",
            language="en",
            on_text=on_text,
        )
        engine._ws = mock_ws

        engine._receiver_loop()

        on_text.assert_called_once_with("hello")

    def test_receiver_handles_timeout(self):
        mock_ws = MagicMock()
        call_count = [0]

        def mock_recv(timeout=None):
            call_count[0] += 1
            if call_count[0] == 1:
                raise TimeoutError
            raise RuntimeError("closed")

        mock_ws.recv = mock_recv
        engine = WebSocketEngine(
            server_url="http://localhost:10300",
            model="tiny",
            language="en",
            on_text=MagicMock(),
        )
        engine._ws = mock_ws

        engine._receiver_loop()  # should not raise
        assert call_count[0] == 2

    def test_receiver_handles_invalid_json(self):
        mock_ws = MagicMock()
        mock_ws.recv.side_effect = ["not json", RuntimeError("closed")]

        engine = WebSocketEngine(
            server_url="http://localhost:10300",
            model="tiny",
            language="en",
            on_text=MagicMock(),
        )
        engine._ws = mock_ws

        engine._receiver_loop()  # should not raise

    def test_receiver_exits_on_stop_event(self):
        engine = WebSocketEngine(
            server_url="http://localhost:10300",
            model="tiny",
            language="en",
            on_text=MagicMock(),
        )
        engine._stop_event.set()
        engine._ws = MagicMock()

        engine._receiver_loop()  # should exit immediately
