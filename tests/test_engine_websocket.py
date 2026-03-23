"""Tests for whisper_dictation.engine.websocket — WhisperLive WebSocketEngine."""

from __future__ import annotations

import json
import threading
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from whisper_dictation.engine.websocket import (
    WebSocketEngine,
    _audio_to_float32_bytes,
    _build_config_message,
    _http_to_ws_url,
)

# ---------------------------------------------------------------------------
# URL conversion
# ---------------------------------------------------------------------------


class TestHttpToWsUrl:
    def test_http_to_ws(self):
        assert _http_to_ws_url("http://localhost:9090") == "ws://localhost:9090"

    def test_https_to_wss(self):
        assert _http_to_ws_url("https://api.example.com") == "wss://api.example.com"

    def test_preserves_path(self):
        assert _http_to_ws_url("http://host:8080/api") == "ws://host:8080/api"

    def test_trailing_slash(self):
        assert _http_to_ws_url("http://host:8080/") == "ws://host:8080/"


# ---------------------------------------------------------------------------
# Config message
# ---------------------------------------------------------------------------


class TestBuildConfigMessage:
    def test_default_values(self):
        msg = _build_config_message(uid="test-123", language="en", model="small")
        assert msg["uid"] == "test-123"
        assert msg["language"] == "en"
        assert msg["task"] == "transcribe"
        assert msg["model"] == "small"
        assert msg["use_vad"] is True

    def test_vad_disabled(self):
        msg = _build_config_message(uid="x", language="de", model="large-v3", use_vad=False)
        assert msg["use_vad"] is False
        assert msg["language"] == "de"
        assert msg["model"] == "large-v3"


# ---------------------------------------------------------------------------
# Audio encoding
# ---------------------------------------------------------------------------


class TestAudioToFloat32Bytes:
    def test_float32_passthrough(self):
        audio = np.array([0.5, -0.5], dtype=np.float32)
        data = _audio_to_float32_bytes(audio)
        result = np.frombuffer(data, dtype=np.float32)
        np.testing.assert_array_almost_equal(result, [0.5, -0.5])

    def test_float64_converts(self):
        audio = np.array([0.5, -0.5], dtype=np.float64)
        data = _audio_to_float32_bytes(audio)
        result = np.frombuffer(data, dtype=np.float32)
        np.testing.assert_array_almost_equal(result, [0.5, -0.5])

    def test_int16_converts(self):
        audio = np.array([16384, -16384], dtype=np.int16)
        data = _audio_to_float32_bytes(audio)
        result = np.frombuffer(data, dtype=np.float32)
        np.testing.assert_array_almost_equal(result, [0.5, -0.5], decimal=3)


# ---------------------------------------------------------------------------
# WebSocketEngine init
# ---------------------------------------------------------------------------


class TestWebSocketEngineInit:
    def test_ws_url_construction(self):
        engine = WebSocketEngine(
            server_url="http://localhost:9090",
            model="tiny",
            language="en",
            on_text=MagicMock(),
        )
        assert engine.ws_url == "ws://localhost:9090"

    def test_ws_url_trailing_slash_stripped(self):
        engine = WebSocketEngine(
            server_url="http://host:8080/",
            model="tiny",
            language="en",
            on_text=MagicMock(),
        )
        assert engine.ws_url == "ws://host:8080"


# ---------------------------------------------------------------------------
# Connect
# ---------------------------------------------------------------------------


class TestWebSocketEngineConnect:
    def test_connect_sends_config(self):
        mock_ws = MagicMock()
        engine = WebSocketEngine(
            server_url="http://localhost:9090",
            model="tiny",
            language="en",
            on_text=MagicMock(),
        )

        with patch("whisper_dictation.engine.websocket.ws_sync") as mock_ws_mod:
            mock_ws_mod.connect.return_value = mock_ws
            engine.connect()

        sent = mock_ws.send.call_args[0][0]
        msg = json.loads(sent)
        assert msg["task"] == "transcribe"
        assert msg["model"] == "tiny"
        assert msg["language"] == "en"
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
        engine = WebSocketEngine(
            server_url="http://localhost:9090",
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

        assert mock_ws_mod.connect.call_count == 3

    def test_connect_succeeds_on_retry(self):
        mock_ws = MagicMock()
        engine = WebSocketEngine(
            server_url="http://localhost:9090",
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
        engine.close()


# ---------------------------------------------------------------------------
# Send audio
# ---------------------------------------------------------------------------


class TestWebSocketEngineSendAudio:
    def test_send_audio_queues_bytes(self):
        engine = WebSocketEngine(
            server_url="http://localhost:9090",
            model="tiny",
            language="en",
            on_text=MagicMock(),
        )
        engine._connected.set()

        audio = np.zeros(512, dtype=np.float32)
        engine.send_audio(audio)

        data = engine._send_queue.get_nowait()
        assert isinstance(data, bytes)
        assert len(data) == 512 * 4  # float32 = 4 bytes per sample

    def test_send_audio_drops_when_queue_full(self):
        engine = WebSocketEngine(
            server_url="http://localhost:9090",
            model="tiny",
            language="en",
            on_text=MagicMock(),
        )
        engine._connected.set()

        for _ in range(engine._send_queue.maxsize):
            engine.send_audio(np.zeros(16, dtype=np.float32))

        engine.send_audio(np.zeros(16, dtype=np.float32))
        assert engine._send_queue.full()

    def test_send_audio_when_disconnected_is_noop(self):
        engine = WebSocketEngine(
            server_url="http://localhost:9090",
            model="tiny",
            language="en",
            on_text=MagicMock(),
        )
        engine.send_audio(np.zeros(512, dtype=np.float32))
        assert engine._send_queue.empty()


# ---------------------------------------------------------------------------
# Flush
# ---------------------------------------------------------------------------


class TestWebSocketEngineFlush:
    def test_flush_queues_end_signal(self):
        engine = WebSocketEngine(
            server_url="http://localhost:9090",
            model="tiny",
            language="en",
            on_text=MagicMock(),
        )
        engine._connected.set()

        engine.flush()

        data = engine._send_queue.get_nowait()
        assert data == b"__END__"

    def test_flush_when_disconnected_is_noop(self):
        engine = WebSocketEngine(
            server_url="http://localhost:9090",
            model="tiny",
            language="en",
            on_text=MagicMock(),
        )
        engine.flush()
        assert engine._send_queue.empty()

    def test_flush_when_queue_full(self):
        engine = WebSocketEngine(
            server_url="http://localhost:9090",
            model="tiny",
            language="en",
            on_text=MagicMock(),
        )
        engine._connected.set()
        for _ in range(engine._send_queue.maxsize):
            engine._send_queue.put_nowait(b"filler")
        engine.flush()  # should not raise


# ---------------------------------------------------------------------------
# Close
# ---------------------------------------------------------------------------


class TestWebSocketEngineClose:
    def test_close_cleans_up(self):
        mock_ws = MagicMock()
        engine = WebSocketEngine(
            server_url="http://localhost:9090",
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
            server_url="http://localhost:9090",
            model="tiny",
            language="en",
            on_text=MagicMock(),
        )
        engine.close()

    def test_close_handles_ws_error(self):
        mock_ws = MagicMock()
        mock_ws.close.side_effect = RuntimeError("already closed")
        engine = WebSocketEngine(
            server_url="http://localhost:9090",
            model="tiny",
            language="en",
            on_text=MagicMock(),
        )
        engine._ws = mock_ws
        engine._connected.set()
        engine.close()
        assert engine._ws is None

    def test_close_warns_on_stuck_threads(self):
        engine = WebSocketEngine(
            server_url="http://localhost:9090",
            model="tiny",
            language="en",
            on_text=MagicMock(),
        )
        mock_sender = MagicMock(spec=threading.Thread)
        mock_sender.is_alive.return_value = True
        mock_receiver = MagicMock(spec=threading.Thread)
        mock_receiver.is_alive.return_value = True
        engine._sender_thread = mock_sender
        engine._receiver_thread = mock_receiver
        engine.close()
        mock_sender.join.assert_called_once()
        mock_receiver.join.assert_called_once()

    def test_close_with_full_queue(self):
        engine = WebSocketEngine(
            server_url="http://localhost:9090",
            model="tiny",
            language="en",
            on_text=MagicMock(),
        )
        engine._connected.set()
        for _ in range(engine._send_queue.maxsize):
            engine._send_queue.put_nowait(b"filler")
        engine.close()

    def test_close_drains_queue(self):
        engine = WebSocketEngine(
            server_url="http://localhost:9090",
            model="tiny",
            language="en",
            on_text=MagicMock(),
        )
        engine._send_queue.put(b"item1")
        engine._send_queue.put(b"item2")
        engine.close()
        assert engine._send_queue.empty()

    def test_close_drain_handles_concurrent_empty(self):
        import queue as queue_mod

        engine = WebSocketEngine(
            server_url="http://localhost:9090",
            model="tiny",
            language="en",
            on_text=MagicMock(),
        )
        mock_queue = MagicMock(spec=queue_mod.Queue)
        empty_calls = [False, True]
        mock_queue.empty.side_effect = lambda: empty_calls.pop(0) if empty_calls else True
        mock_queue.get_nowait.side_effect = queue_mod.Empty
        mock_queue.put_nowait = MagicMock()
        engine._send_queue = mock_queue
        engine.close()


# ---------------------------------------------------------------------------
# is_available
# ---------------------------------------------------------------------------


class TestWebSocketEngineIsAvailable:
    def test_available(self):
        engine = WebSocketEngine(
            server_url="http://localhost:9090",
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


# ---------------------------------------------------------------------------
# wait_for_completion
# ---------------------------------------------------------------------------


class TestWebSocketEngineWaitForCompletion:
    def test_returns_true_when_completed(self):
        engine = WebSocketEngine(
            server_url="http://localhost:9090",
            model="tiny",
            language="en",
            on_text=MagicMock(),
        )
        engine._flush_done.set()
        assert engine.wait_for_completion(timeout=0.1) is True

    def test_returns_false_on_timeout(self):
        engine = WebSocketEngine(
            server_url="http://localhost:9090",
            model="tiny",
            language="en",
            on_text=MagicMock(),
        )
        assert engine.wait_for_completion(timeout=0.01) is False


# ---------------------------------------------------------------------------
# _handle_message — WhisperLive protocol
# ---------------------------------------------------------------------------


class TestWebSocketEngineHandleMessage:
    def _make_engine(self):
        on_text = MagicMock()
        engine = WebSocketEngine(
            server_url="http://localhost:9090",
            model="tiny",
            language="en",
            on_text=on_text,
        )
        return engine, on_text

    def test_server_ready(self):
        engine, on_text = self._make_engine()
        engine._handle_message({"message": "SERVER_READY", "backend": "faster_whisper"})
        on_text.assert_not_called()

    def test_wait_message(self):
        engine, on_text = self._make_engine()
        engine._handle_message({"status": "WAIT", "message": 5})
        on_text.assert_not_called()

    def test_language_detection(self):
        engine, on_text = self._make_engine()
        engine._handle_message({"language": "en", "language_prob": 0.95})
        on_text.assert_not_called()

    def test_unknown_message(self):
        engine, on_text = self._make_engine()
        engine._handle_message({"unknown": "field"})
        on_text.assert_not_called()


# ---------------------------------------------------------------------------
# _process_segments
# ---------------------------------------------------------------------------


class TestProcessSegments:
    def _make_engine(self):
        on_text = MagicMock()
        engine = WebSocketEngine(
            server_url="http://localhost:9090",
            model="tiny",
            language="en",
            on_text=on_text,
        )
        return engine, on_text

    def test_completed_segment_emits_text(self):
        engine, on_text = self._make_engine()
        segments = [{"text": "hello world", "completed": True}]
        engine._process_segments(segments)
        on_text.assert_called_once_with("hello world")

    def test_partial_segment_emits_text(self):
        engine, on_text = self._make_engine()
        segments = [{"text": "hello", "completed": False}]
        engine._process_segments(segments)
        on_text.assert_called_once_with("hello")

    def test_incremental_emission(self):
        engine, on_text = self._make_engine()

        # First update
        engine._process_segments([{"text": "hello", "completed": False}])
        on_text.assert_called_with("hello")

        # Second update extends text
        engine._process_segments([{"text": "hello world", "completed": True}])
        assert on_text.call_count == 2
        on_text.assert_called_with("world")

    def test_same_text_not_re_emitted(self):
        engine, on_text = self._make_engine()
        engine._process_segments([{"text": "hello", "completed": False}])
        engine._process_segments([{"text": "hello", "completed": False}])
        assert on_text.call_count == 1  # only first call

    def test_empty_segments_ignored(self):
        engine, on_text = self._make_engine()
        engine._process_segments([])
        on_text.assert_not_called()

    def test_empty_text_ignored(self):
        engine, on_text = self._make_engine()
        engine._process_segments([{"text": "", "completed": True}])
        on_text.assert_not_called()

    def test_multiple_segments_concatenated(self):
        engine, on_text = self._make_engine()
        segments = [
            {"text": "hello", "completed": True},
            {"text": "world", "completed": True},
        ]
        engine._process_segments(segments)
        on_text.assert_called_once_with("hello world")

    def test_completed_sets_flush_done(self):
        engine, _ = self._make_engine()
        engine._flush_done.clear()
        segments = [{"text": "done", "completed": True}]
        engine._process_segments(segments)
        assert engine._flush_done.is_set()

    def test_partial_does_not_set_flush_done(self):
        engine, _ = self._make_engine()
        engine._flush_done.clear()
        segments = [{"text": "partial", "completed": False}]
        engine._process_segments(segments)
        assert not engine._flush_done.is_set()


# ---------------------------------------------------------------------------
# Sender loop
# ---------------------------------------------------------------------------


class TestWebSocketEngineSenderLoop:
    def test_sender_sends_binary_audio(self):
        mock_ws = MagicMock()
        engine = WebSocketEngine(
            server_url="http://localhost:9090",
            model="tiny",
            language="en",
            on_text=MagicMock(),
        )
        engine._ws = mock_ws

        engine._send_queue.put(b"\x00" * 1024)
        engine._send_queue.put(None)  # shutdown

        engine._sender_loop()
        mock_ws.send.assert_called_once_with(b"\x00" * 1024)

    def test_sender_sends_end_of_audio(self):
        mock_ws = MagicMock()
        engine = WebSocketEngine(
            server_url="http://localhost:9090",
            model="tiny",
            language="en",
            on_text=MagicMock(),
        )
        engine._ws = mock_ws

        engine._send_queue.put(b"__END__")
        engine._send_queue.put(None)

        engine._sender_loop()
        mock_ws.send.assert_called_once_with("END_OF_AUDIO")

    def test_sender_exits_on_send_error(self):
        mock_ws = MagicMock()
        mock_ws.send.side_effect = RuntimeError("broken pipe")
        engine = WebSocketEngine(
            server_url="http://localhost:9090",
            model="tiny",
            language="en",
            on_text=MagicMock(),
        )
        engine._ws = mock_ws
        engine._send_queue.put(b"\x00" * 100)
        engine._sender_loop()

    def test_sender_exits_when_ws_none(self):
        engine = WebSocketEngine(
            server_url="http://localhost:9090",
            model="tiny",
            language="en",
            on_text=MagicMock(),
        )
        engine._ws = None
        engine._send_queue.put(b"\x00" * 100)
        engine._sender_loop()

    def test_sender_handles_empty_queue_then_stops(self):
        engine = WebSocketEngine(
            server_url="http://localhost:9090",
            model="tiny",
            language="en",
            on_text=MagicMock(),
        )
        engine._ws = MagicMock()

        def delayed_stop():
            import time
            time.sleep(0.15)
            engine._stop_event.set()

        t = threading.Thread(target=delayed_stop, daemon=True)
        t.start()
        engine._sender_loop()
        t.join(timeout=1.0)


# ---------------------------------------------------------------------------
# Receiver loop
# ---------------------------------------------------------------------------


class TestWebSocketEngineReceiverLoop:
    def test_receiver_dispatches_segments(self):
        on_text = MagicMock()
        mock_ws = MagicMock()

        msg = json.dumps({"segments": [{"text": "hello", "completed": True}]})
        mock_ws.recv.side_effect = [msg, RuntimeError("closed")]

        engine = WebSocketEngine(
            server_url="http://localhost:9090",
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
            server_url="http://localhost:9090",
            model="tiny",
            language="en",
            on_text=MagicMock(),
        )
        engine._ws = mock_ws
        engine._receiver_loop()
        assert call_count[0] == 2

    def test_receiver_handles_invalid_json(self):
        mock_ws = MagicMock()
        mock_ws.recv.side_effect = ["not json", RuntimeError("closed")]

        engine = WebSocketEngine(
            server_url="http://localhost:9090",
            model="tiny",
            language="en",
            on_text=MagicMock(),
        )
        engine._ws = mock_ws
        engine._receiver_loop()

    def test_receiver_ignores_binary_frames(self):
        mock_ws = MagicMock()
        mock_ws.recv.side_effect = [b"\x00\x00", RuntimeError("closed")]

        engine = WebSocketEngine(
            server_url="http://localhost:9090",
            model="tiny",
            language="en",
            on_text=MagicMock(),
        )
        engine._ws = mock_ws
        engine._receiver_loop()

    def test_receiver_exits_on_stop_event(self):
        engine = WebSocketEngine(
            server_url="http://localhost:9090",
            model="tiny",
            language="en",
            on_text=MagicMock(),
        )
        engine._stop_event.set()
        engine._ws = MagicMock()
        engine._receiver_loop()

    def test_receiver_exits_when_ws_none(self):
        engine = WebSocketEngine(
            server_url="http://localhost:9090",
            model="tiny",
            language="en",
            on_text=MagicMock(),
        )
        engine._ws = None
        engine._receiver_loop()
