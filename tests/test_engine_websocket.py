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
    _is_loopback,
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


class TestIsLoopback:
    def test_localhost(self):
        assert _is_loopback("http://localhost:9090") is True

    def test_ipv4_loopback(self):
        assert _is_loopback("http://127.0.0.1:9090") is True

    def test_remote_host(self):
        assert _is_loopback("http://whisper.example.com") is False

    def test_empty_host(self):
        assert _is_loopback("http://") is True


# ---------------------------------------------------------------------------
# Config message
# ---------------------------------------------------------------------------


class TestBuildConfigMessage:
    def test_default_values(self):
        msg = _build_config_message(uid="test-123", language="en", model="small")
        assert msg["uid"] == "test-123"
        assert msg["task"] == "transcribe"
        assert msg["use_vad"] is True

    def test_custom_values(self):
        msg = _build_config_message(uid="x", language="de", model="large-v3", use_vad=False)
        assert msg["use_vad"] is False
        assert msg["model"] == "large-v3"


# ---------------------------------------------------------------------------
# Audio encoding
# ---------------------------------------------------------------------------


class TestAudioToFloat32Bytes:
    def test_float32_passthrough(self):
        audio = np.array([0.5, -0.5], dtype=np.float32)
        result = np.frombuffer(_audio_to_float32_bytes(audio), dtype=np.float32)
        np.testing.assert_array_almost_equal(result, [0.5, -0.5])

    def test_float64_converts(self):
        audio = np.array([0.5, -0.5], dtype=np.float64)
        result = np.frombuffer(_audio_to_float32_bytes(audio), dtype=np.float32)
        np.testing.assert_array_almost_equal(result, [0.5, -0.5])

    def test_int16_converts(self):
        audio = np.array([16384, -16384], dtype=np.int16)
        result = np.frombuffer(_audio_to_float32_bytes(audio), dtype=np.float32)
        np.testing.assert_array_almost_equal(result, [0.5, -0.5], decimal=3)


# ---------------------------------------------------------------------------
# WebSocketEngine init and URL
# ---------------------------------------------------------------------------


class TestWebSocketEngineInit:
    def test_ws_url(self):
        engine = WebSocketEngine(
            server_url="http://localhost:9090", model="tiny",
            language="en", on_text=MagicMock(),
        )
        assert engine.ws_url == "ws://localhost:9090"


# ---------------------------------------------------------------------------
# Connect
# ---------------------------------------------------------------------------


class TestConnect:
    def test_connect_sends_config_and_waits_for_ready(self):
        mock_ws = MagicMock()
        engine = WebSocketEngine(
            server_url="http://localhost:9090", model="tiny",
            language="en", on_text=MagicMock(),
        )
        # Pre-set server_ready so connect doesn't timeout
        engine._server_ready.set()

        with patch("whisper_dictation.engine.websocket.ws_sync") as m:
            m.connect.return_value = mock_ws
            engine.connect()

        sent = json.loads(mock_ws.send.call_args[0][0])
        assert sent["task"] == "transcribe"
        assert sent["model"] == "tiny"
        assert "uid" in sent
        assert engine._connected.is_set()
        engine.close()

    def test_connect_failure_raises(self):
        engine = WebSocketEngine(
            server_url="http://localhost:99999", model="tiny",
            language="en", reconnect_attempts=0, on_text=MagicMock(),
        )
        with patch("whisper_dictation.engine.websocket.ws_sync") as m:
            m.connect.side_effect = ConnectionRefusedError("refused")
            with pytest.raises(ConnectionRefusedError):
                engine.connect()
        assert not engine._connected.is_set()

    def test_connect_retries(self):
        engine = WebSocketEngine(
            server_url="http://localhost:9090", model="tiny",
            language="en", reconnect_attempts=2, reconnect_delay=0.01,
            on_text=MagicMock(),
        )
        with patch("whisper_dictation.engine.websocket.ws_sync") as m:
            m.connect.side_effect = ConnectionRefusedError("refused")
            with pytest.raises(ConnectionRefusedError):
                engine.connect()
        assert m.connect.call_count == 3

    def test_connect_succeeds_on_retry(self):
        mock_ws = MagicMock()
        engine = WebSocketEngine(
            server_url="http://localhost:9090", model="tiny",
            language="en", reconnect_attempts=2, reconnect_delay=0.01,
            on_text=MagicMock(),
        )
        engine._server_ready.set()
        with patch("whisper_dictation.engine.websocket.ws_sync") as m:
            m.connect.side_effect = [ConnectionRefusedError("fail"), mock_ws]
            engine.connect()
        assert engine._connected.is_set()
        engine.close()

    def test_connect_warns_on_non_localhost_ws(self):
        engine = WebSocketEngine(
            server_url="http://remote-host:9090", model="tiny",
            language="en", reconnect_attempts=0, on_text=MagicMock(),
        )
        with patch("whisper_dictation.engine.websocket.ws_sync") as m:
            m.connect.side_effect = ConnectionRefusedError("fail")
            with pytest.raises(ConnectionRefusedError):
                engine.connect()
        # Warning was logged (can't easily assert log content without caplog)


# ---------------------------------------------------------------------------
# Send audio
# ---------------------------------------------------------------------------


class TestSendAudio:
    def test_queues_bytes(self):
        engine = WebSocketEngine(
            server_url="http://localhost:9090", model="tiny",
            language="en", on_text=MagicMock(),
        )
        engine._connected.set()
        engine.send_audio(np.zeros(512, dtype=np.float32))
        data = engine._send_queue.get_nowait()
        assert isinstance(data, bytes)
        assert len(data) == 512 * 4

    def test_drops_when_full(self):
        engine = WebSocketEngine(
            server_url="http://localhost:9090", model="tiny",
            language="en", on_text=MagicMock(),
        )
        engine._connected.set()
        for _ in range(engine._send_queue.maxsize):
            engine.send_audio(np.zeros(16, dtype=np.float32))
        engine.send_audio(np.zeros(16, dtype=np.float32))  # dropped
        assert engine._send_queue.full()

    def test_noop_when_disconnected(self):
        engine = WebSocketEngine(
            server_url="http://localhost:9090", model="tiny",
            language="en", on_text=MagicMock(),
        )
        engine.send_audio(np.zeros(512, dtype=np.float32))
        assert engine._send_queue.empty()


# ---------------------------------------------------------------------------
# Flush
# ---------------------------------------------------------------------------


class TestFlush:
    def test_queues_end_signal(self):
        engine = WebSocketEngine(
            server_url="http://localhost:9090", model="tiny",
            language="en", on_text=MagicMock(),
        )
        engine._connected.set()
        engine.flush()
        assert engine._send_queue.get_nowait() == b"__END__"

    def test_noop_when_disconnected(self):
        engine = WebSocketEngine(
            server_url="http://localhost:9090", model="tiny",
            language="en", on_text=MagicMock(),
        )
        engine.flush()
        assert engine._send_queue.empty()

    def test_warns_when_queue_full(self):
        engine = WebSocketEngine(
            server_url="http://localhost:9090", model="tiny",
            language="en", on_text=MagicMock(),
        )
        engine._connected.set()
        for _ in range(engine._send_queue.maxsize):
            engine._send_queue.put_nowait(b"x")
        engine.flush()  # should not raise


# ---------------------------------------------------------------------------
# Close
# ---------------------------------------------------------------------------


class TestClose:
    def test_close_cleans_up(self):
        mock_ws = MagicMock()
        engine = WebSocketEngine(
            server_url="http://localhost:9090", model="tiny",
            language="en", on_text=MagicMock(),
        )
        engine._server_ready.set()
        with patch("whisper_dictation.engine.websocket.ws_sync") as m:
            m.connect.return_value = mock_ws
            engine.connect()
        engine.close()
        assert engine._ws is None
        mock_ws.close.assert_called_once()

    def test_close_without_connect(self):
        engine = WebSocketEngine(
            server_url="http://localhost:9090", model="tiny",
            language="en", on_text=MagicMock(),
        )
        engine.close()

    def test_close_handles_ws_error(self):
        mock_ws = MagicMock()
        mock_ws.close.side_effect = RuntimeError("already closed")
        engine = WebSocketEngine(
            server_url="http://localhost:9090", model="tiny",
            language="en", on_text=MagicMock(),
        )
        engine._ws = mock_ws
        engine.close()
        assert engine._ws is None

    def test_close_warns_on_stuck_threads(self):
        engine = WebSocketEngine(
            server_url="http://localhost:9090", model="tiny",
            language="en", on_text=MagicMock(),
        )
        mock_t = MagicMock(spec=threading.Thread)
        mock_t.is_alive.return_value = True
        engine._sender_thread = mock_t
        engine._receiver_thread = MagicMock(spec=threading.Thread)
        engine._receiver_thread.is_alive.return_value = True
        engine.close()

    def test_close_drains_queue(self):
        engine = WebSocketEngine(
            server_url="http://localhost:9090", model="tiny",
            language="en", on_text=MagicMock(),
        )
        engine._send_queue.put(b"x")
        engine.close()
        assert engine._send_queue.empty()

    def test_close_drain_handles_race(self):
        import queue as queue_mod
        engine = WebSocketEngine(
            server_url="http://localhost:9090", model="tiny",
            language="en", on_text=MagicMock(),
        )
        mock_q = MagicMock(spec=queue_mod.Queue)
        empty_calls = [False, True]
        mock_q.empty.side_effect = lambda: empty_calls.pop(0) if empty_calls else True
        mock_q.get_nowait.side_effect = queue_mod.Empty
        mock_q.put_nowait = MagicMock()
        engine._send_queue = mock_q
        engine.close()

    def test_close_with_full_queue(self):
        engine = WebSocketEngine(
            server_url="http://localhost:9090", model="tiny",
            language="en", on_text=MagicMock(),
        )
        for _ in range(engine._send_queue.maxsize):
            engine._send_queue.put_nowait(b"x")
        engine.close()


# ---------------------------------------------------------------------------
# is_available
# ---------------------------------------------------------------------------


class TestIsAvailable:
    def test_available(self):
        engine = WebSocketEngine(
            server_url="http://localhost:9090", model="tiny",
            language="en", on_text=MagicMock(),
        )
        with patch("whisper_dictation.engine.websocket.ws_sync") as m:
            m.connect.return_value = MagicMock()
            assert engine.is_available() is True

    def test_unavailable(self):
        engine = WebSocketEngine(
            server_url="http://localhost:99999", model="tiny",
            language="en", on_text=MagicMock(),
        )
        with patch("whisper_dictation.engine.websocket.ws_sync") as m:
            m.connect.side_effect = ConnectionRefusedError
            assert engine.is_available() is False


# ---------------------------------------------------------------------------
# wait_for_completion
# ---------------------------------------------------------------------------


class TestWaitForCompletion:
    def test_returns_true(self):
        engine = WebSocketEngine(
            server_url="http://localhost:9090", model="tiny",
            language="en", on_text=MagicMock(),
        )
        engine._flush_done.set()
        assert engine.wait_for_completion(timeout=0.1) is True

    def test_returns_false_on_timeout(self):
        engine = WebSocketEngine(
            server_url="http://localhost:9090", model="tiny",
            language="en", on_text=MagicMock(),
        )
        assert engine.wait_for_completion(timeout=0.01) is False


# ---------------------------------------------------------------------------
# _handle_message
# ---------------------------------------------------------------------------


class TestHandleMessage:
    def _make(self):
        on_text = MagicMock()
        engine = WebSocketEngine(
            server_url="http://localhost:9090", model="tiny",
            language="en", on_text=on_text,
        )
        return engine, on_text

    def test_server_ready_sets_event(self):
        engine, _ = self._make()
        engine._handle_message({"message": "SERVER_READY", "backend": "faster_whisper"})
        assert engine._server_ready.is_set()

    def test_wait_message(self):
        engine, on_text = self._make()
        engine._handle_message({"status": "WAIT", "message": 5})
        on_text.assert_not_called()

    def test_language_detection(self):
        engine, _ = self._make()
        engine._handle_message({"language": "en", "language_prob": 0.95})

    def test_language_detection_bad_prob(self):
        engine, _ = self._make()
        engine._handle_message({"language": "en", "language_prob": "high"})

    def test_unknown_message(self):
        engine, _ = self._make()
        engine._handle_message({"unknown": "field"})

    def test_non_dict_ignored(self):
        engine, _ = self._make()
        engine._handle_message("not a dict")  # type: ignore

    def test_segments_non_list_ignored(self):
        engine, on_text = self._make()
        engine._handle_message({"segments": "not a list"})
        on_text.assert_not_called()


# ---------------------------------------------------------------------------
# _process_segments — completed-only emission
# ---------------------------------------------------------------------------


class TestProcessSegments:
    def _make(self):
        on_text = MagicMock()
        engine = WebSocketEngine(
            server_url="http://localhost:9090", model="tiny",
            language="en", on_text=on_text,
        )
        return engine, on_text

    def test_completed_segment_emits(self):
        engine, on_text = self._make()
        engine._process_segments([{"text": "hello world", "completed": True}])
        on_text.assert_called_once_with("hello world")

    def test_partial_segment_not_emitted(self):
        engine, on_text = self._make()
        engine._process_segments([{"text": "hello", "completed": False}])
        on_text.assert_not_called()

    def test_incremental_completed(self):
        engine, on_text = self._make()
        # First completed segment
        engine._process_segments([{"text": "hello", "completed": True}])
        on_text.assert_called_with("hello")
        # Second completed segment arrives
        engine._process_segments([
            {"text": "hello", "completed": True},
            {"text": "world", "completed": True},
        ])
        assert on_text.call_count == 2
        on_text.assert_called_with("world")

    def test_correction_does_not_retype(self):
        """Server corrects partial → final: only final typed, not the partial."""
        engine, on_text = self._make()
        # Partial arrives — not typed
        engine._process_segments([{"text": "hello word", "completed": False}])
        on_text.assert_not_called()
        # Correction arrives as completed
        engine._process_segments([{"text": "hello world", "completed": True}])
        on_text.assert_called_once_with("hello world")

    def test_empty_segments(self):
        engine, on_text = self._make()
        engine._process_segments([])
        on_text.assert_not_called()

    def test_empty_text_ignored(self):
        engine, on_text = self._make()
        engine._process_segments([{"text": "", "completed": True}])
        on_text.assert_not_called()

    def test_non_dict_segments_filtered(self):
        engine, on_text = self._make()
        engine._process_segments(["not a dict", {"text": "ok", "completed": True}])
        on_text.assert_called_once_with("ok")

    def test_completed_sets_flush_done(self):
        engine, _ = self._make()
        engine._flush_done.clear()
        engine._process_segments([{"text": "done", "completed": True}])
        assert engine._flush_done.is_set()

    def test_partial_does_not_set_flush_done(self):
        engine, _ = self._make()
        engine._flush_done.clear()
        engine._process_segments([{"text": "partial", "completed": False}])
        assert not engine._flush_done.is_set()

    def test_multiple_completed_emitted_in_order(self):
        engine, on_text = self._make()
        engine._process_segments([
            {"text": "one", "completed": True},
            {"text": "two", "completed": True},
            {"text": "three", "completed": True},
        ])
        assert on_text.call_count == 3
        calls = [c[0][0] for c in on_text.call_args_list]
        assert calls == ["one", "two", "three"]


# ---------------------------------------------------------------------------
# Sender loop
# ---------------------------------------------------------------------------


class TestSenderLoop:
    def test_sends_binary_audio(self):
        mock_ws = MagicMock()
        engine = WebSocketEngine(
            server_url="http://localhost:9090", model="tiny",
            language="en", on_text=MagicMock(),
        )
        engine._ws = mock_ws
        engine._send_queue.put(b"\x00" * 1024)
        engine._send_queue.put(None)
        engine._sender_loop()
        mock_ws.send.assert_called_once_with(b"\x00" * 1024)

    def test_sends_end_of_audio(self):
        mock_ws = MagicMock()
        engine = WebSocketEngine(
            server_url="http://localhost:9090", model="tiny",
            language="en", on_text=MagicMock(),
        )
        engine._ws = mock_ws
        engine._send_queue.put(b"__END__")
        engine._send_queue.put(None)
        engine._sender_loop()
        mock_ws.send.assert_called_once_with("END_OF_AUDIO")

    def test_exits_on_send_error(self):
        mock_ws = MagicMock()
        mock_ws.send.side_effect = RuntimeError("broken")
        engine = WebSocketEngine(
            server_url="http://localhost:9090", model="tiny",
            language="en", on_text=MagicMock(),
        )
        engine._ws = mock_ws
        engine._send_queue.put(b"\x00")
        engine._sender_loop()

    def test_exits_when_ws_none(self):
        engine = WebSocketEngine(
            server_url="http://localhost:9090", model="tiny",
            language="en", on_text=MagicMock(),
        )
        engine._ws = None
        engine._send_queue.put(b"\x00")
        engine._sender_loop()

    def test_handles_empty_queue_then_stops(self):
        engine = WebSocketEngine(
            server_url="http://localhost:9090", model="tiny",
            language="en", on_text=MagicMock(),
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


class TestReceiverLoop:
    def test_dispatches_segments(self):
        on_text = MagicMock()
        mock_ws = MagicMock()
        msg = json.dumps({"segments": [{"text": "hello", "completed": True}]})
        mock_ws.recv.side_effect = [msg, RuntimeError("closed")]

        engine = WebSocketEngine(
            server_url="http://localhost:9090", model="tiny",
            language="en", on_text=on_text,
        )
        engine._ws = mock_ws
        engine._receiver_loop()
        on_text.assert_called_once_with("hello")

    def test_handles_timeout(self):
        mock_ws = MagicMock()
        calls = [0]

        def mock_recv(timeout=None):
            calls[0] += 1
            if calls[0] == 1:
                raise TimeoutError
            raise RuntimeError("closed")

        mock_ws.recv = mock_recv
        engine = WebSocketEngine(
            server_url="http://localhost:9090", model="tiny",
            language="en", on_text=MagicMock(),
        )
        engine._ws = mock_ws
        engine._receiver_loop()
        assert calls[0] == 2

    def test_handles_invalid_json(self):
        mock_ws = MagicMock()
        mock_ws.recv.side_effect = ["not json", RuntimeError("closed")]
        engine = WebSocketEngine(
            server_url="http://localhost:9090", model="tiny",
            language="en", on_text=MagicMock(),
        )
        engine._ws = mock_ws
        engine._receiver_loop()

    def test_ignores_binary_frames(self):
        mock_ws = MagicMock()
        mock_ws.recv.side_effect = [b"\x00", RuntimeError("closed")]
        engine = WebSocketEngine(
            server_url="http://localhost:9090", model="tiny",
            language="en", on_text=MagicMock(),
        )
        engine._ws = mock_ws
        engine._receiver_loop()

    def test_exits_on_stop_event(self):
        engine = WebSocketEngine(
            server_url="http://localhost:9090", model="tiny",
            language="en", on_text=MagicMock(),
        )
        engine._stop_event.set()
        engine._ws = MagicMock()
        engine._receiver_loop()

    def test_exits_when_ws_none(self):
        engine = WebSocketEngine(
            server_url="http://localhost:9090", model="tiny",
            language="en", on_text=MagicMock(),
        )
        engine._ws = None
        engine._receiver_loop()

    def test_handle_message_exception_does_not_kill_loop(self):
        """Bad message should warn and continue, not crash receiver."""
        mock_ws = MagicMock()
        msg_good = json.dumps({"segments": [{"text": "ok", "completed": True}]})
        mock_ws.recv.side_effect = [msg_good, RuntimeError("closed")]

        on_text = MagicMock()
        engine = WebSocketEngine(
            server_url="http://localhost:9090", model="tiny",
            language="en", on_text=on_text,
        )
        engine._ws = mock_ws

        # Monkey-patch _handle_message to raise on first call, then work normally
        original = engine._handle_message
        call_count = [0]

        def raising_handler(msg):
            call_count[0] += 1
            if call_count[0] == 1:
                raise ValueError("simulated handler error")
            return original(msg)

        engine._handle_message = raising_handler
        # Need two good messages — first raises, second processes normally
        msg_good2 = json.dumps({"segments": [{"text": "ok", "completed": True}]})
        mock_ws.recv.side_effect = [msg_good, msg_good2, RuntimeError("closed")]
        engine._receiver_loop()
        on_text.assert_called_once_with("ok")
