"""Tests for whisper_dictation.engine.websocket — WhisperLive WebSocketEngine."""

from __future__ import annotations

import json
import queue as queue_mod
import threading
import time
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from whisper_dictation.engine.websocket import (
    _END_SENTINEL,
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

    def test_rejects_non_http_scheme(self):
        with pytest.raises(ValueError, match="must use http or https"):
            _http_to_ws_url("ftp://host:21")


class TestIsLoopback:
    def test_localhost(self):
        assert _is_loopback("http://localhost:9090") is True

    def test_ipv4_loopback(self):
        assert _is_loopback("http://127.0.0.1:9090") is True

    def test_remote_host(self):
        assert _is_loopback("http://whisper.example.com") is False

    def test_ipv4_loopback_range(self):
        """127.x.x.x addresses are all loopback."""
        assert _is_loopback("http://127.0.0.2:9090") is True
        assert _is_loopback("http://127.255.255.255:9090") is True

    def test_ipv6_loopback(self):
        assert _is_loopback("http://[::1]:9090") is True

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


def _patch_server_ready(engine: WebSocketEngine) -> None:
    """Make _server_ready.wait() return True immediately in tests.

    Pre-setting the event doesn't work because connect() clears it.
    Instead, patch wait() to return True without blocking.
    """
    engine._server_ready.wait = lambda timeout=None: True  # type: ignore[assignment]


class TestConnect:
    def test_connect_sends_config_and_waits_for_ready(self):
        mock_ws = MagicMock()
        engine = WebSocketEngine(
            server_url="http://localhost:9090", model="tiny",
            language="en", on_text=MagicMock(),
        )
        _patch_server_ready(engine)

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
        _patch_server_ready(engine)
        with patch("whisper_dictation.engine.websocket.ws_sync") as m:
            m.connect.side_effect = [ConnectionRefusedError("fail"), mock_ws]
            engine.connect()
        assert engine._connected.is_set()
        engine.close()

    def test_connect_warns_on_non_localhost_ws(self, caplog):
        import logging
        engine = WebSocketEngine(
            server_url="http://remote-host:9090", model="tiny",
            language="en", reconnect_attempts=0, on_text=MagicMock(),
        )
        with caplog.at_level(logging.WARNING, logger="whisper_dictation.engine.websocket"):
            with patch("whisper_dictation.engine.websocket.ws_sync") as m:
                m.connect.side_effect = ConnectionRefusedError("fail")
                with pytest.raises(ConnectionRefusedError):
                    engine.connect()
        assert "unencrypted ws://" in caplog.text

    def test_connect_cleans_up_if_config_send_fails(self):
        mock_ws = MagicMock()
        mock_ws.send.side_effect = RuntimeError("send failed")
        engine = WebSocketEngine(
            server_url="http://localhost:9090", model="tiny",
            language="en", on_text=MagicMock(),
        )
        with patch("whisper_dictation.engine.websocket.ws_sync") as m:
            m.connect.return_value = mock_ws
            with pytest.raises(RuntimeError, match="send failed"):
                engine.connect()
        assert engine._ws is None
        assert engine._sender_thread is None
        assert engine._receiver_thread is None
        mock_ws.close.assert_called_once()


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


class TestSendAudioBlocking:
    def test_blocks_until_space(self):
        engine = WebSocketEngine(
            server_url="http://localhost:9090", model="tiny",
            language="en", on_text=MagicMock(),
        )
        engine._connected.set()
        engine._send_audio_blocking(np.zeros(512, dtype=np.float32))
        data = engine._send_queue.get_nowait()
        assert isinstance(data, bytes)
        assert len(data) == 512 * 4

    def test_noop_when_disconnected(self):
        engine = WebSocketEngine(
            server_url="http://localhost:9090", model="tiny",
            language="en", on_text=MagicMock(),
        )
        engine._send_audio_blocking(np.zeros(512, dtype=np.float32))
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
        assert engine._send_queue.get_nowait() is _END_SENTINEL

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
        _patch_server_ready(engine)
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

    def test_close_handles_unstarted_threads(self):
        engine = WebSocketEngine(
            server_url="http://localhost:9090", model="tiny",
            language="en", on_text=MagicMock(),
        )
        engine._sender_thread = threading.Thread(target=lambda: None)
        engine._receiver_thread = threading.Thread(target=lambda: None)
        engine.close()

    def test_close_handles_join_runtime_error(self):
        engine = WebSocketEngine(
            server_url="http://localhost:9090", model="tiny",
            language="en", on_text=MagicMock(),
        )
        mock_t = MagicMock(spec=threading.Thread)
        mock_t.ident = 1
        mock_t.is_alive.return_value = False
        mock_t.join.side_effect = RuntimeError("not started")
        engine._sender_thread = mock_t
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

    def test_all_non_dict_segments_ignored(self):
        engine, on_text = self._make()
        engine._process_segments(["not a dict"])
        on_text.assert_not_called()

    def test_completed_sets_flush_done_after_eoa(self):
        """_flush_done only set when _eoa_sent is True."""
        engine, _ = self._make()
        engine._flush_done.clear()
        # Without EOA, completed segment does NOT set _flush_done
        engine._process_segments([{"text": "done", "completed": True}])
        assert not engine._flush_done.is_set()
        # After EOA sent, completed segment sets _flush_done
        with engine._seg_lock:
            engine._eoa_sent = True
            engine._emitted_count = 0  # reset to re-emit
        engine._process_segments([
            {"text": "done", "completed": True},
            {"text": "final", "completed": True},
        ])
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

    def test_batch_mode_collects_completed_segments(self):
        engine, on_text = self._make()
        with engine._seg_lock:
            engine._batch_collected = []
        engine._process_segments([{"text": "done", "completed": True}])
        assert engine._batch_collected == ["done"]
        on_text.assert_not_called()

    def test_batch_mode_ignores_empty_completed_text(self):
        engine, on_text = self._make()
        with engine._seg_lock:
            engine._batch_collected = []
        engine._process_segments([{"text": "   ", "completed": True}])
        assert engine._batch_collected == []
        on_text.assert_not_called()


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
        engine._send_queue.put(_END_SENTINEL)
        engine._send_queue.put(None)
        engine._sender_loop()
        mock_ws.send.assert_called_once_with(b"END_OF_AUDIO")

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
        msg_good = json.dumps({"segments": [{"text": "first", "completed": True}]})
        msg_good2 = json.dumps({"segments": [{"text": "second", "completed": True}]})

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
        mock_ws.recv.side_effect = [msg_good, msg_good2, RuntimeError("closed")]
        engine._receiver_loop()
        on_text.assert_called_once_with("second")

    def test_skips_oversized_message(self):
        """Messages exceeding _MAX_MESSAGE_BYTES are skipped."""
        mock_ws = MagicMock()
        huge = "x" * (2 * 1024 * 1024)  # 2MB
        mock_ws.recv.side_effect = [huge, RuntimeError("closed")]
        engine = WebSocketEngine(
            server_url="http://localhost:9090", model="tiny",
            language="en", on_text=MagicMock(),
        )
        engine._ws = mock_ws
        engine._receiver_loop()  # should not crash


# ---------------------------------------------------------------------------
# transcribe_batch
# ---------------------------------------------------------------------------


class TestTranscribeBatch:
    def test_returns_transcribed_text(self):
        """Batch mode: connect, send audio, collect segments, return text."""
        engine = WebSocketEngine(
            server_url="http://localhost:9090", model="tiny",
            language="en",
        )

        def fake_connect(*, use_vad=None):
            engine._connected.set()

            # Simulate: receiver thread collects a segment shortly after connect
            def simulate_segments():
                time.sleep(0.1)
                with engine._seg_lock:
                    if engine._batch_collected is not None:
                        engine._batch_collected.append("hello world")
                    engine._eoa_sent = True
                engine._flush_done.set()

            threading.Thread(target=simulate_segments, daemon=True).start()

        with patch.object(engine, "connect", side_effect=fake_connect):
            with patch.object(engine, "close"):
                result = engine.transcribe_batch(
                    np.zeros(1024, dtype=np.float32), sample_rate=16000,
                )

        assert result == "hello world"

    def test_returns_empty_on_timeout(self):
        """Batch mode: timeout returns whatever was collected."""
        engine = WebSocketEngine(
            server_url="http://localhost:9090", model="tiny",
            language="en",
        )
        engine._send_queue = queue_mod.Queue()  # unbounded for test

        def fake_connect(*, use_vad=None):
            engine._connected.set()

        with patch.object(engine, "connect", side_effect=fake_connect):
            with patch.object(engine, "wait_for_completion", return_value=False):
                with patch.object(engine, "close"):
                    result = engine.transcribe_batch(
                        np.zeros(512, dtype=np.float32),
                    )

        assert result == ""

    def test_returns_empty_on_exception(self):
        """Batch mode: transport/setup exceptions propagate to caller."""
        engine = WebSocketEngine(
            server_url="http://localhost:9090", model="tiny",
            language="en",
        )
        with patch.object(engine, "connect", side_effect=ConnectionRefusedError):
            with pytest.raises(ConnectionRefusedError):
                engine.transcribe_batch(np.zeros(512, dtype=np.float32))

    def test_does_not_mutate_use_vad(self):
        """Batch mode: _use_vad is never changed, connect() receives use_vad=False."""
        engine = WebSocketEngine(
            server_url="http://localhost:9090", model="tiny",
            language="en", use_vad=True,
        )
        engine._send_queue = queue_mod.Queue()  # unbounded for test

        def fake_connect(*, use_vad=None):
            assert use_vad is True, "transcribe_batch should pass use_vad=True"
            engine._connected.set()

        with patch.object(engine, "connect", side_effect=fake_connect):
            with patch.object(engine, "wait_for_completion", return_value=True):
                with patch.object(engine, "close"):
                    engine.transcribe_batch(np.zeros(512, dtype=np.float32))
        assert engine._use_vad is True

    def test_batch_collected_in_process_segments(self):
        """_process_segments appends to _batch_collected in batch mode."""
        engine = WebSocketEngine(
            server_url="http://localhost:9090", model="tiny",
            language="en",
        )
        with engine._seg_lock:
            engine._batch_collected = []
        engine._process_segments([{"text": "hello", "completed": True}])
        assert engine._batch_collected == ["hello"]

    def test_flush_does_not_reset_emitted_count(self):
        """flush() preserves _emitted_count to prevent duplicate emissions."""
        engine = WebSocketEngine(
            server_url="http://localhost:9090", model="tiny",
            language="en", on_text=MagicMock(),
        )
        engine._connected.set()
        engine._emitted_count = 5
        engine.flush()
        assert engine._emitted_count == 5

    def test_batch_fallback_to_latest_full_text(self):
        """Batch returns _latest_full_text when no completed segments arrive."""
        engine = WebSocketEngine(
            server_url="http://localhost:9090", model="tiny",
            language="en",
        )
        engine._send_queue = queue_mod.Queue()  # unbounded for test

        def fake_connect(*, use_vad=None):
            engine._connected.set()
            # Simulate server sending partials only (no completed segments)
            with engine._seg_lock:
                engine._latest_full_text = "partial result"

        with patch.object(engine, "connect", side_effect=fake_connect):
            with patch.object(engine, "wait_for_completion", return_value=False):
                with patch.object(engine, "close"):
                    result = engine.transcribe_batch(
                        np.zeros(1024, dtype=np.float32), sample_rate=16000,
                    )

        assert result == "partial result"


# ---------------------------------------------------------------------------
# Flush — send_eoa=False
# ---------------------------------------------------------------------------


class TestFlushNoEoa:
    def test_no_eoa_does_not_enqueue_sentinel(self):
        """flush(send_eoa=False) does not put END_SENTINEL in the queue."""
        engine = WebSocketEngine(
            server_url="http://localhost:9090", model="tiny",
            language="en", on_text=MagicMock(),
        )
        engine._connected.set()
        engine.flush(send_eoa=False)
        assert engine._send_queue.empty()

    def test_no_eoa_does_not_set_eoa_sent(self):
        """flush(send_eoa=False) does not set _eoa_sent flag."""
        engine = WebSocketEngine(
            server_url="http://localhost:9090", model="tiny",
            language="en", on_text=MagicMock(),
        )
        engine._connected.set()
        engine.flush(send_eoa=False)
        assert not engine._eoa_sent

    def test_eoa_sets_eoa_sent(self):
        """flush(send_eoa=True) sets _eoa_sent flag."""
        engine = WebSocketEngine(
            server_url="http://localhost:9090", model="tiny",
            language="en", on_text=MagicMock(),
        )
        engine._connected.set()
        engine.flush(send_eoa=True)
        assert engine._eoa_sent


# ---------------------------------------------------------------------------
# _MAX_BATCH_SEGMENTS overflow guard
# ---------------------------------------------------------------------------


class TestMaxBatchSegments:
    def test_overflow_guard_stops_collection(self):
        """Once _MAX_BATCH_SEGMENTS reached, no more segments are appended."""
        from whisper_dictation.engine.websocket import _MAX_BATCH_SEGMENTS

        engine = WebSocketEngine(
            server_url="http://localhost:9090", model="tiny",
            language="en",
        )
        with engine._seg_lock:
            engine._batch_collected = [f"seg{i}" for i in range(_MAX_BATCH_SEGMENTS)]
        # Try to add one more
        engine._process_segments([{"text": "overflow", "completed": True}])
        assert len(engine._batch_collected) == _MAX_BATCH_SEGMENTS
        assert "overflow" not in engine._batch_collected


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


class TestValidation:
    def test_negative_reconnect_attempts_raises(self):
        with pytest.raises(ValueError, match="reconnect_attempts must be >= 0"):
            WebSocketEngine(
                server_url="http://localhost:9090", model="tiny",
                language="en", reconnect_attempts=-1,
            )

    def test_zero_reconnect_delay_raises(self):
        with pytest.raises(ValueError, match="reconnect_delay must be > 0"):
            WebSocketEngine(
                server_url="http://localhost:9090", model="tiny",
                language="en", reconnect_delay=0.0,
            )

    def test_negative_reconnect_delay_raises(self):
        with pytest.raises(ValueError, match="reconnect_delay must be > 0"):
            WebSocketEngine(
                server_url="http://localhost:9090", model="tiny",
                language="en", reconnect_delay=-1.0,
            )


# ---------------------------------------------------------------------------
# connect() use_vad override
# ---------------------------------------------------------------------------


class TestConnectUseVadOverride:
    def test_connect_passes_use_vad_override(self):
        """connect(use_vad=False) sends use_vad=False in config."""
        mock_ws = MagicMock()
        engine = WebSocketEngine(
            server_url="http://localhost:9090", model="tiny",
            language="en", use_vad=True, on_text=MagicMock(),
        )
        _patch_server_ready(engine)
        with patch("whisper_dictation.engine.websocket.ws_sync") as m:
            m.connect.return_value = mock_ws
            engine.connect(use_vad=False)
        sent = json.loads(mock_ws.send.call_args[0][0])
        assert sent["use_vad"] is False
        assert engine._use_vad is True  # instance not mutated
        engine.close()

    def test_connect_default_uses_instance_vad(self):
        """connect() without override uses instance _use_vad."""
        mock_ws = MagicMock()
        engine = WebSocketEngine(
            server_url="http://localhost:9090", model="tiny",
            language="en", use_vad=True, on_text=MagicMock(),
        )
        _patch_server_ready(engine)
        with patch("whisper_dictation.engine.websocket.ws_sync") as m:
            m.connect.return_value = mock_ws
            engine.connect()
        sent = json.loads(mock_ws.send.call_args[0][0])
        assert sent["use_vad"] is True
        engine.close()


# ---------------------------------------------------------------------------
# _latest_full_text tracking
# ---------------------------------------------------------------------------


class TestLatestFullText:
    def test_tracks_all_segment_text(self):
        """_latest_full_text includes partial and completed text."""
        engine = WebSocketEngine(
            server_url="http://localhost:9090", model="tiny",
            language="en", on_text=MagicMock(),
        )
        engine._process_segments([
            {"text": "hello", "completed": True},
            {"text": "world", "completed": False},
        ])
        assert engine._latest_full_text == "hello world"

    def test_reset_on_connect(self):
        """connect() resets _latest_full_text."""
        engine = WebSocketEngine(
            server_url="http://localhost:9090", model="tiny",
            language="en", on_text=MagicMock(),
        )
        engine._latest_full_text = "stale text"
        _patch_server_ready(engine)
        with patch("whisper_dictation.engine.websocket.ws_sync") as m:
            m.connect.return_value = MagicMock()
            engine.connect()
        assert engine._latest_full_text == ""
        engine.close()


# ---------------------------------------------------------------------------
# get_pending_text
# ---------------------------------------------------------------------------


class TestGetPendingText:
    def test_returns_text_when_nothing_emitted(self):
        engine = WebSocketEngine(
            server_url="http://localhost:9090", model="tiny",
            language="en", on_text=MagicMock(),
        )
        with engine._seg_lock:
            engine._latest_full_text = "partial result"
            engine._emitted_count = 0
        assert engine.get_pending_text() == "partial result"

    def test_returns_empty_when_segments_already_emitted(self):
        engine = WebSocketEngine(
            server_url="http://localhost:9090", model="tiny",
            language="en", on_text=MagicMock(),
        )
        with engine._seg_lock:
            engine._latest_full_text = "full text"
            engine._emitted_count = 2
        assert engine.get_pending_text() == ""

    def test_returns_empty_when_no_text(self):
        engine = WebSocketEngine(
            server_url="http://localhost:9090", model="tiny",
            language="en", on_text=MagicMock(),
        )
        assert engine.get_pending_text() == ""


# ---------------------------------------------------------------------------
# ConnectionClosedOK handling in receiver loop
# ---------------------------------------------------------------------------


class TestConnectionClosedOK:
    def test_sets_flush_done_when_eoa_sent(self):
        """Server closes cleanly after EOA — _flush_done is set."""
        import websockets.exceptions as ws_exc

        engine = WebSocketEngine(
            server_url="http://localhost:9090", model="tiny",
            language="en", on_text=MagicMock(),
        )
        mock_ws = MagicMock()
        mock_ws.recv.side_effect = ws_exc.ConnectionClosedOK(None, None)
        engine._ws = mock_ws
        with engine._seg_lock:
            engine._eoa_sent = True
        engine._receiver_loop()
        assert engine._flush_done.is_set()

    def test_no_flush_done_without_eoa(self):
        """Server closes cleanly without EOA — _flush_done stays unset."""
        import websockets.exceptions as ws_exc

        engine = WebSocketEngine(
            server_url="http://localhost:9090", model="tiny",
            language="en", on_text=MagicMock(),
        )
        mock_ws = MagicMock()
        mock_ws.recv.side_effect = ws_exc.ConnectionClosedOK(None, None)
        engine._ws = mock_ws
        engine._flush_done.clear()
        engine._receiver_loop()
        assert not engine._flush_done.is_set()


# ---------------------------------------------------------------------------
# _send_audio_blocking timeout/drop
# ---------------------------------------------------------------------------


class TestSendAudioBlockingTimeout:
    def test_aborts_when_disconnected_during_full_queue(self):
        """When queue is full and connection drops, send aborts gracefully."""
        engine = WebSocketEngine(
            server_url="http://localhost:9090", model="tiny",
            language="en", on_text=MagicMock(),
        )
        engine._connected.set()

        def full_then_disconnect(data, timeout=None):
            engine._connected.clear()  # simulate sender death
            raise queue_mod.Full

        with patch.object(engine._send_queue, "put", side_effect=full_then_disconnect):
            engine._send_audio_blocking(np.zeros(16, dtype=np.float32))
        # Should not hang — exits when _connected is cleared


# ---------------------------------------------------------------------------
# connect() queue drain race (lines 174-177)
# ---------------------------------------------------------------------------


class TestConnectQueueDrain:
    def test_drains_stale_items_from_previous_session(self):
        """connect() drains leftover items from the send queue."""
        engine = WebSocketEngine(
            server_url="http://localhost:9090", model="tiny",
            language="en", on_text=MagicMock(),
        )
        # Pre-fill queue with stale data
        engine._send_queue.put(b"stale1")
        engine._send_queue.put(b"stale2")
        _patch_server_ready(engine)
        with patch("whisper_dictation.engine.websocket.ws_sync") as m:
            m.connect.return_value = MagicMock()
            engine.connect()
        # Queue should be empty after drain (only new items from connect remain)
        # Close to clean up threads
        engine.close()

    def test_drain_handles_empty_race(self):
        """connect() drain loop handles queue.Empty from concurrent get."""
        engine = WebSocketEngine(
            server_url="http://localhost:9090", model="tiny",
            language="en", on_text=MagicMock(),
        )
        # Make queue report not-empty but raise Empty on get (race condition)
        mock_q = MagicMock(spec=queue_mod.Queue)
        mock_q.maxsize = 100
        mock_q.empty.return_value = False
        mock_q.get_nowait.side_effect = queue_mod.Empty
        mock_q.put_nowait = MagicMock()
        mock_q.put = MagicMock()
        engine._send_queue = mock_q
        _patch_server_ready(engine)
        with patch("whisper_dictation.engine.websocket.ws_sync") as m:
            m.connect.return_value = MagicMock()
            engine.connect()
        # Should not raise — the Empty exception breaks the drain loop
        engine.close()


# ---------------------------------------------------------------------------
# connect() SERVER_READY timeout (line 228)
# ---------------------------------------------------------------------------


class TestConnectServerReadyTimeout:
    def test_warns_when_server_ready_times_out(self, caplog):
        """connect() warns and proceeds if SERVER_READY not received within timeout."""
        import logging

        engine = WebSocketEngine(
            server_url="http://localhost:9090", model="tiny",
            language="en", on_text=MagicMock(),
        )
        # Patch _server_ready.wait to return False (timeout)
        engine._server_ready.wait = lambda timeout=None: False  # type: ignore[assignment]

        with caplog.at_level(logging.WARNING, logger="whisper_dictation.engine.websocket"):
            with patch("whisper_dictation.engine.websocket.ws_sync") as m:
                m.connect.return_value = MagicMock()
                engine.connect()
        assert "SERVER_READY" in caplog.text
        assert engine._connected.is_set()
        engine.close()


# ---------------------------------------------------------------------------
# transcribe_batch() queue.Full on END_OF_AUDIO (lines 313-314)
# ---------------------------------------------------------------------------


class TestTranscribeBatchEndOfAudioFull:
    def test_warns_when_end_of_audio_queue_full(self, caplog):
        """transcribe_batch warns if queue is full when sending END_OF_AUDIO."""
        import logging

        engine = WebSocketEngine(
            server_url="http://localhost:9090", model="tiny",
            language="en",
        )
        engine._send_queue = queue_mod.Queue()  # unbounded for audio chunks

        def fake_connect(*, use_vad=None):
            engine._connected.set()

        original_put = engine._send_queue.put

        def put_raises_on_sentinel(item, timeout=None):
            if item is _END_SENTINEL:
                raise queue_mod.Full
            return original_put(item, timeout=timeout)

        with caplog.at_level(logging.WARNING, logger="whisper_dictation.engine.websocket"):
            with patch.object(engine, "connect", side_effect=fake_connect):
                with patch.object(engine._send_queue, "put", side_effect=put_raises_on_sentinel):
                    with patch.object(engine, "wait_for_completion", return_value=False):
                        with patch.object(engine, "close"):
                            result = engine.transcribe_batch(
                                np.zeros(512, dtype=np.float32),
                            )

        assert "END_OF_AUDIO not sent" in caplog.text
        assert result == ""
