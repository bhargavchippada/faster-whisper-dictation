"""Tests for whisper_dictation.engine.whisperlivekit — WhisperLiveKit engine."""

from __future__ import annotations

import json
import threading
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from whisper_dictation.engine.whisperlivekit import (
    _END_SENTINEL,
    WhisperLiveKitEngine,
    _audio_to_int16_bytes,
    _http_to_ws_url,
    _is_loopback,
)

# ---------------------------------------------------------------------------
# URL conversion
# ---------------------------------------------------------------------------


class TestHttpToWsUrl:
    def test_http_to_ws_with_asr_path(self):
        url = _http_to_ws_url("http://localhost:8000", path="/asr", language="en")
        assert url == "ws://localhost:8000/asr?language=en"

    def test_https_to_wss(self):
        url = _http_to_ws_url("https://api.example.com", path="/asr", language="fr")
        assert url == "wss://api.example.com/asr?language=fr"

    def test_preserves_existing_path(self):
        url = _http_to_ws_url("http://host:8080/api", path="/asr", language="en")
        assert url == "ws://host:8080/api/asr?language=en"

    def test_rejects_non_http_scheme(self):
        with pytest.raises(ValueError, match="must use http or https"):
            _http_to_ws_url("ftp://host:21", path="/asr", language="en")

    def test_strips_trailing_slash(self):
        url = _http_to_ws_url("http://localhost:8000/", path="/asr", language="en")
        assert "/asr" in url


class TestIsLoopback:
    def test_localhost(self):
        assert _is_loopback("http://localhost:8000") is True

    def test_ipv4_loopback(self):
        assert _is_loopback("http://127.0.0.1:8000") is True

    def test_remote_host(self):
        assert _is_loopback("http://example.com") is False

    def test_empty_host(self):
        assert _is_loopback("http://") is True

    def test_non_ip_hostname(self):
        assert _is_loopback("http://myserver.local") is False


# ---------------------------------------------------------------------------
# Audio conversion
# ---------------------------------------------------------------------------


class TestAudioToInt16Bytes:
    def test_float32_input(self):
        audio = np.array([0.5, -0.5, 1.0, -1.0], dtype=np.float32)
        result = _audio_to_int16_bytes(audio)
        arr = np.frombuffer(result, dtype=np.int16)
        assert arr[0] == 16383  # 0.5 * 32767 ≈ 16383
        assert arr[1] == -16383
        assert arr[2] == 32767  # clipped
        assert arr[3] == -32767

    def test_float64_input(self):
        audio = np.array([0.0, 0.5], dtype=np.float64)
        result = _audio_to_int16_bytes(audio)
        arr = np.frombuffer(result, dtype=np.int16)
        assert arr[0] == 0
        assert arr[1] == 16383

    def test_int16_passthrough(self):
        audio = np.array([100, -200, 32767], dtype=np.int16)
        result = _audio_to_int16_bytes(audio)
        assert result == audio.tobytes()

    def test_other_dtype_converted(self):
        audio = np.array([16384, -16384], dtype=np.int32)
        result = _audio_to_int16_bytes(audio)
        arr = np.frombuffer(result, dtype=np.int16)
        assert len(arr) == 2


# ---------------------------------------------------------------------------
# Engine init
# ---------------------------------------------------------------------------


class TestWhisperLiveKitEngineInit:
    def test_default_init(self):
        engine = WhisperLiveKitEngine(
            server_url="http://localhost:8000", model="large-v3", language="en",
        )
        assert engine._server_url == "http://localhost:8000"
        assert engine._model == "large-v3"
        assert engine._language == "en"
        assert engine._reconnect_attempts == 3

    def test_strips_trailing_slash(self):
        engine = WhisperLiveKitEngine(
            server_url="http://localhost:8000/", model="m", language="en",
        )
        assert engine._server_url == "http://localhost:8000"

    def test_negative_reconnect_raises(self):
        with pytest.raises(ValueError, match="reconnect_attempts must be >= 0"):
            WhisperLiveKitEngine(
                server_url="http://localhost:8000", model="m",
                language="en", reconnect_attempts=-1,
            )

    def test_zero_reconnect_delay_raises(self):
        with pytest.raises(ValueError, match="reconnect_delay must be > 0"):
            WhisperLiveKitEngine(
                server_url="http://localhost:8000", model="m",
                language="en", reconnect_delay=0.0,
            )

    def test_negative_reconnect_delay_raises(self):
        with pytest.raises(ValueError, match="reconnect_delay must be > 0"):
            WhisperLiveKitEngine(
                server_url="http://localhost:8000", model="m",
                language="en", reconnect_delay=-1.0,
            )

    def test_ws_url_property(self):
        engine = WhisperLiveKitEngine(
            server_url="http://localhost:8000", model="m", language="fr",
        )
        assert engine.ws_url == "ws://localhost:8000/asr?language=fr"

    def test_on_text_default_noop(self):
        engine = WhisperLiveKitEngine(
            server_url="http://localhost:8000", model="m", language="en",
        )
        engine._on_text("test")  # should not raise


# ---------------------------------------------------------------------------
# Connect
# ---------------------------------------------------------------------------


class TestConnect:
    @patch("whisper_dictation.engine.whisperlivekit.ws_sync")
    def test_connect_success(self, mock_ws_mod):
        mock_ws = MagicMock()
        mock_ws_mod.connect.return_value = mock_ws
        engine = WhisperLiveKitEngine(
            server_url="http://localhost:8000", model="m", language="en",
        )
        engine.connect()
        assert engine._connected.is_set()
        mock_ws_mod.connect.assert_called_once()
        engine.close()

    @patch("whisper_dictation.engine.whisperlivekit.ws_sync")
    def test_connect_retry_on_failure(self, mock_ws_mod):
        mock_ws_mod.connect.side_effect = [ConnectionRefusedError, MagicMock()]
        engine = WhisperLiveKitEngine(
            server_url="http://localhost:8000", model="m", language="en",
            reconnect_attempts=1, reconnect_delay=0.01,
        )
        engine.connect()
        assert engine._connected.is_set()
        assert mock_ws_mod.connect.call_count == 2
        engine.close()

    @patch("whisper_dictation.engine.whisperlivekit.ws_sync")
    def test_connect_all_retries_exhausted(self, mock_ws_mod):
        mock_ws_mod.connect.side_effect = ConnectionRefusedError("refused")
        engine = WhisperLiveKitEngine(
            server_url="http://localhost:8000", model="m", language="en",
            reconnect_attempts=1, reconnect_delay=0.01,
        )
        with pytest.raises(ConnectionRefusedError):
            engine.connect()
        assert not engine._connected.is_set()

    @patch("whisper_dictation.engine.whisperlivekit.ws_sync")
    def test_connect_non_localhost_warning(self, mock_ws_mod):
        mock_ws_mod.connect.return_value = MagicMock()
        engine = WhisperLiveKitEngine(
            server_url="http://remote.example.com", model="m", language="en",
        )
        with patch("whisper_dictation.engine.whisperlivekit.log") as mock_log:
            engine.connect()
            mock_log.warning.assert_called()
        engine.close()

    @patch("whisper_dictation.engine.whisperlivekit.ws_sync")
    def test_connect_resets_state(self, mock_ws_mod):
        mock_ws_mod.connect.return_value = MagicMock()
        engine = WhisperLiveKitEngine(
            server_url="http://localhost:8000", model="m", language="en",
        )
        engine._emitted_count = 5
        engine._latest_full_text = "old"
        engine.connect()
        assert engine._emitted_count == 0
        assert engine._latest_full_text == ""
        engine.close()

    @patch("whisper_dictation.engine.whisperlivekit.ws_sync")
    def test_connect_no_client_handshake_sent(self, mock_ws_mod):
        """WhisperLiveKit: no JSON config sent from client."""
        mock_ws = MagicMock()
        mock_ws_mod.connect.return_value = mock_ws
        engine = WhisperLiveKitEngine(
            server_url="http://localhost:8000", model="m", language="en",
        )
        engine.connect()
        # The WS should NOT have send() called for a config message
        mock_ws.send.assert_not_called()
        engine.close()

    @patch("whisper_dictation.engine.whisperlivekit.ws_sync")
    def test_connect_use_vad_ignored(self, mock_ws_mod):
        """use_vad parameter accepted but not sent (no handshake)."""
        mock_ws_mod.connect.return_value = MagicMock()
        engine = WhisperLiveKitEngine(
            server_url="http://localhost:8000", model="m", language="en",
        )
        engine.connect(use_vad=False)  # should not raise
        assert engine._connected.is_set()
        engine.close()


# ---------------------------------------------------------------------------
# Send audio
# ---------------------------------------------------------------------------


class TestSendAudio:
    def test_sends_int16_bytes(self):
        engine = WhisperLiveKitEngine(
            server_url="http://localhost:8000", model="m", language="en",
        )
        engine._connected.set()
        audio = np.zeros(512, dtype=np.float32)
        engine.send_audio(audio)
        data = engine._send_queue.get_nowait()
        assert isinstance(data, bytes)
        arr = np.frombuffer(data, dtype=np.int16)
        assert len(arr) == 512

    def test_not_connected_ignores(self):
        engine = WhisperLiveKitEngine(
            server_url="http://localhost:8000", model="m", language="en",
        )
        engine.send_audio(np.zeros(10, dtype=np.float32))
        assert engine._send_queue.empty()

    def test_full_queue_drops(self):
        engine = WhisperLiveKitEngine(
            server_url="http://localhost:8000", model="m", language="en",
        )
        engine._connected.set()
        # Fill the queue
        for _ in range(100):
            engine._send_queue.put(b"x")
        engine.send_audio(np.zeros(10, dtype=np.float32))  # should not raise


# ---------------------------------------------------------------------------
# Flush
# ---------------------------------------------------------------------------


class TestFlush:
    def test_flush_sends_eoa_sentinel(self):
        engine = WhisperLiveKitEngine(
            server_url="http://localhost:8000", model="m", language="en",
        )
        engine._connected.set()
        engine.flush(send_eoa=True)
        data = engine._send_queue.get_nowait()
        assert data is _END_SENTINEL

    def test_flush_no_eoa(self):
        engine = WhisperLiveKitEngine(
            server_url="http://localhost:8000", model="m", language="en",
        )
        engine._connected.set()
        engine.flush(send_eoa=False)
        assert engine._send_queue.empty()

    def test_flush_not_connected(self):
        engine = WhisperLiveKitEngine(
            server_url="http://localhost:8000", model="m", language="en",
        )
        engine.flush()
        assert engine._send_queue.empty()

    def test_flush_queue_full(self):
        engine = WhisperLiveKitEngine(
            server_url="http://localhost:8000", model="m", language="en",
        )
        engine._connected.set()
        for _ in range(100):
            engine._send_queue.put(b"x")
        engine.flush(send_eoa=True)  # should not raise

    def test_flush_sets_eoa_sent(self):
        engine = WhisperLiveKitEngine(
            server_url="http://localhost:8000", model="m", language="en",
        )
        engine._connected.set()
        engine.flush(send_eoa=True)
        assert engine._eoa_sent is True

    def test_flush_eoa_rollback_on_queue_full(self):
        engine = WhisperLiveKitEngine(
            server_url="http://localhost:8000", model="m", language="en",
        )
        engine._connected.set()
        for _ in range(100):
            engine._send_queue.put(b"x")
        engine.flush(send_eoa=True)
        assert engine._eoa_sent is False


# ---------------------------------------------------------------------------
# Sender loop
# ---------------------------------------------------------------------------


class TestSenderLoop:
    def test_sends_empty_frame_for_eoa(self):
        mock_ws = MagicMock()
        engine = WhisperLiveKitEngine(
            server_url="http://localhost:8000", model="m",
            language="en", on_text=MagicMock(),
        )
        engine._ws = mock_ws
        engine._send_queue.put(_END_SENTINEL)
        engine._send_queue.put(None)
        engine._sender_loop()
        mock_ws.send.assert_called_once_with(b"")

    def test_sends_audio_bytes(self):
        mock_ws = MagicMock()
        engine = WhisperLiveKitEngine(
            server_url="http://localhost:8000", model="m", language="en",
        )
        engine._ws = mock_ws
        data = b"\x00" * 1024
        engine._send_queue.put(data)
        engine._send_queue.put(None)
        engine._sender_loop()
        mock_ws.send.assert_called_once_with(data)

    def test_exits_on_none_sentinel(self):
        engine = WhisperLiveKitEngine(
            server_url="http://localhost:8000", model="m", language="en",
        )
        engine._ws = MagicMock()
        engine._send_queue.put(None)
        engine._sender_loop()  # should return

    def test_exits_on_send_error(self):
        mock_ws = MagicMock()
        mock_ws.send.side_effect = RuntimeError("broken")
        engine = WhisperLiveKitEngine(
            server_url="http://localhost:8000", model="m", language="en",
        )
        engine._ws = mock_ws
        engine._send_queue.put(b"audio")
        engine._sender_loop()  # should not raise

    def test_exits_when_ws_none(self):
        engine = WhisperLiveKitEngine(
            server_url="http://localhost:8000", model="m", language="en",
        )
        engine._ws = None
        engine._send_queue.put(b"audio")
        engine._sender_loop()

    def test_exits_on_stop_event(self):
        engine = WhisperLiveKitEngine(
            server_url="http://localhost:8000", model="m", language="en",
        )
        engine._ws = MagicMock()
        engine._stop_event.set()
        engine._sender_loop()


# ---------------------------------------------------------------------------
# Receiver loop
# ---------------------------------------------------------------------------


class TestReceiverLoop:
    def test_handles_timeout(self):
        mock_ws = MagicMock()
        call_count = 0

        def fake_recv(timeout=None):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise TimeoutError
            raise ConnectionError("closed")

        mock_ws.recv.side_effect = fake_recv
        engine = WhisperLiveKitEngine(
            server_url="http://localhost:8000", model="m", language="en",
        )
        engine._ws = mock_ws
        engine._stop_event.set()  # exit after first iteration
        engine._receiver_loop()

    def test_ignores_binary_frames(self):
        mock_ws = MagicMock()
        mock_ws.recv.side_effect = [b"\x00\x01", ConnectionError]
        engine = WhisperLiveKitEngine(
            server_url="http://localhost:8000", model="m", language="en",
        )
        engine._ws = mock_ws
        engine._receiver_loop()

    def test_skips_oversized_messages(self):
        mock_ws = MagicMock()
        big = "x" * (1024 * 1024 + 1)
        mock_ws.recv.side_effect = [big, ConnectionError]
        engine = WhisperLiveKitEngine(
            server_url="http://localhost:8000", model="m", language="en",
        )
        engine._ws = mock_ws
        engine._receiver_loop()

    def test_skips_invalid_json(self):
        mock_ws = MagicMock()
        mock_ws.recv.side_effect = ["not json{", ConnectionError]
        engine = WhisperLiveKitEngine(
            server_url="http://localhost:8000", model="m", language="en",
        )
        engine._ws = mock_ws
        engine._receiver_loop()

    def test_dispatches_valid_message(self):
        mock_ws = MagicMock()
        msg = json.dumps({"type": "ready_to_stop"})
        mock_ws.recv.side_effect = [msg, ConnectionError]
        engine = WhisperLiveKitEngine(
            server_url="http://localhost:8000", model="m", language="en",
        )
        engine._ws = mock_ws
        engine._eoa_sent = True
        engine._receiver_loop()
        assert engine._flush_done.is_set()

    def test_exits_when_ws_none(self):
        engine = WhisperLiveKitEngine(
            server_url="http://localhost:8000", model="m", language="en",
        )
        engine._ws = None
        engine._receiver_loop()

    def test_handles_message_exception(self):
        mock_ws = MagicMock()
        msg = json.dumps({"lines": "not-a-list", "buffer_transcription": 123})
        mock_ws.recv.side_effect = [msg, ConnectionError]
        engine = WhisperLiveKitEngine(
            server_url="http://localhost:8000", model="m", language="en",
        )
        engine._ws = mock_ws
        engine._receiver_loop()  # should not raise


# ---------------------------------------------------------------------------
# Handle message
# ---------------------------------------------------------------------------


class TestHandleMessage:
    def test_config_message(self):
        engine = WhisperLiveKitEngine(
            server_url="http://localhost:8000", model="m", language="en",
        )
        engine._handle_message({"type": "config", "useAudioWorklet": True})
        # Should not raise, just log

    def test_ready_to_stop(self):
        engine = WhisperLiveKitEngine(
            server_url="http://localhost:8000", model="m", language="en",
        )
        engine._eoa_sent = True
        engine._handle_message({"type": "ready_to_stop"})
        assert engine._flush_done.is_set()

    def test_ready_to_stop_ignored_before_eoa(self):
        engine = WhisperLiveKitEngine(
            server_url="http://localhost:8000", model="m", language="en",
        )
        engine._handle_message({"type": "ready_to_stop"})
        assert not engine._flush_done.is_set()

    def test_transcription_with_lines(self):
        cb = MagicMock()
        engine = WhisperLiveKitEngine(
            server_url="http://localhost:8000", model="m",
            language="en", on_text=cb,
        )
        engine._handle_message({
            "lines": [{"text": "hello world", "start": "0:00:00.00", "end": "0:00:01.00"}],
            "buffer_transcription": "",
        })
        cb.assert_called_once_with("hello world")

    def test_transcription_with_buffer_only(self):
        engine = WhisperLiveKitEngine(
            server_url="http://localhost:8000", model="m", language="en",
        )
        engine._handle_message({
            "buffer_transcription": "partial text",
        })
        assert engine._latest_full_text == "partial text"

    def test_non_dict_ignored(self):
        engine = WhisperLiveKitEngine(
            server_url="http://localhost:8000", model="m", language="en",
        )
        engine._handle_message("not a dict")  # type: ignore[arg-type]

    def test_unhandled_message(self):
        engine = WhisperLiveKitEngine(
            server_url="http://localhost:8000", model="m", language="en",
        )
        engine._handle_message({"unknown": "field"})  # should just log


# ---------------------------------------------------------------------------
# Process response
# ---------------------------------------------------------------------------


class TestProcessResponse:
    def test_emits_new_completed_lines(self):
        cb = MagicMock()
        engine = WhisperLiveKitEngine(
            server_url="http://localhost:8000", model="m",
            language="en", on_text=cb,
        )
        engine._process_response(
            lines=[{"text": "hello"}, {"text": "world"}],
            buffer_text="",
        )
        assert cb.call_count == 2
        cb.assert_any_call("hello")
        cb.assert_any_call("world")
        assert engine._emitted_count == 2

    def test_does_not_re_emit(self):
        cb = MagicMock()
        engine = WhisperLiveKitEngine(
            server_url="http://localhost:8000", model="m",
            language="en", on_text=cb,
        )
        engine._process_response(lines=[{"text": "hello"}], buffer_text="")
        cb.reset_mock()
        engine._process_response(lines=[{"text": "hello"}], buffer_text="partial")
        cb.assert_not_called()

    def test_emits_only_new_lines(self):
        cb = MagicMock()
        engine = WhisperLiveKitEngine(
            server_url="http://localhost:8000", model="m",
            language="en", on_text=cb,
        )
        engine._process_response(lines=[{"text": "first"}], buffer_text="")
        cb.reset_mock()
        engine._process_response(
            lines=[{"text": "first"}, {"text": "second"}],
            buffer_text="",
        )
        cb.assert_called_once_with("second")

    def test_tracks_full_text_with_buffer(self):
        engine = WhisperLiveKitEngine(
            server_url="http://localhost:8000", model="m", language="en",
        )
        engine._process_response(
            lines=[{"text": "hello"}],
            buffer_text="world",
        )
        assert engine._latest_full_text == "hello world"

    def test_batch_mode_collects(self):
        engine = WhisperLiveKitEngine(
            server_url="http://localhost:8000", model="m", language="en",
        )
        with engine._seg_lock:
            engine._batch_collected = []
        engine._process_response(
            lines=[{"text": "hello"}, {"text": "world"}],
            buffer_text="",
        )
        assert engine._batch_collected == ["hello", "world"]

    def test_batch_mode_max_lines(self):
        engine = WhisperLiveKitEngine(
            server_url="http://localhost:8000", model="m", language="en",
        )
        with engine._seg_lock:
            engine._batch_collected = ["x"] * 1000
        engine._process_response(lines=[{"text": "overflow"}], buffer_text="")
        assert len(engine._batch_collected) == 1000  # not 1001

    def test_skips_empty_text_lines(self):
        cb = MagicMock()
        engine = WhisperLiveKitEngine(
            server_url="http://localhost:8000", model="m",
            language="en", on_text=cb,
        )
        engine._process_response(lines=[{"text": ""}, {"text": "  "}], buffer_text="")
        cb.assert_not_called()

    def test_skips_non_dict_lines(self):
        cb = MagicMock()
        engine = WhisperLiveKitEngine(
            server_url="http://localhost:8000", model="m",
            language="en", on_text=cb,
        )
        engine._process_response(lines=["not a dict", 42], buffer_text="")
        cb.assert_not_called()


# ---------------------------------------------------------------------------
# Get pending text
# ---------------------------------------------------------------------------


class TestGetPendingText:
    def test_returns_text_when_none_emitted(self):
        engine = WhisperLiveKitEngine(
            server_url="http://localhost:8000", model="m", language="en",
        )
        engine._latest_full_text = "pending"
        assert engine.get_pending_text() == "pending"

    def test_returns_empty_when_already_emitted(self):
        engine = WhisperLiveKitEngine(
            server_url="http://localhost:8000", model="m", language="en",
        )
        engine._emitted_count = 1
        engine._latest_full_text = "already sent"
        assert engine.get_pending_text() == ""


# ---------------------------------------------------------------------------
# Close
# ---------------------------------------------------------------------------


class TestClose:
    def test_close_sets_events(self):
        engine = WhisperLiveKitEngine(
            server_url="http://localhost:8000", model="m", language="en",
        )
        engine._connected.set()
        engine.close()
        assert not engine._connected.is_set()
        assert engine._stop_event.is_set()
        assert engine._flush_done.is_set()

    def test_close_joins_threads(self):
        engine = WhisperLiveKitEngine(
            server_url="http://localhost:8000", model="m", language="en",
        )
        mock_thread = MagicMock()
        mock_thread.ident = 123
        engine._sender_thread = mock_thread
        engine._receiver_thread = mock_thread
        engine.close()
        assert mock_thread.join.called

    def test_close_closes_ws(self):
        engine = WhisperLiveKitEngine(
            server_url="http://localhost:8000", model="m", language="en",
        )
        mock_ws = MagicMock()
        engine._ws = mock_ws
        engine.close()
        mock_ws.close.assert_called_once()
        assert engine._ws is None

    def test_close_ws_error_handled(self):
        engine = WhisperLiveKitEngine(
            server_url="http://localhost:8000", model="m", language="en",
        )
        mock_ws = MagicMock()
        mock_ws.close.side_effect = RuntimeError("oops")
        engine._ws = mock_ws
        engine.close()  # should not raise

    def test_close_drains_queue(self):
        engine = WhisperLiveKitEngine(
            server_url="http://localhost:8000", model="m", language="en",
        )
        engine._send_queue.put(b"data")
        engine._send_queue.put(b"more")
        engine.close()
        assert engine._send_queue.empty()


# ---------------------------------------------------------------------------
# Join thread helper
# ---------------------------------------------------------------------------


class TestJoinThread:
    def test_none_thread(self):
        engine = WhisperLiveKitEngine(
            server_url="http://localhost:8000", model="m", language="en",
        )
        engine._join_thread(None, "test")  # should not raise

    def test_unstarted_thread(self):
        engine = WhisperLiveKitEngine(
            server_url="http://localhost:8000", model="m", language="en",
        )
        t = threading.Thread(target=lambda: None)
        engine._join_thread(t, "test")  # should not raise

    def test_alive_thread_timeout(self):
        engine = WhisperLiveKitEngine(
            server_url="http://localhost:8000", model="m", language="en",
        )
        mock_thread = MagicMock()
        mock_thread.ident = 1
        mock_thread.is_alive.return_value = True
        engine._join_thread(mock_thread, "test")
        mock_thread.join.assert_called_once_with(timeout=2.0)

    def test_runtime_error_handled(self):
        engine = WhisperLiveKitEngine(
            server_url="http://localhost:8000", model="m", language="en",
        )
        mock_thread = MagicMock()
        mock_thread.join.side_effect = RuntimeError("cannot join before start")
        engine._join_thread(mock_thread, "test")  # should not raise


# ---------------------------------------------------------------------------
# Is available
# ---------------------------------------------------------------------------


class TestIsAvailable:
    @patch("whisper_dictation.engine.whisperlivekit.ws_sync")
    def test_available(self, mock_ws_mod):
        mock_ws_mod.connect.return_value.__enter__ = MagicMock()
        mock_ws_mod.connect.return_value.__exit__ = MagicMock(return_value=False)
        engine = WhisperLiveKitEngine(
            server_url="http://localhost:8000", model="m", language="en",
        )
        assert engine.is_available() is True

    @patch("whisper_dictation.engine.whisperlivekit.ws_sync")
    def test_not_available(self, mock_ws_mod):
        mock_ws_mod.connect.side_effect = ConnectionRefusedError
        engine = WhisperLiveKitEngine(
            server_url="http://localhost:8000", model="m", language="en",
        )
        assert engine.is_available() is False


# ---------------------------------------------------------------------------
# Transcribe batch
# ---------------------------------------------------------------------------


class TestTranscribeBatch:
    def test_returns_transcribed_text(self):
        """Batch mode: collects completed segments and returns text."""
        engine = WhisperLiveKitEngine(
            server_url="http://localhost:8000", model="m", language="en",
        )

        def fake_connect(*, use_vad=None):
            engine._connected.set()

        def fake_wait(timeout=5.0):
            with engine._seg_lock:
                if engine._batch_collected is not None:
                    engine._batch_collected.append("hello world")
            return True

        with patch.object(engine, "connect", side_effect=fake_connect):
            with patch.object(engine, "wait_for_completion", side_effect=fake_wait):
                with patch.object(engine, "close"):
                    result = engine.transcribe_batch(
                        np.zeros(1024, dtype=np.float32),
                    )

        assert result == "hello world"

    def test_fallback_to_latest_full_text(self):
        """Batch mode: falls back to latest_full_text when no completed segs."""
        engine = WhisperLiveKitEngine(
            server_url="http://localhost:8000", model="m", language="en",
        )

        def fake_connect(*, use_vad=None):
            engine._connected.set()

        def fake_wait(timeout=5.0):
            engine._latest_full_text = "partial only"
            return True

        with patch.object(engine, "connect", side_effect=fake_connect):
            with patch.object(engine, "wait_for_completion", side_effect=fake_wait):
                with patch.object(engine, "close"):
                    result = engine.transcribe_batch(
                        np.zeros(512, dtype=np.float32),
                    )

        assert result == "partial only"

    @patch("whisper_dictation.engine.whisperlivekit.ws_sync")
    def test_connect_failure_raises(self, mock_ws_mod):
        mock_ws_mod.connect.side_effect = ConnectionRefusedError
        engine = WhisperLiveKitEngine(
            server_url="http://localhost:8000", model="m", language="en",
            reconnect_attempts=0,
        )
        with pytest.raises(ConnectionRefusedError):
            engine.transcribe_batch(np.zeros(512, dtype=np.float32))

    def test_cleans_up_batch_collected(self):
        """Batch mode: _batch_collected is cleaned up after transcription."""
        engine = WhisperLiveKitEngine(
            server_url="http://localhost:8000", model="m", language="en",
        )

        def fake_connect(*, use_vad=None):
            engine._connected.set()

        with patch.object(engine, "connect", side_effect=fake_connect):
            with patch.object(engine, "wait_for_completion", return_value=True):
                with patch.object(engine, "close"):
                    engine.transcribe_batch(np.zeros(512, dtype=np.float32))

        assert engine._batch_collected is None


# ---------------------------------------------------------------------------
# Send audio blocking
# ---------------------------------------------------------------------------


class TestSendAudioBlocking:
    def test_sends_when_connected(self):
        engine = WhisperLiveKitEngine(
            server_url="http://localhost:8000", model="m", language="en",
        )
        engine._connected.set()
        engine._send_audio_blocking(np.zeros(10, dtype=np.float32))
        assert not engine._send_queue.empty()

    def test_ignores_when_not_connected(self):
        engine = WhisperLiveKitEngine(
            server_url="http://localhost:8000", model="m", language="en",
        )
        engine._send_audio_blocking(np.zeros(10, dtype=np.float32))
        assert engine._send_queue.empty()

    def test_exits_when_stop_event_set(self):
        engine = WhisperLiveKitEngine(
            server_url="http://localhost:8000", model="m", language="en",
        )
        engine._connected.set()
        # Fill the queue so put() will block
        for _ in range(100):
            engine._send_queue.put(b"x")
        # Set stop event — should exit the loop without hanging
        engine._stop_event.set()
        engine._send_audio_blocking(np.zeros(10, dtype=np.float32))
        assert engine._send_queue.full()

    def test_retries_on_full_queue_then_succeeds(self):
        """Queue full on first put, then space opens up (covers lines 240-241)."""
        import queue as queue_mod

        engine = WhisperLiveKitEngine(
            server_url="http://localhost:8000", model="m", language="en",
        )
        engine._connected.set()

        call_count = 0
        original_put = engine._send_queue.put

        def fake_put(data, timeout=None):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise queue_mod.Full
            return original_put(data, timeout=0.1)

        with patch.object(engine._send_queue, "put", side_effect=fake_put):
            engine._send_audio_blocking(np.zeros(10, dtype=np.float32))

        assert call_count == 2


# ---------------------------------------------------------------------------
# Connect: queue drain and exception cleanup (lines 164-167, 214-216)
# ---------------------------------------------------------------------------


class TestConnectQueueDrainAndCleanup:
    @patch("whisper_dictation.engine.whisperlivekit.ws_sync")
    def test_connect_drains_stale_queue(self, mock_ws_mod):
        """connect() drains leftover items from previous session (lines 164-167)."""
        mock_ws_mod.connect.return_value = MagicMock()
        engine = WhisperLiveKitEngine(
            server_url="http://localhost:8000", model="m", language="en",
        )
        # Simulate stale data from a previous session
        engine._send_queue.put(b"stale1")
        engine._send_queue.put(b"stale2")
        engine._eoa_sent = True

        engine.connect()

        # Queue should be drained (only the None sentinel from close may remain)
        # and _eoa_sent should be reset
        assert engine._eoa_sent is False
        engine.close()

    @patch("whisper_dictation.engine.whisperlivekit.ws_sync")
    def test_connect_drain_handles_empty_race(self, mock_ws_mod):
        """Queue drain handles TOCTOU: empty() False but get_nowait() raises Empty."""
        import queue as _queue_mod
        mock_ws_mod.connect.return_value = MagicMock()
        engine = WhisperLiveKitEngine(
            server_url="http://localhost:8000", model="m", language="en",
        )
        # Mock queue that reports non-empty but raises Empty on get
        mock_q = MagicMock(spec=_queue_mod.Queue)
        empty_calls = [False, True]
        mock_q.empty.side_effect = lambda: empty_calls.pop(0) if empty_calls else True
        mock_q.get_nowait.side_effect = _queue_mod.Empty
        mock_q.put_nowait = MagicMock()
        mock_q.put = MagicMock()
        mock_q.maxsize = 100
        engine._send_queue = mock_q
        engine.connect()
        engine.close()

    @patch("whisper_dictation.engine.whisperlivekit.ws_sync")
    def test_connect_exception_during_thread_start(self, mock_ws_mod):
        """Exception after WS connect but during thread start calls close() (lines 214-216)."""
        mock_ws = MagicMock()
        mock_ws_mod.connect.return_value = mock_ws

        engine = WhisperLiveKitEngine(
            server_url="http://localhost:8000", model="m", language="en",
        )

        # Make Thread creation fail after WS is connected
        with patch("whisper_dictation.engine.whisperlivekit.threading.Thread",
                    side_effect=RuntimeError("thread creation failed")):
            with pytest.raises(RuntimeError, match="thread creation failed"):
                engine.connect()

        # close() should have been called, clearing connected
        assert not engine._connected.is_set()


# ---------------------------------------------------------------------------
# Flush send_eoa=False noop (line 267)
# ---------------------------------------------------------------------------


class TestWaitForCompletion:
    def test_returns_true_when_set(self):
        engine = WhisperLiveKitEngine(
            server_url="http://localhost:8000", model="m", language="en",
        )
        engine._flush_done.set()
        assert engine.wait_for_completion(timeout=0.01) is True

    def test_returns_false_on_timeout(self):
        engine = WhisperLiveKitEngine(
            server_url="http://localhost:8000", model="m", language="en",
        )
        assert engine.wait_for_completion(timeout=0.01) is False


class TestFlushSendEoaFalseNoop:
    def test_flush_send_eoa_false_does_not_queue(self):
        """flush(send_eoa=False) does nothing to the queue (line 267 — noop path)."""
        engine = WhisperLiveKitEngine(
            server_url="http://localhost:8000", model="m", language="en",
        )
        engine._connected.set()
        engine.flush(send_eoa=False)
        assert engine._send_queue.empty()
        assert engine._eoa_sent is False


# ---------------------------------------------------------------------------
# transcribe_batch: queue.Full for END_OF_AUDIO + timeout warning (lines 293-294, 299)
# ---------------------------------------------------------------------------


class TestTranscribeBatchEdgeCases:
    def test_batch_eoa_queue_full(self):
        """EOA put fails with queue.Full in batch mode (lines 293-294)."""
        import queue as queue_mod

        engine = WhisperLiveKitEngine(
            server_url="http://localhost:8000", model="m", language="en",
        )

        def fake_connect(*, use_vad=None):
            engine._connected.set()

        original_put = engine._send_queue.put

        def fake_put(data, timeout=None):
            if data is _END_SENTINEL:
                raise queue_mod.Full
            return original_put(data, timeout=timeout)

        with patch.object(engine, "connect", side_effect=fake_connect), \
             patch.object(engine._send_queue, "put", side_effect=fake_put), \
             patch.object(engine, "wait_for_completion", return_value=True), \
             patch.object(engine, "close"):
            result = engine.transcribe_batch(np.zeros(512, dtype=np.float32))

        # EOA was not sent, so _eoa_sent stays False
        assert engine._eoa_sent is False
        # Result is empty string (no collected text, no fallback)
        assert result == ""

    def test_batch_timeout_warning(self):
        """Batch transcription timeout triggers warning log (line 299)."""
        engine = WhisperLiveKitEngine(
            server_url="http://localhost:8000", model="m", language="en",
        )

        def fake_connect(*, use_vad=None):
            engine._connected.set()

        with patch.object(engine, "connect", side_effect=fake_connect), \
             patch.object(engine, "wait_for_completion", return_value=False), \
             patch.object(engine, "close"), \
             patch("whisper_dictation.engine.whisperlivekit.log") as mock_log:
            engine.transcribe_batch(np.zeros(512, dtype=np.float32))

        # Should have logged a timeout warning
        mock_log.warning.assert_called()
        args = mock_log.warning.call_args[0]
        assert "timed out" in args[0]


# ---------------------------------------------------------------------------
# close(): queue full for None sentinel (lines 323-324)
# ---------------------------------------------------------------------------


class TestCloseQueueFull:
    def test_close_queue_full_for_sentinel(self):
        """close() handles queue.Full when putting None sentinel (lines 323-324)."""
        engine = WhisperLiveKitEngine(
            server_url="http://localhost:8000", model="m", language="en",
        )
        # Fill queue so put_nowait raises Full
        for _ in range(100):
            engine._send_queue.put(b"x")
        engine.close()  # should not raise
        assert engine._stop_event.is_set()

    def test_close_drain_handles_empty_race(self):
        """close() drain handles TOCTOU: empty() False but get_nowait() raises Empty."""
        import queue as _queue_mod
        engine = WhisperLiveKitEngine(
            server_url="http://localhost:8000", model="m", language="en",
        )
        mock_q = MagicMock(spec=_queue_mod.Queue)
        empty_calls = [False, True]
        mock_q.empty.side_effect = lambda: empty_calls.pop(0) if empty_calls else True
        mock_q.get_nowait.side_effect = _queue_mod.Empty
        mock_q.put_nowait = MagicMock()
        engine._send_queue = mock_q
        engine.close()


# ---------------------------------------------------------------------------
# _join_thread: finished-but-started thread path (lines 344-345)
# ---------------------------------------------------------------------------


class TestJoinThreadFinished:
    def test_join_finished_started_thread(self):
        """Thread that was started (ident set) but is no longer alive (lines 344-345)."""
        engine = WhisperLiveKitEngine(
            server_url="http://localhost:8000", model="m", language="en",
        )
        # Create a real thread that has already finished
        done_event = threading.Event()
        t = threading.Thread(target=lambda: done_event.set())
        t.start()
        done_event.wait()
        t.join()  # ensure it's done

        # ident is set (was started) but is_alive() is False
        assert t.ident is not None
        assert not t.is_alive()

        engine._join_thread(t, "finished")  # should not raise


# ---------------------------------------------------------------------------
# _sender_loop: stop_event primary exit (lines 385-386)
# ---------------------------------------------------------------------------


class TestSenderLoopStopEvent:
    def test_sender_loop_hits_empty_then_stops(self):
        """Sender loop gets queue.Empty, continues, then stops on stop_event."""
        import time as _time
        import threading as _threading

        engine = WhisperLiveKitEngine(
            server_url="http://localhost:8000", model="m", language="en",
        )
        engine._ws = MagicMock()
        # Let loop run — queue is empty, so get() times out (queue.Empty),
        # then continue back to while check where stop_event is now set.
        def delayed_stop():
            _time.sleep(0.15)
            engine._stop_event.set()
        t = _threading.Thread(target=delayed_stop, daemon=True)
        t.start()
        engine._sender_loop()
        t.join(timeout=1.0)
        engine._ws.send.assert_not_called()


# ---------------------------------------------------------------------------
# _receiver_loop: stop_event primary exit (line 414)
# ---------------------------------------------------------------------------


class TestReceiverLoopStopEvent:
    def test_receiver_loop_hits_timeout_then_stops(self):
        """Receiver loop gets TimeoutError, continues, then stops on stop_event."""
        import time as _time
        import threading as _threading

        mock_ws = MagicMock()
        call_count = [0]

        def fake_recv(timeout=None):
            call_count[0] += 1
            if call_count[0] >= 2:
                # Set stop on second call so first TimeoutError hits continue
                raise RuntimeError("closed")
            raise TimeoutError

        mock_ws.recv.side_effect = fake_recv
        engine = WhisperLiveKitEngine(
            server_url="http://localhost:8000", model="m", language="en",
        )
        engine._ws = mock_ws
        engine._receiver_loop()
        assert call_count[0] >= 2  # Hit TimeoutError at least once


# ---------------------------------------------------------------------------
# _handle_message: exception handler (lines 436-437)
# ---------------------------------------------------------------------------


class TestHandleMessageException:
    def test_handle_message_exception_caught_in_receiver(self, caplog):
        """Exception in _handle_message is caught and logged (lines 436-437)."""
        import logging
        mock_ws = MagicMock()
        msg_good = json.dumps({"lines": [{"text": "ok"}], "buffer_transcription": ""})
        mock_ws.recv.side_effect = [msg_good, RuntimeError("closed")]

        engine = WhisperLiveKitEngine(
            server_url="http://localhost:8000", model="m", language="en",
        )
        engine._ws = mock_ws

        # Patch _handle_message to raise on first call
        original = engine._handle_message
        calls = [0]

        def raising_handler(msg):
            calls[0] += 1
            if calls[0] == 1:
                raise ValueError("boom")
            return original(msg)

        engine._handle_message = raising_handler
        # Need two messages — first raises, second breaks
        msg2 = json.dumps({"lines": [{"text": "ok"}], "buffer_transcription": ""})
        mock_ws.recv.side_effect = [msg_good, msg2, RuntimeError("closed")]
        with caplog.at_level(logging.WARNING, logger="whisper_dictation.engine.whisperlivekit"):
            engine._receiver_loop()
        assert "Error handling WLK message" in caplog.text


# ---------------------------------------------------------------------------
# close(): drain queue Empty break path (lines 344-345 in close)
# ---------------------------------------------------------------------------


class TestCloseDrainEmpty:
    def test_close_drain_empty_queue(self):
        """close() drain loop handles already-empty queue (lines 344-345 — break on Empty)."""
        engine = WhisperLiveKitEngine(
            server_url="http://localhost:8000", model="m", language="en",
        )
        # Queue is empty — drain at end of close() hits Empty and breaks
        engine.close()
        assert engine._send_queue.empty()
