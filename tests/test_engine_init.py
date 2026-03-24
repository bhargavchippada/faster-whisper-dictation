"""Tests for whisper_dictation.engine.__init__ — create_ws_engine factory."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from whisper_dictation.config import Config, WebSocketConfig


class TestCreateWsEngine:
    def test_unknown_backend_raises(self):
        """create_ws_engine raises ValueError for unknown backend."""
        from whisper_dictation.engine import create_ws_engine

        cfg = Config(websocket=WebSocketConfig(backend="nonexistent"))
        with pytest.raises(ValueError, match="Unknown websocket.backend.*'nonexistent'"):
            create_ws_engine(
                cfg,
                server_url="http://localhost:9090",
                model="tiny",
                language="en",
                reconnect_attempts=0,
                reconnect_delay=1.0,
            )

    def test_whisperlive_backend(self):
        """create_ws_engine returns WebSocketEngine for 'whisperlive' backend."""
        from whisper_dictation.engine import create_ws_engine

        cfg = Config(websocket=WebSocketConfig(backend="whisperlive"))
        engine = create_ws_engine(
            cfg,
            server_url="http://localhost:9090",
            model="tiny",
            language="en",
            reconnect_attempts=0,
            reconnect_delay=1.0,
        )
        from whisper_dictation.engine.websocket import WebSocketEngine

        assert isinstance(engine, WebSocketEngine)

    def test_whisperlivekit_backend(self):
        """create_ws_engine returns WhisperLiveKitEngine for 'whisperlivekit' backend."""
        from whisper_dictation.engine import create_ws_engine

        cfg = Config(websocket=WebSocketConfig(backend="whisperlivekit"))
        with patch(
            "whisper_dictation.engine.whisperlivekit.WhisperLiveKitEngine"
        ) as mock_cls:
            mock_cls.return_value = MagicMock()
            engine = create_ws_engine(
                cfg,
                server_url="http://localhost:9090",
                model="tiny",
                language="en",
                reconnect_attempts=0,
                reconnect_delay=1.0,
            )
            mock_cls.assert_called_once()
