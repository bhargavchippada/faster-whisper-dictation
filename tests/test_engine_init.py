"""Tests for whisper_dictation.engine.__init__ — create_ws_engine factory."""

from __future__ import annotations

from whisper_dictation.config import Config


class TestCreateWsEngine:
    def test_creates_whisperlivekit_engine(self):
        """create_ws_engine always returns a WhisperLiveKitEngine."""
        from whisper_dictation.engine import create_ws_engine
        from whisper_dictation.engine.whisperlivekit import WhisperLiveKitEngine

        cfg = Config()
        engine = create_ws_engine(
            cfg,
            server_url="http://localhost:8000",
            language="en",
            reconnect_attempts=0,
            reconnect_delay=1.0,
        )
        assert isinstance(engine, WhisperLiveKitEngine)
