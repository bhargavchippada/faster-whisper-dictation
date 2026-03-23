"""Tests for whisper_dictation.engine.server — ServerEngine."""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest
import requests

from whisper_dictation.config import ServerConfig
from whisper_dictation.engine.server import ServerEngine


@pytest.fixture
def server_config():
    return ServerConfig(
        url="http://localhost:9090",
        model="tiny",
        language="en",
        timeout=10,
    )


@pytest.fixture
def engine(server_config):
    return ServerEngine(server_config)


# ---------------------------------------------------------------------------
# Init
# ---------------------------------------------------------------------------


class TestServerEngineInit:
    def test_url_construction(self, server_config):
        engine = ServerEngine(server_config)
        assert engine._url == "http://localhost:9090/v1/audio/transcriptions"
        assert engine._health_url == "http://localhost:9090/health"

    def test_url_trailing_slash_stripped(self):
        cfg = ServerConfig(url="http://host:8080/")
        engine = ServerEngine(cfg)
        assert engine._url == "http://host:8080/v1/audio/transcriptions"
        assert engine._health_url == "http://host:8080/health"


# ---------------------------------------------------------------------------
# transcribe
# ---------------------------------------------------------------------------


class TestTranscribe:
    def test_successful_transcription(self, engine):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"text": " Hello world "}
        mock_resp.raise_for_status.return_value = None
        engine._session.post = MagicMock(return_value=mock_resp)

        audio = np.zeros(16000, dtype=np.float32)
        text = engine.transcribe(audio, 16000)

        assert text == "Hello world"
        engine._session.post.assert_called_once()
        call_kwargs = engine._session.post.call_args[1]
        assert call_kwargs["data"]["model"] == "tiny"
        assert call_kwargs["data"]["language"] == "en"
        assert call_kwargs["timeout"] == 10

    def test_empty_text_response(self, engine):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"text": ""}
        mock_resp.raise_for_status.return_value = None
        engine._session.post = MagicMock(return_value=mock_resp)

        text = engine.transcribe(np.zeros(16000, dtype=np.float32))
        assert text == ""

    def test_missing_text_key(self, engine):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {}
        mock_resp.raise_for_status.return_value = None
        engine._session.post = MagicMock(return_value=mock_resp)

        text = engine.transcribe(np.zeros(16000, dtype=np.float32))
        assert text == ""

    def test_timeout_returns_empty(self, engine):
        engine._session.post = MagicMock(side_effect=requests.Timeout("timeout"))
        text = engine.transcribe(np.zeros(16000, dtype=np.float32))
        assert text == ""

    def test_connection_error_returns_empty(self, engine):
        engine._session.post = MagicMock(side_effect=requests.ConnectionError("refused"))
        text = engine.transcribe(np.zeros(16000, dtype=np.float32))
        assert text == ""

    def test_http_error_returns_empty(self, engine):
        mock_resp = MagicMock()
        mock_resp.raise_for_status.side_effect = requests.HTTPError("500")
        engine._session.post = MagicMock(return_value=mock_resp)

        text = engine.transcribe(np.zeros(16000, dtype=np.float32))
        assert text == ""

    def test_invalid_json_returns_empty(self, engine):
        """Non-JSON response (e.g. HTML error page) returns empty string."""
        mock_resp = MagicMock()
        mock_resp.raise_for_status.return_value = None
        mock_resp.json.side_effect = ValueError("No JSON")
        engine._session.post = MagicMock(return_value=mock_resp)

        text = engine.transcribe(np.zeros(16000, dtype=np.float32))
        assert text == ""

    def test_sends_wav_file(self, engine):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"text": "hi"}
        mock_resp.raise_for_status.return_value = None
        engine._session.post = MagicMock(return_value=mock_resp)

        audio = np.array([0.5, -0.5], dtype=np.float32)
        engine.transcribe(audio, 16000)

        files_arg = engine._session.post.call_args[1]["files"]
        filename, data, mime = files_arg["file"]
        assert filename == "audio.wav"
        assert mime == "audio/wav"
        # data should be valid WAV bytes (starts with RIFF)
        assert data[:4] == b"RIFF"


# ---------------------------------------------------------------------------
# is_available
# ---------------------------------------------------------------------------


class TestIsAvailable:
    def test_available_on_ok(self, engine):
        mock_resp = MagicMock()
        mock_resp.ok = True
        engine._session.get = MagicMock(return_value=mock_resp)

        assert engine.is_available() is True
        engine._session.get.assert_called_once_with(
            "http://localhost:9090/health",
            timeout=3,
        )

    def test_unavailable_on_error(self, engine):
        engine._session.get = MagicMock(side_effect=requests.ConnectionError("refused"))
        assert engine.is_available() is False

    def test_unavailable_on_timeout(self, engine):
        engine._session.get = MagicMock(side_effect=requests.Timeout("timeout"))
        assert engine.is_available() is False

    def test_unavailable_on_not_ok(self, engine):
        mock_resp = MagicMock()
        mock_resp.ok = False
        engine._session.get = MagicMock(return_value=mock_resp)
        assert engine.is_available() is False

    def test_health_url_trailing_slash(self):
        cfg = ServerConfig(url="http://host:9090/")
        engine = ServerEngine(cfg)
        mock_resp = MagicMock()
        mock_resp.ok = True
        engine._session.get = MagicMock(return_value=mock_resp)

        engine.is_available()
        engine._session.get.assert_called_once_with("http://host:9090/health", timeout=3)


class TestClose:
    def test_close_closes_session(self, engine):
        engine._session.close = MagicMock()
        engine.close()
        engine._session.close.assert_called_once()
