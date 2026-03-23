"""Tests for whisper_dictation.engine.server — ServerEngine."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import requests

from whisper_dictation.config import ServerConfig
from whisper_dictation.engine.server import ServerEngine


@pytest.fixture
def server_config():
    return ServerConfig(
        url="http://localhost:10300",
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
        assert engine._url == "http://localhost:10300/v1/audio/transcriptions"

    def test_url_trailing_slash_stripped(self):
        cfg = ServerConfig(url="http://host:8080/")
        engine = ServerEngine(cfg)
        assert engine._url == "http://host:8080/v1/audio/transcriptions"


# ---------------------------------------------------------------------------
# transcribe
# ---------------------------------------------------------------------------


class TestTranscribe:
    @patch("whisper_dictation.engine.server.requests.post")
    def test_successful_transcription(self, mock_post, engine):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"text": " Hello world "}
        mock_resp.raise_for_status.return_value = None
        mock_post.return_value = mock_resp

        audio = np.zeros(16000, dtype=np.float32)
        text = engine.transcribe(audio, 16000)

        assert text == "Hello world"
        mock_post.assert_called_once()
        call_kwargs = mock_post.call_args[1]
        assert call_kwargs["data"]["model"] == "tiny"
        assert call_kwargs["data"]["language"] == "en"
        assert call_kwargs["timeout"] == 10

    @patch("whisper_dictation.engine.server.requests.post")
    def test_empty_text_response(self, mock_post, engine):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"text": ""}
        mock_resp.raise_for_status.return_value = None
        mock_post.return_value = mock_resp

        text = engine.transcribe(np.zeros(16000, dtype=np.float32))
        assert text == ""

    @patch("whisper_dictation.engine.server.requests.post")
    def test_missing_text_key(self, mock_post, engine):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {}
        mock_resp.raise_for_status.return_value = None
        mock_post.return_value = mock_resp

        text = engine.transcribe(np.zeros(16000, dtype=np.float32))
        assert text == ""

    @patch("whisper_dictation.engine.server.requests.post")
    def test_timeout_returns_empty(self, mock_post, engine):
        mock_post.side_effect = requests.Timeout("timeout")
        text = engine.transcribe(np.zeros(16000, dtype=np.float32))
        assert text == ""

    @patch("whisper_dictation.engine.server.requests.post")
    def test_connection_error_returns_empty(self, mock_post, engine):
        mock_post.side_effect = requests.ConnectionError("refused")
        text = engine.transcribe(np.zeros(16000, dtype=np.float32))
        assert text == ""

    @patch("whisper_dictation.engine.server.requests.post")
    def test_http_error_returns_empty(self, mock_post, engine):
        mock_resp = MagicMock()
        mock_resp.raise_for_status.side_effect = requests.HTTPError("500")
        mock_post.return_value = mock_resp

        text = engine.transcribe(np.zeros(16000, dtype=np.float32))
        assert text == ""

    @patch("whisper_dictation.engine.server.requests.post")
    def test_sends_wav_file(self, mock_post, engine):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"text": "hi"}
        mock_resp.raise_for_status.return_value = None
        mock_post.return_value = mock_resp

        audio = np.array([0.5, -0.5], dtype=np.float32)
        engine.transcribe(audio, 16000)

        files_arg = mock_post.call_args[1]["files"]
        filename, data, mime = files_arg["file"]
        assert filename == "audio.wav"
        assert mime == "audio/wav"
        # data should be valid WAV bytes (starts with RIFF)
        assert data[:4] == b"RIFF"


# ---------------------------------------------------------------------------
# is_available
# ---------------------------------------------------------------------------


class TestIsAvailable:
    @patch("whisper_dictation.engine.server.requests.get")
    def test_available_on_ok(self, mock_get, engine):
        mock_resp = MagicMock()
        mock_resp.ok = True
        mock_get.return_value = mock_resp

        assert engine.is_available() is True
        mock_get.assert_called_once_with(
            "http://localhost:10300/health",
            timeout=3,
        )

    @patch("whisper_dictation.engine.server.requests.get")
    def test_unavailable_on_error(self, mock_get, engine):
        mock_get.side_effect = requests.ConnectionError("refused")
        assert engine.is_available() is False

    @patch("whisper_dictation.engine.server.requests.get")
    def test_unavailable_on_timeout(self, mock_get, engine):
        mock_get.side_effect = requests.Timeout("timeout")
        assert engine.is_available() is False

    @patch("whisper_dictation.engine.server.requests.get")
    def test_unavailable_on_not_ok(self, mock_get, engine):
        mock_resp = MagicMock()
        mock_resp.ok = False
        mock_get.return_value = mock_resp
        assert engine.is_available() is False

    @patch("whisper_dictation.engine.server.requests.get")
    def test_health_url_trailing_slash(self, mock_get):
        cfg = ServerConfig(url="http://host:9090/")
        engine = ServerEngine(cfg)
        mock_resp = MagicMock()
        mock_resp.ok = True
        mock_get.return_value = mock_resp

        engine.is_available()
        mock_get.assert_called_once_with("http://host:9090/health", timeout=3)
