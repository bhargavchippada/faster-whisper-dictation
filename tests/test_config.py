"""Tests for whisper_dictation.config — TOML loading, env overrides, defaults."""

from __future__ import annotations

import textwrap
from pathlib import Path
from unittest.mock import patch

import pytest

from whisper_dictation.config import (
    AudioConfig,
    Config,
    EngineConfig,
    HotkeyConfig,
    ServerConfig,
    VADConfig,
    _apply_env_overrides,
    _build_section,
    load_config,
)


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------


class TestDefaults:
    def test_server_defaults(self):
        cfg = ServerConfig()
        assert cfg.url == "http://localhost:10300"
        assert cfg.model == "Systran/faster-whisper-large-v3"
        assert cfg.language == "en"
        assert cfg.timeout == 30

    def test_hotkey_defaults(self):
        cfg = HotkeyConfig()
        assert cfg.binding == "alt+v"
        assert cfg.mode == "toggle"

    def test_vad_defaults(self):
        cfg = VADConfig()
        assert cfg.threshold == 0.5
        assert cfg.silence_ms == 800
        assert cfg.min_speech_ms == 250

    def test_audio_defaults(self):
        cfg = AudioConfig()
        assert cfg.sample_rate == 16000
        assert cfg.channels == 1
        assert cfg.device is None

    def test_engine_defaults(self):
        cfg = EngineConfig()
        assert cfg.type == "server"
        assert cfg.compute_type == "auto"
        assert cfg.device == "auto"

    def test_config_defaults(self):
        cfg = Config()
        assert isinstance(cfg.server, ServerConfig)
        assert isinstance(cfg.hotkey, HotkeyConfig)
        assert isinstance(cfg.vad, VADConfig)
        assert isinstance(cfg.audio, AudioConfig)
        assert isinstance(cfg.engine, EngineConfig)


# ---------------------------------------------------------------------------
# Frozen dataclasses
# ---------------------------------------------------------------------------


class TestFrozen:
    def test_server_config_is_frozen(self):
        cfg = ServerConfig()
        with pytest.raises(AttributeError):
            cfg.url = "http://other"  # type: ignore[misc]

    def test_config_is_frozen(self):
        cfg = Config()
        with pytest.raises(AttributeError):
            cfg.server = ServerConfig()  # type: ignore[misc]


# ---------------------------------------------------------------------------
# _build_section
# ---------------------------------------------------------------------------


class TestBuildSection:
    def test_known_keys_only(self):
        data = {"url": "http://x", "model": "tiny", "unknown_key": "ignored"}
        section = _build_section(data, ServerConfig)
        assert section.url == "http://x"
        assert section.model == "tiny"
        assert not hasattr(section, "unknown_key")

    def test_empty_dict_gives_defaults(self):
        section = _build_section({}, ServerConfig)
        assert section == ServerConfig()

    def test_partial_override(self):
        section = _build_section({"language": "fr"}, ServerConfig)
        assert section.language == "fr"
        assert section.url == "http://localhost:10300"  # default preserved

    def test_all_sections(self):
        for cls in (ServerConfig, HotkeyConfig, VADConfig, AudioConfig, EngineConfig):
            section = _build_section({}, cls)
            assert section == cls()


# ---------------------------------------------------------------------------
# _apply_env_overrides
# ---------------------------------------------------------------------------


class TestApplyEnvOverrides:
    def test_no_env_returns_same_config(self):
        cfg = Config()
        with patch.dict("os.environ", {}, clear=True):
            result = _apply_env_overrides(cfg)
        assert result == cfg

    def test_server_url_override(self):
        cfg = Config()
        with patch.dict("os.environ", {"WHISPER_SERVER_URL": "http://remote:9000"}, clear=True):
            result = _apply_env_overrides(cfg)
        assert result.server.url == "http://remote:9000"
        # Other fields unchanged
        assert result.server.model == cfg.server.model

    def test_multiple_overrides(self):
        cfg = Config()
        env = {
            "WHISPER_SERVER_URL": "http://a",
            "WHISPER_MODEL": "tiny",
            "WHISPER_LANG": "de",
            "DICTATION_HOTKEY": "ctrl+d",
            "DICTATION_MODE": "hold",
            "DICTATION_ENGINE": "local",
            "DICTATION_AUDIO_DEVICE": "hw:2",
        }
        with patch.dict("os.environ", env, clear=True):
            result = _apply_env_overrides(cfg)

        assert result.server.url == "http://a"
        assert result.server.model == "tiny"
        assert result.server.language == "de"
        assert result.hotkey.binding == "ctrl+d"
        assert result.hotkey.mode == "hold"
        assert result.engine.type == "local"
        assert result.audio.device == "hw:2"

    def test_numeric_coercion_float(self):
        cfg = Config()
        with patch.dict("os.environ", {"DICTATION_VAD_THRESHOLD": "0.8"}, clear=True):
            result = _apply_env_overrides(cfg)
        assert result.vad.threshold == 0.8
        assert isinstance(result.vad.threshold, float)

    def test_numeric_coercion_int(self):
        cfg = Config()
        with patch.dict("os.environ", {"DICTATION_VAD_SILENCE_MS": "1200"}, clear=True):
            result = _apply_env_overrides(cfg)
        assert result.vad.silence_ms == 1200
        assert isinstance(result.vad.silence_ms, int)

    def test_min_speech_ms_override(self):
        cfg = Config()
        with patch.dict("os.environ", {"DICTATION_VAD_MIN_SPEECH_MS": "500"}, clear=True):
            result = _apply_env_overrides(cfg)
        assert result.vad.min_speech_ms == 500


# ---------------------------------------------------------------------------
# load_config
# ---------------------------------------------------------------------------


class TestLoadConfig:
    def test_missing_file_returns_defaults(self, tmp_path):
        missing = tmp_path / "does_not_exist.toml"
        with patch.dict("os.environ", {}, clear=True):
            cfg = load_config(missing)
        assert cfg == Config()

    def test_load_from_toml(self, tmp_path):
        toml_path = tmp_path / "config.toml"
        toml_path.write_text(textwrap.dedent("""\
            [server]
            url = "http://myserver:8080"
            language = "ja"

            [hotkey]
            binding = "ctrl+shift+r"
            mode = "hold"

            [vad]
            threshold = 0.3
            silence_ms = 600

            [audio]
            sample_rate = 44100

            [engine]
            type = "local"
        """))
        with patch.dict("os.environ", {}, clear=True):
            cfg = load_config(toml_path)

        assert cfg.server.url == "http://myserver:8080"
        assert cfg.server.language == "ja"
        assert cfg.hotkey.binding == "ctrl+shift+r"
        assert cfg.hotkey.mode == "hold"
        assert cfg.vad.threshold == 0.3
        assert cfg.vad.silence_ms == 600
        assert cfg.audio.sample_rate == 44100
        assert cfg.engine.type == "local"

    def test_partial_toml(self, tmp_path):
        toml_path = tmp_path / "config.toml"
        toml_path.write_text("[server]\nmodel = \"tiny\"\n")
        with patch.dict("os.environ", {}, clear=True):
            cfg = load_config(toml_path)
        assert cfg.server.model == "tiny"
        assert cfg.server.url == "http://localhost:10300"  # default
        assert cfg.hotkey == HotkeyConfig()  # all defaults

    def test_toml_with_unknown_keys(self, tmp_path):
        toml_path = tmp_path / "config.toml"
        toml_path.write_text("[server]\nurl = \"http://x\"\nfoo = \"bar\"\n")
        with patch.dict("os.environ", {}, clear=True):
            cfg = load_config(toml_path)
        assert cfg.server.url == "http://x"

    def test_env_overrides_toml(self, tmp_path):
        toml_path = tmp_path / "config.toml"
        toml_path.write_text("[server]\nurl = \"http://toml\"\n")
        with patch.dict("os.environ", {"WHISPER_SERVER_URL": "http://env"}, clear=True):
            cfg = load_config(toml_path)
        assert cfg.server.url == "http://env"

    def test_empty_toml(self, tmp_path):
        toml_path = tmp_path / "config.toml"
        toml_path.write_text("")
        with patch.dict("os.environ", {}, clear=True):
            cfg = load_config(toml_path)
        assert cfg == Config()
