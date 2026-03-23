"""Configuration management for whisper-dictation."""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib

from platformdirs import user_config_dir


APP_NAME = "whisper-dictation"
CONFIG_DIR = Path(user_config_dir(APP_NAME))
CONFIG_FILE = CONFIG_DIR / "config.toml"
PID_FILE = CONFIG_DIR / "daemon.pid"
STATE_FILE = CONFIG_DIR / "state.json"


@dataclass(frozen=True)
class ServerConfig:
    url: str = "http://localhost:10300"
    model: str = "Systran/faster-whisper-large-v3"
    language: str = "en"
    timeout: int = 30


@dataclass(frozen=True)
class HotkeyConfig:
    binding: str = "alt+v"
    mode: str = "toggle"  # "toggle" or "hold"


@dataclass(frozen=True)
class VADConfig:
    threshold: float = 0.5
    silence_ms: int = 800
    min_speech_ms: int = 250


@dataclass(frozen=True)
class AudioConfig:
    sample_rate: int = 16000
    channels: int = 1
    device: str | None = None


@dataclass(frozen=True)
class EngineConfig:
    type: str = "server"  # "server" or "local"
    compute_type: str = "auto"
    device: str = "auto"


@dataclass(frozen=True)
class Config:
    server: ServerConfig = field(default_factory=ServerConfig)
    hotkey: HotkeyConfig = field(default_factory=HotkeyConfig)
    vad: VADConfig = field(default_factory=VADConfig)
    audio: AudioConfig = field(default_factory=AudioConfig)
    engine: EngineConfig = field(default_factory=EngineConfig)


def _apply_env_overrides(config: Config) -> Config:
    """Apply environment variable overrides to config."""
    env_map = {
        "WHISPER_SERVER_URL": ("server", "url"),
        "WHISPER_MODEL": ("server", "model"),
        "WHISPER_LANG": ("server", "language"),
        "DICTATION_HOTKEY": ("hotkey", "binding"),
        "DICTATION_MODE": ("hotkey", "mode"),
        "DICTATION_ENGINE": ("engine", "type"),
        "DICTATION_AUDIO_DEVICE": ("audio", "device"),
        "DICTATION_VAD_THRESHOLD": ("vad", "threshold"),
        "DICTATION_VAD_SILENCE_MS": ("vad", "silence_ms"),
        "DICTATION_VAD_MIN_SPEECH_MS": ("vad", "min_speech_ms"),
    }

    sections: dict[str, dict] = {
        "server": {},
        "hotkey": {},
        "vad": {},
        "audio": {},
        "engine": {},
    }

    for env_key, (section, key) in env_map.items():
        value = os.environ.get(env_key)
        if value is not None:
            sections[section][key] = value

    if not any(sections.values()):
        return config

    def _merge_section(current, overrides, cls):
        if not overrides:
            return current
        merged = {}
        for f in current.__dataclass_fields__:
            val = overrides.get(f, getattr(current, f))
            # Coerce types
            expected = type(getattr(current, f)) if getattr(current, f) is not None else str
            if isinstance(val, str) and expected in (int, float):
                val = expected(val)
            merged[f] = val
        return cls(**merged)

    return Config(
        server=_merge_section(config.server, sections["server"], ServerConfig),
        hotkey=_merge_section(config.hotkey, sections["hotkey"], HotkeyConfig),
        vad=_merge_section(config.vad, sections["vad"], VADConfig),
        audio=_merge_section(config.audio, sections["audio"], AudioConfig),
        engine=_merge_section(config.engine, sections["engine"], EngineConfig),
    )


def _build_section(data: dict, cls):
    """Build a dataclass from a dict, ignoring unknown keys."""
    known = {f for f in cls.__dataclass_fields__}
    filtered = {k: v for k, v in data.items() if k in known}
    return cls(**filtered)


def load_config(config_path: Path | None = None) -> Config:
    """Load configuration from file, then apply env overrides."""
    path = config_path or CONFIG_FILE

    if path.exists():
        with open(path, "rb") as f:
            raw = tomllib.load(f)

        config = Config(
            server=_build_section(raw.get("server", {}), ServerConfig),
            hotkey=_build_section(raw.get("hotkey", {}), HotkeyConfig),
            vad=_build_section(raw.get("vad", {}), VADConfig),
            audio=_build_section(raw.get("audio", {}), AudioConfig),
            engine=_build_section(raw.get("engine", {}), EngineConfig),
        )
    else:
        config = Config()

    return _apply_env_overrides(config)
