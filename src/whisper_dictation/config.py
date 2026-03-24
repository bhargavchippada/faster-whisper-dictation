"""Configuration management for whisper-dictation."""

from __future__ import annotations

import math
import os
import re
import sys
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib

from platformdirs import user_config_dir

APP_NAME = "faster-whisper-dictation"
CONFIG_DIR = Path(user_config_dir(APP_NAME))
CONFIG_FILE = CONFIG_DIR / "config.toml"
PID_FILE = CONFIG_DIR / "daemon.pid"
STATE_FILE = CONFIG_DIR / "state.json"
LOG_FILE = CONFIG_DIR / "daemon.log"
_HOTKEY_MODIFIERS = {"alt", "ctrl", "control", "shift", "cmd", "super", "meta"}


@dataclass(frozen=True)
class ServerConfig:
    url: str = "http://localhost:8000"
    model: str = "Systran/faster-whisper-large-v3"
    language: str = "en"
    timeout: int = 10
    prompt: str = ""  # optional: domain vocabulary or style example for Whisper
    temperature: float = 0.0  # 0.0 = most accurate, higher = more creative
    hotwords: str = ""  # comma-separated words to boost recognition


@dataclass(frozen=True)
class HotkeyConfig:
    binding: str = "alt+v"
    mode: str = "toggle"  # "toggle" or "hold"


@dataclass(frozen=True)
class VADConfig:
    threshold: float = 0.6
    silence_ms: int = 200
    min_speech_ms: int = 250
    max_speech_s: float = 90.0


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
class WebSocketConfig:
    reconnect_attempts: int = 3
    reconnect_delay: float = 1.0


@dataclass(frozen=True)
class Config:
    server: ServerConfig = field(default_factory=ServerConfig)
    hotkey: HotkeyConfig = field(default_factory=HotkeyConfig)
    vad: VADConfig = field(default_factory=VADConfig)
    audio: AudioConfig = field(default_factory=AudioConfig)
    engine: EngineConfig = field(default_factory=EngineConfig)
    websocket: WebSocketConfig = field(default_factory=WebSocketConfig)


def _apply_env_overrides(config: Config) -> Config:
    """Apply environment variable overrides to config."""
    env_map = {
        "WHISPER_SERVER_URL": ("server", "url"),
        "WHISPER_MODEL": ("server", "model"),
        "WHISPER_LANG": ("server", "language"),
        "WHISPER_TIMEOUT": ("server", "timeout"),
        "WHISPER_PROMPT": ("server", "prompt"),
        "WHISPER_TEMPERATURE": ("server", "temperature"),
        "WHISPER_HOTWORDS": ("server", "hotwords"),
        "DICTATION_HOTKEY": ("hotkey", "binding"),
        "DICTATION_MODE": ("hotkey", "mode"),
        "DICTATION_ENGINE": ("engine", "type"),
        "DICTATION_ENGINE_COMPUTE": ("engine", "compute_type"),
        "DICTATION_ENGINE_DEVICE": ("engine", "device"),
        "DICTATION_AUDIO_DEVICE": ("audio", "device"),
        "DICTATION_SAMPLE_RATE": ("audio", "sample_rate"),
        "DICTATION_VAD_THRESHOLD": ("vad", "threshold"),
        "DICTATION_VAD_SILENCE_MS": ("vad", "silence_ms"),
        "DICTATION_VAD_MIN_SPEECH_MS": ("vad", "min_speech_ms"),
        "DICTATION_VAD_MAX_SPEECH_S": ("vad", "max_speech_s"),
        "DICTATION_WS_RECONNECT_ATTEMPTS": ("websocket", "reconnect_attempts"),
        "DICTATION_WS_RECONNECT_DELAY": ("websocket", "reconnect_delay"),
    }

    sections: dict[str, dict] = {
        "server": {},
        "hotkey": {},
        "vad": {},
        "audio": {},
        "engine": {},
        "websocket": {},
    }

    for env_key, (section, key) in env_map.items():
        value = os.environ.get(env_key)
        if value is not None:
            sections[section][key] = value

    if not any(sections.values()):
        return config

    def _merge_section(current: Any, overrides: dict[str, Any], cls: type) -> Any:
        if not overrides:
            return current
        merged = {}
        for f in fields(current):
            val = overrides.get(f.name, getattr(current, f.name))
            # Coerce types
            current_val = getattr(current, f.name)
            expected = type(current_val) if current_val is not None else str
            if isinstance(val, str) and expected is bool:
                val = val.lower() in ("1", "true", "yes")
            elif isinstance(val, str) and expected in (int, float):
                try:
                    val = expected(val)
                except (ValueError, TypeError) as exc:
                    raise ValueError(
                        f"Invalid value for environment variable "
                        f"(expected {expected.__name__}): {val!r}"
                    ) from exc
            merged[f.name] = val
        return cls(**merged)

    return Config(
        server=_merge_section(config.server, sections["server"], ServerConfig),
        hotkey=_merge_section(config.hotkey, sections["hotkey"], HotkeyConfig),
        vad=_merge_section(config.vad, sections["vad"], VADConfig),
        audio=_merge_section(config.audio, sections["audio"], AudioConfig),
        engine=_merge_section(config.engine, sections["engine"], EngineConfig),
        websocket=_merge_section(config.websocket, sections["websocket"], WebSocketConfig),
    )


def _build_section(data: dict[str, Any], cls: type) -> Any:
    """Build a dataclass from a dict, ignoring unknown keys."""
    known = {f.name for f in fields(cls)}
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
            websocket=_build_section(raw.get("websocket", {}), WebSocketConfig),
        )
    else:
        config = Config()

    config = _apply_env_overrides(config)
    return config


def _is_supported_hotkey_binding(binding: str) -> bool:
    """Return True for portable hotkeys supported across current backends."""
    parts = [part.strip().lower() for part in binding.split("+")]
    if not parts or any(not part for part in parts):
        return False
    key = parts[-1]
    modifiers = parts[:-1]
    return key.isalpha() and len(key) == 1 and all(mod in _HOTKEY_MODIFIERS for mod in modifiers)


def validate(config: Config) -> None:
    """Validate config values and raise ValueError with clear messages."""
    errors: list[str] = []

    if not config.server.url:
        errors.append("server.url must not be empty")

    if config.hotkey.mode not in ("toggle", "hold"):
        errors.append(f"hotkey.mode must be 'toggle' or 'hold', got '{config.hotkey.mode}'")

    binding = config.hotkey.binding
    if not _is_supported_hotkey_binding(binding):
        errors.append(
            "hotkey.binding must use zero or more modifiers "
            f"({', '.join(sorted(_HOTKEY_MODIFIERS))}) plus a single letter key, got {binding!r}"
        )

    if config.engine.type not in ("server", "local"):
        errors.append(f"engine.type must be 'server' or 'local', got '{config.engine.type}'")

    if not (0.0 <= config.vad.threshold <= 1.0) or not math.isfinite(config.vad.threshold):
        errors.append(f"vad.threshold must be 0.0-1.0, got {config.vad.threshold}")

    if config.vad.silence_ms <= 0:
        errors.append(f"vad.silence_ms must be positive, got {config.vad.silence_ms}")

    if config.vad.min_speech_ms <= 0:
        errors.append(f"vad.min_speech_ms must be positive, got {config.vad.min_speech_ms}")

    max_s = config.vad.max_speech_s
    if not math.isfinite(max_s) or max_s <= 0:
        errors.append(f"vad.max_speech_s must be finite and positive, got {max_s}")

    timeout = float(config.server.timeout)
    if not math.isfinite(timeout) or timeout <= 0:
        errors.append(f"server.timeout must be finite and positive, got {timeout}")

    temp = config.server.temperature
    if not math.isfinite(temp) or not (0.0 <= temp <= 1.0):
        errors.append(
            f"server.temperature must be 0.0-1.0, got {config.server.temperature}"
        )

    if config.audio.sample_rate <= 0:
        errors.append(f"audio.sample_rate must be positive, got {config.audio.sample_rate}")

    # Validate server URL scheme
    try:
        parsed = urlparse(config.server.url)
        if parsed.scheme not in ("http", "https"):
            errors.append(f"server.url must use http or https scheme, got '{parsed.scheme}'")
        if not parsed.hostname:
            errors.append("server.url must have a valid hostname")
    except (ValueError, TypeError):  # pragma: no cover — defensive, urlparse rarely raises
        errors.append(f"server.url is not a valid URL: {config.server.url!r}")

    # Language code validation (BCP-47 subset: 2-8 letter codes, optional subtags)
    lang = config.server.language
    if lang and not re.match(r"^[a-zA-Z]{2,8}(-[a-zA-Z0-9]{1,8})*$", lang):
        errors.append(
            f"server.language must be a valid language code (e.g. 'en', 'zh-CN'), "
            f"got {lang!r}"
        )

    # WebSocket config
    ws = config.websocket
    if ws.reconnect_attempts < 0:
        errors.append(f"websocket.reconnect_attempts must be >= 0, got {ws.reconnect_attempts}")
    delay = ws.reconnect_delay
    if not math.isfinite(delay) or not (0.1 <= delay <= 30.0):
        errors.append(f"websocket.reconnect_delay must be 0.1-30.0, got {delay}")

    if errors:
        raise ValueError("Invalid configuration:\n  - " + "\n  - ".join(errors))
