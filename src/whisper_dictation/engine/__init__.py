"""Transcription engine package."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .base import TranscriptionEngine
from .server import ServerEngine

if TYPE_CHECKING:
    from ..config import Config
    from .websocket import WebSocketEngine
    from .whisperlivekit import WhisperLiveKitEngine

__all__ = [
    "ServerEngine",
    "TranscriptionEngine",
    "create_engine",
    "create_ws_engine",
]


def create_engine(config: Config) -> TranscriptionEngine:
    """Create a transcription engine based on config.

    Returns a ServerEngine or LocalEngine depending on config.engine.type.
    """
    if config.engine.type == "local":
        from .local import LocalEngine

        return LocalEngine(config.server, config.engine)

    return ServerEngine(config.server)


def create_ws_engine(config: Config, **kwargs: Any) -> WebSocketEngine | WhisperLiveKitEngine:
    """Create the configured WebSocket engine based on config.websocket.backend.

    Required kwargs: server_url, model, language, reconnect_attempts, reconnect_delay.
    Optional kwargs: use_vad (bool), on_text (Callable[[str], None]).
    """
    backend = config.websocket.backend
    if backend == "whisperlivekit":
        from .whisperlivekit import WhisperLiveKitEngine

        return WhisperLiveKitEngine(**kwargs)

    if backend != "whisperlive":
        raise ValueError(f"Unknown websocket.backend: {backend!r}")

    from .websocket import WebSocketEngine

    return WebSocketEngine(**kwargs)
