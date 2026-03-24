"""Transcription engine package."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .base import TranscriptionEngine
from .server import ServerEngine

if TYPE_CHECKING:
    from ..config import Config
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


def create_ws_engine(config: Config, **kwargs: Any) -> WhisperLiveKitEngine:
    """Create the WhisperLiveKit WebSocket engine for streaming/batch transcription.

    ``config`` is accepted for future extensibility (e.g. selecting between
    backends) but currently unused — all engine parameters come from kwargs.

    Required kwargs: server_url, language, reconnect_attempts, reconnect_delay.
    Optional kwargs: on_text (Callable[[str], None]).
    """
    from .whisperlivekit import WhisperLiveKitEngine

    return WhisperLiveKitEngine(**kwargs)
