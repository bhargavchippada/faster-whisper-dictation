"""Transcription engine package."""

from __future__ import annotations

from typing import TYPE_CHECKING

from .base import TranscriptionEngine
from .server import ServerEngine

if TYPE_CHECKING:
    from ..config import Config

__all__ = ["ServerEngine", "TranscriptionEngine", "create_engine"]


def create_engine(config: Config) -> TranscriptionEngine:
    """Create a transcription engine based on config.

    Returns a ServerEngine or LocalEngine depending on config.engine.type.
    """
    if config.engine.type == "local":
        from .local import LocalEngine

        return LocalEngine(config.server, config.engine)

    return ServerEngine(config.server)
