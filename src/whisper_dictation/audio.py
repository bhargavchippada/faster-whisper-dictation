"""Audio capture using sounddevice."""

from __future__ import annotations

import io
import logging
import wave
from collections.abc import Callable

import numpy as np
import sounddevice as sd

from .config import AudioConfig

log = logging.getLogger(__name__)


def list_devices() -> list[dict]:
    """List available audio input devices."""
    devices = sd.query_devices()
    inputs = []
    for i, d in enumerate(devices):
        if d["max_input_channels"] > 0:
            inputs.append(
                {
                    "index": i,
                    "name": d["name"],
                    "channels": d["max_input_channels"],
                    "sample_rate": d["default_samplerate"],
                }
            )
    return inputs


def audio_to_wav(audio: np.ndarray, sample_rate: int = 16000) -> bytes:
    """Convert float32 audio array to WAV bytes."""
    if audio.dtype == np.float32:
        audio_int16 = (audio * 32768).clip(-32768, 32767).astype(np.int16)
    else:
        audio_int16 = audio.astype(np.int16)

    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(audio_int16.tobytes())
    return buf.getvalue()


class AudioStream:
    """Streaming audio capture from microphone.

    Calls the provided callback with audio chunks as they arrive.
    """

    def __init__(
        self,
        config: AudioConfig,
        callback: Callable[[np.ndarray], None],
        block_ms: int = 32,
    ):
        self.config = config
        self.callback = callback
        self.block_size = config.sample_rate * block_ms // 1000
        self._stream: sd.InputStream | None = None

    def _audio_callback(
        self,
        indata: np.ndarray,
        frames: int,
        time_info: object,
        status: object,
    ) -> None:
        if status:
            log.debug("Audio status: %s", status)
        self.callback(indata.copy().flatten())

    def start(self) -> None:
        """Start capturing audio."""
        device = self.config.device
        if isinstance(device, str):
            stripped = device.strip()
            if stripped.isdigit():
                device = int(stripped)

        # Try to find device by name if string
        if isinstance(device, str):
            matched = False
            for d in sd.query_devices():
                if device.lower() in d["name"].lower() and d["max_input_channels"] > 0:
                    device = d["name"]
                    matched = True
                    break
            if not matched:
                log.warning("Audio device %r not found, passing as-is to sounddevice", device)

        log.info(
            "Starting audio capture: device=%s, rate=%d, block=%d",
            device or "default",
            self.config.sample_rate,
            self.block_size,
        )

        self._stream = sd.InputStream(
            samplerate=self.config.sample_rate,
            channels=self.config.channels,
            dtype="float32",
            blocksize=self.block_size,
            device=device,
            callback=self._audio_callback,
        )
        try:
            self._stream.start()
        except Exception:
            self._stream.close()
            self._stream = None
            raise

    def stop(self) -> None:
        """Stop capturing audio."""
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None
            log.info("Audio capture stopped")

    @property
    def active(self) -> bool:
        return self._stream is not None and self._stream.active
