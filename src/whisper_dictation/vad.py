"""Voice Activity Detection using Silero VAD."""

from __future__ import annotations

import hashlib
import logging
import os
import threading
from collections import deque
from urllib.parse import urlparse

import numpy as np

log = logging.getLogger(__name__)

# Silero VAD requires exactly 32ms audio windows for inference.
_SILERO_CHUNK_MS = 32

# Silero VAD ONNX model — defaults to a known release.
# Override URL with DICTATION_VAD_MODEL_URL env var.
# Enable SHA-256 verification with DICTATION_VAD_VERIFY_HASH=true.
_ONNX_MODEL_SHA256 = "2623a2953f6ff3d2c1e61740c6cdb7168133479b267dfef114a4a3cc5bdd788f"
_DEFAULT_ONNX_MODEL_URL = (
    "https://github.com/snakers4/silero-vad/raw/v5.1.2/src/silero_vad/data/silero_vad.onnx"
)
_ONNX_MODEL_URL = os.environ.get("DICTATION_VAD_MODEL_URL", _DEFAULT_ONNX_MODEL_URL)
_VERIFY_HASH = os.environ.get("DICTATION_VAD_VERIFY_HASH", "").lower() in ("1", "true", "yes")

# Validate custom URL scheme at import time
if _ONNX_MODEL_URL != _DEFAULT_ONNX_MODEL_URL:
    _parsed = urlparse(_ONNX_MODEL_URL)
    if _parsed.scheme not in ("http", "https"):
        raise ValueError(
            f"DICTATION_VAD_MODEL_URL must use http or https scheme, got: {_parsed.scheme!r}"
        )

# Lazy-loaded model (protected by _model_lock).
# Typed as union because torch.hub.load returns an opaque callable, not OnnxVAD.
_model: OnnxVAD | object | None = None
_model_lock = threading.Lock()


def _load_model() -> None:
    """Load Silero VAD model (lazy, thread-safe, first call only)."""
    global _model

    if _model is not None:
        return

    with _model_lock:
        if _model is not None:
            return  # double-checked locking

        try:
            import torch

            model, _utils = torch.hub.load(
                repo_or_dir="snakers4/silero-vad",
                model="silero_vad",
                force_reload=False,
                onnx=True,
            )
            _model = model
            log.info("Silero VAD model loaded (torch)")
        except ImportError:
            log.info("torch not available, using ONNX Runtime fallback")
            _load_onnx_model()


def _verify_model_hash(path: str | bytes, expected: str) -> None:
    """Verify SHA-256 hash of a downloaded model file."""
    from pathlib import Path

    data = Path(path).read_bytes()
    actual = hashlib.sha256(data).hexdigest()
    if actual != expected:
        Path(path).unlink(missing_ok=True)
        raise RuntimeError(
            f"Model integrity check failed: expected SHA-256 {expected[:16]}..., "
            f"got {actual[:16]}... — file deleted. Retry to re-download."
        )


def _load_onnx_model() -> None:
    """Load Silero VAD via ONNX Runtime (no PyTorch needed)."""
    global _model

    import urllib.request
    from pathlib import Path

    from platformdirs import user_cache_dir

    from .config import APP_NAME

    cache = Path(user_cache_dir(APP_NAME)) / "silero_vad.onnx"
    if not cache.exists():
        cache.parent.mkdir(parents=True, exist_ok=True)
        tmp = cache.with_suffix(".tmp")
        log.info("Downloading Silero VAD ONNX model...")
        try:
            with urllib.request.urlopen(_ONNX_MODEL_URL, timeout=60) as resp:
                tmp.write_bytes(resp.read())
            if _VERIFY_HASH:
                _verify_model_hash(tmp, _ONNX_MODEL_SHA256)
            else:
                log.debug("SHA-256 verification skipped (DICTATION_VAD_VERIFY_HASH=true to enable)")
            tmp.rename(cache)
        except Exception:
            tmp.unlink(missing_ok=True)
            raise
        log.info("Downloaded to %s", cache)

    _model = OnnxVAD(str(cache))
    log.info("Silero VAD ONNX model loaded")


class OnnxVAD:
    """Minimal Silero VAD v5 wrapper using ONNX Runtime.

    Silero VAD v5 uses a single 'state' tensor (shape [2, 1, 128]) instead
    of separate 'h' and 'c' tensors used in v4.
    """

    def __init__(self, model_path: str) -> None:
        import onnxruntime as ort

        opts = ort.SessionOptions()
        opts.inter_op_num_threads = 1
        opts.intra_op_num_threads = 1
        self.session = ort.InferenceSession(model_path, sess_options=opts)
        self._state = np.zeros((2, 1, 128), dtype=np.float32)
        self._sr = np.array(16000, dtype=np.int64)

    def reset_states(self) -> None:
        self._state = np.zeros((2, 1, 128), dtype=np.float32)

    def __call__(self, audio: np.ndarray, sr: int) -> float:
        """Run VAD on a chunk, return speech probability (0.0-1.0)."""
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32) / 32768.0

        if audio.ndim == 1:
            audio = audio[np.newaxis, :]

        ort_inputs = {
            "input": audio,
            "state": self._state,
            "sr": self._sr,
        }
        out, new_state = self.session.run(None, ort_inputs)
        self._state = new_state
        return float(out[0][0])


class SpeechDetector:
    """Streaming speech detector using Silero VAD.

    Processes audio chunks and detects speech boundaries (start/end of utterances).
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        threshold: float = 0.5,
        silence_ms: int = 800,
        min_speech_ms: int = 250,
        max_speech_s: float = 90.0,
    ) -> None:
        self.sample_rate = sample_rate
        self.threshold = threshold
        self.silence_chunks = silence_ms // _SILERO_CHUNK_MS
        self.min_speech_chunks = min_speech_ms // _SILERO_CHUNK_MS
        self.max_speech_chunks = int(max_speech_s * 1000 / _SILERO_CHUNK_MS)
        self.chunk_size = sample_rate * _SILERO_CHUNK_MS // 1000  # 512 samples at 16kHz
        self._lock = threading.Lock()

        # Pre-speech ring buffer: keeps last N chunks so the start of speech
        # isn't clipped (VAD needs ~200ms to detect speech onset)
        self._pre_speech_chunks = 8  # ~256ms of lookback at 32ms/chunk
        self._ring_buffer: deque[np.ndarray] = deque(maxlen=self._pre_speech_chunks)

        self._is_speaking = False
        self._silence_count = 0
        self._speech_count = 0
        self._speech_frames: list[np.ndarray] = []
        self._buffer = np.array([], dtype=np.float32)
        self._model_loaded = False

    def _ensure_model(self) -> None:
        if not self._model_loaded:
            _load_model()
            self._model_loaded = True

    def reset(self) -> None:
        """Reset detector state for a new session. Thread-safe."""
        with self._lock:
            self._is_speaking = False
            self._silence_count = 0
            self._speech_count = 0
            self._speech_frames.clear()
            self._ring_buffer.clear()
            self._buffer = np.array([], dtype=np.float32)
            if _model is not None and hasattr(_model, "reset_states"):
                _model.reset_states()

    def process_chunk(self, audio: np.ndarray) -> tuple[bool, np.ndarray | None]:
        """Process an audio chunk. Thread-safe.

        Returns:
            (utterance_complete, audio_data)
            - If utterance_complete is True, audio_data contains the full utterance.
            - Otherwise audio_data is None.
        """
        self._ensure_model()
        with self._lock:
            return self._process_chunk_impl(audio)

    def _process_chunk_impl(self, audio: np.ndarray) -> tuple[bool, np.ndarray | None]:

        # Convert int16 to float32
        if audio.dtype == np.int16:
            audio = audio.astype(np.float32) / 32768.0

        # Buffer incoming audio and process in chunk_size pieces
        self._buffer = np.concatenate([self._buffer, audio.flatten()])

        completed_utterance = None

        while len(self._buffer) >= self.chunk_size:
            chunk = self._buffer[: self.chunk_size]
            self._buffer = self._buffer[self.chunk_size :]

            prob = _model(chunk, self.sample_rate)

            if prob >= self.threshold:
                if not self._is_speaking:
                    self._is_speaking = True
                    self._silence_count = 0
                    self._speech_count = 0
                    # Prepend pre-speech audio so word onsets aren't clipped
                    self._speech_frames.extend(self._ring_buffer)
                    self._ring_buffer.clear()
                    log.debug("Speech started (prob=%.2f)", prob)

                self._speech_count += 1
                self._silence_count = 0
                self._speech_frames.append(chunk)

                # Guard against unbounded buffer growth
                if self._speech_count >= self.max_speech_chunks:
                    completed_utterance = np.concatenate(self._speech_frames)
                    log.warning(
                        "Utterance exceeded max duration (%.1fs), flushing",
                        len(completed_utterance) / self.sample_rate,
                    )
                    self._speech_frames.clear()
                    self._is_speaking = False
                    self._silence_count = 0
                    self._speech_count = 0
                    break  # return immediately to avoid overwriting

            elif self._is_speaking:
                self._silence_count += 1
                self._speech_frames.append(chunk)

                if self._silence_count >= self.silence_chunks:
                    if self._speech_count >= self.min_speech_chunks:
                        completed_utterance = np.concatenate(self._speech_frames)
                        log.debug(
                            "Utterance complete: %.2fs, %d speech chunks",
                            len(completed_utterance) / self.sample_rate,
                            self._speech_count,
                        )
                    else:
                        log.debug("Rejected short utterance: %d chunks", self._speech_count)

                    self._speech_frames.clear()
                    self._is_speaking = False
                    self._silence_count = 0
                    self._speech_count = 0

                    if completed_utterance is not None:
                        break  # return immediately to avoid overwriting

            else:
                # Not speaking — maintain ring buffer for pre-speech lookback
                self._ring_buffer.append(chunk)

        return (completed_utterance is not None, completed_utterance)

    def flush(self) -> np.ndarray | None:
        """Flush any buffered speech frames and return them. Thread-safe.

        Returns the concatenated audio if there are enough speech frames,
        otherwise None. Resets internal state afterward.
        """
        with self._lock:
            if not self._is_speaking or not self._speech_frames:
                return None

            audio = np.concatenate(self._speech_frames)
            has_enough = self._speech_count >= self.min_speech_chunks
            self._speech_frames.clear()
            self._is_speaking = False
            self._silence_count = 0
            self._speech_count = 0

            return audio if has_enough else None

    @property
    def is_speaking(self) -> bool:
        """Whether speech is currently being detected.

        Note: reads _is_speaking without the lock for performance in the
        audio callback hot path. Safe on CPython due to the GIL, but the
        value may be briefly stale on other implementations.
        """
        return self._is_speaking
