"""Voice Activity Detection using Silero VAD."""

from __future__ import annotations

import logging
import threading

import numpy as np

log = logging.getLogger(__name__)

# Lazy-loaded model (protected by _model_lock)
_model = None
_model_lock = threading.Lock()


def _load_model():
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


def _load_onnx_model():
    """Load Silero VAD via ONNX Runtime (no PyTorch needed)."""
    global _model

    import urllib.request
    from pathlib import Path

    from platformdirs import user_cache_dir

    cache = Path(user_cache_dir("whisper-dictation")) / "silero_vad.onnx"
    if not cache.exists():
        cache.parent.mkdir(parents=True, exist_ok=True)
        url = "https://github.com/snakers4/silero-vad/raw/master/src/silero_vad/data/silero_vad.onnx"
        log.info("Downloading Silero VAD ONNX model...")
        urllib.request.urlretrieve(url, cache)
        log.info("Downloaded to %s", cache)

    import onnxruntime as ort

    _model = OnnxVAD(str(cache))
    log.info("Silero VAD ONNX model loaded")


class OnnxVAD:
    """Minimal Silero VAD wrapper using ONNX Runtime."""

    def __init__(self, model_path: str):
        import onnxruntime as ort

        opts = ort.SessionOptions()
        opts.inter_op_num_threads = 1
        opts.intra_op_num_threads = 1
        self.session = ort.InferenceSession(model_path, sess_options=opts)
        self._h = np.zeros((2, 1, 64), dtype=np.float32)
        self._c = np.zeros((2, 1, 64), dtype=np.float32)
        self._sr = np.array(16000, dtype=np.int64)

    def reset_states(self):
        self._h = np.zeros((2, 1, 64), dtype=np.float32)
        self._c = np.zeros((2, 1, 64), dtype=np.float32)

    def __call__(self, audio: np.ndarray, sr: int) -> float:
        """Run VAD on a chunk, return speech probability (0.0-1.0)."""
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32) / 32768.0

        if audio.ndim == 1:
            audio = audio[np.newaxis, :]

        ort_inputs = {
            "input": audio,
            "h": self._h,
            "c": self._c,
            "sr": self._sr,
        }
        out, hn, cn = self.session.run(None, ort_inputs)
        self._h = hn
        self._c = cn
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
    ):
        self.sample_rate = sample_rate
        self.threshold = threshold
        self.silence_chunks = silence_ms // 32  # 32ms per chunk for Silero
        self.min_speech_chunks = min_speech_ms // 32
        self.max_speech_chunks = int(max_speech_s * 1000 / 32)
        self.chunk_size = sample_rate * 32 // 1000  # 512 samples at 16kHz

        self._is_speaking = False
        self._silence_count = 0
        self._speech_count = 0
        self._speech_frames: list[np.ndarray] = []
        self._buffer = np.array([], dtype=np.float32)
        self._model_loaded = False

    def _ensure_model(self):
        if not self._model_loaded:
            _load_model()
            self._model_loaded = True

    def reset(self):
        """Reset detector state for a new session."""
        self._is_speaking = False
        self._silence_count = 0
        self._speech_count = 0
        self._speech_frames.clear()
        self._buffer = np.array([], dtype=np.float32)
        if _model is not None and hasattr(_model, "reset_states"):
            _model.reset_states()

    def process_chunk(self, audio: np.ndarray) -> tuple[bool, np.ndarray | None]:
        """Process an audio chunk.

        Returns:
            (utterance_complete, audio_data)
            - If utterance_complete is True, audio_data contains the full utterance.
            - Otherwise audio_data is None.
        """
        self._ensure_model()

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

        return (completed_utterance is not None, completed_utterance)

    def flush(self) -> np.ndarray | None:
        """Flush any buffered speech frames and return them.

        Returns the concatenated audio if there are enough speech frames,
        otherwise None. Resets internal state afterward.
        """
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
        return self._is_speaking
