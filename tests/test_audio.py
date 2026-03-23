"""Tests for whisper_dictation.audio — AudioStream, audio_to_wav, list_devices."""

from __future__ import annotations

import struct
import wave
from io import BytesIO
from unittest.mock import MagicMock, patch

import numpy as np

from whisper_dictation.audio import AudioStream, audio_to_wav, list_devices
from whisper_dictation.config import AudioConfig

# ---------------------------------------------------------------------------
# audio_to_wav
# ---------------------------------------------------------------------------


class TestAudioToWav:
    def test_float32_conversion(self):
        audio = np.array([0.0, 0.5, -0.5, 1.0, -1.0], dtype=np.float32)
        wav_bytes = audio_to_wav(audio, sample_rate=16000)

        buf = BytesIO(wav_bytes)
        with wave.open(buf, "rb") as w:
            assert w.getnchannels() == 1
            assert w.getsampwidth() == 2
            assert w.getframerate() == 16000
            assert w.getnframes() == 5

            raw = w.readframes(5)
            samples = struct.unpack(f"<{5}h", raw)
            # 0.5 * 32768 = 16384
            assert samples[0] == 0
            assert samples[1] == 16384
            assert samples[2] == -16384

    def test_int16_passthrough(self):
        audio = np.array([0, 1000, -1000, 32767, -32768], dtype=np.int16)
        wav_bytes = audio_to_wav(audio, sample_rate=44100)

        buf = BytesIO(wav_bytes)
        with wave.open(buf, "rb") as w:
            assert w.getframerate() == 44100
            raw = w.readframes(5)
            samples = struct.unpack(f"<{5}h", raw)
            assert samples[1] == 1000
            assert samples[3] == 32767
            assert samples[4] == -32768

    def test_clipping(self):
        # Values beyond [-1, 1] should be clipped
        audio = np.array([2.0, -2.0], dtype=np.float32)
        wav_bytes = audio_to_wav(audio, sample_rate=16000)

        buf = BytesIO(wav_bytes)
        with wave.open(buf, "rb") as w:
            raw = w.readframes(2)
            samples = struct.unpack("<2h", raw)
            assert samples[0] == 32767
            assert samples[1] == -32768

    def test_empty_audio(self):
        audio = np.array([], dtype=np.float32)
        wav_bytes = audio_to_wav(audio, sample_rate=16000)

        buf = BytesIO(wav_bytes)
        with wave.open(buf, "rb") as w:
            assert w.getnframes() == 0

    def test_custom_sample_rate(self):
        audio = np.zeros(100, dtype=np.float32)
        wav_bytes = audio_to_wav(audio, sample_rate=48000)

        buf = BytesIO(wav_bytes)
        with wave.open(buf, "rb") as w:
            assert w.getframerate() == 48000


# ---------------------------------------------------------------------------
# list_devices
# ---------------------------------------------------------------------------


class TestListDevices:
    @patch("whisper_dictation.audio.sd")
    def test_filters_input_devices(self, mock_sd):
        mock_sd.query_devices.return_value = [
            {"name": "Mic1", "max_input_channels": 2, "default_samplerate": 44100.0},
            {"name": "Speakers", "max_input_channels": 0, "default_samplerate": 48000.0},
            {"name": "Mic2", "max_input_channels": 1, "default_samplerate": 16000.0},
        ]
        result = list_devices()
        assert len(result) == 2
        assert result[0]["name"] == "Mic1"
        assert result[0]["index"] == 0
        assert result[0]["channels"] == 2
        assert result[0]["sample_rate"] == 44100.0
        assert result[1]["name"] == "Mic2"
        assert result[1]["index"] == 2

    @patch("whisper_dictation.audio.sd")
    def test_no_input_devices(self, mock_sd):
        mock_sd.query_devices.return_value = [
            {"name": "Out", "max_input_channels": 0, "default_samplerate": 48000.0},
        ]
        result = list_devices()
        assert result == []

    @patch("whisper_dictation.audio.sd")
    def test_empty_device_list(self, mock_sd):
        mock_sd.query_devices.return_value = []
        result = list_devices()
        assert result == []


# ---------------------------------------------------------------------------
# AudioStream
# ---------------------------------------------------------------------------


class TestAudioStream:
    def test_init(self):
        cfg = AudioConfig(sample_rate=16000, channels=1, device=None)
        cb = MagicMock()
        stream = AudioStream(cfg, cb, block_ms=32)
        assert stream.block_size == 512  # 16000 * 32 // 1000
        assert stream._stream is None

    def test_block_size_calculation(self):
        cfg = AudioConfig(sample_rate=48000, channels=1, device=None)
        stream = AudioStream(cfg, MagicMock(), block_ms=20)
        assert stream.block_size == 960  # 48000 * 20 // 1000

    @patch("whisper_dictation.audio.sd")
    def test_start_creates_input_stream(self, mock_sd):
        cfg = AudioConfig(sample_rate=16000, channels=1, device=None)
        cb = MagicMock()
        stream = AudioStream(cfg, cb)

        mock_input_stream = MagicMock()
        mock_sd.InputStream.return_value = mock_input_stream

        stream.start()

        mock_sd.InputStream.assert_called_once()
        call_kwargs = mock_sd.InputStream.call_args[1]
        assert call_kwargs["samplerate"] == 16000
        assert call_kwargs["channels"] == 1
        assert call_kwargs["dtype"] == "float32"
        mock_input_stream.start.assert_called_once()

    @patch("whisper_dictation.audio.sd")
    def test_start_with_named_device(self, mock_sd):
        mock_sd.query_devices.return_value = [
            {"name": "My USB Microphone", "max_input_channels": 1},
            {"name": "Built-in Output", "max_input_channels": 0},
        ]
        mock_sd.InputStream.return_value = MagicMock()

        cfg = AudioConfig(sample_rate=16000, channels=1, device="usb")
        stream = AudioStream(cfg, MagicMock())
        stream.start()

        call_kwargs = mock_sd.InputStream.call_args[1]
        assert call_kwargs["device"] == "My USB Microphone"

    @patch("whisper_dictation.audio.sd")
    def test_start_with_unmatched_device_warns(self, mock_sd, caplog):
        mock_sd.query_devices.return_value = [
            {"name": "Built-in Mic", "max_input_channels": 1},
        ]
        mock_sd.InputStream.return_value = MagicMock()

        cfg = AudioConfig(sample_rate=16000, channels=1, device="nonexistent")
        stream = AudioStream(cfg, MagicMock())

        import logging

        with caplog.at_level(logging.WARNING):
            stream.start()

        assert "not found" in caplog.text
        # Should pass the original string as-is to sounddevice
        call_kwargs = mock_sd.InputStream.call_args[1]
        assert call_kwargs["device"] == "nonexistent"

    @patch("whisper_dictation.audio.sd")
    def test_stop(self, mock_sd):
        cfg = AudioConfig(sample_rate=16000, channels=1, device=None)
        stream = AudioStream(cfg, MagicMock())

        mock_input_stream = MagicMock()
        mock_sd.InputStream.return_value = mock_input_stream
        stream.start()
        stream.stop()

        mock_input_stream.stop.assert_called_once()
        mock_input_stream.close.assert_called_once()
        assert stream._stream is None

    def test_stop_without_start(self):
        cfg = AudioConfig(sample_rate=16000, channels=1, device=None)
        stream = AudioStream(cfg, MagicMock())
        # Should not raise
        stream.stop()

    @patch("whisper_dictation.audio.sd")
    def test_active_property(self, mock_sd):
        cfg = AudioConfig(sample_rate=16000, channels=1, device=None)
        stream = AudioStream(cfg, MagicMock())
        assert stream.active is False

        mock_input_stream = MagicMock()
        mock_input_stream.active = True
        mock_sd.InputStream.return_value = mock_input_stream
        stream.start()
        assert stream.active is True

    def test_audio_callback_invokes_user_callback(self):
        cfg = AudioConfig(sample_rate=16000, channels=1, device=None)
        cb = MagicMock()
        stream = AudioStream(cfg, cb)

        indata = np.ones((512, 1), dtype=np.float32)
        stream._audio_callback(indata, 512, None, None)

        cb.assert_called_once()
        passed = cb.call_args[0][0]
        assert passed.shape == (512,)
        np.testing.assert_array_equal(passed, np.ones(512, dtype=np.float32))

    def test_audio_callback_with_status(self):
        cfg = AudioConfig(sample_rate=16000, channels=1, device=None)
        cb = MagicMock()
        stream = AudioStream(cfg, cb)

        indata = np.zeros((512, 1), dtype=np.float32)
        # Should not raise even with status message
        stream._audio_callback(indata, 512, None, "input overflow")
        cb.assert_called_once()
