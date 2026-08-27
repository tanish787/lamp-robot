"""Speech-to-text via faster-whisper (CTranslate2, CPU-friendly, free
open-weight model). The model is injectable so unit tests never load a
real model; scripts/smoke_stt.py exercises the real thing."""

import numpy as np

# Whisper is trained on 16 kHz mono; faster-whisper assumes an array at
# that rate and has no way to be told otherwise, so anything else has to
# be resampled before it is handed over.
TARGET_SAMPLE_RATE = 16000


def _resample(audio_array: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
    """Linear resample. Good enough for speech at these ratios, and avoids
    pulling in scipy/librosa for one call."""
    if audio_array.size == 0:
        return audio_array
    target_length = max(1, round(audio_array.size * target_rate / source_rate))
    source_positions = np.arange(audio_array.size, dtype=np.float64)
    target_positions = np.linspace(0, audio_array.size - 1, target_length)
    return np.interp(target_positions, source_positions, audio_array).astype(np.float32)


class SpeechToText:
    def __init__(self, model=None, model_size: str = "tiny"):
        if model is None:
            from faster_whisper import WhisperModel

            model = WhisperModel(model_size, device="cpu", compute_type="int8")
        self._model = model

    def transcribe(self, audio: bytes, sample_rate: int = TARGET_SAMPLE_RATE) -> str:
        """Transcribe 16-bit mono PCM. `sample_rate` is honoured: audio at
        any other rate is resampled to 16 kHz rather than silently
        mis-transcribed (a 44.1 kHz clip would otherwise be heard at
        roughly a third speed)."""
        if sample_rate <= 0:
            raise ValueError(f"sample_rate must be positive, got {sample_rate}")
        audio_array = np.frombuffer(audio, dtype=np.int16).astype(np.float32) / 32768.0
        if sample_rate != TARGET_SAMPLE_RATE:
            audio_array = _resample(audio_array, sample_rate, TARGET_SAMPLE_RATE)
        segments, _ = self._model.transcribe(audio_array, language="en")
        return "".join(segment.text for segment in segments).strip()
