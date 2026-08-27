"""Speech-to-text via faster-whisper (CTranslate2, CPU-friendly, free
open-weight model). The model is injectable so unit tests never load a
real model; scripts/smoke_stt.py exercises the real thing."""

import numpy as np


class SpeechToText:
    def __init__(self, model=None, model_size: str = "tiny"):
        if model is None:
            from faster_whisper import WhisperModel

            model = WhisperModel(model_size, device="cpu", compute_type="int8")
        self._model = model

    def transcribe(self, audio: bytes, sample_rate: int = 16000) -> str:
        audio_array = np.frombuffer(audio, dtype=np.int16).astype(np.float32) / 32768.0
        segments, _ = self._model.transcribe(audio_array, language="en")
        return "".join(segment.text for segment in segments).strip()
