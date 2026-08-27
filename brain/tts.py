"""Text-to-speech via Piper (free, offline, CPU-friendly). Both the
synthesizer and the audio player are injectable so unit tests never touch
a real model or audio device; scripts/smoke_tts.py exercises the real
thing end to end."""

from typing import Callable


class TextToSpeech:
    def __init__(
        self,
        synthesizer=None,
        player: Callable[[bytes], None] | None = None,
        voice_path: str = "models/piper/en_US.onnx",
    ):
        if synthesizer is None:
            from piper import PiperVoice

            synthesizer = PiperVoice.load(voice_path)
        if player is None:
            player = _play_wav_bytes
        self._synthesizer = synthesizer
        self._player = player

    def synthesize(self, text: str) -> bytes:
        return self._synthesizer.synthesize(text)

    def speak(self, text: str) -> None:
        self._player(self.synthesize(text))


def _play_wav_bytes(audio: bytes) -> None:
    import io

    import simpleaudio

    simpleaudio.WaveObject.from_wave_file(io.BytesIO(audio)).play().wait_done()
