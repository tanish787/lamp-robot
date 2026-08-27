"""Text-to-speech via Piper (free, offline, CPU-friendly). Both the
synthesizer and the audio player are injectable so unit tests never touch
a real model or audio device; scripts/smoke_tts.py exercises the real
thing end to end.

The injected synthesizer's contract is `synthesize(text) -> WAV bytes`.
Note that this is *not* piper's own API: `PiperVoice.synthesize()` (piper
1.7) returns an iterable of `AudioChunk` objects, not bytes, so
`PiperSynthesizer` below adapts it by rendering through
`PiperVoice.synthesize_wav()` into an in-memory WAV file.
"""

import io
import wave
from typing import Callable


class PiperSynthesizer:
    """Adapts a piper `PiperVoice` to `synthesize(text) -> WAV bytes`.

    `voice` is injectable purely so this adapter can be unit-tested; in
    normal use `TextToSpeech` builds one from `voice_path`.
    """

    def __init__(self, voice=None, voice_path: str = "models/piper/en_US.onnx"):
        if voice is None:
            from piper import PiperVoice

            voice = PiperVoice.load(voice_path)
        self._voice = voice

    def synthesize(self, text: str) -> bytes:
        buffer = io.BytesIO()
        with wave.open(buffer, "wb") as wav_file:
            self._voice.synthesize_wav(text, wav_file)
        return buffer.getvalue()


class TextToSpeech:
    def __init__(
        self,
        synthesizer=None,
        player: Callable[[bytes], None] | None = None,
        voice_path: str = "models/piper/en_US.onnx",
    ):
        if synthesizer is None:
            synthesizer = PiperSynthesizer(voice_path=voice_path)
        if player is None:
            player = _play_wav_bytes
        self._synthesizer = synthesizer
        self._player = player

    def synthesize(self, text: str) -> bytes:
        return self._synthesizer.synthesize(text)

    def speak(self, text: str) -> None:
        self._player(self.synthesize(text))


def _play_wav_bytes(audio: bytes) -> None:
    # `simpleaudio`'s native bindings segfaulted reliably on real Ubuntu
    # target hardware (confirmed during deployment testing); the system's
    # own `aplay` plays the same audio without issue, so we shell out to
    # it via a temp file instead of depending on a fragile compiled
    # extension. This call is expected to block until playback finishes
    # (matching simpleaudio's `.wait_done()`), since the caller already
    # runs it in a thread via `asyncio.to_thread`.
    import subprocess
    import tempfile

    with tempfile.NamedTemporaryFile(suffix=".wav") as tmp:
        tmp.write(audio)
        tmp.flush()
        subprocess.run(["aplay", "-q", tmp.name], check=False)
