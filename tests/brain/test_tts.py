import io
import wave

from brain.tts import PiperSynthesizer, TextToSpeech


class FakeSynthesizer:
    def __init__(self):
        self.received_text = None

    def synthesize(self, text):
        self.received_text = text
        return b"FAKEWAVDATA"


def test_synthesize_returns_synthesizer_output():
    fake = FakeSynthesizer()
    tts = TextToSpeech(synthesizer=fake, player=lambda audio: None)
    audio = tts.synthesize("hello")
    assert audio == b"FAKEWAVDATA"
    assert fake.received_text == "hello"


def test_speak_synthesizes_and_plays():
    played = []
    fake = FakeSynthesizer()
    tts = TextToSpeech(synthesizer=fake, player=lambda audio: played.append(audio))
    tts.speak("hi there")
    assert played == [b"FAKEWAVDATA"]


class FakePiperVoice:
    """Mimics piper 1.7's PiperVoice.synthesize_wav, which writes into a
    caller-supplied wave.Wave_write rather than returning bytes."""

    def __init__(self):
        self.received_text = None

    def synthesize_wav(self, text, wav_file, **kwargs):
        self.received_text = text
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(22050)
        wav_file.writeframes(b"\x00\x01" * 100)


def test_piper_synthesizer_renders_a_real_wav_container():
    """Regression: the wrapper assumed PiperVoice.synthesize(text) returned
    bytes. It returns an iterable of AudioChunk, so the player was handed
    a generator and could never play anything."""
    voice = FakePiperVoice()
    audio = PiperSynthesizer(voice=voice).synthesize("hello")

    assert voice.received_text == "hello"
    assert audio.startswith(b"RIFF")
    with wave.open(io.BytesIO(audio), "rb") as wav:
        assert wav.getnchannels() == 1
        assert wav.getframerate() == 22050
        assert wav.getnframes() == 100
