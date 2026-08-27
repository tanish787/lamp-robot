from brain.tts import TextToSpeech


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
