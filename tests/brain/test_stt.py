import pytest

from brain.stt import TARGET_SAMPLE_RATE, SpeechToText


class FakeWhisperModel:
    def __init__(self):
        self.received = None

    def transcribe(self, audio_array, **kwargs):
        self.received = audio_array
        segment = type("Segment", (), {"text": " hello there "})()
        return [segment], None


def test_transcribe_joins_segment_text_and_strips_whitespace():
    fake_model = FakeWhisperModel()
    stt = SpeechToText(model=fake_model)
    result = stt.transcribe(b"\x00\x01" * 8000, sample_rate=16000)
    assert result == "hello there"
    assert fake_model.received is not None


def test_transcribe_joins_multiple_segments():
    class MultiSegmentModel:
        def transcribe(self, audio_array, **kwargs):
            seg1 = type("S", (), {"text": "hello"})()
            seg2 = type("S", (), {"text": " world"})()
            return [seg1, seg2], None

    stt = SpeechToText(model=MultiSegmentModel())
    assert stt.transcribe(b"\x00\x01" * 8000) == "hello world"


def test_transcribe_passes_16khz_audio_through_untouched():
    fake_model = FakeWhisperModel()
    SpeechToText(model=fake_model).transcribe(b"\x00\x01" * 8000, sample_rate=16000)
    assert len(fake_model.received) == 8000


def test_transcribe_resamples_audio_that_is_not_16khz():
    """scripts/smoke_stt.py passes a WAV's real framerate; a 44.1 kHz clip
    used to be handed to whisper unchanged and mis-transcribed."""
    fake_model = FakeWhisperModel()
    stt = SpeechToText(model=fake_model)
    stt.transcribe(b"\x00\x01" * 44100, sample_rate=44100)
    expected = round(44100 * TARGET_SAMPLE_RATE / 44100)
    assert len(fake_model.received) == expected == 44100 * 16000 // 44100

    fake_model = FakeWhisperModel()
    SpeechToText(model=fake_model).transcribe(b"\x00\x01" * 8000, sample_rate=32000)
    assert len(fake_model.received) == 4000


def test_transcribe_rejects_a_nonsense_sample_rate():
    with pytest.raises(ValueError):
        SpeechToText(model=FakeWhisperModel()).transcribe(b"\x00\x01" * 10, sample_rate=0)
