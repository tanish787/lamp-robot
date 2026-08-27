from brain.audio_capture import segment_utterance


def test_returns_none_if_no_speech_detected():
    frames = [b"silence"] * 5
    assert segment_utterance(frames, is_speech=lambda f: False) is None


def test_captures_speech_and_stops_after_hangover():
    # speech, speech, silence x3 (>= hangover of 2) -> utterance ends
    frames = [b"s1", b"s2", b"q1", b"q2", b"q3", b"s3"]
    is_speech = lambda f: f.startswith(b"s")
    result = segment_utterance(frames, is_speech=is_speech, silence_hangover_frames=2)
    assert result == b"s1s2"


def test_ignores_leading_silence_before_speech_starts():
    frames = [b"q1", b"q2", b"s1", b"s2", b"q3", b"q4"]
    is_speech = lambda f: f.startswith(b"s")
    result = segment_utterance(frames, is_speech=is_speech, silence_hangover_frames=2)
    assert result == b"s1s2"


def test_returns_none_if_speech_never_hits_hangover_before_frames_end():
    frames = [b"s1", b"s2", b"q1"]
    is_speech = lambda f: f.startswith(b"s")
    assert segment_utterance(frames, is_speech=is_speech, silence_hangover_frames=5) is None
