import pytest

from brain.audio_capture import MicStream, segment_utterance, split_frames


def test_split_frames_drops_a_trailing_partial_frame():
    """webrtcvad rejects a frame that is not exactly 10/20/30 ms, so a
    short final frame must never be handed to it."""
    frames = split_frames(b"x" * 2500, frame_bytes=960)
    assert [len(f) for f in frames] == [960, 960]


def test_split_frames_handles_an_exact_multiple():
    assert len(split_frames(b"x" * 1920, frame_bytes=960)) == 2


def test_split_frames_handles_audio_shorter_than_one_frame():
    assert split_frames(b"x" * 100, frame_bytes=960) == []


def test_split_frames_rejects_a_nonsense_frame_size():
    with pytest.raises(ValueError):
        split_frames(b"x" * 100, frame_bytes=0)


def test_mic_stream_frame_bytes_matches_a_30ms_16khz_frame():
    assert MicStream(sample_rate=16000, frame_ms=30).frame_bytes == 960


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
