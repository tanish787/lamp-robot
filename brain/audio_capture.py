"""Voice-activity segmentation: pure logic over a stream of fixed-size audio
frames, independent of the VAD implementation and the audio device — both
are injected so this is fully unit-testable without real audio."""

from typing import Callable


def segment_utterance(
    frames: list[bytes],
    is_speech: Callable[[bytes], bool],
    silence_hangover_frames: int = 10,
) -> bytes | None:
    """Return the concatenated speech frames of the first utterance found,
    or None if no complete utterance (speech followed by a hangover of
    silence) appears in `frames`."""
    speech_frames: list[bytes] = []
    trailing_silence = 0
    started = False

    for frame in frames:
        if is_speech(frame):
            started = True
            trailing_silence = 0
            speech_frames.append(frame)
        elif started:
            trailing_silence += 1
            if trailing_silence >= silence_hangover_frames:
                return b"".join(speech_frames)

    return None


class Vad:
    """Thin wrapper around webrtcvad, exercised by the smoke script."""

    def __init__(self, sample_rate: int = 16000, aggressiveness: int = 2):
        import webrtcvad

        self._vad = webrtcvad.Vad(aggressiveness)
        self._sample_rate = sample_rate

    def is_speech(self, frame: bytes) -> bool:
        return self._vad.is_speech(frame, self._sample_rate)


def split_frames(raw: bytes, frame_bytes: int) -> list[bytes]:
    """Cut raw PCM into fixed-size frames, discarding a trailing partial
    frame. webrtcvad only accepts exactly 10/20/30 ms of audio and raises
    on anything else, so a short final frame would fail the whole
    listening window rather than just being ignored."""
    if frame_bytes < 1:
        raise ValueError("frame_bytes must be >= 1")
    usable = len(raw) - (len(raw) % frame_bytes)
    return [raw[i : i + frame_bytes] for i in range(0, usable, frame_bytes)]


class MicStream:
    """Thin wrapper around sounddevice, exercised against a real
    microphone in the manual live pass."""

    def __init__(self, sample_rate: int = 16000, frame_ms: int = 30):
        self._sample_rate = sample_rate
        self._frame_samples = int(sample_rate * frame_ms / 1000)

    @property
    def frame_bytes(self) -> int:
        return self._frame_samples * 2  # int16 = 2 bytes/sample

    def read_frames(self, duration_s: float) -> list[bytes]:
        import sounddevice as sd

        recording = sd.rec(
            int(duration_s * self._sample_rate),
            samplerate=self._sample_rate,
            channels=1,
            dtype="int16",
        )
        sd.wait()
        return split_frames(recording.tobytes(), self.frame_bytes)
