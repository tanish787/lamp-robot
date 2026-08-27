"""The two live loops in brain/main.py, driven with fake devices.

The real camera/mic path is only exercised in the manual live pass (per
the spec's testing strategy), but the *wiring* — frame -> face monitor ->
debouncer -> engagement transition, and engaged -> mic -> VAD -> utterance
handler — is deterministic and is asserted here, so a regression in the
loop structure shows up without a human in front of the camera.
"""

import asyncio

import pytest

from brain.main import LatestFrame, _dialogue_loop, _engagement_loop


class FakeCamera:
    def __init__(self, frames):
        self._frames = list(frames)
        self.closed = False

    def read(self):
        if len(self._frames) > 1:
            return self._frames.pop(0)
        return self._frames[0] if self._frames else "IDLE_FRAME"

    def close(self):
        self.closed = True


class FakeFaceMonitor:
    """Reports a face for every frame whose name starts with 'FACE'."""

    def detect(self, frame):
        return str(frame).startswith("FACE")


class ImmediateDebouncer:
    """Flips on the first change, so a test doesn't have to sleep out a
    real hold window (the hold logic itself is tested in test_engagement)."""

    def __init__(self):
        self._engaged = False

    def update(self, face_detected, now):
        if face_detected == self._engaged:
            return None
        self._engaged = face_detected
        return face_detected


class RecordingOrchestrator:
    def __init__(self):
        self.engagement_changes = []
        self.observed = []
        self.utterances = []

    async def on_engagement_change(self, engaged):
        self.engagement_changes.append(engaged)

    async def observe_scene(self, frame, timestamp=None):
        self.observed.append(frame)
        return ["mug"]

    async def handle_utterance(self, audio):
        self.utterances.append(audio)
        return "heard"


async def _run_briefly(coro, seconds=0.6):
    task = asyncio.create_task(coro)
    await asyncio.sleep(seconds)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task


@pytest.mark.asyncio
async def test_engagement_loop_fires_a_transition_and_observes_the_scene():
    orchestrator = RecordingOrchestrator()
    camera = FakeCamera(["EMPTY", "FACE_1", "FACE_2"])
    engaged = asyncio.Event()
    latest = LatestFrame()

    await _run_briefly(_engagement_loop(
        orchestrator, FakeFaceMonitor(), ImmediateDebouncer(), camera, latest, engaged
    ))

    assert orchestrator.engagement_changes[0] is True
    assert engaged.is_set()
    # Moment 4: the scene is observed as soon as someone engages.
    assert orchestrator.observed[0] == "FACE_1"
    assert latest.get() is not None


@pytest.mark.asyncio
async def test_engagement_loop_survives_a_failing_frame_grab():
    class ExplodingCamera(FakeCamera):
        def __init__(self):
            super().__init__([])
            self.calls = 0

        def read(self):
            self.calls += 1
            if self.calls == 1:
                raise OSError("device fell over")
            return "FACE_after_recovery"

    orchestrator = RecordingOrchestrator()
    camera = ExplodingCamera()
    await _run_briefly(_engagement_loop(
        orchestrator, FakeFaceMonitor(), ImmediateDebouncer(), camera,
        LatestFrame(), asyncio.Event()
    ))
    assert camera.calls > 1
    assert orchestrator.engagement_changes == [True]


@pytest.mark.asyncio
async def test_dialogue_loop_waits_for_engagement_then_handles_an_utterance():
    class FakeMic:
        def __init__(self):
            self.calls = 0

        def read_frames(self, duration_s):
            self.calls += 1
            return [b"speech", b"speech"] + [b""] * 12

    class FakeVad:
        def is_speech(self, frame):
            return frame == b"speech"

    orchestrator = RecordingOrchestrator()
    engaged = asyncio.Event()
    mic = FakeMic()

    task = asyncio.create_task(_dialogue_loop(orchestrator, mic, FakeVad(), engaged))
    await asyncio.sleep(0.1)
    assert mic.calls == 0  # nothing recorded while disengaged

    engaged.set()
    await asyncio.sleep(0.2)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert orchestrator.utterances
    assert orchestrator.utterances[0] == b"speechspeech"


@pytest.mark.asyncio
async def test_dialogue_loop_keeps_listening_after_a_bad_chunk():
    class FlakyMic:
        def __init__(self):
            self.calls = 0

        def read_frames(self, duration_s):
            self.calls += 1
            if self.calls == 1:
                raise RuntimeError("audio underrun")
            return [b"speech"] + [b""] * 12

    class FakeVad:
        def is_speech(self, frame):
            return frame == b"speech"

    orchestrator = RecordingOrchestrator()
    engaged = asyncio.Event()
    engaged.set()
    mic = FlakyMic()

    # Longer than the loop's post-error backoff, so recovery is observable.
    await _run_briefly(_dialogue_loop(orchestrator, mic, FakeVad(), engaged), seconds=0.9)
    assert mic.calls > 1
    assert orchestrator.utterances
