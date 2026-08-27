import asyncio
import logging

import pytest

from brain.memory import SceneMemory
from brain.orchestrator import Orchestrator, looks_like_goal


class FakeTts:
    def __init__(self):
        self.spoken = []

    def speak(self, text):
        self.spoken.append(text)


class FakeProtocolClient:
    def __init__(self, errors=()):
        self.sent = []
        self._errors = set(errors)

    async def send_command(self, name, params):
        self.sent.append((name, params))
        if name in self._errors:
            return {"id": 1, "status": "error", "pose": None, "error": f"no such thing: {name}"}
        return {"id": 1, "status": "done", "pose": [0.0] * 5, "error": None}


class FakeStt:
    def __init__(self, text):
        self.text = text
        self.received = None

    def transcribe(self, audio, sample_rate=16000):
        self.received = audio
        return self.text


class FakeReasoner:
    def __init__(self, plan=None, reply_text="sure"):
        self.plan = plan or [{"name": "idle_sway", "params": {}}]
        self.reply_text = reply_text
        self.replied = []
        self.planned = []

    def reply(self, user_text, memory):
        self.replied.append(user_text)
        return self.reply_text

    def plan_actions(self, goal_text, memory):
        self.planned.append(goal_text)
        return self.plan


class FakePerception:
    def __init__(self, labels=("mug",)):
        self.labels = list(labels)
        self.frames = []

    def observe(self, frame, memory, timestamp):
        self.frames.append(frame)
        for label in self.labels:
            memory.observe(label, {"position": (0.1, 0.2)}, timestamp)
        return self.labels


def test_execute_actions_routes_speak_to_local_tts_not_body():
    tts, client = FakeTts(), FakeProtocolClient()
    orchestrator = Orchestrator(tts=tts, protocol_client=client)
    asyncio.run(orchestrator.execute_actions([{"name": "speak", "params": {"text": "hello"}}]))
    assert tts.spoken == ["hello"]
    assert client.sent == []


def test_execute_actions_forwards_body_actions_to_protocol_client():
    tts, client = FakeTts(), FakeProtocolClient()
    orchestrator = Orchestrator(tts=tts, protocol_client=client)
    asyncio.run(orchestrator.execute_actions([{"name": "nod", "params": {}}]))
    assert client.sent == [("nod", {})]
    assert tts.spoken == []


def test_execute_actions_runs_a_mixed_sequence_in_order():
    tts, client = FakeTts(), FakeProtocolClient()
    orchestrator = Orchestrator(tts=tts, protocol_client=client)
    asyncio.run(orchestrator.execute_actions([
        {"name": "scan_sweep", "params": {}},
        {"name": "speak", "params": {"text": "found it"}},
        {"name": "look_at", "params": {"direction": "left"}},
    ]))
    assert client.sent == [("scan_sweep", {}), ("look_at", {"direction": "left"})]
    assert tts.spoken == ["found it"]


def test_on_engagement_change_true_plays_fixed_greeting():
    tts, client = FakeTts(), FakeProtocolClient()
    orchestrator = Orchestrator(tts=tts, protocol_client=client)
    asyncio.run(orchestrator.on_engagement_change(True))
    sent_names = [name for name, _ in client.sent]
    assert "curious_lean" in sent_names
    assert any(name == "set_light" for name in sent_names)
    assert any(name == "play_sfx" for name in sent_names)


def test_on_engagement_change_false_returns_to_idle():
    tts, client = FakeTts(), FakeProtocolClient()
    orchestrator = Orchestrator(tts=tts, protocol_client=client)
    asyncio.run(orchestrator.on_engagement_change(False))
    sent_names = [name for name, _ in client.sent]
    assert "idle_sway" in sent_names
    assert any(name == "set_light" for name in sent_names)


def test_on_engagement_change_false_resets_pose_before_idling():
    """Otherwise a prior curious_lean is never undone and the disengaged
    pose looks identical to the engaged one."""
    tts, client = FakeTts(), FakeProtocolClient()
    orchestrator = Orchestrator(tts=tts, protocol_client=client)
    asyncio.run(orchestrator.on_engagement_change(False))
    sent_names = [name for name, _ in client.sent]
    assert sent_names.index("neutral") < sent_names.index("idle_sway")


# ----------------------------------------------------------------------
# Ack handling (I1)
# ----------------------------------------------------------------------

def test_execute_actions_surfaces_an_error_ack_and_keeps_going(caplog):
    tts, client = FakeTts(), FakeProtocolClient(errors={"nod"})
    orchestrator = Orchestrator(tts=tts, protocol_client=client)
    with caplog.at_level(logging.ERROR, logger="brain.orchestrator"):
        results = asyncio.run(orchestrator.execute_actions([
            {"name": "nod", "params": {}},
            {"name": "set_light", "params": {"state": "dim"}},
        ]))
    assert [r["status"] for r in results] == ["error", "done"]
    assert results[0]["error"] == "no such thing: nod"
    assert "no such thing: nod" in caplog.text
    # The failing step did not abort the sequence.
    assert [name for name, _ in client.sent] == ["nod", "set_light"]


def test_execute_actions_reports_done_for_successful_acks():
    tts, client = FakeTts(), FakeProtocolClient()
    orchestrator = Orchestrator(tts=tts, protocol_client=client)
    results = asyncio.run(orchestrator.execute_actions([{"name": "nod", "params": {}}]))
    assert results == [{"name": "nod", "status": "done"}]


# ----------------------------------------------------------------------
# Utterance handling (C2)
# ----------------------------------------------------------------------

@pytest.mark.parametrize("text", [
    "find the mug", "look at the window", "scan the room", "point at the plant",
    "show me the desk",
])
def test_looks_like_goal_accepts_instructions(text):
    assert looks_like_goal(text)


@pytest.mark.parametrize("text", [
    "what did you see?", "where is the mug", "hello there", "is that a mug",
    "how are you", "", "   ",
])
def test_looks_like_goal_rejects_questions_and_chatter(text):
    assert not looks_like_goal(text)


def _dialogue_orchestrator(transcript, plan=None, reply_text="sure", perception=None,
                           frame=None, client=None):
    tts = FakeTts()
    client = client or FakeProtocolClient()
    reasoner = FakeReasoner(plan=plan, reply_text=reply_text)
    orchestrator = Orchestrator(
        tts=tts,
        protocol_client=client,
        stt=FakeStt(transcript),
        reasoner=reasoner,
        memory=SceneMemory(),
        perception=perception,
        frame_source=(lambda: frame) if frame is not None else None,
    )
    return orchestrator, tts, client, reasoner


def test_handle_utterance_replies_to_a_question_and_does_not_plan():
    orchestrator, tts, client, reasoner = _dialogue_orchestrator(
        "what is on the desk?", reply_text="A red mug."
    )
    heard = asyncio.run(orchestrator.handle_utterance(b"audio"))
    assert heard == "what is on the desk?"
    assert reasoner.replied == ["what is on the desk?"]
    assert reasoner.planned == []
    assert tts.spoken == ["A red mug."]


def test_handle_utterance_plans_and_executes_a_spoken_goal():
    orchestrator, tts, client, reasoner = _dialogue_orchestrator(
        "find the mug",
        plan=[
            {"name": "look_at", "params": {"direction": "left"}},
            {"name": "speak", "params": {"text": "There it is."}},
        ],
    )
    asyncio.run(orchestrator.handle_utterance(b"audio"))
    assert reasoner.planned == ["find the mug"]
    assert reasoner.replied == []
    assert [name for name, _ in client.sent] == ["look_at"]
    assert tts.spoken == ["There it is."]


def test_handle_utterance_reobserves_and_confirms_after_a_scan_sweep():
    """Spec moment 5: the goal plan sweeps, then the scene is observed
    again before the character confirms."""
    perception = FakePerception(labels=("mug",))
    orchestrator, tts, client, reasoner = _dialogue_orchestrator(
        "find the mug",
        plan=[{"name": "scan_sweep", "params": {}}],
        reply_text="Found the mug.",
        perception=perception,
        frame="FRAME",
    )
    asyncio.run(orchestrator.handle_utterance(b"audio"))
    assert perception.frames == ["FRAME"]
    assert reasoner.replied == ["find the mug"]
    assert tts.spoken == ["Found the mug."]


def test_handle_utterance_ignores_an_empty_transcript():
    orchestrator, tts, client, reasoner = _dialogue_orchestrator("   ")
    assert asyncio.run(orchestrator.handle_utterance(b"")) == ""
    assert tts.spoken == []
    assert client.sent == []
    assert reasoner.replied == []


def test_handle_utterance_is_a_no_op_without_stt_or_reasoner():
    orchestrator = Orchestrator(tts=FakeTts(), protocol_client=FakeProtocolClient())
    assert asyncio.run(orchestrator.handle_utterance(b"audio")) == ""


# ----------------------------------------------------------------------
# Scene perception (C2)
# ----------------------------------------------------------------------

def test_observe_scene_writes_detections_into_memory():
    memory = SceneMemory()
    perception = FakePerception(labels=("mug", "plant"))
    orchestrator = Orchestrator(
        tts=FakeTts(), protocol_client=FakeProtocolClient(),
        perception=perception, memory=memory,
    )
    labels = asyncio.run(orchestrator.observe_scene("FRAME", timestamp=1.0))
    assert labels == ["mug", "plant"]
    assert {r["label"] for r in memory.records()} == {"mug", "plant"}


def test_observe_scene_is_a_no_op_without_a_frame_or_perception():
    orchestrator = Orchestrator(
        tts=FakeTts(), protocol_client=FakeProtocolClient(),
        perception=FakePerception(), memory=SceneMemory(),
    )
    assert asyncio.run(orchestrator.observe_scene(None)) == []
    bare = Orchestrator(tts=FakeTts(), protocol_client=FakeProtocolClient())
    assert asyncio.run(bare.observe_scene("FRAME")) == []
    assert asyncio.run(bare.observe_current_frame()) == []


# ----------------------------------------------------------------------
# Concurrency (I5)
# ----------------------------------------------------------------------

def test_speak_with_does_not_block_the_event_loop():
    """TTS is blocking; if it ran inline nothing else could make progress
    while the character talks."""
    import threading
    import time as _time

    class BlockingTts:
        def __init__(self):
            self.done = threading.Event()

        def speak(self, text):
            _time.sleep(0.2)
            self.done.set()

    tts, client = BlockingTts(), FakeProtocolClient()
    orchestrator = Orchestrator(tts=tts, protocol_client=client)

    async def scenario():
        speaking = asyncio.create_task(orchestrator.speak_with("hello", [
            {"name": "nod", "params": {}},
        ]))
        # The Body action lands while speech is still playing.
        while not client.sent:
            await asyncio.sleep(0.001)
        overlapped = not tts.done.is_set()
        await speaking
        return overlapped

    assert asyncio.run(scenario()) is True
    assert client.sent == [("nod", {})]


def test_speak_with_awaits_the_speaking_task_even_if_execute_actions_raises():
    """If execute_actions blows up mid-sequence, speak_with must still
    await the in-flight speaking task rather than leaving the TTS thread
    running orphaned and unawaited."""

    class RaisingProtocolClient:
        async def send_command(self, name, params):
            raise RuntimeError("body rejected the command")

    tts = FakeTts()
    orchestrator = Orchestrator(tts=tts, protocol_client=RaisingProtocolClient())

    async def scenario():
        with pytest.raises(RuntimeError):
            await orchestrator.speak_with("hello", [{"name": "nod", "params": {}}])
        # Check this *inside* the still-running event loop, immediately
        # after speak_with re-raised. asyncio.run() force-finishes any
        # orphaned tasks during its own teardown, which would mask the
        # bug if this assertion ran after asyncio.run() returned instead.
        assert tts.spoken == ["hello"]

    asyncio.run(scenario())
