import asyncio

from brain.memory import SceneMemory
from brain.orchestrator import Orchestrator


class FakeTts:
    def __init__(self):
        self.spoken = []

    def speak(self, text):
        self.spoken.append(text)


class FakeProtocolClient:
    def __init__(self):
        self.sent = []

    async def send_command(self, name, params):
        self.sent.append((name, params))
        return {"status": "done", "pose": [0.0] * 5}


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
