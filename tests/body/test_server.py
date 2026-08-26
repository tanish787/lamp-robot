import asyncio

import pytest

from shared.protocol_schema import decode_ack, encode_command
from body.server import BodyServer


class FakeSim:
    def apply_action(self, name, params):
        return [0.1, 0.0, 0.0, 0.0, 0.0]


class FakeLight:
    def __init__(self):
        self.state = None

    def set(self, state):
        self.state = state


class FakeSfx:
    def __init__(self):
        self.played = None

    def play(self, name):
        self.played = name


class FakeMusic:
    def play(self, track, loop=True):
        pass

    def stop(self):
        pass


@pytest.fixture
def server():
    return BodyServer(FakeSim(), FakeLight(), FakeSfx(), FakeMusic())


def test_handle_message_executes_motion_action(server):
    raw = encode_command(1, "look_at", {"direction": "left"})
    reply = asyncio.run(server.handle_message(raw))
    ack = decode_ack(reply)
    assert ack["id"] == 1
    assert ack["status"] == "done"
    assert ack["pose"] == [0.1, 0.0, 0.0, 0.0, 0.0]


def test_handle_message_executes_light_action(server):
    raw = encode_command(2, "set_light", {"state": "pulse"})
    asyncio.run(server.handle_message(raw))
    assert server._light.state == "pulse"


def test_handle_message_executes_sfx_action(server):
    raw = encode_command(3, "play_sfx", {"name": "chime"})
    asyncio.run(server.handle_message(raw))
    assert server._sfx.played == "chime"


def test_handle_message_rejects_unknown_action_without_crashing(server):
    raw = encode_command(4, "teleport", {})
    reply = asyncio.run(server.handle_message(raw))
    ack = decode_ack(reply)
    assert ack["status"] == "error"
    assert ack["id"] == 4


def test_handle_message_rejects_brain_local_action(server):
    raw = encode_command(5, "speak", {"text": "hi"})
    reply = asyncio.run(server.handle_message(raw))
    ack = decode_ack(reply)
    assert ack["status"] == "error"


def test_handle_message_rejects_malformed_json(server):
    reply = asyncio.run(server.handle_message("{not json"))
    ack = decode_ack(reply)
    assert ack["status"] == "error"
    assert ack["id"] is None
