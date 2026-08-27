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


# Regression: these four frames are all valid JSON but the wrong *shape*.
# Before the fix each one raised out of handle_message (TypeError from a
# `in`/subscript against a non-dict/non-str), which killed the whole
# WebSocket connection because Brain has no reconnect.
MALFORMED_BUT_JSON_VALID = [
    '{"id": 1, "cmd": "set_light", "params": null}',
    '{"id": 2, "cmd": "look_at", "params": "direction"}',
    '{"id": 3, "cmd": ["nod"], "params": {}}',
    '{"id": 4, "cmd": "set_light", "params": ["state"]}',
]


@pytest.mark.parametrize("raw", MALFORMED_BUT_JSON_VALID)
def test_handle_message_error_acks_malformed_but_json_valid_frames(server, raw):
    reply = asyncio.run(server.handle_message(raw))
    ack = decode_ack(reply)
    assert ack["status"] == "error"
    assert ack["error"]


def test_connection_survives_every_malformed_frame(server):
    """Each bad frame gets an error ack and the next good command still
    works, i.e. nothing escapes handle_message to close the socket."""
    async def scenario():
        for raw in MALFORMED_BUT_JSON_VALID:
            ack = decode_ack(await server.handle_message(raw))
            assert ack["status"] == "error"
        good = decode_ack(await server.handle_message(
            encode_command(99, "look_at", {"direction": "right"})
        ))
        return good

    ack = asyncio.run(scenario())
    assert ack["status"] == "done"
    assert ack["id"] == 99
