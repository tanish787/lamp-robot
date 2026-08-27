import pytest

from shared.protocol_schema import (
    ProtocolError, encode_command, decode_command, encode_ack, decode_ack,
)


def test_command_round_trips():
    raw = encode_command(1, "look_at", {"direction": "left"})
    decoded = decode_command(raw)
    assert decoded == {"id": 1, "cmd": "look_at", "params": {"direction": "left"}}


def test_decode_command_defaults_missing_params_to_empty_dict():
    raw = '{"id": 2, "cmd": "idle_sway"}'
    decoded = decode_command(raw)
    assert decoded["params"] == {}


def test_decode_command_rejects_malformed_json():
    with pytest.raises(ProtocolError):
        decode_command("{not json")


def test_decode_command_rejects_missing_required_fields():
    with pytest.raises(ProtocolError):
        decode_command('{"params": {}}')


@pytest.mark.parametrize("raw", [
    '{"id": 1, "cmd": "set_light", "params": null}',
    '{"id": 2, "cmd": "look_at", "params": "direction"}',
    '{"id": 4, "cmd": "set_light", "params": ["state"]}',
])
def test_decode_command_rejects_non_object_params(raw):
    with pytest.raises(ProtocolError):
        decode_command(raw)


@pytest.mark.parametrize("raw", [
    '{"id": 3, "cmd": ["nod"], "params": {}}',
    '{"id": 3, "cmd": 7, "params": {}}',
    '{"id": 3, "cmd": null, "params": {}}',
])
def test_decode_command_rejects_non_string_cmd(raw):
    with pytest.raises(ProtocolError):
        decode_command(raw)


def test_ack_round_trips():
    raw = encode_ack(1, "done", pose=[0.0, 0.1, 0.0, 0.0, 0.0])
    decoded = decode_ack(raw)
    assert decoded == {"id": 1, "status": "done", "pose": [0.0, 0.1, 0.0, 0.0, 0.0], "error": None}


def test_decode_ack_rejects_missing_required_fields():
    with pytest.raises(ProtocolError):
        decode_ack('{"id": 1}')
