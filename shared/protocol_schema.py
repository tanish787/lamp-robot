"""Wire format for the local Brain <-> Body WebSocket protocol.

Command (Brain -> Body): {"id": int, "cmd": str, "params": dict}
Ack     (Body -> Brain): {"id": int, "status": "done"|"error", "pose": list|None, "error": str|None}
"""

import json


class ProtocolError(ValueError):
    """A message did not conform to the wire schema."""


def encode_command(id_: int, cmd: str, params: dict) -> str:
    return json.dumps({"id": id_, "cmd": cmd, "params": params})


def decode_command(raw: str) -> dict:
    data = _parse_json(raw)
    if "id" not in data or "cmd" not in data:
        raise ProtocolError("command missing required fields 'id'/'cmd'")
    data.setdefault("params", {})
    return data


def encode_ack(id_: int | None, status: str, pose: list | None = None, error: str | None = None) -> str:
    return json.dumps({"id": id_, "status": status, "pose": pose, "error": error})


def decode_ack(raw: str) -> dict:
    data = _parse_json(raw)
    if "id" not in data or "status" not in data:
        raise ProtocolError("ack missing required fields 'id'/'status'")
    data.setdefault("pose", None)
    data.setdefault("error", None)
    return data


def _parse_json(raw: str) -> dict:
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ProtocolError(f"invalid JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise ProtocolError("message must be a JSON object")
    return data
