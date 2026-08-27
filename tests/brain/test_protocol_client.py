import asyncio

import pytest

from shared.protocol_schema import decode_command, encode_ack
from brain.protocol_client import AckMismatchError, ProtocolClient


class FakeConnection:
    def __init__(self):
        self.sent = []

    async def send(self, raw):
        self.sent.append(raw)

    async def recv(self):
        command = decode_command(self.sent[-1])
        return encode_ack(command["id"], "done", pose=[0.0] * 5)


def test_send_command_encodes_and_returns_decoded_ack():
    fake = FakeConnection()
    client = ProtocolClient(connection=fake)
    ack = asyncio.run(client.send_command("nod", {}))
    assert ack["status"] == "done"
    assert ack["pose"] == [0.0] * 5
    sent_command = decode_command(fake.sent[0])
    assert sent_command["cmd"] == "nod"


def test_send_command_increments_id_each_call():
    fake = FakeConnection()
    client = ProtocolClient(connection=fake)
    asyncio.run(client.send_command("nod", {}))
    asyncio.run(client.send_command("shake", {}))
    first = decode_command(fake.sent[0])["id"]
    second = decode_command(fake.sent[1])["id"]
    assert second == first + 1


# ----------------------------------------------------------------------
# Concurrency (brain/main.py's engagement and dialogue loops both call
# send_command against the same connection)
# ----------------------------------------------------------------------

class ConcurrencyGuardError(RuntimeError):
    """Stands in for websockets' own ConcurrencyError, which real
    `websockets` connections raise when two coroutines call recv()
    concurrently on the same connection."""


class SerializingFakeConnection:
    """Behaves like a real single WebSocket connection: acks arrive in
    the same order commands were sent (FIFO), and calling recv() while
    another recv() is already in flight raises -- exactly like
    `websockets` does. Multiple `asyncio.sleep(0)` yield points inside
    send()/recv() stand in for real network latency, so an unlocked
    caller has every opportunity to interleave with another one."""

    def __init__(self):
        self.sent = []
        self._pending_acks = []
        self._recv_in_flight = False

    async def send(self, raw):
        self.sent.append(raw)
        command = decode_command(raw)
        await asyncio.sleep(0)
        self._pending_acks.append(encode_ack(command["id"], "done", pose=[0.0] * 5))

    async def recv(self):
        if self._recv_in_flight:
            raise ConcurrencyGuardError(
                "cannot call recv while another coroutine is already running recv"
            )
        self._recv_in_flight = True
        try:
            await asyncio.sleep(0)
            await asyncio.sleep(0)
            return self._pending_acks.pop(0)
        finally:
            self._recv_in_flight = False


def test_send_command_serializes_concurrent_callers():
    """Two coroutines racing send_command (as the engagement and dialogue
    loops in brain/main.py do) must not crash, and each must get back the
    ack for the command it actually sent -- never a stale or swapped one
    left over from the other caller."""
    fake = SerializingFakeConnection()
    client = ProtocolClient(connection=fake)

    async def scenario():
        return await asyncio.gather(
            client.send_command("nod", {}),
            client.send_command("shake", {}),
        )

    ack_a, ack_b = asyncio.run(scenario())

    sent_ids = [decode_command(raw)["id"] for raw in fake.sent]
    assert len(sent_ids) == 2
    assert ack_a["id"] == sent_ids[0]
    assert ack_b["id"] == sent_ids[1]
    assert ack_a["id"] != ack_b["id"]


def test_send_command_raises_on_mismatched_ack_id():
    """If an ack ever arrives whose id doesn't match the command that was
    sent, that must be a loud failure, not a silently accepted mismatch
    (which is exactly how the permanent desync went unnoticed)."""

    class MismatchedConnection:
        async def send(self, raw):
            pass

        async def recv(self):
            return encode_ack(999, "done", pose=[0.0] * 5)

    client = ProtocolClient(connection=MismatchedConnection())
    with pytest.raises(AckMismatchError):
        asyncio.run(client.send_command("nod", {}))
