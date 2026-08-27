import asyncio

from shared.protocol_schema import decode_command, encode_ack
from brain.protocol_client import ProtocolClient


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
