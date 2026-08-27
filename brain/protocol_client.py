"""Brain's WebSocket client to Body. `connection` is injectable (must
provide async send(str) and async recv() -> str) so unit tests never open
a real socket; brain/main.py wires a real `websockets` connection."""

from shared.protocol_schema import decode_ack, encode_command


class ProtocolClient:
    def __init__(self, uri: str = "ws://127.0.0.1:8765", connection=None):
        self._uri = uri
        self._connection = connection
        self._next_id = 1

    async def connect(self) -> None:
        if self._connection is None:
            import websockets

            self._connection = await websockets.connect(self._uri)

    async def send_command(self, name: str, params: dict) -> dict:
        command_id = self._next_id
        self._next_id += 1
        await self._connection.send(encode_command(command_id, name, params))
        return decode_ack(await self._connection.recv())

    async def close(self) -> None:
        if self._connection is not None:
            await self._connection.close()
