"""Brain's WebSocket client to Body. `connection` is injectable (must
provide async send(str) and async recv() -> str) so unit tests never open
a real socket; brain/main.py wires a real `websockets` connection.

`send_command` is called concurrently by brain/main.py's engagement and
dialogue loops (they run together in a TaskGroup). A single WebSocket
connection can only have one `recv()` in flight at a time — a second
concurrent `recv()` raises in `websockets`, and worse, if one caller's
`recv()` is ever the one to pick up another caller's ack, every ack from
then on is off by one, permanently, for the rest of the process's life.
`_lock` serialises the whole send+recv round trip per command so only one
command is ever outstanding on the wire at a time, and the id check below
turns any future violation of that invariant into a loud error instead of
a silent, permanent desync."""

import asyncio

from shared.protocol_schema import decode_ack, encode_command


class AckMismatchError(RuntimeError):
    """Raised when the ack received does not match the command sent.

    Should be unreachable in practice now that `_lock` serialises the
    round trip, but the check stays in as defense in depth: a future
    change that weakens the locking should fail loudly here rather than
    silently misattributing one command's result to another."""


class ProtocolClient:
    def __init__(self, uri: str = "ws://127.0.0.1:8765", connection=None):
        self._uri = uri
        self._connection = connection
        self._next_id = 1
        self._lock = asyncio.Lock()

    async def connect(self) -> None:
        if self._connection is None:
            import websockets

            self._connection = await websockets.connect(self._uri)

    async def send_command(self, name: str, params: dict) -> dict:
        command_id = self._next_id
        self._next_id += 1
        # Hold the lock across the full send+recv round trip: only one
        # command may be outstanding on the connection at a time, so a
        # concurrent caller's recv() can never pick up this command's ack
        # (or vice versa).
        async with self._lock:
            await self._connection.send(encode_command(command_id, name, params))
            ack = decode_ack(await self._connection.recv())
        if ack.get("id") != command_id:
            raise AckMismatchError(
                f"ack id mismatch: sent command id={command_id}, got ack id={ack.get('id')!r}"
            )
        return ack

    async def close(self) -> None:
        if self._connection is not None:
            await self._connection.close()
