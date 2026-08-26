"""WebSocket protocol server: the only entry point into Body. Validates and
dispatches every incoming command, and never trusts it blindly — an unknown
or malformed command gets an error ack, never a crash or a silent drop.
"""

import asyncio

import websockets

from shared.action_vocabulary import BODY_ACTIONS, is_valid_action
from shared.protocol_schema import ProtocolError, decode_command, encode_ack


class BodyServer:
    def __init__(self, sim, light, sfx, music, host: str = "127.0.0.1", port: int = 8765):
        self._sim = sim
        self._light = light
        self._sfx = sfx
        self._music = music
        self._host = host
        self._port = port

    async def handle_message(self, raw: str) -> str:
        try:
            command = decode_command(raw)
        except ProtocolError as exc:
            return encode_ack(None, "error", error=str(exc))

        id_, name, params = command["id"], command["cmd"], command["params"]

        if name not in BODY_ACTIONS or not is_valid_action(name, params):
            return encode_ack(id_, "error", error=f"invalid command: {name!r} {params!r}")

        try:
            if name == "set_light":
                self._light.set(params["state"])
                pose = None
            elif name == "play_sfx":
                self._sfx.play(params["name"])
                pose = None
            elif name == "play_music":
                if params["on"]:
                    self._music.play(params["track"])
                else:
                    self._music.stop()
                pose = None
            else:
                pose = self._sim.apply_action(name, params)
        except Exception as exc:  # noqa: BLE001 - never let a bad command crash Body
            return encode_ack(id_, "error", error=str(exc))

        return encode_ack(id_, "done", pose=pose)

    async def serve(self) -> None:
        async def _handler(websocket):
            async for raw in websocket:
                await websocket.send(await self.handle_message(raw))

        async with websockets.serve(_handler, self._host, self._port):
            await asyncio.Future()  # run forever
