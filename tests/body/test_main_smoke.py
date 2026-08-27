import asyncio

import pytest
import websockets

from body.light_sfx import LightState, MusicPlayer, SfxPlayer
from body.server import BodyServer
from body.simulation import LampSimulation
from shared.protocol_schema import decode_ack, encode_command


@pytest.mark.asyncio
async def test_full_stack_command_round_trip(tmp_path, unused_tcp_port):
    sim = LampSimulation(gui=False, cache_dir=tmp_path)
    light = LightState()
    sfx = SfxPlayer(tmp_path, player=lambda p: None)
    music = MusicPlayer(tmp_path, player=lambda p, loop: None)
    server = BodyServer(sim, light, sfx, music, port=unused_tcp_port)

    serve_task = asyncio.create_task(server.serve())
    await asyncio.sleep(0.2)  # let the server bind

    try:
        async with websockets.connect(f"ws://127.0.0.1:{unused_tcp_port}") as ws:
            await ws.send(encode_command(1, "curious_lean", {}))
            reply = await ws.recv()
            ack = decode_ack(reply)
            assert ack["status"] == "done"
            assert len(ack["pose"]) == 5
    finally:
        serve_task.cancel()
        sim.close()
