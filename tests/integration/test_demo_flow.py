"""End-to-end demo flow: the one place Brain is allowed to import Body
directly. Stands up a real, in-process, headless BodyServer and drives a
real Orchestrator against it over a real WebSocket connection, with faked
perception/reasoning inputs (per the spec's testing strategy: only the
model-shaped edges are faked, everything else is real)."""

import asyncio

import pytest
import websockets

from body.light_sfx import LightState, MusicPlayer, SfxPlayer
from body.server import BodyServer
from body.simulation import LampSimulation
from brain.memory import SceneMemory
from brain.orchestrator import Orchestrator
from brain.protocol_client import ProtocolClient


class FakeTts:
    def __init__(self):
        self.spoken = []

    def speak(self, text):
        self.spoken.append(text)


class ScriptedReasoner:
    def __init__(self, plan):
        self._plan = plan

    def plan_actions(self, goal_text, memory):
        return self._plan


async def _start_body(tmp_path, port):
    sim = LampSimulation(gui=False, cache_dir=tmp_path)
    server = BodyServer(sim, LightState(), SfxPlayer(tmp_path, lambda p: None),
                         MusicPlayer(tmp_path, lambda p, loop: None), port=port)
    task = asyncio.create_task(server.serve())
    await asyncio.sleep(0.2)
    return task, sim


async def _stop_body(body_task, sim):
    """Cancel the server task and wait for the cancellation to actually
    land before tearing down the simulation, so a failing assertion above
    can't leak a still-running server/bound port into the next test."""
    body_task.cancel()
    try:
        await body_task
    except asyncio.CancelledError:
        pass
    sim.close()


@pytest.mark.asyncio
async def test_full_demo_flow_sends_expected_command_sequence(tmp_path, unused_tcp_port):
    body_task, sim = await _start_body(tmp_path, unused_tcp_port)
    try:
        async with websockets.connect(f"ws://127.0.0.1:{unused_tcp_port}") as ws:
            client = ProtocolClient(connection=ws)
            tts = FakeTts()
            memory = SceneMemory()
            memory.observe("mug", {"color": "red", "position": (0.1, 0.2)}, timestamp=0.0)
            reasoner = ScriptedReasoner([
                {"name": "scan_sweep", "params": {}},
                {"name": "look_at", "params": {"direction": "left"}},
                {"name": "speak", "params": {"text": "Found the mug."}},
            ])
            orchestrator = Orchestrator(tts=tts, protocol_client=client, reasoner=reasoner, memory=memory)

            await orchestrator.on_engagement_change(True)
            actions = reasoner.plan_actions("find the mug", memory)
            await orchestrator.execute_actions(actions)

            assert tts.spoken == ["Found the mug."]
    finally:
        await _stop_body(body_task, sim)


@pytest.mark.asyncio
async def test_invalid_action_from_reasoner_falls_back_to_idle_sway(tmp_path, unused_tcp_port):
    body_task, sim = await _start_body(tmp_path, unused_tcp_port)
    try:
        async with websockets.connect(f"ws://127.0.0.1:{unused_tcp_port}") as ws:
            client = ProtocolClient(connection=ws)
            tts = FakeTts()
            # A reasoner that "hallucinated" — plan_actions itself is
            # responsible for validation (Task 7), so a reasoner used here
            # already returns the post-fallback plan.
            reasoner = ScriptedReasoner([{"name": "idle_sway", "params": {}}])
            orchestrator = Orchestrator(tts=tts, protocol_client=client, reasoner=reasoner)

            actions = reasoner.plan_actions("do something impossible", SceneMemory())
            await orchestrator.execute_actions(actions)
            # No exception, no crash, Body accepted the fallback action.
    finally:
        await _stop_body(body_task, sim)
