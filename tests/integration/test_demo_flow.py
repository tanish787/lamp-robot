"""End-to-end demo flow: the one place Brain is allowed to import Body
directly. Stands up a real, in-process, headless BodyServer and drives a
real Orchestrator against it over a real WebSocket connection, with faked
perception/reasoning inputs (per the spec's testing strategy: only the
model-shaped edges are faked, everything else is real)."""

import asyncio

import pytest
import websockets

from body.light_sfx import LightState, MusicPlayer, SfxPlayer
from body.motion import NEUTRAL
from body.server import BodyServer
from body.simulation import LampSimulation
from brain.memory import SceneMemory
from brain.orchestrator import Orchestrator
from brain.protocol_client import ProtocolClient
from brain.reasoning import Reasoner


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

    def reply(self, user_text, memory):
        return "Done."


async def _start_body(tmp_path, port):
    sim = LampSimulation(gui=False, cache_dir=tmp_path)
    sfx_dir = tmp_path / "sfx"
    sfx_dir.mkdir()
    (sfx_dir / "chime.wav").write_bytes(b"RIFF....WAVEfmt ")
    server = BodyServer(sim, LightState(), SfxPlayer(sfx_dir, lambda p: None),
                         MusicPlayer(tmp_path, lambda p, loop: None), port=port)
    task = asyncio.create_task(server.serve())
    await asyncio.sleep(0.2)
    return task, sim, server


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
    body_task, sim, server = await _start_body(tmp_path, unused_tcp_port)
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
            orchestrator = Orchestrator(
                tts=tts, protocol_client=client, reasoner=reasoner, memory=memory
            )

            engage_results = await orchestrator.on_engagement_change(True)
            actions = reasoner.plan_actions("find the mug", memory)
            goal_results = await orchestrator.execute_actions(actions)

            # Every command Body actually received was accepted.
            assert [r["status"] for r in engage_results] == ["done"] * 3
            assert [r["status"] for r in goal_results] == ["done"] * 3
            assert [r["name"] for r in engage_results] == [
                "curious_lean", "set_light", "play_sfx",
            ]

            # Body-side state really changed, not just Brain-side bookkeeping.
            assert server._light.get() == "pulse"
            pose = sim.get_pose()
            assert pose != pytest.approx([NEUTRAL[i] for i in sorted(NEUTRAL)])
            # look_at("left") was the last motion, so base_yaw ended negative
            # and the earlier scan_sweep did not leave the head parked at a limit.
            assert pose[0] < -0.5

            assert tts.spoken == ["Found the mug."]

            # Disengaging visibly resets the pose.
            await orchestrator.on_engagement_change(False)
            assert server._light.get() == "dim"
            assert sim.get_pose() == pytest.approx([0.0] * 5, abs=1e-6)
    finally:
        await _stop_body(body_task, sim)


@pytest.mark.asyncio
async def test_invalid_action_from_reasoner_falls_back_to_idle_sway(tmp_path, unused_tcp_port):
    """The real Reasoner is given an LLM response naming an action that
    does not exist; validation must drop it and fall back, and the fallback
    must be something Body actually accepts."""
    body_task, sim, server = await _start_body(tmp_path, unused_tcp_port)
    try:
        async with websockets.connect(f"ws://127.0.0.1:{unused_tcp_port}") as ws:
            client = ProtocolClient(connection=ws)
            tts = FakeTts()
            memory = SceneMemory()

            hallucinated = '[{"name": "teleport", "params": {}}]'
            reasoner = Reasoner(llm_call=lambda prompt: hallucinated)
            actions = reasoner.plan_actions("do something impossible", memory)
            assert actions == [{"name": "idle_sway", "params": {}}]

            orchestrator = Orchestrator(tts=tts, protocol_client=client, memory=memory)
            results = await orchestrator.execute_actions(actions)
            assert results == [{"name": "idle_sway", "status": "done"}]
    finally:
        await _stop_body(body_task, sim)


@pytest.mark.asyncio
async def test_reasoner_plan_with_an_out_of_range_param_is_dropped(tmp_path, unused_tcp_port):
    """A plausible-looking but invalid parameter (a direction and an sfx
    name outside the vocabulary) must not reach Body."""
    body_task, sim, server = await _start_body(tmp_path, unused_tcp_port)
    try:
        async with websockets.connect(f"ws://127.0.0.1:{unused_tcp_port}") as ws:
            client = ProtocolClient(connection=ws)
            raw = (
                '[{"name": "look_at", "params": {"direction": "sideways"}},'
                ' {"name": "play_sfx", "params": {"name": "../../etc/passwd"}},'
                ' {"name": "nod", "params": {}}]'
            )
            reasoner = Reasoner(llm_call=lambda prompt: raw)
            actions = reasoner.plan_actions("look sideways", SceneMemory())
            assert actions == [{"name": "nod", "params": {}}]

            orchestrator = Orchestrator(tts=FakeTts(), protocol_client=client)
            results = await orchestrator.execute_actions(actions)
            assert [r["status"] for r in results] == ["done"]
    finally:
        await _stop_body(body_task, sim)


@pytest.mark.asyncio
async def test_body_error_ack_is_surfaced_and_does_not_stop_the_sequence(
    tmp_path, unused_tcp_port
):
    """Body refuses a clip it has no file for; Brain reports it and carries
    on to the next action rather than silently assuming success."""
    body_task, sim, server = await _start_body(tmp_path, unused_tcp_port)
    try:
        async with websockets.connect(f"ws://127.0.0.1:{unused_tcp_port}") as ws:
            orchestrator = Orchestrator(tts=FakeTts(), protocol_client=ProtocolClient(connection=ws))
            results = await orchestrator.execute_actions([
                # "alert" is a legal vocabulary name but this test's assets
                # dir only contains chime.wav, so Body raises FileNotFound.
                {"name": "play_sfx", "params": {"name": "alert"}},
                {"name": "set_light", "params": {"state": "bright"}},
            ])
            assert results[0]["status"] == "error"
            assert "alert" in results[0]["error"]
            assert results[1]["status"] == "done"
            assert server._light.get() == "bright"
    finally:
        await _stop_body(body_task, sim)


@pytest.mark.asyncio
async def test_malformed_frame_does_not_kill_the_connection(tmp_path, unused_tcp_port):
    """The C1 regression, end to end over a real socket: a bad frame gets
    an error ack and the same connection still serves the next command."""
    body_task, sim, server = await _start_body(tmp_path, unused_tcp_port)
    try:
        async with websockets.connect(f"ws://127.0.0.1:{unused_tcp_port}") as ws:
            for raw in [
                '{"id": 1, "cmd": "set_light", "params": null}',
                '{"id": 2, "cmd": "look_at", "params": "direction"}',
                '{"id": 3, "cmd": ["nod"], "params": {}}',
                '{"id": 4, "cmd": "set_light", "params": ["state"]}',
            ]:
                await ws.send(raw)
                reply = await asyncio.wait_for(ws.recv(), timeout=2)
                assert '"status": "error"' in reply

            client = ProtocolClient(connection=ws)
            ack = await client.send_command("curious_lean", {})
            assert ack["status"] == "done"
            assert len(ack["pose"]) == 5
    finally:
        await _stop_body(body_task, sim)
