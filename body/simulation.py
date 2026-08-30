"""PyBullet simulation runtime: loads the patched URDF, applies a named
action by interpolating each affected joint toward its clamped target, and
reports the resulting full pose. Ported from the lab's src/animate_lamp.py
stepping approach; headless (DIRECT) by default so it runs in CI/tests
without a display.
"""

import time
from pathlib import Path
from typing import Callable

import pybullet as p

from body.motion import NEUTRAL, SOFT_LIMITS, plan_trajectory, resolve_waypoints
from body.urdf_prep import prepare_urdf

_DEFAULT_CACHE = Path(__file__).resolve().parent.parent / ".cache"

# Matches PyBullet's own default simulation timestep (1/240 s). stepSimulation
# only advances *simulated* time; nothing paces it to wall-clock time on its
# own, so a live viewer (the GUI window, or a MeshCat mirror) sees an action
# snap through its whole trajectory almost instantly unless something sleeps
# between steps. Headless runs (gui=False, no on_step) never pace, so the
# test suite's speed is unaffected.
_STEP_DT = 1.0 / 240.0


class LampSimulation:
    def __init__(
        self,
        gui: bool = False,
        cache_dir: Path | None = None,
        on_step: Callable[[], None] | None = None,
        sleep_fn: Callable[[float], None] = time.sleep,
    ):
        self._client = p.connect(p.GUI if gui else p.DIRECT)
        try:
            p.setGravity(0, 0, -9.81, physicsClientId=self._client)
            patched_urdf = prepare_urdf(cache_dir or _DEFAULT_CACHE)
            self._robot = p.loadURDF(
                str(patched_urdf), useFixedBase=True, physicsClientId=self._client
            )
        except Exception:
            # Don't leak a connected PyBullet client if asset prep or the
            # URDF load fails partway through construction.
            p.disconnect(physicsClientId=self._client)
            raise
        self._pose = dict(NEUTRAL)
        # `on_step` lets an external viewer (MeshCat) mirror every
        # intermediate pose, not just the final one, so it animates instead
        # of snapping to the end of the trajectory. Real-time pacing is
        # wanted whenever something is actually watching live — the GUI
        # window or an attached viewer — not during headless testing.
        self._on_step = on_step
        self._sleep_fn = sleep_fn
        self._paced = gui or on_step is not None

    @property
    def robot_id(self) -> int:
        return self._robot

    @property
    def client_id(self) -> int:
        return self._client

    def apply_action(self, name: str, params: dict, steps: int = 30) -> list[float]:
        for overlay in resolve_waypoints(name, params, self._pose):
            for joint, target in overlay.items():
                trajectory = plan_trajectory(self._pose[joint], target, steps)
                for step_target in trajectory:
                    p.setJointMotorControl2(
                        self._robot,
                        joint,
                        p.POSITION_CONTROL,
                        targetPosition=step_target,
                        physicsClientId=self._client,
                    )
                    p.stepSimulation(physicsClientId=self._client)
                    if self._on_step is not None:
                        self._on_step()
                    if self._paced:
                        self._sleep_fn(_STEP_DT)
                self._pose[joint] = target
        return self.get_pose()

    def get_pose(self) -> list[float]:
        return [self._pose[i] for i in sorted(SOFT_LIMITS)]

    def close(self) -> None:
        p.disconnect(physicsClientId=self._client)
