"""PyBullet simulation runtime: loads the patched URDF, applies a named
action by interpolating each affected joint toward its clamped target, and
reports the resulting full pose. Ported from the lab's src/animate_lamp.py
stepping approach; headless (DIRECT) by default so it runs in CI/tests
without a display.
"""

from pathlib import Path

import pybullet as p

from body.motion import NEUTRAL, SOFT_LIMITS, plan_trajectory, resolve_action
from body.urdf_prep import prepare_urdf

_DEFAULT_CACHE = Path(__file__).resolve().parent.parent / ".cache"


class LampSimulation:
    def __init__(self, gui: bool = False, cache_dir: Path | None = None):
        self._client = p.connect(p.GUI if gui else p.DIRECT)
        p.setGravity(0, 0, -9.81, physicsClientId=self._client)
        patched_urdf = prepare_urdf(cache_dir or _DEFAULT_CACHE)
        self._robot = p.loadURDF(
            str(patched_urdf), useFixedBase=True, physicsClientId=self._client
        )
        self._pose = dict(NEUTRAL)

    def apply_action(self, name: str, params: dict, steps: int = 30) -> list[float]:
        overlay = resolve_action(name, params)
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
            self._pose[joint] = target
        return self.get_pose()

    def get_pose(self) -> list[float]:
        return [self._pose[i] for i in sorted(SOFT_LIMITS)]

    def close(self) -> None:
        p.disconnect(physicsClientId=self._client)
