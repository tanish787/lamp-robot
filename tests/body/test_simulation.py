import pytest

from body.motion import SOFT_LIMITS
from body.simulation import LampSimulation


@pytest.fixture
def sim(tmp_path):
    s = LampSimulation(gui=False, cache_dir=tmp_path)
    yield s
    s.close()


def test_apply_action_returns_full_pose(sim):
    pose = sim.apply_action("idle_sway", {})
    assert len(pose) == len(SOFT_LIMITS)


def test_apply_action_respects_soft_limits(sim):
    pose = sim.apply_action("scan_sweep", {})
    for joint, target in enumerate(pose):
        _, lower, upper = SOFT_LIMITS[joint]
        assert lower - 1e-6 <= target <= upper + 1e-6


def test_apply_action_moves_only_targeted_joints_from_neutral(sim):
    pose = sim.apply_action("look_at", {"direction": "left"})
    # base_yaw (0) and neck_yaw (3) move for a look_at; others stay near 0.
    assert pose[0] < -0.5
    assert pose[1] == pytest.approx(0.0, abs=0.05)


def test_get_pose_matches_last_apply_action(sim):
    applied = sim.apply_action("nod", {})
    assert sim.get_pose() == pytest.approx(applied, abs=1e-6)
