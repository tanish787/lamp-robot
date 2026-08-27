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


def test_nod_oscillates_and_returns_to_the_pre_nod_pose(sim):
    sim.apply_action("curious_lean", {})
    before = sim.get_pose()
    after = sim.apply_action("nod", {})
    assert after == pytest.approx(before, abs=1e-6)


def test_neutral_visibly_undoes_a_curious_lean(sim):
    leaned = sim.apply_action("curious_lean", {})
    assert any(abs(v) > 0.1 for v in leaned)
    assert sim.apply_action("neutral", {}) == pytest.approx([0.0] * 5, abs=1e-6)


def test_idle_sway_after_a_lean_is_not_the_same_pose_as_engaged(sim):
    """Disengaging must be visibly different from the engaged pose: the
    orchestrator's disengage sequence resets to neutral first."""
    leaned = sim.apply_action("curious_lean", {})
    sim.apply_action("neutral", {})
    idle = sim.apply_action("idle_sway", {})
    assert idle != pytest.approx(leaned, abs=1e-6)
