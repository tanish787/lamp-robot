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


def test_headless_apply_action_never_sleeps(tmp_path):
    """gui=False, on_step=None (the default the whole test suite runs
    under) must not pace itself against wall-clock time — a live viewer
    isn't watching, so there is nothing to animate for."""
    sleeps: list[float] = []
    sim = LampSimulation(gui=False, cache_dir=tmp_path, sleep_fn=sleeps.append)
    try:
        sim.apply_action("scan_sweep", {})
    finally:
        sim.close()
    assert sleeps == []


def test_on_step_fires_once_per_simulation_step(tmp_path):
    """A MeshCat mirror (or anything else) attached via on_step must see
    every intermediate pose, not just the final one, or it would snap to
    the end of the trajectory instead of animating."""
    calls = 0

    def count():
        nonlocal calls
        calls += 1

    sim = LampSimulation(gui=False, cache_dir=tmp_path, on_step=count)
    try:
        sim.apply_action("nod", {}, steps=10)
    finally:
        sim.close()
    # "nod" is 4 oscillation waypoints, one joint each, steps=10 -> 40 steps.
    assert calls == 40


def test_on_step_without_gui_still_paces_real_time(tmp_path):
    """Attaching a viewer should pace playback even with no native GUI
    window, so a MeshCat-only run animates instead of finishing instantly."""
    sleeps: list[float] = []
    sim = LampSimulation(
        gui=False, cache_dir=tmp_path, on_step=lambda: None, sleep_fn=sleeps.append
    )
    try:
        sim.apply_action("nod", {}, steps=10)
    finally:
        sim.close()
    assert len(sleeps) == 40
    assert all(s == pytest.approx(1.0 / 240.0) for s in sleeps)


def test_robot_id_and_client_id_are_exposed(sim):
    """A MeshCat mirror queries PyBullet's own forward kinematics directly,
    so it needs the robot and client ids LampSimulation already holds."""
    assert isinstance(sim.robot_id, int)
    assert isinstance(sim.client_id, int)
