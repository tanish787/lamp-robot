import pytest

from body.motion import (
    NEUTRAL, SOFT_LIMITS, clamp_target, clamp_pose, resolve_action,
    resolve_waypoints, plan_trajectory,
)


def test_clamp_target_passes_through_in_range_value():
    assert clamp_target(0, 0.5) == 0.5


def test_clamp_target_clamps_above_upper_soft_limit():
    _, _, upper = SOFT_LIMITS[0]
    assert clamp_target(0, 999.0) == upper


def test_clamp_target_clamps_below_lower_soft_limit():
    _, lower, _ = SOFT_LIMITS[1]
    assert clamp_target(1, -999.0) == lower


def test_clamp_pose_clamps_every_joint_given():
    _, lower0, upper0 = SOFT_LIMITS[0]
    pose = clamp_pose({0: 999.0, 4: -999.0})
    _, lower4, _ = SOFT_LIMITS[4]
    assert pose[0] == upper0
    assert pose[4] == lower4


def test_resolve_action_look_left_stays_within_limits():
    pose = resolve_action("look_at", {"direction": "left"})
    for joint, target in pose.items():
        _, lower, upper = SOFT_LIMITS[joint]
        assert lower <= target <= upper


def test_resolve_action_unknown_name_raises():
    with pytest.raises(ValueError):
        resolve_action("teleport", {})


def test_resolve_action_rejects_brain_local_action():
    with pytest.raises(ValueError):
        resolve_action("speak", {"text": "hi"})


@pytest.mark.parametrize("name", ["nod", "shake", "idle_sway"])
def test_oscillating_actions_return_several_waypoints(name):
    waypoints = resolve_waypoints(name, {})
    assert len(waypoints) > 1


@pytest.mark.parametrize("name,joint", [("nod", 4), ("shake", 3), ("idle_sway", 0)])
def test_oscillating_actions_end_where_they_started(name, joint):
    """A nod is motion, not a new resting pose — it must come back."""
    start = 0.25
    waypoints = resolve_waypoints(name, {}, {joint: start})
    assert waypoints[-1][joint] == pytest.approx(start)
    assert any(w[joint] != pytest.approx(start) for w in waypoints[:-1])


def test_nod_after_curious_lean_actually_moves():
    """Regression: nod used to target head_pitch 0.4, exactly where
    curious_lean leaves it, so chaining them produced zero visible motion."""
    lean = resolve_action("curious_lean", {})
    waypoints = resolve_waypoints("nod", {}, lean)
    assert any(w[4] != pytest.approx(lean[4]) for w in waypoints)


def test_oscillation_waypoints_stay_within_soft_limits():
    for name in ("nod", "shake", "idle_sway", "scan_sweep"):
        for waypoint in resolve_waypoints(name, {}, resolve_action("curious_lean", {})):
            for joint, target in waypoint.items():
                _, lower, upper = SOFT_LIMITS[joint]
                assert lower <= target <= upper


def test_scan_sweep_sweeps_both_ways_and_recentres():
    waypoints = resolve_waypoints("scan_sweep", {})
    yaws = [w[0] for w in waypoints]
    assert len(yaws) >= 3
    assert min(yaws) < 0 < max(yaws)
    assert yaws[-1] == pytest.approx(0.0)


def test_neutral_returns_every_joint_to_origin():
    assert resolve_action("neutral", {}) == NEUTRAL


def test_plan_trajectory_interpolates_to_target():
    steps = plan_trajectory(0.0, 1.0, 4)
    assert len(steps) == 4
    assert steps[-1] == pytest.approx(1.0)
    assert steps[0] < steps[-1]


def test_plan_trajectory_rejects_zero_steps():
    with pytest.raises(ValueError):
        plan_trajectory(0.0, 1.0, 0)
