import pytest

from body.motion import (
    SOFT_LIMITS, clamp_target, clamp_pose, resolve_action, plan_trajectory,
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


def test_plan_trajectory_interpolates_to_target():
    steps = plan_trajectory(0.0, 1.0, 4)
    assert len(steps) == 4
    assert steps[-1] == pytest.approx(1.0)
    assert steps[0] < steps[-1]


def test_plan_trajectory_rejects_zero_steps():
    with pytest.raises(ValueError):
        plan_trajectory(0.0, 1.0, 0)
