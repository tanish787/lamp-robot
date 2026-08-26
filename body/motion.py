"""Kinematics: maps action-vocabulary names to joint targets and clamps
every target to the URDF's soft joint limits before anything is executed.

Soft limits and the expressive "curious" pose are ported from the lab's
src/animate_lamp.py and src/render_lamp.py (safety_controller blocks in
the URDF, and the hand-tuned pose used there).
"""

from shared.action_vocabulary import BODY_ACTIONS

# Joint index -> (name, soft_lower, soft_upper)
SOFT_LIMITS: dict[int, tuple[str, float, float]] = {
    0: ("base_yaw", -2.45, 2.45),
    1: ("shoulder_pitch", -0.65, 0.95),
    2: ("elbow_pitch", -1.70, 0.30),
    3: ("neck_yaw", -1.25, 1.25),
    4: ("head_pitch", -0.80, 0.60),
}

NEUTRAL: dict[int, float] = {i: 0.0 for i in SOFT_LIMITS}
CURIOUS_POSE: dict[int, float] = {0: 0.4, 1: 0.5, 2: -0.9, 3: -0.3, 4: 0.4}

_LOOK_TARGETS: dict[str, dict[int, float]] = {
    "left": {0: -1.2, 3: -0.6},
    "right": {0: 1.2, 3: 0.6},
    "center": {0: 0.0, 3: 0.0},
    "up": {4: -0.6},
    "down": {4: 0.5},
}

# Motion-only subset of BODY_ACTIONS: excludes set_light/play_sfx/play_music,
# which body.light_sfx handles instead of body.motion.
_MOTION_ACTIONS = BODY_ACTIONS - {"set_light", "play_sfx", "play_music"}


def clamp_target(joint_index: int, target: float) -> float:
    """Clamp a single joint target to its soft limit range."""
    _, lower, upper = SOFT_LIMITS[joint_index]
    return max(lower, min(upper, target))


def clamp_pose(pose: dict[int, float]) -> dict[int, float]:
    """Clamp every joint in a pose overlay to its soft limits."""
    return {i: clamp_target(i, v) for i, v in pose.items()}


def resolve_action(name: str, params: dict) -> dict[int, float]:
    """Turn a named motion action + params into a clamped pose overlay.

    Returns only the joints this action moves; the caller (simulation.py)
    leaves every other joint at its current position.
    """
    if name not in _MOTION_ACTIONS:
        raise ValueError(f"resolve_action: {name!r} is not a Body motion action")
    if name == "look_at":
        direction = params.get("direction", "center")
        pose = _LOOK_TARGETS.get(direction, _LOOK_TARGETS["center"])
    elif name == "curious_lean":
        pose = CURIOUS_POSE
    elif name == "nod":
        pose = {4: 0.4}
    elif name == "shake":
        pose = {0: 0.3}
    elif name == "scan_sweep":
        pose = {0: SOFT_LIMITS[0][2]}
    else:  # idle_sway
        pose = {0: 0.05}
    return clamp_pose(pose)


def plan_trajectory(current: float, target: float, steps: int) -> list[float]:
    """Linearly interpolate a single joint from current to target."""
    if steps < 1:
        raise ValueError("steps must be >= 1")
    return [current + (target - current) * (i / steps) for i in range(1, steps + 1)]
