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

# Actions whose character comes from motion over time, not from a resting
# pose. Each entry is a list of *incremental* joint offsets applied from
# wherever the joint currently is; every sequence sums to zero, so the
# action oscillates and then returns to its starting pose. Without this a
# `nod` right after a `curious_lean` was a literal no-op (both target
# head_pitch 0.4) and `idle_sway` was a single static 0.05 offset.
_OSCILLATIONS: dict[str, list[dict[int, float]]] = {
    # head_pitch down-up-down-up
    "nod": [{4: 0.30}, {4: -0.30}, {4: 0.20}, {4: -0.20}],
    # neck_yaw left-right-left-centre
    "shake": [{3: -0.35}, {3: 0.70}, {3: -0.70}, {3: 0.35}],
    # slow, small base_yaw drift either side of where it already is
    "idle_sway": [{0: 0.08}, {0: -0.16}, {0: 0.16}, {0: -0.08}],
}

_STATIC_POSES: dict[str, dict[int, float]] = {
    "curious_lean": CURIOUS_POSE,
    "neutral": NEUTRAL,
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


def resolve_waypoints(
    name: str, params: dict, current: dict[int, float] | None = None
) -> list[dict[int, float]]:
    """Turn a named motion action + params into the ordered list of clamped
    pose overlays it should pass through.

    Each overlay names only the joints this action moves; the caller
    (simulation.py) leaves every other joint at its current position.
    Static actions return a single waypoint; oscillating ones
    (nod/shake/idle_sway) and scan_sweep return several, so they actually
    animate instead of snapping to one target and stopping there.
    """
    if name not in _MOTION_ACTIONS:
        raise ValueError(f"resolve_waypoints: {name!r} is not a Body motion action")

    if name == "look_at":
        direction = params.get("direction", "center")
        return [clamp_pose(_LOOK_TARGETS.get(direction, _LOOK_TARGETS["center"]))]

    if name in _STATIC_POSES:
        return [clamp_pose(_STATIC_POSES[name])]

    if name == "scan_sweep":
        _, lower, upper = SOFT_LIMITS[0]
        return [clamp_pose({0: lower}), clamp_pose({0: upper}), clamp_pose({0: 0.0})]

    base = dict(NEUTRAL) if current is None else dict(current)
    waypoints: list[dict[int, float]] = []
    running: dict[int, float] = {}
    for offset in _OSCILLATIONS[name]:
        for joint, delta in offset.items():
            # Accumulate unclamped so a clipped extreme doesn't shift where
            # the oscillation lands; only the emitted waypoint is clamped.
            running[joint] = running.get(joint, base.get(joint, 0.0)) + delta
        waypoints.append(clamp_pose(dict(running)))
    return waypoints


def resolve_action(
    name: str, params: dict, current: dict[int, float] | None = None
) -> dict[int, float]:
    """The resting pose overlay an action ends on (its last waypoint)."""
    return resolve_waypoints(name, params, current)[-1]


def plan_trajectory(current: float, target: float, steps: int) -> list[float]:
    """Linearly interpolate a single joint from current to target."""
    if steps < 1:
        raise ValueError("steps must be >= 1")
    return [current + (target - current) * (i / steps) for i in range(1, steps + 1)]
