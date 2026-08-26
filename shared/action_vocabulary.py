"""The only interface between Brain and Body: names, parameter schemas,
and which side executes each action.

Body executes BODY_ACTIONS (motion, light, sound/music playback) — none of
them require a model. BRAIN_LOCAL_ACTIONS are handled by Brain itself
(currently just `speak`, which drives local TTS) and never cross the wire
to Body.
"""

ACTIONS: dict[str, dict[str, type]] = {
    "look_at": {"direction": str},
    "curious_lean": {},
    "nod": {},
    "shake": {},
    "scan_sweep": {},
    "idle_sway": {},
    "set_light": {"state": str},
    "play_sfx": {"name": str},
    "play_music": {"on": bool, "track": str},
    "speak": {"text": str},
}

BRAIN_LOCAL_ACTIONS: set[str] = {"speak"}
BODY_ACTIONS: set[str] = set(ACTIONS) - BRAIN_LOCAL_ACTIONS

LOOK_DIRECTIONS: set[str] = {"left", "right", "center", "up", "down"}
LIGHT_STATES: set[str] = {"off", "dim", "bright", "pulse"}


def is_valid_action(name: str, params: dict) -> bool:
    """True if `name` is a known action and `params` matches its schema."""
    schema = ACTIONS.get(name)
    if schema is None:
        return False
    for key, expected_type in schema.items():
        if key not in params or not isinstance(params[key], expected_type):
            return False
    if name == "look_at" and params.get("direction") not in LOOK_DIRECTIONS:
        return False
    if name == "set_light" and params.get("state") not in LIGHT_STATES:
        return False
    return True
