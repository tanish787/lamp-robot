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
    "neutral": {},
    "set_light": {"state": str},
    "play_sfx": {"name": str},
    "play_music": {"on": bool, "track": str},
    "speak": {"text": str},
}

BRAIN_LOCAL_ACTIONS: set[str] = {"speak"}
BODY_ACTIONS: set[str] = set(ACTIONS) - BRAIN_LOCAL_ACTIONS

LOOK_DIRECTIONS: set[str] = {"left", "right", "center", "up", "down"}
LIGHT_STATES: set[str] = {"off", "dim", "bright", "pulse"}

# Clip names are enumerated for the same reason directions and light states
# are: Body turns them straight into a filesystem path
# (body/light_sfx.py -> assets_dir / f"{name}.wav"), so an unconstrained
# string would let an LLM-authored or hand-crafted command walk out of the
# assets directory. These must stay in sync with body/assets/.
SFX_NAMES: set[str] = {"chime", "confirm", "alert"}
MUSIC_TRACKS: set[str] = {"ambient"}

# Enumerated-value checks, keyed by (action name, parameter name).
_ENUMERATED: dict[tuple[str, str], set[str]] = {
    ("look_at", "direction"): LOOK_DIRECTIONS,
    ("set_light", "state"): LIGHT_STATES,
    ("play_sfx", "name"): SFX_NAMES,
    ("play_music", "track"): MUSIC_TRACKS,
}


def is_valid_action(name: str, params: dict) -> bool:
    """True if `name` is a known action and `params` matches its schema."""
    schema = ACTIONS.get(name)
    if schema is None:
        return False
    if not isinstance(params, dict):
        return False
    for key, expected_type in schema.items():
        if key not in params or not isinstance(params[key], expected_type):
            return False
        allowed = _ENUMERATED.get((name, key))
        if allowed is not None and params[key] not in allowed:
            return False
    return True
