from pathlib import Path

from shared.action_vocabulary import (
    ACTIONS, BODY_ACTIONS, BRAIN_LOCAL_ACTIONS, LOOK_DIRECTIONS,
    LIGHT_STATES, MUSIC_TRACKS, SFX_NAMES, is_valid_action,
)

_ASSETS = Path(__file__).resolve().parents[2] / "body" / "assets"


def test_speak_is_brain_local_not_body():
    assert "speak" in BRAIN_LOCAL_ACTIONS
    assert "speak" not in BODY_ACTIONS


def test_body_actions_cover_motion_light_and_sound():
    expected = {
        "look_at", "curious_lean", "nod", "shake", "scan_sweep",
        "idle_sway", "neutral", "set_light", "play_sfx", "play_music",
    }
    assert expected == BODY_ACTIONS


def test_is_valid_action_accepts_known_good_command():
    assert is_valid_action("look_at", {"direction": "left"})
    assert is_valid_action("set_light", {"state": "pulse"})
    assert is_valid_action("play_sfx", {"name": "chime"})


def test_is_valid_action_rejects_unknown_name():
    assert not is_valid_action("teleport", {})


def test_is_valid_action_rejects_bad_direction():
    assert not is_valid_action("look_at", {"direction": "sideways"})


def test_is_valid_action_rejects_bad_light_state():
    assert not is_valid_action("set_light", {"state": "rainbow"})


def test_is_valid_action_rejects_missing_required_param():
    assert not is_valid_action("play_sfx", {})


def test_is_valid_action_rejects_non_dict_params():
    # Body's server relies on this: params that are not a mapping must be
    # rejected, not raise (see the malformed-frame regression in
    # tests/body/test_server.py).
    for params in (None, "direction", ["state"], 7):
        assert not is_valid_action("look_at", params)


def test_is_valid_action_rejects_sfx_name_outside_the_enumeration():
    assert is_valid_action("play_sfx", {"name": "chime"})
    assert not is_valid_action("play_sfx", {"name": "nope"})


def test_is_valid_action_rejects_path_traversal_in_clip_names():
    assert not is_valid_action("play_sfx", {"name": "../../x"})
    assert not is_valid_action("play_sfx", {"name": "/etc/passwd"})
    assert not is_valid_action("play_music", {"on": True, "track": "../../x"})


def test_is_valid_action_rejects_music_track_outside_the_enumeration():
    assert is_valid_action("play_music", {"on": True, "track": "ambient"})
    assert not is_valid_action("play_music", {"on": True, "track": "mixtape"})


def test_every_enumerated_clip_name_has_a_committed_asset():
    for name in SFX_NAMES:
        assert (_ASSETS / "sfx" / f"{name}.wav").exists()
    for track in MUSIC_TRACKS:
        assert (_ASSETS / "music" / f"{track}.wav").exists()
