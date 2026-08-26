from shared.action_vocabulary import (
    ACTIONS, BODY_ACTIONS, BRAIN_LOCAL_ACTIONS, LOOK_DIRECTIONS,
    LIGHT_STATES, is_valid_action,
)


def test_speak_is_brain_local_not_body():
    assert "speak" in BRAIN_LOCAL_ACTIONS
    assert "speak" not in BODY_ACTIONS


def test_body_actions_cover_motion_light_and_sound():
    expected = {
        "look_at", "curious_lean", "nod", "shake", "scan_sweep",
        "idle_sway", "set_light", "play_sfx", "play_music",
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
