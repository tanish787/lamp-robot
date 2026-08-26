import pytest

from body.light_sfx import LightState, SfxPlayer, MusicPlayer


def test_light_state_defaults_to_off():
    assert LightState().get() == "off"


def test_light_state_set_and_get_round_trips():
    light = LightState()
    light.set("pulse")
    assert light.get() == "pulse"


def test_light_state_rejects_unknown_state():
    with pytest.raises(ValueError):
        LightState().set("rainbow")


def test_sfx_player_plays_existing_clip(tmp_path):
    (tmp_path / "chime.wav").write_bytes(b"RIFF....WAVEfmt ")
    calls = []
    player = SfxPlayer(tmp_path, player=lambda path: calls.append(path))
    player.play("chime")
    assert calls == [tmp_path / "chime.wav"]


def test_sfx_player_raises_for_missing_clip(tmp_path):
    player = SfxPlayer(tmp_path, player=lambda path: None)
    with pytest.raises(FileNotFoundError):
        player.play("nonexistent")


def test_music_player_play_and_stop(tmp_path):
    (tmp_path / "ambient.wav").write_bytes(b"RIFF....WAVEfmt ")
    started, stopped = [], []
    music = MusicPlayer(
        tmp_path,
        player=lambda path, loop: started.append((path, loop)),
    )
    music.stop_fn = lambda: stopped.append(True)
    music.play("ambient")
    music.stop()
    assert started == [(tmp_path / "ambient.wav", True)]
    assert stopped == [True]
