"""Simulated light state plus sound-effect/music playback. Neither depends
on any model — light is a tracked attribute (there's no real bulb on a
simulated lamp; PyBullet's debug-visualizer color on the shade link is
driven from this state), and audio playback is local clip playback only.
"""

from pathlib import Path
from typing import Callable

from shared.action_vocabulary import LIGHT_STATES


def _resolve_clip(assets_dir: Path, name: str) -> Path:
    """Resolve `<assets_dir>/<name>.wav`, refusing anything that escapes
    `assets_dir`. Defence in depth: shared.action_vocabulary already
    restricts clip names to SFX_NAMES/MUSIC_TRACKS, but Body must not
    depend on the caller having validated its input."""
    root = Path(assets_dir).resolve()
    path = (root / f"{name}.wav").resolve()
    if path.parent != root:
        raise ValueError(f"clip name {name!r} escapes the assets directory")
    return path


class LightState:
    def __init__(self):
        self._state = "off"

    def set(self, state: str) -> None:
        if state not in LIGHT_STATES:
            raise ValueError(f"unknown light state: {state!r}")
        self._state = state

    def get(self) -> str:
        return self._state


class SfxPlayer:
    def __init__(self, assets_dir: Path, player: Callable[[Path], None] = None):
        self._assets_dir = assets_dir
        self._player = player

    def play(self, name: str) -> None:
        path = _resolve_clip(self._assets_dir, name)
        if not path.exists():
            raise FileNotFoundError(f"no sound effect named {name!r} at {path}")
        self._player(path)


class MusicPlayer:
    def __init__(self, assets_dir: Path, player: Callable[[Path, bool], None] = None):
        self._assets_dir = assets_dir
        self._player = player
        self.stop_fn: Callable[[], None] = lambda: None

    def play(self, track: str, loop: bool = True) -> None:
        path = _resolve_clip(self._assets_dir, track)
        if not path.exists():
            raise FileNotFoundError(f"no music track named {track!r} at {path}")
        self._player(path, loop)

    def stop(self) -> None:
        self.stop_fn()
