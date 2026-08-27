import argparse
import asyncio
from pathlib import Path

import simpleaudio

from body.light_sfx import LightState, MusicPlayer, SfxPlayer
from body.server import BodyServer
from body.simulation import LampSimulation

SFX_DIR = Path(__file__).resolve().parent / "assets" / "sfx"
MUSIC_DIR = Path(__file__).resolve().parent / "assets" / "music"


def _play_clip(path: Path) -> None:
    simpleaudio.WaveObject.from_wave_file(str(path)).play()


def _play_loop(path: Path, loop: bool) -> None:
    # Minimal loop: simpleaudio has no native loop flag; re-triggering on
    # completion is left as a documented follow-up (see technical note).
    _play_clip(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gui", action="store_true", help="show the PyBullet GUI window")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()

    sim = LampSimulation(gui=args.gui)
    light = LightState()
    sfx = SfxPlayer(SFX_DIR, player=_play_clip)
    music = MusicPlayer(MUSIC_DIR, player=_play_loop)
    server = BodyServer(sim, light, sfx, music, port=args.port)

    print(f"Body listening on ws://127.0.0.1:{args.port}")
    asyncio.run(server.serve())


if __name__ == "__main__":
    main()
