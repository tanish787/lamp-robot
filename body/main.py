import argparse
import asyncio
import subprocess
import sys
from pathlib import Path

from body.light_sfx import LightState, MusicPlayer, SfxPlayer
from body.server import BodyServer
from body.simulation import LampSimulation

SFX_DIR = Path(__file__).resolve().parent / "assets" / "sfx"
MUSIC_DIR = Path(__file__).resolve().parent / "assets" / "music"


def _play_clip(path: Path) -> None:
    # `simpleaudio`'s native bindings segfaulted reliably on real Ubuntu
    # target hardware (confirmed during deployment testing), while the
    # system's own `aplay` plays the same files without issue — so we
    # shell out to it instead of depending on a fragile compiled
    # extension. Non-blocking (Popen, not run), matching simpleaudio's
    # original fire-and-forget playback so a sound effect never stalls
    # Body's asyncio event loop.
    #
    # Windows has no `aplay`; this branch exists purely for running the
    # demo locally on a Windows dev machine (the actual deployment target
    # is Ubuntu, where the branch below is what actually runs).
    # `winsound` is stdlib on Windows only.
    if sys.platform.startswith("win"):
        import winsound

        winsound.PlaySound(str(path), winsound.SND_FILENAME | winsound.SND_ASYNC)
    else:
        subprocess.Popen(["aplay", "-q", str(path)])


def _play_loop(path: Path, loop: bool) -> None:
    # Minimal loop: aplay has no native loop flag; re-triggering on
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
    try:
        asyncio.run(server.serve())
    except KeyboardInterrupt:
        print("\nBody stopping.")
    finally:
        # Always disconnect the PyBullet client, including on Ctrl+C.
        sim.close()


if __name__ == "__main__":
    main()
