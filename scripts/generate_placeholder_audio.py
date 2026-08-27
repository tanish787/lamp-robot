"""Regenerate the placeholder sfx/music clips in body/assets.

These are stand-ins so the demo runs out of the box; they are meant to be
replaced with real recordings before the demo is recorded (see
body/assets/README.md). Every name written here must also appear in
shared.action_vocabulary.SFX_NAMES / MUSIC_TRACKS.

    python -m scripts.generate_placeholder_audio
"""

import math
import struct
import wave
from pathlib import Path

from shared.action_vocabulary import MUSIC_TRACKS, SFX_NAMES

SAMPLE_RATE = 22050
ASSETS = Path(__file__).resolve().parent.parent / "body" / "assets"

# name -> list of (frequency_hz, duration_s, amplitude 0..1) segments
CLIPS: dict[str, list[tuple[float, float, float]]] = {
    "chime": [(880.0, 0.15, 0.35), (1318.5, 0.20, 0.30)],
    "confirm": [(660.0, 0.20, 0.30)],
    "alert": [(220.0, 0.30, 0.35)],
    "ambient": [(110.0, 2.00, 0.12)],
}


def _tone(frequency: float, duration_s: float, amplitude: float) -> bytes:
    total = int(SAMPLE_RATE * duration_s)
    fade = max(1, int(SAMPLE_RATE * 0.01))  # 10 ms fades, avoids clicks
    samples = bytearray()
    for i in range(total):
        envelope = min(1.0, i / fade, (total - i) / fade)
        value = amplitude * envelope * math.sin(2 * math.pi * frequency * i / SAMPLE_RATE)
        samples += struct.pack("<h", int(value * 32767))
    return bytes(samples)


def _write(path: Path, segments: list[tuple[float, float, float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frames = b"".join(_tone(*segment) for segment in segments)
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(SAMPLE_RATE)
        wav.writeframes(frames)
    print(f"wrote {path} ({len(frames) // 2} samples)")


def main() -> None:
    for name in sorted(SFX_NAMES):
        _write(ASSETS / "sfx" / f"{name}.wav", CLIPS[name])
    for name in sorted(MUSIC_TRACKS):
        _write(ASSETS / "music" / f"{name}.wav", CLIPS[name])


if __name__ == "__main__":
    main()
