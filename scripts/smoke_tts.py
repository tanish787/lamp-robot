"""Synthesize and play a sample line with the real Piper voice, printing
latency. Run manually on the target hardware (or a VM configured to
match the target spec):
    python -m scripts.smoke_tts "Hello, I can see you."
"""

import sys
import time

from brain.tts import TextToSpeech


def main() -> None:
    text = sys.argv[1]
    tts = TextToSpeech()
    start = time.monotonic()
    tts.speak(text)
    print(f"Latency: {time.monotonic() - start:.2f}s")


if __name__ == "__main__":
    main()
