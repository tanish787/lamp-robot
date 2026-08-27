"""Run the real detector on a sample image, printing detections and
latency. Run manually:
    python -m scripts.smoke_detector path/to/sample.jpg
"""

import sys
import time

import cv2

from brain.memory import SceneMemory
from brain.perception import ScenePerception


def main() -> None:
    frame = cv2.imread(sys.argv[1])
    perception = ScenePerception()
    memory = SceneMemory()

    start = time.monotonic()
    labels = perception.observe(frame, memory, timestamp=0.0)
    elapsed = time.monotonic() - start

    print(f"Detected: {labels}")
    print(f"Latency: {elapsed:.2f}s")


if __name__ == "__main__":
    main()
