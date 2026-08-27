"""Load the full local model stack simultaneously and report peak RSS,
per spec section 9's resource-ceiling test. Run manually on the target
(or target-equivalent VM) hardware:
    python scripts/resource_ceiling_check.py
"""

import os

import psutil

from brain.perception import ScenePerception
from brain.reasoning import Reasoner
from brain.stt import SpeechToText
from brain.tts import TextToSpeech
from body.simulation import LampSimulation

BUDGET_BYTES = 7 * 1024 * 1024 * 1024  # 7 GB


def main() -> None:
    process = psutil.Process(os.getpid())

    sim = LampSimulation(gui=False)
    stt = SpeechToText()
    tts = TextToSpeech()
    perception = ScenePerception()
    reasoner = Reasoner()

    peak_rss = process.memory_info().rss
    print(f"Peak RSS with full stack loaded: {peak_rss / 1e9:.2f} GB")
    print(f"Budget: {BUDGET_BYTES / 1e9:.2f} GB")
    if peak_rss > BUDGET_BYTES:
        print("OVER BUDGET")
    else:
        print("within budget")

    sim.close()
    del stt, tts, perception, reasoner


if __name__ == "__main__":
    main()
