"""Exercise the real local LLM on a canned prompt, printing latency. Run
manually on the target hardware:
    python scripts/smoke_llm.py
"""

import time

from brain.memory import SceneMemory
from brain.reasoning import Reasoner


def main() -> None:
    memory = SceneMemory()
    memory.observe("mug", {"color": "red", "position": (0.1, 0.2)}, timestamp=0.0)
    reasoner = Reasoner()

    start = time.monotonic()
    reply = reasoner.reply("What do you see?", memory)
    print(f"Reply: {reply!r} ({time.monotonic() - start:.2f}s)")

    start = time.monotonic()
    actions = reasoner.plan_actions("Look at the mug", memory)
    print(f"Plan: {actions} ({time.monotonic() - start:.2f}s)")


if __name__ == "__main__":
    main()
