"""Local-LLM reasoning: dialogue replies, and turning a spoken goal into a
validated action sequence. The LLM never controls kinematics directly — it
only ever names actions from shared.action_vocabulary, and every name is
checked before it can reach Body (spec section 8's fallback path)."""

import json
from typing import Callable

from shared.action_vocabulary import (
    ACTIONS, LIGHT_STATES, LOOK_DIRECTIONS, MUSIC_TRACKS, SFX_NAMES,
    is_valid_action,
)

# Every enumerated parameter is spelled out for the model. Validation drops
# anything outside these sets anyway, so naming them up front is the
# difference between a usable plan and a fallback to idle_sway.
_ALLOWED_VALUES = "\n".join([
    f"  look_at.direction: {sorted(LOOK_DIRECTIONS)}",
    f"  set_light.state: {sorted(LIGHT_STATES)}",
    f"  play_sfx.name: {sorted(SFX_NAMES)}",
    f"  play_music.on: true/false, play_music.track: {sorted(MUSIC_TRACKS)}",
    "  speak.text: any short sentence",
])

_PLAN_PROMPT = """You control a lamp robot. Given the scene memory and a
spoken goal, respond with a JSON array of actions to accomplish it. Each
action is {"name": one of %s, "params": {...}}. Return ONLY the JSON array.

Allowed parameter values (anything else is rejected):
%s

Scene memory:
%s

Goal: %s
"""


class Reasoner:
    def __init__(self, llm_call: Callable[[str], str] | None = None):
        if llm_call is None:
            llm_call = _default_llm_call()
        self._llm_call = llm_call

    def reply(self, user_text: str, memory) -> str:
        prompt = f"Scene memory:\n{memory.as_prompt_text()}\n\nUser said: {user_text}\nReply briefly."
        return self._llm_call(prompt).strip()

    def plan_actions(self, goal_text: str, memory) -> list[dict]:
        prompt = _PLAN_PROMPT % (list(ACTIONS), _ALLOWED_VALUES, memory.as_prompt_text(), goal_text)
        raw = self._llm_call(prompt)
        actions = self._parse_and_validate(raw)
        return actions if actions else [{"name": "idle_sway", "params": {}}]

    def _parse_and_validate(self, raw: str) -> list[dict]:
        try:
            candidates = json.loads(raw)
        except json.JSONDecodeError:
            return []
        if not isinstance(candidates, list):
            return []
        valid = []
        for item in candidates:
            if not isinstance(item, dict):
                continue
            name, params = item.get("name"), item.get("params", {})
            if not isinstance(params, dict):
                continue
            if isinstance(name, str) and is_valid_action(name, params):
                valid.append({"name": name, "params": params})
        return valid


def _default_llm_call() -> Callable[[str], str]:
    from llama_cpp import Llama

    llm = Llama(model_path="models/llm/model.gguf", n_ctx=2048, verbose=False)

    def call(prompt: str) -> str:
        result = llm(prompt, max_tokens=256, stop=["\n\n"])
        return result["choices"][0]["text"]

    return call
