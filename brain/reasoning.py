"""Local-LLM reasoning: dialogue replies, and turning a spoken goal into a
validated action sequence. The LLM never controls kinematics directly — it
only ever names actions from shared.action_vocabulary, and every name is
checked before it can reach Body (spec section 8's fallback path)."""

import json
from typing import Callable

from shared.action_vocabulary import ACTIONS, is_valid_action

_FALLBACK_ACTIONS = [{"name": "idle_sway", "params": {}}]

_PLAN_PROMPT = """You control a lamp robot. Given the scene memory and a
spoken goal, respond with a JSON array of actions to accomplish it. Each
action is {{"name": one of %s, "params": {{...}}}}. Return ONLY the JSON array.

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
        prompt = _PLAN_PROMPT % (list(ACTIONS), memory.as_prompt_text(), goal_text)
        raw = self._llm_call(prompt)
        actions = self._parse_and_validate(raw)
        return actions if actions else list(_FALLBACK_ACTIONS)

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
