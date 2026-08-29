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

_PLAN_SYSTEM = """You control a lamp robot. Given the scene memory and a
spoken goal, respond with a JSON array of actions to accomplish it. Each
action is {"name": one of %s, "params": {...}}. Return ONLY the JSON
array — no explanation, no markdown, nothing else.

Allowed parameter values (anything else is rejected):
%s"""

# Qwen2.5-1.5B-Instruct (the pinned default model, see scripts/setup.sh)
# is fine-tuned specifically on this ChatML template; feeding it plain
# unstructured text causes it to ignore instructions and produce empty or
# unparseable output instead of following them (confirmed during
# deployment testing on the target hardware — this is not hypothetical;
# an earlier, smaller pinned model, TinyLlama-1.1B-Chat, was replaced
# after failing this same test even with its own correct chat template
# and grammar-constrained decoding — see KNOWN_LIMITATIONS.md).
_CHAT_TEMPLATE = "<|im_start|>system\n{system}<|im_end|>\n<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n"


def _chat_prompt(system: str, user: str) -> str:
    return _CHAT_TEMPLATE.format(system=system, user=user)


class Reasoner:
    def __init__(self, llm_call: Callable[[str], str] | None = None):
        # A single injected `llm_call` (used throughout the test suite) is
        # deliberately reused for BOTH reply() and plan_actions() — this
        # preserves the existing dependency-injection contract exactly.
        # Only the real default path (llm_call=None) gets the upgrade: one
        # shared model, wrapped as two call sites, one grammar-constrained.
        if llm_call is None:
            llm = _load_default_llm()
            self._llm_call = _make_llm_call(llm)
            self._plan_llm_call = _make_llm_call(llm, grammar=_ACTION_PLAN_GRAMMAR)
        else:
            self._llm_call = llm_call
            self._plan_llm_call = llm_call

    def reply(self, user_text: str, memory) -> str:
        system = (
            "You are a lamp robot. Reply briefly and naturally to what the "
            "person said, using the scene memory as context if it's relevant."
        )
        user = f"Scene memory:\n{memory.as_prompt_text()}\n\nThe person said: {user_text}"
        prompt = _chat_prompt(system, user)
        return self._llm_call(prompt).strip()

    def plan_actions(self, goal_text: str, memory) -> list[dict]:
        system = _PLAN_SYSTEM % (list(ACTIONS), _ALLOWED_VALUES)
        user = f"Scene memory:\n{memory.as_prompt_text()}\n\nGoal: {goal_text}"
        prompt = _chat_prompt(system, user)
        raw = self._plan_llm_call(prompt)
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


# A minimal JSON grammar restricted at the top level to "array of {name,
# params} objects" — the exact shape plan_actions() expects. This is
# grammar-CONSTRAINED decoding (llama.cpp's GBNF), not prompt engineering:
# every token the model emits is checked against this grammar as it's
# generated, so the output is *structurally* guaranteed to parse as JSON
# matching this shape, regardless of how well the underlying model
# naturally follows instructions. shared.action_vocabulary.is_valid_action
# still does the semantic check afterward (is "look_at" a real action, is
# "direction" one of the allowed values) — the grammar only guarantees
# syntax, not semantics. This is the actual fix for the failure mode
# confirmed during deployment testing: a small local model producing
# prose, markdown-fenced JSON, or truncated JSON instead of a clean array.
_ACTION_PLAN_GRAMMAR = r'''
root   ::= "[" ws (action ("," ws action)*)? ws "]"
action ::= "{" ws "\"name\"" ws ":" ws string ws "," ws "\"params\"" ws ":" ws object ws "}"
object ::= "{" ws (pair ("," ws pair)*)? ws "}"
pair   ::= string ws ":" ws value
value  ::= string | number | object | array | "true" | "false" | "null"
array  ::= "[" ws (value ("," ws value)*)? ws "]"
string ::= "\"" ([^"\\] | "\\" .)* "\""
number ::= "-"? [0-9]+ ("." [0-9]+)?
ws     ::= [ \t\n]*
'''


def _load_default_llm():
    from llama_cpp import Llama

    return Llama(model_path="models/llm/model.gguf", n_ctx=2048, verbose=False)


def _make_llm_call(llm, grammar: str | None = None) -> Callable[[str], str]:
    compiled_grammar = None
    if grammar is not None:
        from llama_cpp import LlamaGrammar

        compiled_grammar = LlamaGrammar.from_string(grammar)

    def call(prompt: str) -> str:
        # "<|im_end|>" is ChatML's real end-of-turn token; "<|im_start|>" is
        # a safety net in case the model tries to hallucinate a new turn
        # instead of stopping cleanly.
        result = llm(
            prompt,
            max_tokens=256,
            stop=["<|im_end|>", "<|im_start|>"],
            grammar=compiled_grammar,
        )
        return result["choices"][0]["text"]

    return call
