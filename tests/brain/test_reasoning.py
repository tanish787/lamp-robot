import json

from brain.memory import SceneMemory
from brain.reasoning import Reasoner


def test_reply_returns_llm_text_directly():
    reasoner = Reasoner(llm_call=lambda prompt: "Hi! I see a red mug on the table.")
    assert reasoner.reply("what do you see?", SceneMemory()) == "Hi! I see a red mug on the table."


def test_plan_actions_parses_valid_json_action_list():
    valid_plan = json.dumps([
        {"name": "scan_sweep", "params": {}},
        {"name": "look_at", "params": {"direction": "left"}},
    ])
    reasoner = Reasoner(llm_call=lambda prompt: valid_plan)
    actions = reasoner.plan_actions("look at the mug", SceneMemory())
    assert actions == [
        {"name": "scan_sweep", "params": {}},
        {"name": "look_at", "params": {"direction": "left"}},
    ]


def test_plan_actions_drops_invalid_action_and_keeps_valid_ones():
    mixed_plan = json.dumps([
        {"name": "teleport", "params": {}},
        {"name": "nod", "params": {}},
    ])
    calls = []

    def fake_llm(prompt):
        calls.append(prompt)
        return mixed_plan

    reasoner = Reasoner(llm_call=fake_llm)
    actions = reasoner.plan_actions("greet them", SceneMemory())
    assert actions == [{"name": "nod", "params": {}}]
    assert len(calls) == 1  # no retry needed once a valid action remains


def test_plan_actions_falls_back_to_idle_sway_when_nothing_valid_survives():
    reasoner = Reasoner(llm_call=lambda prompt: json.dumps([{"name": "teleport", "params": {}}]))
    actions = reasoner.plan_actions("do something impossible", SceneMemory())
    assert actions == [{"name": "idle_sway", "params": {}}]


def test_plan_actions_falls_back_on_malformed_json():
    reasoner = Reasoner(llm_call=lambda prompt: "not json at all")
    actions = reasoner.plan_actions("goal", SceneMemory())
    assert actions == [{"name": "idle_sway", "params": {}}]


def test_plan_actions_drops_action_with_non_dict_params():
    plan_with_null_params = json.dumps([{"name": "look_at", "params": None}])
    reasoner = Reasoner(llm_call=lambda prompt: plan_with_null_params)
    actions = reasoner.plan_actions("look somewhere", SceneMemory())
    assert actions == [{"name": "idle_sway", "params": {}}]


def test_plan_actions_fallback_returns_independent_objects_across_calls():
    reasoner = Reasoner(llm_call=lambda prompt: "not json at all")
    first = reasoner.plan_actions("goal", SceneMemory())
    first[0]["params"]["poisoned"] = True
    first[0]["name"] = "mutated"

    second = reasoner.plan_actions("goal", SceneMemory())
    assert second == [{"name": "idle_sway", "params": {}}]
    assert second[0] is not first[0]
    assert second[0]["params"] is not first[0]["params"]
