from brain.memory import SceneMemory


def test_observe_creates_a_new_record():
    mem = SceneMemory()
    record_id = mem.observe("mug", {"color": "red", "position": (0.1, 0.2)}, timestamp=1.0)
    record = mem.get(record_id)
    assert record["label"] == "mug"
    assert record["attributes"]["color"] == "red"
    assert record["first_seen_ts"] == 1.0
    assert record["last_seen_ts"] == 1.0


def test_observe_updates_existing_record_for_same_object():
    mem = SceneMemory()
    first_id = mem.observe("mug", {"color": "red", "position": (0.1, 0.2)}, timestamp=1.0)
    second_id = mem.observe("mug", {"color": "red", "position": (0.11, 0.19)}, timestamp=2.0)
    assert first_id == second_id
    assert len(mem.records()) == 1
    assert mem.get(first_id)["last_seen_ts"] == 2.0


def test_observe_treats_far_apart_same_label_objects_as_distinct():
    mem = SceneMemory()
    mem.observe("mug", {"color": "red", "position": (0.0, 0.0)}, timestamp=1.0)
    mem.observe("mug", {"color": "blue", "position": (5.0, 5.0)}, timestamp=1.0)
    assert len(mem.records()) == 2


def test_as_prompt_text_includes_every_record_label():
    mem = SceneMemory()
    mem.observe("mug", {"color": "red", "position": (0.0, 0.0)}, timestamp=1.0)
    mem.observe("bottle", {"color": "blue", "position": (2.0, 2.0)}, timestamp=1.0)
    text = mem.as_prompt_text()
    assert "mug" in text
    assert "bottle" in text


def test_as_prompt_text_on_empty_memory_says_nothing_observed():
    assert "nothing" in SceneMemory().as_prompt_text().lower()
