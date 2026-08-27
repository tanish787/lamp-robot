from brain.memory import SceneMemory
from brain.perception import ScenePerception


class FakeDetector:
    def __init__(self, detections):
        self._detections = detections

    def detect(self, frame):
        return self._detections


def test_observe_writes_each_detection_into_memory():
    detector = FakeDetector([
        {"label": "mug", "color": "red", "position": (0.1, 0.2)},
        {"label": "bottle", "color": "blue", "position": (0.5, 0.5)},
    ])
    memory = SceneMemory()
    perception = ScenePerception(detector=detector)

    labels = perception.observe(frame=None, memory=memory, timestamp=1.0)

    assert labels == ["mug", "bottle"]
    assert len(memory.records()) == 2


def test_observe_with_no_detections_returns_empty_list():
    perception = ScenePerception(detector=FakeDetector([]))
    memory = SceneMemory()
    assert perception.observe(frame=None, memory=memory, timestamp=1.0) == []
