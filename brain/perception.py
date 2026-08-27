"""Vision is composed, not fused: the detector produces plain text labels
+ attributes, which brain.reasoning later reasons over as text. No vision-
language model is loaded here."""


class ScenePerception:
    def __init__(self, detector=None):
        if detector is None:
            detector = _Yolov8nDetector()
        self._detector = detector

    def observe(self, frame, memory, timestamp: float) -> list[str]:
        detections = self._detector.detect(frame)
        labels = []
        for detection in detections:
            label = detection["label"]
            attributes = {k: v for k, v in detection.items() if k != "label"}
            memory.observe(label, attributes, timestamp)
            labels.append(label)
        return labels


class _Yolov8nDetector:
    """Thin wrapper around a YOLOv8n model, exercised by
    scripts/smoke_detector.py against a real camera frame."""

    def __init__(self):
        from ultralytics import YOLO

        self._model = YOLO("yolov8n.pt")

    def detect(self, frame) -> list[dict]:
        results = self._model(frame, verbose=False)[0]
        detections = []
        for box in results.boxes:
            label = results.names[int(box.cls[0])]
            x_center, y_center, _, _ = box.xywhn[0].tolist()
            detections.append({"label": label, "position": (x_center, y_center)})
        return detections
