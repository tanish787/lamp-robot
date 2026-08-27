"""Turns a noisy per-frame face-detected signal into stable engaged/
disengaged transitions. A face must be (continuously) present or absent
for `hold_seconds` before the state actually flips, so a brief glance away
or a missed detection doesn't trigger a spurious disengage/re-engage.
"""


class EngagementDebouncer:
    def __init__(self, hold_seconds: float = 0.75):
        self._hold_seconds = hold_seconds
        self._engaged = False
        self._candidate: bool | None = None
        self._candidate_since: float | None = None

    def update(self, face_detected: bool, now: float) -> bool | None:
        if face_detected == self._engaged:
            # Signal matches current state; any pending flip is stale.
            self._candidate = None
            self._candidate_since = None
            return None

        if self._candidate != face_detected:
            self._candidate = face_detected
            self._candidate_since = now
            return None

        if now - self._candidate_since >= self._hold_seconds:
            self._engaged = face_detected
            self._candidate = None
            self._candidate_since = None
            return face_detected

        return None


class MediaPipeFaceMonitor:
    """Thin wrapper around mediapipe face detection. Exercised by
    scripts/smoke_engagement.py against a real camera, not unit-tested."""

    def __init__(self, camera_index: int = 0):
        import mediapipe as mp

        self._camera_index = camera_index
        self._detector = mp.solutions.face_detection.FaceDetection(
            model_selection=0, min_detection_confidence=0.6
        )

    def detect(self, frame) -> bool:
        results = self._detector.process(frame)
        return bool(results.detections)
