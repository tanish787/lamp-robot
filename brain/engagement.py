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
    """Thin wrapper around mediapipe's Tasks API face detector. Frames are
    supplied by the caller (brain/main.py owns the single capture device),
    so this holds no camera of its own. Exercised against a real camera in
    the manual live pass, not unit-tested.

    Uses the Tasks API (mediapipe.tasks), not the older mp.solutions API
    this class originally used: testing on the target VM found that
    mp.solutions has been removed from every mediapipe release currently
    installable (confirmed on both 1.0.1 and 0.10.35 — this was not a
    version-boundary issue, the legacy API is simply gone). The Tasks API
    needs an explicit model file rather than a bundled default; see
    scripts/setup.sh, which downloads it to the path below.
    """

    def __init__(
        self,
        min_detection_confidence: float = 0.6,
        model_path: str = "models/mediapipe/blaze_face_short_range.tflite",
    ):
        import mediapipe as mp
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision

        self._mp = mp
        base_options = mp_python.BaseOptions(model_asset_path=model_path)
        options = vision.FaceDetectorOptions(
            base_options=base_options, min_detection_confidence=min_detection_confidence
        )
        self._detector = vision.FaceDetector.create_from_options(options)

    def detect(self, frame) -> bool:
        image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=frame)
        result = self._detector.detect(image)
        return bool(result.detections)
