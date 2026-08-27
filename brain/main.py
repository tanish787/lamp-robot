"""Brain entry point: the live perception/dialogue loop.

Design (kept deliberately small — see docs/technical-note.md):

Two concurrent asyncio tasks share one Orchestrator, one camera and one
connection to Body.

  * `_engagement_loop` is the only thing that touches the camera. It reads
    a frame, hands it to MediaPipe, feeds the boolean through
    `EngagementDebouncer`, and fires `Orchestrator.on_engagement_change`
    on a real transition. It also caches the newest frame so scene
    perception can look at what the camera is seeing without opening a
    second capture device.
  * `_dialogue_loop` idles until engagement is on, then records a listening
    window from the mic, cuts an utterance out of it with the VAD, and
    hands the audio to `Orchestrator.handle_utterance`, which transcribes
    it and either replies or plans and runs an action sequence.

Both loops are supervised: an exception inside one iteration is logged and
the loop continues, because a single dropped frame or a transient audio
error must not end a live demo.

The loops themselves are verified in the manual live pass rather than by
unit tests (real camera/mic, per the spec's testing strategy). Everything
they call — debounce, segmentation, the Orchestrator's routing, action
validation, the protocol — is unit- and integration-tested.
"""

import argparse
import asyncio
import logging
import time

from brain.audio_capture import MicStream, Vad, segment_utterance
from brain.engagement import EngagementDebouncer, MediaPipeFaceMonitor
from brain.memory import SceneMemory
from brain.orchestrator import Orchestrator
from brain.perception import ScenePerception
from brain.protocol_client import ProtocolClient
from brain.reasoning import Reasoner
from brain.stt import SpeechToText
from brain.tts import TextToSpeech

_LOG = logging.getLogger("brain")

FRAME_INTERVAL_S = 0.1      # ~10 fps is plenty for face presence
LISTEN_WINDOW_S = 4.0       # one recording chunk handed to the VAD
OBSERVE_INTERVAL_S = 20.0   # opportunistic scene refresh while engaged


class Camera:
    """Single owner of the capture device.

    Fails fast at construction if the camera cannot be opened, rather than
    silently running a demo that can never engage (spec section 8).
    """

    def __init__(self, index: int = 0):
        import cv2

        self._cv2 = cv2
        self._capture = cv2.VideoCapture(index)
        if not self._capture.isOpened():
            raise RuntimeError(
                f"camera {index} could not be opened — check it is connected and "
                f"that this user has permission to use it (video group on Ubuntu)"
            )

    def read(self):
        """Return the next frame as RGB (MediaPipe's expected layout), or
        None if the grab failed."""
        ok, frame = self._capture.read()
        if not ok or frame is None:
            return None
        return self._cv2.cvtColor(frame, self._cv2.COLOR_BGR2RGB)

    def close(self) -> None:
        self._capture.release()


class LatestFrame:
    """Newest camera frame, published by the engagement loop and read by
    scene perception. Plain attribute assignment is atomic enough here —
    only one writer, and readers only ever want 'whatever is current'."""

    def __init__(self):
        self._frame = None

    def set(self, frame) -> None:
        self._frame = frame

    def get(self):
        return self._frame


async def _engagement_loop(orchestrator, face_monitor, debouncer, camera, latest, engaged):
    last_observed = 0.0
    while True:
        try:
            frame = await asyncio.to_thread(camera.read)
            if frame is None:
                await asyncio.sleep(FRAME_INTERVAL_S)
                continue
            latest.set(frame)

            detected = await asyncio.to_thread(face_monitor.detect, frame)
            transition = debouncer.update(detected, time.monotonic())

            if transition is True:
                engaged.set()
                _LOG.info("engaged")
                await orchestrator.on_engagement_change(True)
                # Moment 4: take stock of the scene as soon as someone is
                # here, so a question asked straight away has context.
                await orchestrator.observe_scene(frame)
                last_observed = time.monotonic()
            elif transition is False:
                engaged.clear()
                _LOG.info("disengaged")
                await orchestrator.on_engagement_change(False)
            elif engaged.is_set() and time.monotonic() - last_observed > OBSERVE_INTERVAL_S:
                await orchestrator.observe_scene(frame)
                last_observed = time.monotonic()
        except asyncio.CancelledError:
            raise
        except Exception:  # noqa: BLE001 - a dropped frame must not end the demo
            _LOG.exception("engagement loop iteration failed")

        await asyncio.sleep(FRAME_INTERVAL_S)


async def _dialogue_loop(orchestrator, mic, vad, engaged):
    while True:
        await engaged.wait()
        try:
            frames = await asyncio.to_thread(mic.read_frames, LISTEN_WINDOW_S)
            utterance = segment_utterance(frames, vad.is_speech)
            if utterance:
                await orchestrator.handle_utterance(utterance)
        except asyncio.CancelledError:
            raise
        except Exception:  # noqa: BLE001 - keep listening after a bad chunk
            _LOG.exception("dialogue loop iteration failed")
            # A device that fails instantly would otherwise spin this loop
            # (a healthy read_frames blocks for the listening window).
            await asyncio.sleep(0.5)


async def main_async(uri: str, camera_index: int = 0) -> None:
    client = ProtocolClient(uri=uri)
    await client.connect()

    camera = Camera(camera_index)
    latest = LatestFrame()
    try:
        debouncer = EngagementDebouncer()
        mic = MicStream()
        vad = Vad()
        face_monitor = MediaPipeFaceMonitor()
        engaged = asyncio.Event()

        orchestrator = Orchestrator(
            tts=TextToSpeech(),
            protocol_client=client,
            engagement=debouncer,
            audio=mic,
            stt=SpeechToText(),
            perception=ScenePerception(),
            reasoner=Reasoner(),
            memory=SceneMemory(),
            frame_source=latest.get,
        )

        print("Brain running. Ctrl+C to stop.")
        await asyncio.gather(
            _engagement_loop(orchestrator, face_monitor, debouncer, camera, latest, engaged),
            _dialogue_loop(orchestrator, mic, vad, engaged),
        )
    finally:
        camera.close()
        await client.close()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--body-uri", default="ws://127.0.0.1:8765")
    parser.add_argument("--camera", type=int, default=0, help="camera device index")
    args = parser.parse_args()
    try:
        asyncio.run(main_async(args.body_uri, args.camera))
    except KeyboardInterrupt:
        print("\nBrain stopped.")


if __name__ == "__main__":
    main()
