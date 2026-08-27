import argparse
import asyncio

from brain.audio_capture import MicStream, Vad
from brain.engagement import MediaPipeFaceMonitor, EngagementDebouncer
from brain.memory import SceneMemory
from brain.orchestrator import Orchestrator
from brain.perception import ScenePerception
from brain.protocol_client import ProtocolClient
from brain.reasoning import Reasoner
from brain.stt import SpeechToText
from brain.tts import TextToSpeech


async def main_async(uri: str) -> None:
    client = ProtocolClient(uri=uri)
    await client.connect()

    orchestrator = Orchestrator(
        tts=TextToSpeech(),
        protocol_client=client,
        engagement=EngagementDebouncer(),
        audio=MicStream(),
        stt=SpeechToText(),
        perception=ScenePerception(),
        reasoner=Reasoner(),
        memory=SceneMemory(),
    )
    face_monitor = MediaPipeFaceMonitor()

    print("Brain running. Ctrl+C to stop.")
    # Full camera-loop wiring (frame capture -> face_monitor.detect ->
    # orchestrator.on_engagement_change, plus the mic/STT/goal path) is
    # exercised via the manual live demo pass, not unit-tested here — see
    # the spec's testing strategy for why perception loops aren't
    # meaningfully unit-testable. This entry point wires the pieces
    # together; scripts/smoke_*.py and tests/integration validate each
    # piece and the command-sequencing logic independently.
    while True:
        await asyncio.sleep(1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--body-uri", default="ws://127.0.0.1:8765")
    args = parser.parse_args()
    asyncio.run(main_async(args.body_uri))


if __name__ == "__main__":
    main()
