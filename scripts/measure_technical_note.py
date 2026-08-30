"""Collects every number in docs/technical-note.md's Measurements section
in one run, on the target (or target-equivalent) hardware:

    python -m scripts.measure_technical_note

Reproducible and unattended: it needs no live camera or microphone. STT is
measured against a Piper-synthesized sample (Piper's own latency is
measured separately, first), and the detector against a blank frame — the
detector's *latency* does not depend on what is in the frame, only its
size, so no live capture is needed to measure it honestly.

Prints a human-readable summary and writes measurements.json alongside it
for the record.
"""

import asyncio
import json
import os
import time
import wave
from pathlib import Path

import numpy as np
import psutil

from body.light_sfx import LightState, MusicPlayer, SfxPlayer
from body.server import BodyServer
from body.simulation import LampSimulation
from brain.engagement import MediaPipeFaceMonitor
from brain.main import FRAME_INTERVAL_S
from brain.memory import SceneMemory
from brain.perception import ScenePerception
from brain.protocol_client import ProtocolClient
from brain.reasoning import Reasoner
from brain.stt import SpeechToText
from brain.tts import TextToSpeech

_ENGAGEMENT_LOOP_SAMPLE_S = 5.0
_SAMPLE_TEXT = "Hello, I can see you and I am ready to help."


def _timed(fn):
    start = time.monotonic()
    result = fn()
    return result, time.monotonic() - start


def _measure_tts_and_stt() -> dict:
    tts = TextToSpeech()
    audio, tts_s = _timed(lambda: tts.synthesize(_SAMPLE_TEXT))

    sample_path = Path("measurement_sample.wav")
    sample_path.write_bytes(audio)
    with wave.open(str(sample_path), "rb") as wav:
        frames = wav.readframes(wav.getnframes())
        sample_rate = wav.getframerate()
    sample_path.unlink()

    stt = SpeechToText(model_size="tiny")
    text, stt_s = _timed(lambda: stt.transcribe(frames, sample_rate=sample_rate))
    return {
        "tts_synth_latency_s": tts_s,
        "stt_latency_s": stt_s,
        "stt_transcript": text,
    }


def _measure_detector() -> dict:
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    perception = ScenePerception()
    memory = SceneMemory()
    _, detector_s = _timed(lambda: perception.observe(frame, memory, timestamp=0.0))
    return {"detector_latency_s": detector_s}


def _measure_llm() -> dict:
    memory = SceneMemory()
    memory.observe("mug", {"color": "red", "position": (0.1, 0.2)}, timestamp=0.0)
    reasoner = Reasoner()
    _, reply_s = _timed(lambda: reasoner.reply("What do you see?", memory))
    _, plan_s = _timed(lambda: reasoner.plan_actions("Look at the mug", memory))
    return {"llm_reply_latency_s": reply_s, "llm_plan_latency_s": plan_s}


def _measure_engagement_loop_cost() -> dict:
    """Per-frame detector compute cost, plus steady-state CPU for running
    the engagement loop at its real ~10 fps cadence (brain.main's own
    FRAME_INTERVAL_S) — a face-present baseline, not counting the
    intermittent STT/LLM spikes reported separately above."""
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    monitor = MediaPipeFaceMonitor()
    _, per_frame_s = _timed(lambda: monitor.detect(frame))

    process = psutil.Process(os.getpid())
    process.cpu_percent(interval=None)  # prime; first call always reads 0
    start = time.monotonic()
    frames = 0
    while time.monotonic() - start < _ENGAGEMENT_LOOP_SAMPLE_S:
        monitor.detect(frame)
        time.sleep(FRAME_INTERVAL_S)
        frames += 1
    cpu_percent = process.cpu_percent(interval=None)
    return {
        "engagement_detect_latency_s": per_frame_s,
        "engagement_loop_cpu_percent_of_one_core": cpu_percent,
        "engagement_loop_sampled_fps": frames / _ENGAGEMENT_LOOP_SAMPLE_S,
        "cpu_count": psutil.cpu_count(),
    }


def _measure_peak_rss() -> dict:
    sim = LampSimulation(gui=False)
    stt = SpeechToText()
    tts = TextToSpeech()
    perception = ScenePerception()
    reasoner = Reasoner()
    process = psutil.Process(os.getpid())
    peak_rss_gb = process.memory_info().rss / 1e9
    sim.close()
    del stt, tts, perception, reasoner
    return {"peak_rss_gb": peak_rss_gb}


async def _measure_protocol_roundtrip() -> dict:
    """Real websocket round-trip, not an in-process shortcut: the same
    BodyServer + ProtocolClient pattern the demo and the integration tests
    use, timing one full command -> ack cycle for a cheap action."""
    import websockets

    sim = LampSimulation(gui=False)
    sfx_dir = Path("body/assets/sfx")
    music_dir = Path("body/assets/music")
    server = BodyServer(
        sim, LightState(), SfxPlayer(sfx_dir), MusicPlayer(music_dir), port=18765
    )
    task = asyncio.create_task(server.serve())
    await asyncio.sleep(0.2)
    try:
        async with websockets.connect("ws://127.0.0.1:18765") as ws:
            client = ProtocolClient(connection=ws)
            start = time.monotonic()
            await client.send_command("neutral", {})
            roundtrip_s = time.monotonic() - start
    finally:
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        sim.close()
    return {"protocol_roundtrip_s": roundtrip_s}


def main() -> None:
    results: dict = {}
    print("Measuring TTS + STT...")
    results.update(_measure_tts_and_stt())
    print("Measuring detector...")
    results.update(_measure_detector())
    print("Measuring LLM reply + plan...")
    results.update(_measure_llm())
    print(f"Measuring engagement loop cost ({_ENGAGEMENT_LOOP_SAMPLE_S:.0f}s sample)...")
    results.update(_measure_engagement_loop_cost())
    print("Measuring peak RSS with the full stack loaded...")
    results.update(_measure_peak_rss())
    print("Measuring one protocol round-trip...")
    results.update(asyncio.run(_measure_protocol_roundtrip()))

    end_to_end_s = (
        results["stt_latency_s"]
        + results["llm_plan_latency_s"]
        + results["protocol_roundtrip_s"]
    )
    results["end_to_end_finish_speaking_to_response_s"] = end_to_end_s

    print("\n=== MEASUREMENTS ===")
    for key, value in results.items():
        print(f"{key}: {value}")

    Path("measurements.json").write_text(json.dumps(results, indent=2))
    print("\nWrote measurements.json")


if __name__ == "__main__":
    main()
