# HCL Lamp Robot — Live Character

A live, expressive character built around a simulated 5-DOF lamp robot. Uses
a laptop camera to see, a microphone to listen, and a speaker for voice,
sound effects, and music. Runs two local processes — a perception/reasoning
"Brain" and a simulation/motion "Body" — connected over a local protocol.

Everything runs locally on CPU. No cloud services, no API keys.

## Requirements

- Linux (developed against Ubuntu 24.04) or another platform with a working
  camera, microphone and audio output. Windows works for the test suite.
- Python 3.11+
- A webcam and a microphone, plus speakers or headphones
- ~8 GB RAM (the resource-ceiling check budgets 7 GB for the full model
  stack; see `scripts/resource_ceiling_check.py`)
- Roughly 2 GB of disk for the downloaded model weights

On a fresh Ubuntu install the user may need to be in the `video` and `audio`
groups for camera/mic access:

```bash
sudo usermod -aG video,audio "$USER"   # log out and back in
```

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
./scripts/setup.sh
```

`setup.sh` installs `requirements.txt` and downloads the model weights (a
Piper voice and a GGUF LLM into `models/`, plus warming the caches that
`faster-whisper` and `ultralytics` manage themselves). Read the comment at
the top of the script — it is explicit about which weights are genuinely
vendored and which are library-cached. See `KNOWN_LIMITATIONS.md`.

To install dependencies only, without models:

```bash
pip install -r requirements.txt
```

## Running the demo

```bash
./scripts/run_all.sh
```

This starts Body, waits for its WebSocket to accept connections, then starts
Brain in the foreground. Ctrl+C stops both.

- Body listens on `ws://127.0.0.1:8765`. Override with `BODY_PORT=9000
  ./scripts/run_all.sh`.
- Brain opens camera index 0. Override with
  `python -m brain.main --camera 1`.

To run the halves separately — useful when you want the PyBullet window:

```bash
./scripts/run_body.sh --gui              # terminal 1
python -m brain.main                     # terminal 2
```

Then sit in front of the camera. The lamp should lean in, pulse its light
and chime when it notices you; talk to it and it replies; ask it to find
something and it plans and runs an action sequence.

## Testing

```bash
pytest                                   # full suite, no devices needed
python -m scripts.smoke_stt sample.wav   # per-model smoke checks (real models)
python -m scripts.smoke_tts "Hello."
python -m scripts.smoke_detector sample.jpg
python -m scripts.smoke_llm
python -m scripts.resource_ceiling_check # loads the whole stack, exits 1 over budget
```

The test suite runs headless and needs no camera, microphone or model
weights: model-shaped edges are faked and everything else — the protocol,
the simulation, the action vocabulary, the orchestrator — is real. The smoke
scripts are the manual checks that exercise the real models on the target
hardware.

## Layout

| Path | What lives there |
|---|---|
| `shared/` | The Brain↔Body contract: the action vocabulary and the wire schema. The only module both sides import. |
| `body/` | PyBullet simulation, motion kinematics with soft-limit clamping, light/sfx state, and the WebSocket server. |
| `brain/` | Camera engagement, VAD/STT/TTS, the object detector, scene memory, the local LLM reasoner, and the orchestrator that sequences them. |
| `scripts/` | Setup, launchers, per-model smoke checks, the resource-ceiling check. |
| `tests/` | Unit tests per module plus in-process Brain+Body integration tests. |

## Architecture

See `docs/technical-note.md` for the architecture diagram, the protocol,
the model-to-action design, and measurements.

Known gaps and intentional scope boundaries are listed in
`KNOWN_LIMITATIONS.md`.
