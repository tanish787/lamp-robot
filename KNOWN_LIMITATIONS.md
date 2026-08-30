# Known limitations

Honest list of what this repository does not do. Some entries are
intentional scope boundaries from the design spec; others are resilience
work that was consciously deferred while the five demo moments were being
built. Nothing here is a surprise to the authors — it is written down so it
is not a surprise to a reader either.

## Deferred resilience work (design spec §8)

These are named in the spec's error-handling table but are **not
implemented**:

| Gap | What the spec asked for | What happens today |
|---|---|---|
| **LLM call timeout** | an ~8-10 s timeout on the reasoning/planning call, falling back to a canned "still thinking" filler (`speak` + `idle_sway`) | the call runs on a worker thread with no deadline. A slow model leaves the character silent for as long as it takes. The event loop keeps running (engagement still reacts), but nothing fills the pause. |
| **Retry with a stricter re-prompt** | on an invalid/malformed action plan, retry once with a tightened prompt before falling back | `Reasoner.plan_actions` validates, drops invalid steps, and falls straight through to the `idle_sway` fallback. There is no second attempt, and no spoken apology. |
| **Mic fail-fast at startup** | explicit device-open check for camera *and* microphone at Brain startup, failing with a clear named error | the camera is checked (`brain.main.Camera` raises if `cv2.VideoCapture` will not open). The microphone is not: `MicStream` opens the device lazily on the first `read_frames`, so a missing/blocked mic surfaces as a logged error inside the dialogue loop rather than a startup failure. |
| **WebSocket heartbeat and reconnect** | ping/heartbeat on the Brain↔Body connection; Brain pauses and reconnects on disconnect, Body holds its last pose | there is no heartbeat and no reconnect. Body was hardened so a malformed frame can no longer close the connection, but if the socket does drop (Body process killed, for example) Brain does not recover — the run has to be restarted. |

## Model assets and offline operation

The spec calls for pinned weights fetched once into `models/` so that
nothing touches the network at runtime. `scripts/setup.sh` does this for the
Piper voice and the GGUF LLM. It does **not** for the other two:

- `faster-whisper` downloads its CTranslate2 weights from Hugging Face on
  first use, into the Hugging Face cache, not `models/`.
- `ultralytics` downloads `yolov8n.pt` on first use into its own cache.

`setup.sh` warms both caches so the first run after setup is offline in
practice, but they remain library-managed rather than vendored. Making them
genuinely offline-first means passing explicit local paths to
`WhisperModel()` and `YOLO()`.

Dependencies are pinned as version *floors* (`>=`), not exact versions, so
`pip install -r requirements.txt` is reproducible in API terms but not
byte-for-byte.

## Assets

`body/assets/sfx/*.wav` and `body/assets/music/ambient.wav` are generated
placeholder tones, not designed sound effects. They exist so the demo runs
out of the box; replace them before recording (see `body/assets/README.md`).

`body/main.py`'s music playback does not actually loop — `aplay` has no
loop flag and the clip is played once.

## Simulation and motion

- The "light" is a tracked state attribute, not a physical light source or
  a rendered emissive material. `LightState` records `off`/`dim`/`bright`/
  `pulse`; nothing in the PyBullet view changes colour yet.
- `pulse` is a state, not an animation — there is no oscillating brightness.
- Motion actions are open-loop joint interpolation with soft-limit clamping.
  There is no velocity/acceleration limiting beyond the per-step
  interpolation rate, and no collision checking against the environment.
- Each `apply_action` call blocks the Body event loop while it steps the
  simulation. At the demo's action rate this is not noticeable, but Body
  cannot service a second command mid-motion.

## Perception, memory and dialogue

- Scene memory is in-process only, with no persistence across restarts and
  no retrieval/ranking layer — the whole list is serialized into every
  prompt. Fine at a handful of objects; it would need keyword filtering
  before it hit context limits (spec §7's documented escape hatch).
- Goal-vs-question routing in `Orchestrator.handle_utterance` is a keyword
  heuristic, not a classifier. It misroutes unusual phrasings; the result is
  a reply where an action was wanted (or vice versa), never an unsafe action.
- The dialogue loop records in fixed windows (`LISTEN_WINDOW_S`), so an
  utterance that straddles a window boundary can be clipped. There is no
  barge-in: speech captured while the character is talking is not handled
  specially.
- Local small-model quality and latency are materially below a cloud
  LLM/VLM. Multi-second responses on CPU-only hardware are expected.

## Process and test environment

- `.cache/` (the patched-URDF cache) is shared by every process in the
  repository. Two Body processes started simultaneously against the same
  cache directory could race while writing it. Tests avoid this by passing
  a per-test `cache_dir`; the real entry point does not, so run one Body at
  a time. A per-process cache directory would be the fix.
- The live camera and microphone loops in `brain/main.py` are covered by
  tests only with fake devices (`tests/brain/test_main_loops.py`). Real
  device behaviour is verified in the manual live pass.
- No systemd/service packaging. The demo is a manually launched two-process
  run.
- Audio playback originally used `simpleaudio`, which segfaulted reliably
  during real-hardware testing on an Ubuntu 24.04 VM (confirmed in
  isolation: Piper synthesis and the system's own `aplay` both worked
  correctly against the same audio device; only `simpleaudio`'s native
  bindings crashed). It was replaced with a direct `aplay` subprocess call
  in both `body/main.py` and `brain/tts.py`. This is exactly the class of
  bug the deployment target's real hardware surfaces and a dev-machine
  test suite cannot — noted here as a concrete example of why the manual
  live pass on real hardware matters, not just as a fixed bug.
- The originally pinned LLM, TinyLlama-1.1B-Chat, was replaced with
  Qwen2.5-1.5B-Instruct after real-hardware testing. TinyLlama produced
  empty replies with plain-text prompts; after fixing the chat template
  and adding grammar-constrained decoding (GBNF, forcing `plan_actions`'
  output to be structurally valid JSON regardless of the model's own
  instruction-following quality — see `brain/reasoning.py`), it still
  produced syntactically valid but semantically incoherent plans (e.g.
  listing every action in the vocabulary with mismatched, hallucinated
  parameters, rather than reasoning about the actual goal). This confirmed
  the failure was model capacity, not prompt or decoding engineering.
  Qwen2.5-1.5B was chosen as a free, open-weight model known for
  stronger instruction-following relative to its size. The
  grammar-constrained decoding was kept regardless of model choice — it's
  a structural guarantee, not a workaround for one weak model, and is the
  correct tool for reliable structured output from any small local model.
- `MediaPipeFaceMonitor` originally used the legacy `mp.solutions` API
  (`mp.solutions.face_detection`), which real-hardware testing found is
  gone from mediapipe entirely — confirmed by direct introspection
  (`hasattr(mediapipe, 'solutions')` is `False`) on both `1.0.1` and
  `0.10.35`. This was not a version-boundary problem an upper-bound pin
  could fix; the class now uses the Tasks API instead
  (`mediapipe.tasks.python.vision.FaceDetector`), which needs an explicit
  model file — `scripts/setup.sh` downloads `blaze_face_short_range.tflite`
  (Google's official small face-detector model) into `models/mediapipe/`.
