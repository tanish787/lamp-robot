#HCL lamp robot live character

## 1. Architecture

5DOF desk lamp character: notices someone via camera, reacts, converses,
remembers what it has seen, and carries out spoken goals as action
sequences. GPU-less ~8 GB machine, no network access at demo time.

```
                    ┌──────────────────── BRAIN (process 1) ────────────────────┐
   camera ─────────▶│  Camera ─┬─▶ MediaPipe face detect ─▶ EngagementDebouncer │
                    │          │                                    │           │
                    │          └─▶ YOLOv8n detector ─▶ SceneMemory   │           │
                    │                                    ▲   │       │           │
   microphone ─────▶│  MicStream ─▶ WebRTC VAD ─▶ whisper─┘   │       │           │
                    │                     (STT)               ▼       ▼           │
                    │                              ┌──────────────────────┐      │
                    │                              │     Orchestrator     │      │
                    │                              └──────────┬───────────┘      │
                    │                     ┌───────────────────┼──────────┐       │
                    │                     ▼                   ▼          ▼       │
   speaker ◀────────│              Piper (TTS)        Reasoner (GGUF   ProtocolClient
                    │                                  llama.cpp)          │      │
                    └─────────────────────────────────────────────────────│──────┘
                                                                          │
                                            JSON over ws://127.0.0.1:8765 │
                                                                          │
                    ┌──────────────────── BODY (process 2) ───────────────│──────┐
                    │  BodyServer ─┬─▶ motion.resolve_waypoints ─▶ LampSimulation │
                    │              │        (soft-limit clamp)      (PyBullet)    │
                    │              ├─▶ LightState                                 │
                    │              └─▶ SfxPlayer / MusicPlayer                    │
                    └───────────────────────────────────────────────────────────┘
```

Two processes: Brain blocks for seconds at a time on model inference;
Body must stay responsive to the simulation. Either half restarts or gets
replaced independently — Body with a real robot controller, for instance.
`shared/` (the action vocabulary and wire schema) is the only common
import; both sides validate independently, neither trusts the other.

## 2. Protocol and action vocabulary

JSON over a local WebSocket, one command in flight at a time:

```
Brain → Body   {"id": 7, "cmd": "look_at", "params": {"direction": "left"}}
Body  → Brain  {"id": 7, "status": "done", "pose": [0.0, ...], "error": null}
```

Type checked at decode; a malformed frame gets an error ack instead of a
crash, and the connection survives. Commands run in order; `speak` alone
is local to the brain (never crosses the wire) and overlaps with `nod`/light
pulse on a worker thread.

```
look_at(direction) curious_lean() nod() shake() scan_sweep() idle_sway()
neutral() set_light(state) play_sfx(name) play_music(on,track) speak(text)
```

Every finite-range parameter is enumerated — no free-form string reaches
the filesystem via `play_sfx`'s name. **The LLM picks from this
vocabulary; it never computes kinematics** — `body/motion.py` clamps
every target to the URDF's soft limits, so a hallucination yields an
invalid action name (dropped at validation), never an unsafe pose.
`nod`/`shake`/`idle_sway` oscillate via offsets that sum to zero (return
to start); `scan_sweep` sweeps both extremes and recentres; `neutral`
resets every joint, so disengaging looks visibly different from engaging.

## 3. Models

| Role | Model | Why |
|---|---|---|
| Face presence | MediaPipe face detection | Cheap per-frame on CPU; presence is all engagement needs. |
| Object detection | YOLOv8n | Smallest useful detector; plain text labels into memory. |
| STT | faster-whisper `tiny` (CTranslate2, int8) | Fastest whisper variant usable on CPU. |
| TTS | Piper | Offline, CPU-friendly, natural enough to carry a character. |
| Reasoning | small quantized GGUF via llama.cpp | Fits the RAM budget alongside everything else. |

Vision is composed, not fused: the detector emits text labels, the LLM
reasons over them as text — no vision-language model, keeping the stack
inside the memory budget.

## 4. Engagement, concurrency, memory

- **Engagement**: `EngagementDebouncer` holds face presence for 0.75 s
  before flipping state, absorbing per-frame flicker — unit-tested at
  that boundary.
- **Concurrency**: two supervised asyncio tasks share one Orchestrator,
  camera and connection — the engagement loop (sole camera owner, ~10
  fps) and the dialogue loop (records/transcribes once engaged). Blocking
  model/audio calls run via `asyncio.to_thread`, so TTS playback can't
  freeze the camera loop. Either loop logs and continues past an
  exception — a dropped frame must not end a live demo.
- **Memory**: an in-process dict of `{id, label, attributes,
  first_seen_ts, last_seen_ts, notes}`, serialized as plain text into any
  prompt that needs it — fine at a handful of objects; keyword filtering
  is the documented (not built) escape hatch past that.

## 5. The five demo moments

1. **Engagement** — held past debounce → lean + light pulse + chime; lost
   → neutral + sway + dim.
2. **Character response** — the reaction above, instant, no model call.
3. **Spoken interaction** — mic → VAD → whisper → LLM reply → `speak` +
   `nod` + light pulse.
4. **Scene memory** — detector runs on engagement and periodically while
   engaged; later questions draw on the memory list.
5. **Goal-directed action** — a spoken goal is planned against the
   vocabulary and memory, validated step by step; a `scan_sweep` triggers
   a re-observe before the LLM confirms.

Routing (3) vs (5) is a keyword heuristic on the transcript, not a second
model call — a misroute still degrades to a sensible reply.

## 6. Failure handling and limitations

| Failure | Response |
|---|---|
| Malformed/hostile protocol frame | Type-checked at decode; error ack; connection survives. |
| Invalid or malformed model output | Step dropped at validation; falls back to `idle_sway`. |
| Body rejects a command | Error logged; sequence continues. |
| Command exceeds joint limits | Clamped to nearest safe target, never rejected. |
| Camera missing at startup | Brain fails fast with a named error. |
| Dropped frame / audio underrun | Logged; loop continues. |

Deferred (see `KNOWN_LIMITATIONS.md`): LLM timeout with filler,
retry-with-stricter-reprompt, mic fail-fast, WebSocket heartbeat/reconnect.

## 7. Deployment

Plain venv plus `requirements.txt` — no Docker, avoiding container
audio/video passthrough friction. `scripts/setup.sh` fetches model
weights once; `scripts/run_all.sh` starts Body, waits for its port, then
starts Brain.

## 8. Measurements

Collected with `python -m scripts.measure_technical_note` on a 4-core,
8 GB Ubuntu 24.04 VM matching the target spec — reproducible, no live
camera or microphone needed (STT uses a Piper-synthesized sample;
detector latency depends on frame size, not content).

| Metric | Value |
|---|---|
| Engagement detection latency | 0.75 s debounce hold + 19 ms detector compute |
| STT latency (utterance → transcript) | 7.42 s (~4 s utterance) |
| LLM reply latency | 4.29 s |
| LLM plan latency | 8.06 s |
| TTS latency (text → first audio) | 0.76 s |
| End-to-end (finish speaking → response) | 15.49 s (STT + LLM plan + 13 ms round-trip) |
| Peak RSS, full stack loaded | 3.06 GB — budget 7 GB |
| Steady-state CPU while engaged | 9.7% of one core (~10 fps loop, face-present baseline) |

Whisper and the LLM dominate every interaction — expected for CPU-only
inference with no GPU; the demo overlaps `speak` with `nod`/light-pulse
rather than trying to hide the latency.
