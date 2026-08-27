# Technical note — HCL lamp robot live character

First draft. Measurements marked _(to measure)_ are collected on the target
hardware with the smoke scripts and the resource-ceiling check; they are not
yet filled in because they are hardware-specific and must come from the
machine the demo is recorded on.

## 1. What this is

A simulated 5-DOF desk-lamp robot that behaves like a character: it notices
when someone is in front of the camera, reacts, holds a short spoken
conversation, remembers what it has seen in the room, and carries out spoken
goals as sequences of physical actions.

Everything runs locally on CPU. The design constraint that shaped almost
every decision below is a GPU-less ~8 GB machine with no network access at
demo time.

## 2. Architecture

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

### Why two processes

Brain holds four models and blocks for seconds at a time on inference and on
speech playback. Body must stay responsive to run the simulation. Splitting
them means a slow LLM call cannot stall the physics loop, and either half
can be run, restarted or replaced independently — Body with a real robot
controller, for instance, without Brain noticing.

The cost is a serialization boundary, which is also the benefit: it forces
the interface between "what the character decides" and "what the hardware
does" to be an explicit, validated vocabulary rather than a Python method
call.

### Why one shared module

`shared/` is the only thing both processes import. It holds the action
vocabulary and the wire schema — the two things that must not drift. Brain
validates a plan against `is_valid_action` before sending; Body validates
the same way on receipt. Neither trusts the other.

## 3. Protocol

JSON frames over a local WebSocket, one command in flight at a time.

```
Brain → Body   {"id": 7, "cmd": "look_at", "params": {"direction": "left"}}
Body  → Brain  {"id": 7, "status": "done", "pose": [0.0, ...], "error": null}
Body  → Brain  {"id": 7, "status": "error", "pose": null, "error": "..."}
```

`decode_command` type-checks every field — `cmd` must be a string, `params`
an object — so a well-formed-JSON-but-wrong-shape frame is rejected as a
protocol error rather than crashing a downstream lookup. The server's
vocabulary check and its dispatch share one exception guard, and the
connection handler catches anything that still escapes, so a malformed frame
can never close the socket. (There is no reconnect on the Brain side, so a
dropped connection would end the session — see `KNOWN_LIMITATIONS.md`.)

Brain waits for `status: done` before the next command in a sequence, which
is what makes multi-step goals ordered. The exception is `speak`: it is
Brain-local and runs on a worker thread, so `Orchestrator.speak_with` can
fire a `nod` and a light pulse over the top of it and the character looks
like it is talking rather than miming and then speaking.

## 4. The action vocabulary

```
look_at(direction)   curious_lean()   nod()      shake()
scan_sweep()         idle_sway()      neutral()
set_light(state)     play_sfx(name)   play_music(on, track)
speak(text)          # Brain-local, never crosses the wire
```

Every parameter with a finite range is enumerated —
`LOOK_DIRECTIONS`, `LIGHT_STATES`, `SFX_NAMES`, `MUSIC_TRACKS`. That is not
tidiness: `play_sfx`'s name becomes a filesystem path on the Body side, so
an unconstrained string is a path-traversal vector from a model's output
into the filesystem. `light_sfx._resolve_clip` re-checks the resolved path
stays inside the assets directory as defence in depth.

**The LLM chooses from this vocabulary; it never computes kinematics.**
`body/motion.py` owns the mapping from a name to joint targets, and clamps
every target to the URDF's soft limits before anything moves. A model that
hallucinates cannot produce an unsafe pose, only an invalid action name —
which validation drops.

Motion actions resolve to *waypoint sequences*, not single poses.
`nod`, `shake` and `idle_sway` accumulate relative offsets that sum to zero,
so they oscillate and return to wherever they started — a nod after a lean
is still a nod. `scan_sweep` sweeps both extremes and recentres.
`neutral` resets every joint, which is what makes disengaging visibly
different from engaging.

## 5. Models

| Role | Model | Why |
|---|---|---|
| Face presence | MediaPipe face detection | Cheap enough to run every frame on CPU; presence is all the engagement state machine needs. |
| Object detection | YOLOv8n | Smallest useful detector; produces plain labels that go into memory as text. |
| STT | faster-whisper `tiny` (CTranslate2, int8) | Fastest whisper variant that is usable on CPU. |
| TTS | Piper | Offline, CPU-friendly, natural enough to carry a character. |
| Reasoning | small quantized GGUF via llama.cpp | Fits the RAM budget alongside everything else. |

Vision is *composed, not fused*: the detector emits text labels, and the LLM
reasons over them as text. No vision-language model is loaded, which is what
keeps the whole stack inside the memory budget.

## 6. Engagement

Face detection per frame is noisy — a glance away, a missed frame, a hand
across the face. `EngagementDebouncer` requires the signal to hold for
`hold_seconds` (0.75 s default) before the state actually flips, so flicker
never produces a spurious greeting. This is a pure function of
`(face_detected, timestamp)` and is unit-tested at the debounce boundary,
where this class of bug lives.

## 7. Concurrency in Brain

Two supervised asyncio tasks share one Orchestrator, one camera and one
connection:

- **Engagement loop** — the only owner of the capture device. Reads a frame
  (~10 fps), runs face detection, feeds the debouncer, fires transitions,
  and caches the newest frame so scene perception can look at what the
  camera sees without opening a second device.
- **Dialogue loop** — idles on an `asyncio.Event` until engagement is on,
  then records a listening window, cuts an utterance out of it with the VAD,
  and hands it to the orchestrator.

Every blocking model or audio call goes through `asyncio.to_thread`. Without
that, `Piper.speak()`'s `wait_done()` would freeze the camera loop for the
duration of every spoken line. An exception inside either loop is logged and
the loop continues: a dropped frame or an audio underrun must not end a live
demo.

## 8. Memory

An in-process dict of records: `{id, label, attributes, first_seen_ts,
last_seen_ts, notes}`, deduplicated by label plus rough position. The whole
list is serialized as plain text into any prompt that might need scene
knowledge. At a handful of objects this fits comfortably in a small model's
context, and no retrieval or ranking layer is needed. If it grew past dozens
of objects the read path would need keyword filtering — a documented escape
hatch, not built.

## 9. The five demo moments

1. **Engagement** — face held past the debounce window → `curious_lean` +
   `set_light(pulse)` + `play_sfx(chime)`. Losing the face → `neutral` +
   `idle_sway` + `set_light(dim)`.
2. **Character response** — the greeting above is a fixed reaction to a
   fixed trigger, so no model call is involved and it is instant.
3. **Spoken interaction** — mic → VAD segmentation → whisper → LLM reply →
   `speak` overlapped with `nod` and a light pulse.
4. **Scene memory** — the detector runs on engagement and periodically while
   engaged; a later question is answered with the memory list in the prompt,
   so paraphrased references work.
5. **Goal-directed action** — a spoken goal is routed to `plan_actions`,
   which prompts the LLM with the memory list, the goal, and the vocabulary,
   then validates every returned step. If the plan contains `scan_sweep`,
   the scene is observed *again* after the sweep and the LLM confirms
   against the updated memory.

Routing between (3) and (5) is a keyword heuristic on the transcript rather
than a second model call — the latency budget is better spent on the one
call that produces the answer, and a misroute degrades to a sensible reply
either way.

## 10. Failure handling

| Failure | Response |
|---|---|
| Malformed/hostile protocol frame | Type-checked at decode, error ack, connection survives. |
| Model names an action outside the vocabulary | Step dropped at validation; empty plan falls back to `idle_sway`. |
| Model returns malformed JSON | Same fallback. |
| Body rejects a command | Ack status inspected, error logged, sequence continues. |
| Command exceeds joint limits | Clamped to the nearest safe target; never rejected outright. |
| Engagement flicker | Absorbed by the debounce window. |
| Camera missing at startup | Brain fails fast with a named error. |
| Dropped frame / audio underrun | Logged, loop continues. |

Deferred: LLM timeout with filler, retry-with-stricter-reprompt, mic
fail-fast, WebSocket heartbeat/reconnect. See `KNOWN_LIMITATIONS.md`.

## 11. Deployment

Plain venv plus `requirements.txt`; no Docker, which avoids container
audio/video passthrough friction for exactly the devices this project
depends on. `scripts/setup.sh` fetches model weights once.
`scripts/run_all.sh` starts Body, polls its port until it accepts
connections, then starts Brain.

## 12. Measurements

Collected with the smoke scripts (`python -m scripts.smoke_*`) and
`python -m scripts.resource_ceiling_check` on the target machine.

| Metric | Value |
|---|---|
| Engagement detection latency (frame → transition) | _(to measure)_ — bounded below by the 0.75 s debounce hold |
| STT latency (utterance → transcript) | _(to measure)_ |
| LLM reply latency | _(to measure)_ |
| LLM plan latency | _(to measure)_ |
| TTS latency (text → first audio) | _(to measure)_ |
| End-to-end (finish speaking → lamp responds) | _(to measure)_ |
| Peak RSS, full stack loaded | _(to measure)_ — budget 7 GB |
| Steady-state CPU while engaged | _(to measure)_ |
