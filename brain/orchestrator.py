"""Ties every Brain component together. The only place that decides
whether an action name is Brain-local (speak) or forwarded to Body — see
shared.action_vocabulary.BRAIN_LOCAL_ACTIONS.

Everything here is `async`, and every blocking model/audio call is pushed
onto a worker thread with `asyncio.to_thread`. That matters: TTS playback
alone runs for seconds, and if it ran inline the camera/engagement loop in
brain/main.py would stall behind it and the spec's "speak + nod + light
pulse together" would be impossible.
"""

import asyncio
import logging
import time

from shared.action_vocabulary import BRAIN_LOCAL_ACTIONS

_LOG = logging.getLogger(__name__)

# A spoken line that starts like one of these reads as "do something",
# and gets planned into an action sequence. Anything else — a question, a
# statement, a greeting — gets a conversational reply. Deliberately a
# keyword heuristic rather than an extra LLM classification call: it keeps
# the response latency budget for the one call that matters, and a
# misroute still produces a sensible character response either way.
_GOAL_OPENERS: tuple[str, ...] = (
    "find", "look for", "look at", "search", "scan", "point at", "point to",
    "show me", "go to", "turn", "nod", "shake", "sweep", "check the",
    "watch", "focus on", "light", "play",
)
_QUESTION_OPENERS: tuple[str, ...] = (
    "what", "where", "who", "why", "when", "how", "is", "are", "do", "does",
    "did", "can", "could", "would", "should", "will", "tell me",
)


def looks_like_goal(text: str) -> bool:
    """True if `text` reads as an instruction to act rather than something
    to answer. Exposed (and unit-tested) separately from handle_utterance
    so the routing rule can be reasoned about on its own."""
    normalized = text.strip().lower().rstrip(".!")
    if not normalized:
        return False
    if normalized.endswith("?"):
        return False
    if normalized.startswith(_QUESTION_OPENERS):
        return False
    return normalized.startswith(_GOAL_OPENERS)


class Orchestrator:
    def __init__(
        self,
        tts,
        protocol_client,
        engagement=None,
        audio=None,
        stt=None,
        perception=None,
        reasoner=None,
        memory=None,
        frame_source=None,
    ):
        self._tts = tts
        self._protocol_client = protocol_client
        self._engagement = engagement
        self._audio = audio
        self._stt = stt
        self._perception = perception
        self._reasoner = reasoner
        self._memory = memory
        # Callable returning the most recent camera frame (or None). Owned
        # by brain/main.py's engagement loop so only one thing ever touches
        # the capture device.
        self._frame_source = frame_source

    # ------------------------------------------------------------------
    # Action execution
    # ------------------------------------------------------------------

    async def execute_actions(self, actions: list[dict]) -> list[dict]:
        """Run an action sequence in order, returning one result record per
        action. A Body-side error is logged and skipped rather than
        aborting the sequence — one refused step should not silence the
        character mid-response."""
        results: list[dict] = []
        for action in actions:
            name = action.get("name")
            params = action.get("params", {}) or {}
            if name in BRAIN_LOCAL_ACTIONS:
                await asyncio.to_thread(self._tts.speak, params.get("text", ""))
                results.append({"name": name, "status": "done"})
                continue

            ack = await self._protocol_client.send_command(name, params)
            status = ack.get("status") if isinstance(ack, dict) else None
            if status == "error":
                error = ack.get("error")
                _LOG.error("Body rejected %s %r: %s", name, params, error)
                results.append({"name": name, "status": "error", "error": error})
            else:
                results.append({"name": name, "status": status or "done"})
        return results

    async def speak_with(self, text: str, actions: list[dict] | None = None) -> None:
        """Speak while Body performs `actions` at the same time.

        Spec section 5 asks for `speak` + `nod` + a light pulse to overlap
        so the character looks like it is talking, rather than nodding and
        then, separately, speaking."""
        speaking = asyncio.create_task(asyncio.to_thread(self._tts.speak, text))
        if actions:
            await self.execute_actions(actions)
        await speaking

    # ------------------------------------------------------------------
    # Engagement
    # ------------------------------------------------------------------

    async def on_engagement_change(self, engaged: bool) -> list[dict]:
        if engaged:
            return await self.execute_actions([
                {"name": "curious_lean", "params": {}},
                {"name": "set_light", "params": {"state": "pulse"}},
                {"name": "play_sfx", "params": {"name": "chime"}},
            ])
        else:
            # `neutral` first: without an explicit reset, disengaging left
            # the lamp in whatever pose the last character response ended
            # in (e.g. still leaning in), so engaged and disengaged looked
            # identical.
            return await self.execute_actions([
                {"name": "neutral", "params": {}},
                {"name": "idle_sway", "params": {}},
                {"name": "set_light", "params": {"state": "dim"}},
            ])

    # ------------------------------------------------------------------
    # Perception
    # ------------------------------------------------------------------

    async def observe_scene(self, frame, timestamp: float | None = None) -> list[str]:
        """Run the detector over `frame` and fold the results into scene
        memory, returning the labels seen. No-op if this Orchestrator was
        built without perception/memory (as several unit tests are)."""
        if frame is None or self._perception is None or self._memory is None:
            return []
        when = time.monotonic() if timestamp is None else timestamp
        return await asyncio.to_thread(self._perception.observe, frame, self._memory, when)

    async def observe_current_frame(self) -> list[str]:
        """Grab whatever the camera last saw and observe it."""
        if self._frame_source is None:
            return []
        frame = await asyncio.to_thread(self._frame_source)
        return await self.observe_scene(frame)

    # ------------------------------------------------------------------
    # Dialogue
    # ------------------------------------------------------------------

    async def handle_utterance(self, audio_bytes: bytes) -> str:
        """Transcribe one captured utterance and respond to it.

        Returns the transcript (empty if nothing was recognised) so callers
        and tests can see what was heard. Routing: an instruction is turned
        into a validated action plan and executed; anything else gets a
        spoken reply timed with a nod and a light pulse.
        """
        if self._stt is None or self._reasoner is None:
            return ""

        text = (await asyncio.to_thread(self._stt.transcribe, audio_bytes) or "").strip()
        if not text:
            return ""
        _LOG.info("heard: %s", text)

        if looks_like_goal(text):
            await self._handle_goal(text)
        else:
            reply = await asyncio.to_thread(self._reasoner.reply, text, self._memory)
            await self._say(reply)
        return text

    async def _handle_goal(self, text: str) -> None:
        """Plan an action sequence for a spoken goal, execute it, and — if
        the plan swept the scene — look again before confirming, which is
        the spec's 'observes the scene again before completing the goal'."""
        actions = await asyncio.to_thread(self._reasoner.plan_actions, text, self._memory)
        results = await self.execute_actions(actions)

        swept = any(action.get("name") == "scan_sweep" for action in actions)
        if swept:
            labels = await self.observe_current_frame()
            _LOG.info("re-observed after sweep: %s", labels)

        if swept or any(r["status"] == "error" for r in results):
            confirmation = await asyncio.to_thread(self._reasoner.reply, text, self._memory)
            await self._say(confirmation)

    async def _say(self, text: str) -> None:
        """Speak a line with the character's talking beat behind it."""
        if not text:
            return
        await self.speak_with(text, [
            {"name": "nod", "params": {}},
            {"name": "set_light", "params": {"state": "pulse"}},
        ])
