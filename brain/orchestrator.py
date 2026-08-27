"""Ties every Brain component together. The only place that decides
whether an action name is Brain-local (speak) or forwarded to Body — see
shared.action_vocabulary.BRAIN_LOCAL_ACTIONS."""

from shared.action_vocabulary import BRAIN_LOCAL_ACTIONS


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
    ):
        self._tts = tts
        self._protocol_client = protocol_client
        self._engagement = engagement
        self._audio = audio
        self._stt = stt
        self._perception = perception
        self._reasoner = reasoner
        self._memory = memory

    async def execute_actions(self, actions: list[dict]) -> None:
        for action in actions:
            name, params = action["name"], action["params"]
            if name in BRAIN_LOCAL_ACTIONS:
                self._tts.speak(params["text"])
            else:
                await self._protocol_client.send_command(name, params)

    async def on_engagement_change(self, engaged: bool) -> None:
        if engaged:
            await self.execute_actions([
                {"name": "curious_lean", "params": {}},
                {"name": "set_light", "params": {"state": "pulse"}},
                {"name": "play_sfx", "params": {"name": "chime"}},
            ])
        else:
            # `neutral` first: without an explicit reset, disengaging left
            # the lamp in whatever pose the last character response ended
            # in (e.g. still leaning in), so engaged and disengaged looked
            # identical.
            await self.execute_actions([
                {"name": "neutral", "params": {}},
                {"name": "idle_sway", "params": {}},
                {"name": "set_light", "params": {"state": "dim"}},
            ])
