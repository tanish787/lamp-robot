"""In-process scene memory for a single demo session. Deliberately no
vector store or retrieval ranking: at demo scale the whole list is
serialized straight into the LLM prompt (see spec 7)."""

import math
import uuid

_DEDUP_DISTANCE = 0.5  # same units as position tuples; tune per camera setup


class SceneMemory:
    def __init__(self):
        self._records: dict[str, dict] = {}

    def observe(self, label: str, attributes: dict, timestamp: float) -> str:
        existing = self._find_match(label, attributes.get("position"))
        if existing is not None:
            existing["attributes"] = attributes
            existing["last_seen_ts"] = timestamp
            return existing["id"]

        record_id = str(uuid.uuid4())
        self._records[record_id] = {
            "id": record_id,
            "label": label,
            "attributes": attributes,
            "first_seen_ts": timestamp,
            "last_seen_ts": timestamp,
            "notes": "",
        }
        return record_id

    def _find_match(self, label: str, position) -> dict | None:
        if position is None:
            return None
        for record in self._records.values():
            if record["label"] != label:
                continue
            other = record["attributes"].get("position")
            if other is None:
                continue
            distance = math.dist(position, other)
            if distance <= _DEDUP_DISTANCE:
                return record
        return None

    def records(self) -> list[dict]:
        return list(self._records.values())

    def get(self, record_id: str) -> dict | None:
        return self._records.get(record_id)

    def as_prompt_text(self) -> str:
        if not self._records:
            return "Nothing has been observed in the scene yet."
        lines = [
            f"- {r['label']} ({r['attributes']}), last seen at t={r['last_seen_ts']:.1f}s"
            for r in self._records.values()
        ]
        return "Observed objects:\n" + "\n".join(lines)
