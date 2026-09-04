from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class ExperienceEvent:
    id: int
    kind: str
    payload: Any
    created_at: str


class ExperienceLog:
    """Append-only programmatic history for long-horizon agents.

    This is deliberately separate from distilled semantic memory. It records the actual readable
    task/action/result stream so the agent can later search or read old experience without forcing
    the entire history back into active context.
    """

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.path = self.root / "history.jsonl"
        self._next_id = self._initial_count()

    def append(self, kind: str, payload: Any) -> ExperienceEvent:
        if not kind.strip():
            raise ValueError("experience kind cannot be empty")
        event = ExperienceEvent(
            id=self._next_id,
            kind=kind.strip(),
            payload=self._sanitize(payload),
            created_at=datetime.now(timezone.utc).isoformat(),
        )
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(asdict(event), ensure_ascii=False) + "\n")
        self._next_id += 1
        return event

    def events(self) -> list[ExperienceEvent]:
        if not self.path.exists():
            return []
        result: list[ExperienceEvent] = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                result.append(ExperienceEvent(**json.loads(line)))
        return result

    def count(self) -> int:
        return self._next_id

    def search(self, pattern: str, *, limit: int = 12) -> list[ExperienceEvent]:
        if limit < 1:
            return []
        regex = re.compile(pattern, flags=re.IGNORECASE)
        matches: list[ExperienceEvent] = []
        for event in reversed(self.events()):
            if self._is_retrieval_query(event):
                continue
            searchable = f"{event.kind} {json.dumps(event.payload, ensure_ascii=False)}"
            if regex.search(searchable):
                matches.append(event)
                if len(matches) >= limit:
                    break
        return matches

    def read(self, start: int, end: int) -> list[ExperienceEvent]:
        if start < 0 or end < start:
            raise ValueError("expected 0 <= start <= end")
        return self.events()[start:end]

    @staticmethod
    def format(events: list[ExperienceEvent]) -> str:
        if not events:
            return "no events"
        return "\n".join(
            f"#{event.id} [{event.kind}] {json.dumps(event.payload, ensure_ascii=False)}"
            for event in events
        )

    def _initial_count(self) -> int:
        if not self.path.exists():
            return 0
        with self.path.open("r", encoding="utf-8") as handle:
            return sum(1 for line in handle if line.strip())

    @staticmethod
    def _is_retrieval_query(event: ExperienceEvent) -> bool:
        return (
            event.kind == "tool_call"
            and isinstance(event.payload, dict)
            and event.payload.get("name") in {"search_history", "read_history"}
        )

    @classmethod
    def _sanitize(cls, value: Any) -> Any:
        """Keep history readable without duplicating large data-URL image payloads."""
        if isinstance(value, dict):
            sanitized = {key: cls._sanitize(item) for key, item in value.items()}
            if sanitized.get("type") == "image_url":
                image_url = sanitized.get("image_url")
                if isinstance(image_url, dict) and str(image_url.get("url", "")).startswith("data:"):
                    sanitized["image_url"] = {"url": "[embedded image retained in workspace]"}
            return sanitized
        if isinstance(value, list):
            return [cls._sanitize(item) for item in value]
        if isinstance(value, tuple):
            return [cls._sanitize(item) for item in value]
        if isinstance(value, (str, int, float, bool)) or value is None:
            return value
        return repr(value)
