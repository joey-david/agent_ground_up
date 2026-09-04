from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class ArchiveEntry:
    id: str
    parent_id: str | None
    score: float
    novelty: float
    family: str
    fingerprint: str
    created_at: str
    note: str = ""


class Archive:
    def __init__(self, root: str | Path, *, editable: tuple[str, ...]) -> None:
        raise NotImplementedError

    def add(
        self,
        source_root: str | Path,
        *,
        parent_id: str | None,
        score: float,
        novelty: float,
        family: str,
        note: str = "",
    ) -> ArchiveEntry:
        raise NotImplementedError

    def entries(self) -> list[ArchiveEntry]:
        raise NotImplementedError

    def select_parent(self, *, family: str | None = None, exploration: float = 0.20) -> ArchiveEntry:
        raise NotImplementedError

    def materialize(self, entry_id: str, destination: str | Path) -> Path:
        raise NotImplementedError

    def novelty(self, source_root: str | Path) -> float:
        raise NotImplementedError

    def fingerprint(self, source_root: str | Path) -> str:
        raise NotImplementedError
