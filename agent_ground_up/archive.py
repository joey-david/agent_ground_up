from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
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
    """Population archive of valid descendants; old and weaker branches are never overwritten."""

    DEFAULT_EDITABLE = (
        "agent_ground_up",
        "run.py",
        "evolve.py",
        "config.yaml",
        "improvement_policy.md",
        "skills",
    )

    def __init__(self, root: str | Path, *, editable: tuple[str, ...] | None = None) -> None:
        self.root = Path(root).expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.editable = editable or self.DEFAULT_EDITABLE

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
        source = Path(source_root).resolve()
        fingerprint = self.fingerprint(source)
        entry_id = f"{len(self.entries()):04d}-{fingerprint[:8]}"
        destination = self.root / entry_id
        destination.mkdir(parents=True, exist_ok=False)
        for relative in self.editable:
            src = source / relative
            if not src.exists():
                continue
            dst = destination / relative
            dst.parent.mkdir(parents=True, exist_ok=True)
            if src.is_dir():
                shutil.copytree(src, dst)
            else:
                shutil.copy2(src, dst)
        entry = ArchiveEntry(
            id=entry_id,
            parent_id=parent_id,
            score=float(score),
            novelty=float(novelty),
            family=family,
            fingerprint=fingerprint,
            created_at=datetime.now(timezone.utc).isoformat(),
            note=note,
        )
        (destination / "entry.json").write_text(
            json.dumps(asdict(entry), indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        return entry

    def entries(self) -> list[ArchiveEntry]:
        entries = []
        for path in sorted(self.root.glob("*/entry.json")):
            entries.append(ArchiveEntry(**json.loads(path.read_text(encoding="utf-8"))))
        return entries

    def get(self, entry_id: str) -> ArchiveEntry:
        path = self.root / entry_id / "entry.json"
        if not path.exists():
            raise KeyError(entry_id)
        return ArchiveEntry(**json.loads(path.read_text(encoding="utf-8")))

    def select_parent(self, *, family: str | None = None, exploration: float = 0.20) -> ArchiveEntry:
        candidates = [entry for entry in self.entries() if family is None or entry.family == family]
        if not candidates:
            raise ValueError("archive has no eligible parent")
        visits = self._child_counts()

        def value(entry: ArchiveEntry) -> tuple[float, str]:
            branch_bonus = exploration / ((visits.get(entry.id, 0) + 1) ** 0.5)
            return entry.score + 0.25 * entry.novelty + branch_bonus, entry.id

        return max(candidates, key=value)

    def materialize(self, entry_id: str, destination: str | Path) -> Path:
        source = self.root / entry_id
        if not source.exists():
            raise KeyError(entry_id)
        destination_path = Path(destination).resolve()
        destination_path.mkdir(parents=True, exist_ok=True)
        for relative in self.editable:
            src = source / relative
            if not src.exists():
                continue
            dst = destination_path / relative
            dst.parent.mkdir(parents=True, exist_ok=True)
            if src.is_dir():
                shutil.copytree(src, dst, dirs_exist_ok=True)
            else:
                shutil.copy2(src, dst)
        return destination_path

    def novelty(self, source_root: str | Path) -> float:
        fingerprint = self.fingerprint(Path(source_root))
        known = {entry.fingerprint for entry in self.entries()}
        return 0.0 if fingerprint in known else 1.0

    def fingerprint(self, source_root: str | Path) -> str:
        source = Path(source_root).resolve()
        digest = hashlib.sha256()
        for relative in sorted(self.editable):
            path = source / relative
            if not path.exists():
                continue
            if path.is_file():
                self._hash_file(digest, source, path)
                continue
            for child in sorted(p for p in path.rglob("*") if p.is_file()):
                self._hash_file(digest, source, child)
        return digest.hexdigest()

    def _child_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for entry in self.entries():
            if entry.parent_id:
                counts[entry.parent_id] = counts.get(entry.parent_id, 0) + 1
        return counts

    @staticmethod
    def _hash_file(digest, root: Path, path: Path) -> None:
        digest.update(str(path.relative_to(root)).encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
