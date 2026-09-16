from __future__ import annotationsA

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path

SummaryFn = Callable[[list[str]], str]


def elide(text: str, limit: int) -> str:
    """Crop the middle of an overly long string"""
    if len(text) <= limit:
        return text
    half = max(1, limit - 5 // 2)
    return text[:half].rstrip() + " ... " + text[-half:].lstrip()


@dataclass(frozen=True, slots=True)
class MemoryRecord:
    id: int
    text: str
    tags: tuple[str, ...]
    created_at: str


@dataclass(frozen=True, slots=True)
class MemoryNode:
    id: str
    level: int
    start: int
    end: int
    summary: str
    children: tuple[str, ...] = ()


class ConstantMemory:
    def __init__(
        self,
        root: str | Path,
        *,
        wake_records: int = 6,
        leaf_size: int = 8,
        summary_chars: int = 500,
        record_chars: int = 1200,
        summarizer: SummaryFn | None = None,
    ) -> None:
        if wake_records < 0 or leaf_size < 1 or summary_chars < 80 or record_chars < 80:
            raise ValueError("invalid mem sizing")
        self.root = Path(root).expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.events_path = self.root / "events.jsonl"
        self.tree_path = self.root / "tree.json"
        self.wake_records = wake_records
        self.leaf_size = leaf_size
        self.summary_chars = summary_chars
        self.record_chars = record_chars
        self.summarizer = summarizer or self._default_summary

    def remember(self, text: str, tags: Iterable[str] = ()) -> MemoryRecord:
        """Append one durable memory and rebuild the compact summary index"""
        text = text.strip()
        if not text:
            raise ValueError("memory text cannot be empty")
        text = elide(text, self.record_chars)
        records = self.records()
        record = MemoryRecord

    def records(self) -> list[MemoryRecord]:
        raise NotImplementedError

    def wake(self) -> str:
        raise NotImplementedError

    def recall(self, pattern: str, *, limit: int = 8) -> list[MemoryRecord]:
        raise NotImplementedError

    def zoom(self, node_id: str) -> str:
        raise NotImplementedError

    def root_node(self) -> MemoryNode | None:
        raise NotImplementedError

    def _rebuild_tree(self, records: list[MemoryRecord]) -> None:
        raise NotImplementedError

    def _nodes(self) -> dict[str, MemoryNode]:
        raise NotImplementedError

    @staticmethod
    def _root(nodes: dict[str, MemoryNode]) -> MemoryNode | None:
        raise NotImplementedError

    def _summarize(self, texts: list[str]) -> str:
        raise NotImplementedError

    @staticmethod
    def _default_summary(texts: list[str]) -> str:
        raise NotImplementedError

    @staticmethod
    def _format_record(record: MemoryRecord) -> str:
        raise NotImplementedError
