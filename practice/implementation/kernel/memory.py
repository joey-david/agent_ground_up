from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path

SummaryFn = Callable[[list[str]], str]


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
        raise NotImplementedError

    def remember(self, text: str, tags: Iterable[str] = ()) -> MemoryRecord:
        raise NotImplementedError

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
