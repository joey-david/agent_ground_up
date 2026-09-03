from __future__ import annotations

import json
import re
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
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
    """Append-only episodic memory with bounded wake context and zoomable summaries."""

    def __init__(
        self,
        root: str | Path,
        *,
        wake_records: int = 6,
        leaf_size: int = 8,
        summary_chars: int = 500,
        summarizer: SummaryFn | None = None,
    ) -> None:
        if wake_records < 0 or leaf_size < 1 or summary_chars < 80:
            raise ValueError("invalid memory sizing")
        self.root = Path(root).expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.events_path = self.root / "events.jsonl"
        self.tree_path = self.root / "tree.json"
        self.wake_records = wake_records
        self.leaf_size = leaf_size
        self.summary_chars = summary_chars
        self.summarizer = summarizer or self._default_summary

    def remember(self, text: str, tags: Iterable[str] = ()) -> MemoryRecord:
        """Append one durable memory and rebuild the compact summary index."""
        text = text.strip()
        if not text:
            raise ValueError("memory text cannot be empty")
        records = self.records()
        record = MemoryRecord(
            id=len(records),
            text=text,
            tags=tuple(dict.fromkeys(tag.strip() for tag in tags if tag.strip())),
            created_at=datetime.now(timezone.utc).isoformat(),
        )
        with self.events_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(asdict(record), ensure_ascii=False) + "\n")
        self._rebuild_tree([*records, record])
        return record

    def records(self) -> list[MemoryRecord]:
        if not self.events_path.exists():
            return []
        result: list[MemoryRecord] = []
        for line in self.events_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            data = json.loads(line)
            data["tags"] = tuple(data.get("tags", ()))
            result.append(MemoryRecord(**data))
        return result

    def wake(self) -> str:
        """Return constant-sized startup context: root summary plus the newest raw records."""
        records = self.records()
        if not records:
            return "Persistent memory: empty."
        nodes = self._nodes()
        root = self._root(nodes)
        lines = ["Persistent memory (summary; use recall/zoom for detail):"]
        if root:
            lines.append(f"[{root.id}] {root.summary}")
        if self.wake_records:
            lines.append("Recent memories:")
            for record in records[-self.wake_records :]:
                lines.append(self._format_record(record))
        return "\n".join(lines)

    def recall(self, pattern: str, *, limit: int = 8) -> list[MemoryRecord]:
        """Regex-search raw memories newest-first."""
        if limit < 1:
            return []
        regex = re.compile(pattern, flags=re.IGNORECASE)
        matches = [record for record in reversed(self.records()) if regex.search(record.text)]
        return matches[:limit]

    def zoom(self, node_id: str) -> str:
        """Expand one summary node into child summaries or its underlying raw records."""
        nodes = self._nodes()
        node = nodes.get(node_id)
        if node is None:
            raise KeyError(f"unknown memory node: {node_id}")
        if node.level == 0:
            records = self.records()[node.start : node.end]
            return "\n".join(self._format_record(record) for record in records)
        children = [nodes[child] for child in node.children]
        return "\n".join(f"[{child.id}] {child.summary}" for child in children)

    def root_node(self) -> MemoryNode | None:
        return self._root(self._nodes())

    def _rebuild_tree(self, records: list[MemoryRecord]) -> None:
        if not records:
            self.tree_path.write_text("[]\n", encoding="utf-8")
            return
        nodes: list[MemoryNode] = []
        current: list[MemoryNode] = []
        for index, start in enumerate(range(0, len(records), self.leaf_size)):
            chunk = records[start : start + self.leaf_size]
            node = MemoryNode(
                id=f"L0:{index}",
                level=0,
                start=start,
                end=start + len(chunk),
                summary=self._summarize([record.text for record in chunk]),
            )
            nodes.append(node)
            current.append(node)

        level = 1
        while len(current) > 1:
            parents: list[MemoryNode] = []
            for index, start in enumerate(range(0, len(current), 2)):
                children = current[start : start + 2]
                node = MemoryNode(
                    id=f"L{level}:{index}",
                    level=level,
                    start=children[0].start,
                    end=children[-1].end,
                    summary=self._summarize([child.summary for child in children]),
                    children=tuple(child.id for child in children),
                )
                nodes.append(node)
                parents.append(node)
            current = parents
            level += 1

        payload = [asdict(node) for node in nodes]
        self.tree_path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )

    def _nodes(self) -> dict[str, MemoryNode]:
        if not self.tree_path.exists():
            records = self.records()
            if records:
                self._rebuild_tree(records)
            else:
                return {}
        data = json.loads(self.tree_path.read_text(encoding="utf-8"))
        nodes: dict[str, MemoryNode] = {}
        for item in data:
            item["children"] = tuple(item.get("children", ()))
            node = MemoryNode(**item)
            nodes[node.id] = node
        return nodes

    @staticmethod
    def _root(nodes: dict[str, MemoryNode]) -> MemoryNode | None:
        if not nodes:
            return None
        return max(nodes.values(), key=lambda node: (node.level, node.end - node.start))

    def _summarize(self, texts: list[str]) -> str:
        summary = self.summarizer(texts).strip()
        if len(summary) <= self.summary_chars:
            return summary
        half = max(1, (self.summary_chars - 5) // 2)
        return summary[:half].rstrip() + " ... " + summary[-half:].lstrip()

    @staticmethod
    def _default_summary(texts: list[str]) -> str:
        cleaned = [" ".join(text.split()) for text in texts if text.strip()]
        return " | ".join(cleaned)

    @staticmethod
    def _format_record(record: MemoryRecord) -> str:
        tags = f" tags={','.join(record.tags)}" if record.tags else ""
        return f"#{record.id}{tags}: {record.text}"
