"""Grade one V*Bench workspace against its answer key.

    python3 verify.py /path/to/answers/<case>.json answer.txt

The key lives outside the workspace so the agent cannot read it. Exit code 0 means the letter
written to answer.txt matches. A file holding more than a letter is accepted only when exactly
one option letter appears in it, so "The answer is (C)" counts and "maybe B, maybe D" does not.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

LETTERS = "ABCD"


def extract_letter(text: str) -> str | None:
    stripped = text.strip()
    if len(stripped) == 1 and stripped.upper() in LETTERS:
        return stripped.upper()
    found = {match.group(1).upper() for match in re.finditer(r"\b([A-Da-d])\b", stripped)}
    if len(found) == 1:
        return found.pop()
    return None


def main() -> int:
    if len(sys.argv) != 3:
        print("usage: verify.py <answers.json> <answer.txt>", file=sys.stderr)
        return 2
    key_path, answer_path = Path(sys.argv[1]), Path(sys.argv[2])
    if not answer_path.exists():
        print(f"no {answer_path}", file=sys.stderr)
        return 1

    expected = json.loads(key_path.read_text(encoding="utf-8"))["label"].strip().upper()
    written = answer_path.read_text(encoding="utf-8")
    letter = extract_letter(written)
    if letter is None:
        print(f"could not read a single option letter from {written!r}", file=sys.stderr)
        return 1
    if letter != expected:
        print(f"answered {letter}, expected {expected}", file=sys.stderr)
        return 1
    print(f"correct: {letter}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
