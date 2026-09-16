"""Grade one LiveCodeBench workspace against its hidden tests.

Run from inside the (copied) workspace; the hidden tests live outside it so the solution
cannot read what it is graded on:

    python3 verify.py /path/to/hidden/<case>.json solution.py

Exit code 0 means every hidden test passed, which is the only thing the A/B harness records.
Comparison is whitespace-normalized per line, matching how contest judges treat trailing
spaces and a missing final newline.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

PER_TEST_TIMEOUT_S = 10


def normalize(text: str) -> str:
    return "\n".join(line.rstrip() for line in text.strip().splitlines())


def main() -> int:
    if len(sys.argv) != 3:
        print("usage: verify.py <hidden-tests.json> <solution.py>", file=sys.stderr)
        return 2
    hidden_path, solution = Path(sys.argv[1]), Path(sys.argv[2])
    if not solution.exists():
        print(f"no {solution}", file=sys.stderr)
        return 1

    tests = json.loads(hidden_path.read_text(encoding="utf-8"))
    for index, test in enumerate(tests, start=1):
        try:
            completed = subprocess.run(
                [sys.executable, str(solution)],
                input=test["input"],
                capture_output=True,
                text=True,
                timeout=PER_TEST_TIMEOUT_S,
            )
        except subprocess.TimeoutExpired:
            print(f"test {index}: timed out after {PER_TEST_TIMEOUT_S}s", file=sys.stderr)
            return 1
        if completed.returncode != 0:
            print(
                f"test {index}: exit {completed.returncode}\n{completed.stderr[-500:]}",
                file=sys.stderr,
            )
            return 1
        if normalize(completed.stdout) != normalize(test["output"]):
            print(
                f"test {index}: wrong answer\n"
                f"  expected: {normalize(test['output'])[:200]}\n"
                f"  actual:   {normalize(completed.stdout)[:200]}",
                file=sys.stderr,
            )
            return 1

    print(f"all {len(tests)} hidden tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
