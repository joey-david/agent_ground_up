"""Run solution.py against every sample case and report mismatches."""

import subprocess
import sys
from pathlib import Path

TIMEOUT_S = 10


def normalize(text):
    return "\n".join(line.rstrip() for line in text.strip().splitlines())


def main():
    failures = 0
    for input_path in sorted(Path("tests").glob("*.in")):
        expected = input_path.with_suffix(".out").read_text()
        try:
            completed = subprocess.run(
                [sys.executable, "solution.py"],
                input=input_path.read_text(),
                capture_output=True,
                text=True,
                timeout=TIMEOUT_S,
            )
        except subprocess.TimeoutExpired:
            failures += 1
            print(f"FAIL {input_path.name}: timed out after {TIMEOUT_S}s")
            continue
        if completed.returncode != 0:
            failures += 1
            print(f"FAIL {input_path.name}: exit {completed.returncode}")
            print(completed.stderr[-500:])
        elif normalize(completed.stdout) != normalize(expected):
            failures += 1
            print(f"FAIL {input_path.name}")
            print(f"  expected: {normalize(expected)[:200]}")
            print(f"  actual:   {normalize(completed.stdout)[:200]}")
        else:
            print(f"PASS {input_path.name}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
