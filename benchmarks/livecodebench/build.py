"""Materialize LiveCodeBench problems as verifiable workspaces for the A/B harness.

LiveCodeBench is a contest-problem benchmark with a rolling time window, which makes it the
opposite of the bundled coding curriculum: one uniform task shape (read stdin, write stdout),
hundreds of hidden tests per problem, and a difficulty level this model does not saturate.

Every problem becomes one workspace -- statement, sample tests, a stub, and a sample runner --
plus a curriculum entry whose verifier runs the *hidden* tests from outside the workspace, so
the agent cannot read the answers it is being graded against.

    uv run python benchmarks/livecodebench/build.py --difficulty hard --limit 12

Releases are the official `test<N>.jsonl` files: v6 covers contests from 2025-01 to 2025-04.
That window predates this model, so treat absolute scores as contaminated-until-proven and
compare harnesses against each other rather than against a published leaderboard number.
"""

from __future__ import annotations

import argparse
import base64
import json
import pickle
import random
import shutil
import urllib.request
import zlib
from pathlib import Path

RELEASE_URL = (
    "https://huggingface.co/datasets/livecodebench/code_generation_lite/"
    "resolve/main/test{release}.jsonl"
)
HERE = Path(__file__).resolve().parent

PROMPT = """Solve the programming problem described in problem.md.

Write your solution to solution.py. It must read the input from standard input and print the
answer to standard output, exactly in the format the problem specifies.

The sample cases from the statement are in tests/ and `python3 run_samples.py` checks them.
Your solution is graded on hidden tests as well, including large inputs, so make sure the
algorithm is efficient enough for the stated constraints rather than only passing the samples."""

STUB = '''"""Read the input from stdin and print the answer to stdout."""

import sys


def main() -> None:
    data = sys.stdin.read().split()
    raise NotImplementedError


if __name__ == "__main__":
    main()
'''

RUN_SAMPLES = '''"""Run solution.py against every sample case and report mismatches."""

import subprocess
import sys
from pathlib import Path

TIMEOUT_S = 10


def normalize(text):
    return "\\n".join(line.rstrip() for line in text.strip().splitlines())


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
'''


def decode_tests(blob: str) -> list[dict]:
    """Decode a LiveCodeBench test blob, which is JSON, or zlib+base64, sometimes twice over."""
    if not blob:
        return []
    try:
        decoded = json.loads(blob)
    except json.JSONDecodeError:
        raw = zlib.decompress(base64.b64decode(blob.encode("utf-8")))
        try:
            decoded = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            decoded = pickle.loads(raw)
    while isinstance(decoded, str):
        decoded = json.loads(decoded)
    return decoded


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release", default="6", help="LiveCodeBench release number")
    parser.add_argument("--difficulty", default="hard", choices=["easy", "medium", "hard"])
    parser.add_argument("--limit", type=int, default=12, help="problems to materialize")
    parser.add_argument("--hidden-tests", type=int, default=25,
                        help="hidden tests sampled per problem, to keep verification bounded")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--split", default="lcb", help="split name written into the curriculum")
    parser.add_argument("--cache", default=str(HERE / "data"),
                        help="where the downloaded release and decoded hidden tests live")
    parser.add_argument("--out", default="",
                        help="curriculum path; defaults to curriculum_<difficulty>.json")
    args = parser.parse_args()
    # One file per difficulty: the splits are run separately and a single shared name silently
    # overwrites the previous build.
    args.out = args.out or str(HERE / f"curriculum_{args.difficulty}.json")
    return args


def download(release: str, cache: Path) -> Path:
    cache.mkdir(parents=True, exist_ok=True)
    path = cache / f"test{release}.jsonl"
    if not path.exists():
        url = RELEASE_URL.format(release=release)
        print(f"downloading {url}")
        urllib.request.urlopen(url, timeout=600)  # noqa: S310 -- fixed https host
        with urllib.request.urlopen(url, timeout=600) as response, path.open("wb") as handle:
            shutil.copyfileobj(response, handle)
    return path


def main() -> None:
    args = parse_args()
    cache = Path(args.cache)
    release_path = download(args.release, cache)

    problems = [json.loads(line) for line in release_path.read_text().splitlines() if line]
    # stdin problems only: one uniform contract to grade, unlike the functional LeetCode half.
    problems = [
        problem
        for problem in problems
        if problem["difficulty"] == args.difficulty
        and json.loads(problem["public_test_cases"])[0]["testtype"] == "stdin"
    ]
    problems.sort(key=lambda problem: problem["question_id"])
    rng = random.Random(args.seed)
    chosen = problems[: args.limit]
    print(f"{len(problems)} {args.difficulty} stdin problems available; taking {len(chosen)}")

    workspaces = HERE / "workspaces"
    hidden_dir = cache / "hidden"
    hidden_dir.mkdir(parents=True, exist_ok=True)
    verify_script = HERE / "verify.py"
    cases = []

    for problem in chosen:
        case_id = f"lcb_{problem['platform']}_{problem['question_id']}".replace("/", "_")
        workspace = workspaces / case_id
        if workspace.exists():
            shutil.rmtree(workspace)
        (workspace / "tests").mkdir(parents=True)

        title = problem["question_title"]
        (workspace / "problem.md").write_text(
            f"# {title}\n\n{problem['question_content']}\n", encoding="utf-8"
        )
        (workspace / "solution.py").write_text(STUB, encoding="utf-8")
        (workspace / "run_samples.py").write_text(RUN_SAMPLES, encoding="utf-8")

        public = json.loads(problem["public_test_cases"])
        for index, test in enumerate(public, start=1):
            (workspace / "tests" / f"sample_{index}.in").write_text(test["input"], encoding="utf-8")
            (workspace / "tests" / f"sample_{index}.out").write_text(
                test["output"], encoding="utf-8"
            )

        private = decode_tests(problem["private_test_cases"])
        # Hidden tests run on every episode, so sample a bounded, deterministic subset rather
        # than the full bank -- some problems ship thousands of cases.
        sampled = list(private)
        if len(sampled) > args.hidden_tests:
            sampled = rng.sample(sampled, args.hidden_tests)
        hidden = [{"input": t["input"], "output": t["output"]} for t in public + sampled]
        hidden_path = hidden_dir / f"{case_id}.json"
        hidden_path.write_text(json.dumps(hidden), encoding="utf-8")

        cases.append(
            {
                "id": case_id,
                "prompt": PROMPT,
                # Absolute paths on purpose: the workspace is copied to a scratch directory per
                # episode, and the hidden tests must stay outside whatever the agent can read.
                "verifier": f"python3 {verify_script} {hidden_path} solution.py",
                "workspace": str(workspace.relative_to(HERE.parent.parent)),
                "split": args.split,
            }
        )
        print(f"  {case_id:<28} {title[:40]:<42} {len(public)} sample, {len(hidden)} hidden")

    curriculum = {
        "families": [
            {
                "name": "livecodebench",
                "description": (
                    f"LiveCodeBench release v{args.release}, {args.difficulty} stdin problems."
                ),
                "cases": cases,
            }
        ]
    }
    Path(args.out).write_text(json.dumps(curriculum, indent=1) + "\n", encoding="utf-8")
    print(f"\nwrote {args.out} ({len(cases)} cases)")


if __name__ == "__main__":
    main()
