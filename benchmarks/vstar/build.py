"""Materialize V*Bench questions as verifiable workspaces for the A/B harness.

V*Bench asks multiple-choice questions about small objects inside very large photographs. A
model shown the whole image at once has to resolve a target that may occupy a few hundred of
several million pixels, so the benchmark rewards *looking again*: crop the region, enlarge it,
and view the crop. That makes it the opposite of the code benchmarks this model is tuned for,
and the place where an image tool plus a shell should beat a single glance.

    uv run python benchmarks/vstar/build.py --limit 16

Each question becomes a workspace holding the full-resolution image and the question, and the
answer key lives outside the workspace so the agent cannot read what it is graded against.
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import urllib.error
import urllib.request
from pathlib import Path

BASE_URL = "https://huggingface.co/datasets/craigwu/vstar_bench/resolve/main"
HERE = Path(__file__).resolve().parent

PROMPT = """Answer the multiple-choice question in question.md about the image image.jpg.

Write only the letter of your chosen option (A, B, C or D) to answer.txt, with nothing else in
the file.

The image is high resolution and the thing being asked about is often small within it, so
looking at the whole image at once may not settle the question. Python with Pillow is
available if you want to crop or enlarge a region before looking at it again."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=16, help="questions to materialize")
    parser.add_argument("--category", default="all",
                        choices=["all", "direct_attributes", "relative_position"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--split", default="vstar")
    parser.add_argument("--cache", default=str(HERE / "data"))
    parser.add_argument("--out", default=str(HERE / "curriculum.json"))
    return parser.parse_args()


def fetch(url: str, destination: Path) -> None:
    if destination.exists():
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url, timeout=300) as response:  # noqa: S310 -- fixed https host
        with destination.open("wb") as handle:
            shutil.copyfileobj(response, handle)


def main() -> None:
    args = parse_args()
    cache = Path(args.cache)
    index_path = cache / "test_questions.jsonl"
    fetch(f"{BASE_URL}/test_questions.jsonl", index_path)

    questions = [json.loads(line) for line in index_path.read_text().splitlines() if line]
    if args.category != "all":
        questions = [q for q in questions if q["category"] == args.category]
    # Sample rather than take a prefix: the file is ordered by category, and a prefix would
    # quietly make the whole run one subtask.
    rng = random.Random(args.seed)
    chosen = rng.sample(questions, min(args.limit, len(questions)))
    chosen.sort(key=lambda q: int(q["question_id"]))
    print(f"{len(questions)} questions available; taking {len(chosen)}")

    workspaces = HERE / "workspaces"
    key_dir = cache / "answers"
    key_dir.mkdir(parents=True, exist_ok=True)
    verify_script = HERE / "verify.py"
    cases = []

    for question in chosen:
        case_id = f"vstar_{question['category']}_{question['question_id']}"
        image_source = cache / "images" / Path(question["image"]).name
        try:
            fetch(f"{BASE_URL}/{question['image']}", image_source)
        except urllib.error.HTTPError as error:
            print(f"  skip {case_id}: {error}")
            continue

        workspace = workspaces / case_id
        if workspace.exists():
            shutil.rmtree(workspace)
        workspace.mkdir(parents=True)
        # Keep the extension the harness advertises, whatever the source file happened to be:
        # the tool reads the bytes, and a uniform name keeps the prompt identical per case.
        shutil.copyfile(image_source, workspace / "image.jpg")
        (workspace / "question.md").write_text(question["text"] + "\n", encoding="utf-8")

        key_path = key_dir / f"{case_id}.json"
        key_path.write_text(
            json.dumps({"label": question["label"], "question": question["text"]}),
            encoding="utf-8",
        )

        cases.append(
            {
                "id": case_id,
                "prompt": PROMPT,
                "verifier": f"python3 {verify_script} {key_path} answer.txt",
                "workspace": str(workspace.relative_to(HERE.parent.parent)),
                "split": args.split,
            }
        )
        print(f"  {case_id:<34} {question['text'].splitlines()[0][:50]}")

    curriculum = {
        "families": [
            {
                "name": "vstar",
                "description": "V*Bench visual search: small targets inside large photographs.",
                "cases": cases,
            }
        ]
    }
    Path(args.out).write_text(json.dumps(curriculum, indent=1) + "\n", encoding="utf-8")
    print(f"\nwrote {args.out} ({len(cases)} cases)")


if __name__ == "__main__":
    main()
