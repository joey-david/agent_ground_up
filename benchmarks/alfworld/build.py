"""Materialize ALFWorld games as verifiable workspaces for the A/B harness.

ALFWorld is a text-based household environment: the agent is dropped in a room and told to do
something like "heat some apple and put it in garbagecan", then has fifty actions to do it. It
is the standard testbed for agents that learn across episodes, because the six task types recur
with different objects and rooms -- the second "heat X and put it in Y" is the same procedure
as the first, in a different kitchen.

That makes it the right benchmark for the subsystems: a harness that can write down what
worked and reuse it should pull away from one that starts every episode blank, and neither of
them can brute-force the task the way a coding agent can run the tests until they pass.

    uv run python benchmarks/alfworld/build.py --limit 18

Requires the ALFWorld data (`alfworld-download`) and an interpreter with alfworld installed;
point --interpreter at it, since it is deliberately not the harness's own environment.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent

# The six ALFWorld task types, in the order the benchmark names them.
TASK_TYPES = [
    "pick_and_place_simple",
    "look_at_obj_in_light",
    "pick_clean_then_place_in_recep",
    "pick_heat_then_place_in_recep",
    "pick_cool_then_place_in_recep",
    "pick_two_obj_and_place",
]

PROMPT = """You are playing a text-based household game. Your task is stated in task.md.

Interact with the game through the command line:

    python3 act.py <action>     take one action, for example: python3 act.py go to fridge 1
    python3 act.py --look       repeat the current observation without using a turn
    python3 act.py --commands   list the actions the game will accept right now

Only actions the game accepts do anything; anything else wastes a turn and it tells you so.
You have a limited number of turns, shown after every action. The episode is finished when the
status line says the task is won, and that is the only thing that counts as success."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=18, help="games to materialize")
    parser.add_argument("--data", default=os.environ.get("ALFWORLD_DATA", ""),
                        help="ALFWorld data directory (alfworld-download)")
    parser.add_argument("--interpreter", default="",
                        help="python that has alfworld installed")
    parser.add_argument("--subset", default="valid_unseen",
                        choices=["valid_seen", "valid_unseen", "valid_train", "train"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--split", default="alfworld")
    parser.add_argument("--out", default=str(HERE / "curriculum.json"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.data or not args.interpreter:
        raise SystemExit("--data and --interpreter are required (see this file's docstring)")
    data_root = Path(args.data).expanduser() / "json_2.1.1" / args.subset
    if not data_root.is_dir():
        raise SystemExit(f"no ALFWorld games under {data_root}")

    # Group by task type first, then take a round-robin slice: the point of this benchmark is
    # that task types repeat, so every type has to appear more than once.
    by_type: dict[str, list[Path]] = defaultdict(list)
    for game_file in sorted(data_root.glob("*/*/game.tw-pddl")):
        task_type = game_file.parent.parent.name.split("-")[0]
        if task_type in TASK_TYPES:
            by_type[task_type].append(game_file)

    rng = random.Random(args.seed)
    for games in by_type.values():
        rng.shuffle(games)

    chosen: list[Path] = []
    while len(chosen) < args.limit:
        added = False
        for task_type in TASK_TYPES:
            if by_type[task_type] and len(chosen) < args.limit:
                chosen.append(by_type[task_type].pop())
                added = True
        if not added:
            break
    print(f"{sum(len(v) for v in by_type.values()) + len(chosen)} games available; "
          f"taking {len(chosen)} across {len(TASK_TYPES)} task types")

    workspaces = HERE / "workspaces"
    cases = []
    for index, game_file in enumerate(chosen):
        task_type = game_file.parent.parent.name.split("-")[0]
        case_id = f"alfworld_{index:02d}_{task_type}"
        workspace = workspaces / case_id
        if workspace.exists():
            shutil.rmtree(workspace)
        workspace.mkdir(parents=True)

        goal = json.loads((game_file.parent / "traj_data.json").read_text())
        description = goal["turk_annotations"]["anns"][0]["task_desc"]
        (workspace / "task.md").write_text(
            f"# Task\n\n{description}\n\nRun `python3 act.py --look` to see the room.\n",
            encoding="utf-8",
        )
        shutil.copyfile(HERE / "act.py", workspace / "act.py")
        (workspace / "game.json").write_text(
            json.dumps(
                {
                    "game_file": str(game_file),
                    "server": str(HERE / "env_server.py"),
                    "interpreter": str(Path(args.interpreter).expanduser()),
                    "alfworld_data": str(Path(args.data).expanduser()),
                },
                indent=1,
            ),
            encoding="utf-8",
        )

        cases.append(
            {
                "id": case_id,
                "prompt": PROMPT,
                # The game itself is the answer key: it knows whether the goal was reached.
                "verifier": "python3 act.py --status",
                "workspace": str(workspace.relative_to(HERE.parent.parent)),
                "split": args.split,
            }
        )
        print(f"  {case_id:<44} {description[:46]}")

    curriculum = {
        "families": [
            {
                "name": "alfworld",
                "description": "ALFWorld household tasks; six task types that recur.",
                "cases": cases,
            }
        ]
    }
    Path(args.out).write_text(json.dumps(curriculum, indent=1) + "\n", encoding="utf-8")
    print(f"\nwrote {args.out} ({len(cases)} cases)")


if __name__ == "__main__":
    main()
