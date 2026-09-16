"""Take one action in this workspace's ALFWorld game.

    python3 act.py go to fridge 1      # act, and print what happens
    python3 act.py --look              # repeat the current observation without acting
    python3 act.py --commands          # list the actions the game accepts right now

The game itself runs in a background process that this script starts the first time it is
called, so the room keeps its state between commands even though each command is a new shell.
"""

from __future__ import annotations

import hashlib
import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
# A unix socket path is capped near 104 bytes, and the harness runs episodes out of long
# temporary directories, so the socket lives under /tmp keyed by the workspace it belongs to.
SOCKET_PATH = Path("/tmp") / f"alfw-{hashlib.sha1(str(HERE).encode()).hexdigest()[:12]}.sock"
CONFIG_PATH = HERE / "game.json"
RESULT_PATH = HERE / "result.json"
STARTUP_TIMEOUT_S = 180


def start_daemon(config: dict) -> None:
    log = (HERE / "game.log").open("ab")
    subprocess.Popen(
        [
            config["interpreter"],
            config["server"],
            str(SOCKET_PATH),
            config["game_file"],
            str(HERE),
        ],
        stdout=log,
        stderr=log,
        stdin=subprocess.DEVNULL,
        start_new_session=True,
        env={**os.environ, "ALFWORLD_DATA": config["alfworld_data"]},
    )


def request(payload: dict) -> dict:
    config = json.loads(CONFIG_PATH.read_text())
    deadline = time.monotonic() + STARTUP_TIMEOUT_S
    started = False
    while True:
        try:
            with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
                client.connect(str(SOCKET_PATH))
                client.sendall((json.dumps(payload) + "\n").encode("utf-8"))
                data = b""
                while not data.endswith(b"\n"):
                    chunk = client.recv(65536)
                    if not chunk:
                        break
                    data += chunk
            return json.loads(data.decode("utf-8"))
        except (FileNotFoundError, ConnectionRefusedError):
            # The first call in a workspace pays for loading the game; later ones are instant.
            if not started:
                start_daemon(config)
                started = True
            if time.monotonic() > deadline:
                raise SystemExit("the game process did not come up; see game.log")
            time.sleep(1.0)


def render(state: dict, *, with_commands: bool) -> str:
    lines = [state["observation"].strip()]
    if state.get("note"):
        lines.append(f"[{state['note']}]")
    if with_commands:
        lines.append("")
        lines.append("Actions accepted right now:")
        lines.extend(f"  {command}" for command in state["admissible_commands"])
    status = "won" if state["won"] else ("over" if state["done"] else "in progress")
    lines.append("")
    lines.append(
        f"[step {state['steps']}, {state['steps_remaining']} remaining, task {status}]"
    )
    return "\n".join(lines)


def main() -> int:
    arguments = sys.argv[1:]
    if not arguments:
        print(__doc__)
        return 2
    if arguments[0] == "--commands":
        print(render(request({"command": "status"}), with_commands=True))
        return 0
    if arguments[0] == "--look":
        print(render(request({"command": "status"}), with_commands=False))
        return 0
    if arguments[0] == "--status":
        # Used by the verifier: exit code carries the result, so nothing needs parsing. This
        # path never starts the game -- an episode where it was never played is simply not won,
        # and booting a fresh one here would overwrite a finished episode's outcome.
        won = False
        if RESULT_PATH.exists():
            won = bool(json.loads(RESULT_PATH.read_text()).get("won"))
        print("won" if won else "not won")
        return 0 if won else 1

    action = " ".join(arguments)
    print(render(request({"command": "step", "action": action}), with_commands=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
