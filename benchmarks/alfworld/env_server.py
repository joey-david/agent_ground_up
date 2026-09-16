"""Hold one ALFWorld game open across separate shell commands.

The agent's shell tool starts a fresh process per command, but a text game is a long-lived
object: the tenth action only makes sense against the state the first nine produced. So the
game lives in a small daemon listening on a unix socket inside the workspace, and `act.py` is
a thin client that sends one action and prints what came back.

The daemon is started lazily by the first client call and exits after an idle timeout, so an
abandoned episode cannot leave a process behind. It runs under the ALFWorld virtualenv, which
is deliberately separate from the harness's own: ALFWorld drags in textworld and spacy, which
have no business in the agent's environment.
"""

from __future__ import annotations

import json
import socketserver
import sys
import threading
import time
from pathlib import Path

IDLE_TIMEOUT_S = 1800
MAX_EPISODE_STEPS = 50


class Game:
    """One ALFWorld game, registered directly rather than through AlfredTWEnv.

    AlfredTWEnv exists to sample games out of a directory tree according to a large training
    config. A benchmark case is the opposite: one named game, opened the same way every time.
    """

    def __init__(self, game_file: str, workspace: Path) -> None:
        import textworld
        import textworld.gym
        from alfworld.agents.environment.alfred_tw_env import AlfredDemangler, AlfredInfos

        request_infos = textworld.EnvInfos(
            won=True, admissible_commands=True, extras=["gamefile"]
        )
        env_id = textworld.gym.register_games(
            [game_file],
            request_infos,
            batch_size=1,
            asynchronous=True,
            wrappers=[AlfredDemangler(shuffle=False), AlfredInfos],
            max_episode_steps=MAX_EPISODE_STEPS,
        )
        self.env = textworld.gym.make(env_id)
        observations, infos = self.env.reset()
        self.observation = observations[0]
        self.admissible = list(infos["admissible_commands"][0])
        self.steps = 0
        self.done = False
        self.won = False
        self.workspace = workspace
        self._record()

    def step(self, action: str) -> dict:
        if self.done:
            return self.state(note="this episode has already ended")
        observations, _rewards, dones, infos = self.env.step([action])
        self.observation = observations[0]
        self.admissible = list(infos["admissible_commands"][0])
        self.done = bool(dones[0])
        self.won = bool(infos["won"][0])
        self.steps += 1
        self._record()
        return self.state()

    def _record(self) -> None:
        """Write the outcome after every step.

        The verifier runs after the agent has stopped, by which time this process may have
        idled out. Reading a file cannot resurrect a fresh game and score a finished episode
        as a loss, which is what asking a newly started daemon would do.
        """
        (self.workspace / "result.json").write_text(
            json.dumps({"won": self.won, "done": self.done, "steps": self.steps})
        )

    def state(self, note: str = "") -> dict:
        return {
            "observation": self.observation,
            "admissible_commands": self.admissible,
            "steps": self.steps,
            "steps_remaining": max(0, MAX_EPISODE_STEPS - self.steps),
            "done": self.done,
            "won": self.won,
            "note": note,
        }


def serve(socket_path: Path, game_file: str, workspace: Path) -> None:
    game = Game(game_file, workspace)
    last_seen = [time.monotonic()]

    class Handler(socketserver.StreamRequestHandler):
        def handle(self) -> None:
            last_seen[0] = time.monotonic()
            raw = self.rfile.readline().decode("utf-8").strip()
            request = json.loads(raw) if raw else {}
            if request.get("command") == "step":
                reply = game.step(request.get("action", ""))
            else:
                reply = game.state()
            self.wfile.write((json.dumps(reply) + "\n").encode("utf-8"))

    socket_path.unlink(missing_ok=True)
    server = socketserver.ThreadingUnixStreamServer(str(socket_path), Handler)
    server.daemon_threads = True

    def reaper() -> None:
        """Exit when the episode is over, whether it ended politely or not.

        The harness copies each workspace to a temporary directory and deletes it afterwards,
        so a vanished workspace means the episode is gone; the idle timeout covers the case
        where the whole run was interrupted and nothing got cleaned up.
        """
        while True:
            time.sleep(15)
            if not workspace.exists():
                server.shutdown()
                return
            if time.monotonic() - last_seen[0] > IDLE_TIMEOUT_S:
                server.shutdown()
                return

    threading.Thread(target=reaper, daemon=True).start()
    try:
        server.serve_forever()
    finally:
        socket_path.unlink(missing_ok=True)


def main() -> int:
    if len(sys.argv) != 4:
        print("usage: env_server.py <socket-path> <game-file> <workspace>", file=sys.stderr)
        return 2
    serve(Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
