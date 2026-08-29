"""
The supervisor: the one thing that runs on the pod from the moment it boots.

It is started by the pod's container command and does four things -- accept
``remote/``, build the environment the recipe asks for, start the training
session, and report what it is doing.  It never imports the training script and
it holds no credentials.

Two processes, two ports, on purpose:

    7777  supervisor   thin, long-lived, survives the training crashing
    7778  session      the training script, its Link, its dataset and its GPU

The alternative -- one connection multiplexed between control traffic and
gigabytes of tensor data -- would need a multiplexer in the protocol, and would
put the supervisor in the blast radius of every CUDA OOM.  Two sockets cost one
extra exposed port and buy a supervisor that is still there to tell you what
happened.

    python -m any_nn_runpod.agent --port 7777 --token SECRET
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import threading
import time
import traceback

from any_nn_runpod.link import Link
from any_nn_runpod.wire.transport import TcpListener

WORKSPACE = os.environ.get("ANR_WORKSPACE", "/workspace/remote")
SESSION_PORT = int(os.environ.get("ANR_SESSION_PORT", 7778))


class Supervisor:
    """Serves one launcher at a time; outlives every session it starts."""

    def __init__(self, workspace: str, session_port: int, token: str | None):
        self.workspace = os.path.abspath(workspace)
        self.session_port = session_port
        self.token = token
        self.link = None
        self.child = None
        self.child_output: list = []
        self.last_exit: int | None = None
        self._pump = None
        os.makedirs(self.workspace, exist_ok=True)

    # ================================================================
    #  Serving
    # ================================================================
    def serve_forever(self, host: str, port: int):
        listener = TcpListener(host, port)
        _announce(port, self.session_port, listener.port)
        while True:
            channel = listener.accept()
            if channel is None:
                continue
            try:
                self._serve_one(channel)
            except Exception:  # noqa: BLE001 -- one bad client is not fatal
                traceback.print_exc()
            finally:
                self.link = None

    def _serve_one(self, channel):
        link = Link(channel, role="supervisor")
        link.start(initiate=False, info={"workspace": self.workspace})

        if self.token and link.peer.get("token") != self.token:
            print("rejecting a client with a bad token", file=sys.stderr, flush=True)
            link.close("bad token")
            return

        self.link = link
        for name in (
            "sync.manifest",
            "sync.finish",
            "env.build",
            "run.start",
            "run.status",
            "run.stop",
            "info",
        ):
            link.handle(name, getattr(self, "_" + name.replace(".", "_")))

        threading.Thread(
            target=self._accept_sync, args=(link,), name="anr-sync", daemon=True
        ).start()

        print("launcher connected", file=sys.stderr, flush=True)
        link.wait_closed()
        print(f"launcher gone ({link.close_reason})", file=sys.stderr, flush=True)

    # ================================================================
    #  Handlers
    # ================================================================
    def _info(self, _payload=None):
        total, used, free = shutil.disk_usage(self.workspace)
        return {
            "workspace": self.workspace,
            "session_port": self.session_port,
            "disk_free_gb": round(free / (1 << 30), 1),
            "disk_total_gb": round(total / (1 << 30), 1),
            "running": self._running(),
            "last_exit": self.last_exit,
        }

    def _sync_manifest(self, _payload=None):
        from any_nn_runpod.cloud import sync

        return {"manifest": sync.build_manifest(self.workspace)}

    def _sync_finish(self, payload):
        """Confirm the upload actually landed, then remove what is stale.

        Files are written by the sync thread, while this call is answered on
        the handler pool -- so returning immediately would tell the launcher
        the workspace was ready while chunks were still being written to it,
        and the very next step is to run the training script out of it.

        Rather than guess at thread timing, this waits for the goal state: the
        files the launcher says it sent, present with the fingerprints it says
        they should have.  Only those, so the cost is proportional to what
        moved, not to the size of the workspace.
        """
        from any_nn_runpod.cloud import sync

        expect = payload.get("expect") or {}
        deadline = time.monotonic() + float(payload.get("timeout") or 900.0)
        missing = _unmet(self.workspace, expect)
        while missing and time.monotonic() < deadline:
            time.sleep(0.05)
            missing = _unmet(self.workspace, expect)
        if missing:
            raise TimeoutError(
                f"{len(missing)} file(s) never arrived intact, e.g. "
                f"{sorted(missing)[:3]}"
            )

        removed = sync.apply_deletions(self.workspace, payload.get("delete") or [])
        return {"deleted": removed, "verified": len(expect)}

    def _accept_sync(self, link):
        """Take file streams for as long as this launcher is connected."""
        from any_nn_runpod.cloud import sync

        while not link.closed:
            try:
                reader = link.accept_stream("sync", timeout=5)
            except TimeoutError:
                continue
            except Exception:  # noqa: BLE001 -- the link went away
                return
            try:
                summary = sync.receive_files(reader, self.workspace)
                print(
                    f"sync: {summary['files']} files, "
                    f"{summary['bytes'] / (1 << 20):.1f} MiB",
                    file=sys.stderr,
                    flush=True,
                )
            except Exception:  # noqa: BLE001 -- reported, not fatal
                traceback.print_exc()

    def _env_build(self, payload):
        from any_nn_runpod import envs

        recipe = payload["recipe"]
        say = lambda text: self._tell(text + "\n")  # noqa: E731
        python = envs.resolve(
            recipe,
            cache_root=payload.get("cache_root") or envs.DEFAULT_CACHE,
            install=payload.get("install"),
            say=say,
            force=bool(payload.get("force")),
        )
        return {"python": python}

    def _run_start(self, payload):
        if self._running():
            raise RuntimeError(
                "a session is already running on this pod; stop it first"
            )
        entry = os.path.join(self.workspace, payload["entry"])
        if not os.path.isfile(entry):
            raise FileNotFoundError(
                f"{entry} is not on the pod -- has remote/ been synced?"
            )

        command = [
            payload.get("python") or sys.executable,
            "-u",
            "-m",
            "any_nn_runpod.run_session",
            "--entry",
            entry,
            "--output-dir",
            payload.get("output_dir") or os.path.join(self.workspace, "out"),
        ]
        command += (
            ["--no-link"]
            if payload.get("no_link")
            else [
                "--host",
                "0.0.0.0",
                "--port",
                str(self.session_port),
                "--wait",
                str(payload.get("wait", 300)),
            ]
        )

        self.child_output = []
        self.last_exit = None
        self.child = subprocess.Popen(
            command,
            cwd=self.workspace,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            bufsize=0,
            env={**os.environ, "PYTHONUNBUFFERED": "1"},
        )
        self._pump = threading.Thread(
            target=self._pump_child, name="anr-session-output", daemon=True
        )
        self._pump.start()
        return {"pid": self.child.pid, "session_port": self.session_port}

    def _run_status(self, _payload=None):
        return {
            "running": self._running(),
            "exit_code": self.last_exit,
            "pid": self.child.pid if self.child else None,
        }

    def _run_stop(self, _payload=None):
        if not self._running():
            return {"stopped": False}
        self.child.terminate()
        try:
            self.child.wait(timeout=20)
        except subprocess.TimeoutExpired:
            self.child.kill()
            self.child.wait(timeout=10)
        return {"stopped": True, "exit_code": self.child.returncode}

    # ================================================================
    #  Plumbing
    # ================================================================
    def _running(self) -> bool:
        return self.child is not None and self.child.poll() is None

    def _tell(self, text: str):
        """Say something to the launcher, and to the pod's own log."""
        sys.stderr.write(text)
        sys.stderr.flush()
        link = self.link
        if link is not None and link.connected:
            link.notify("session.output", {"text": text})

    def _pump_child(self):
        """Forward the session's output while it runs.

        Kept even when nobody is listening: the tail is what ``run.py logs``
        shows after reconnecting, and it is the only record of why a session
        died if the launcher was not attached when it did.
        """
        child = self.child
        while True:
            chunk = child.stdout.read(4096)
            if not chunk:
                break
            text = chunk.decode("utf-8", "replace")
            self.child_output.append(text)
            del self.child_output[:-2000]
            self._tell(text)
        self.last_exit = child.wait()
        self._tell(f"\n[session exited with code {self.last_exit}]\n")
        link = self.link
        if link is not None and link.connected:
            link.notify("session.exit", {"code": self.last_exit})


def _unmet(workspace: str, expect: dict) -> list:
    """Which expected files are not yet on disk with the right content."""
    from any_nn_runpod.cloud import sync

    missing = []
    for relative, fingerprint in expect.items():
        path = os.path.join(workspace, *relative.split("/"))
        size, digest = fingerprint
        try:
            if os.path.getsize(path) != size:
                missing.append(relative)
                continue
        except OSError:
            missing.append(relative)
            continue
        # Size matches, so the write is at least finished; the hash is what
        # says it is the *right* file.
        if sync._digest(path) != digest:
            missing.append(relative)
    return missing


def _announce(port: int, session_port: int, actual: int):
    """Print what a launcher needs, including RunPod's external mapping."""
    public_ip = os.environ.get("RUNPOD_PUBLIC_IP")
    print(f"any_nn_runpod supervisor listening on {actual}", file=sys.stderr)
    print(f"  session port: {session_port}", file=sys.stderr)
    for container in (port, session_port):
        mapped = os.environ.get(f"RUNPOD_TCP_PORT_{container}")
        if public_ip and mapped:
            print(f"  {container} -> {public_ip}:{mapped}", file=sys.stderr)
    sys.stderr.flush()


def main(argv=None):
    parser = argparse.ArgumentParser(prog="python -m any_nn_runpod.agent")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument(
        "--port", type=int, default=int(os.environ.get("ANR_PORT", 7777))
    )
    parser.add_argument("--session-port", type=int, default=SESSION_PORT)
    parser.add_argument("--workspace", default=WORKSPACE)
    parser.add_argument("--token", default=os.environ.get("ANR_TOKEN"))
    args = parser.parse_args(argv)

    supervisor = Supervisor(args.workspace, args.session_port, args.token)
    supervisor.serve_forever(args.host, args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
