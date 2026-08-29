"""
Driving a pod from here: connect, sync, build, run, wind down.

This is the pod half of what ``cli.command_local`` does with a subprocess.  The
steps are deliberately the same steps in the same order, because that is the
claim ``run.py local`` is making -- if it worked there it works here, and what
is left to go wrong is only the part that needs a pod.
"""

from __future__ import annotations

import os
import time

from tqdm.auto import tqdm

from any_nn_runpod.cloud import lifecycle, sync
from any_nn_runpod.cloud.api import endpoint
from any_nn_runpod.link import Link
from any_nn_runpod.wire.protocol import Channel
from any_nn_runpod.wire.transport import TcpTransport


def connect_supervisor(pod: dict, port: int, token=None, timeout=300.0, say=print):
    """Dial the supervisor, waiting out the pod's boot and pip install.

    A freshly created pod is reachable long before the supervisor is: the
    container has to pull the image, then install the library. Refusing on the
    first connection error would fail every cold start.
    """
    host, mapped = endpoint(pod, port)
    say(f"Connecting to the supervisor at {host}:{mapped}...")
    deadline = time.monotonic() + timeout
    last = None
    while time.monotonic() < deadline:
        try:
            channel = Channel(TcpTransport.connect(host, mapped, timeout=10))
        except OSError as exc:
            last = exc
            time.sleep(5.0)
            continue
        link = Link(channel, role="launcher")
        link.start(initiate=True, info={"token": token} if token else None)
        return link
    raise ConnectionError(
        f"the supervisor on {host}:{mapped} never answered ({timeout:.0f}s, "
        f"last error: {last}).\n"
        "The pod is up but its start command may have failed -- check the pod's "
        "log in the RunPod console. The command installs any_nn_runpod from git; "
        "a private repo needs pod.library_source in anr.toml to point somewhere "
        "the pod can actually reach."
    )


def push_remote(link, remote_dir: str, say=print) -> dict:
    """Send only what differs.  Returns what moved."""
    say("Syncing remote/ ...")
    here = sync.build_manifest(remote_dir)
    there = (link.call("sync.manifest", timeout=600) or {}).get("manifest", {})
    send, delete = sync.plan(here, there)

    if not send and not delete:
        say(f"  up to date ({len(here)} files, nothing to send)")
        return {"sent": 0, "bytes": 0, "deleted": 0}

    moved = 0
    if send:
        total = sync.total_bytes(remote_dir, send)
        say(f"  {len(send)} file(s) to send, {total / (1 << 20):.1f} MiB")
        bar = tqdm(total=total, unit="B", unit_scale=True, desc="  uploading", leave=False)
        stream = link.open_stream("sync", depth=8)
        try:
            moved = sync.send_files(
                stream, remote_dir, send, on_progress=lambda n: bar.update(n - bar.n)
            )
        finally:
            stream.end()
            bar.close()

    # Naming what was sent, so the far side can confirm it landed intact rather
    # than merely that the last chunk was handed to a socket. This call does not
    # return until remote/ is genuinely ready to be run out of.
    answer = (
        link.call(
            "sync.finish",
            {"delete": delete, "expect": {name: here[name] for name in send}},
            timeout=1200,
        )
        or {}
    )
    removed = answer.get("deleted", 0)

    say(f"  sent {moved / (1 << 20):.1f} MiB, removed {removed} stale file(s)")
    return {"sent": len(send), "bytes": moved, "deleted": removed}


def build_environment(link, recipe, library_source: str, force=False, say=print) -> str:
    """Build (or reuse) the venv the recipe describes, on the pod."""
    say("Building the environment on the pod...")
    answer = link.call(
        "env.build",
        {
            "recipe": recipe.environment,
            "install": [library_source] if library_source else None,
            "force": force,
        },
        timeout=3600,
    )
    return (answer or {}).get("python")


def start_session(link, recipe, python, no_link: bool, say=print) -> dict:
    say(f"Starting {recipe.entry} on the pod...")
    return link.call(
        "run.start",
        {
            "entry": recipe.entry,
            "python": python,
            "output_dir": None if not no_link else recipe.output_dir,
            "no_link": no_link,
            "wait": 300,
        },
        timeout=120,
    )


def attach_local(app, pod: dict, port: int, timeout=300.0, say=print):
    """Connect ``local/local.py`` to the session listening on the pod."""
    host, mapped = endpoint(pod, port)
    say(f"Attaching the local side to {host}:{mapped}...")
    deadline = time.monotonic() + timeout
    last = None
    while time.monotonic() < deadline:
        try:
            channel = Channel(TcpTransport.connect(host, mapped, timeout=10))
        except OSError as exc:
            last = exc
            time.sleep(2.0)
            continue
        link = Link(channel, role="local").start(initiate=True)
        app.attach(link)
        return link
    raise ConnectionError(
        f"the session on {host}:{mapped} never accepted a connection "
        f"({timeout:.0f}s, last error: {last}). Its output is above."
    )


def relay_output(link, console, deadline=None):
    """Print what the supervisor forwards from the session."""

    def on_output(payload):
        console.text((payload or {}).get("text", ""))
        if deadline is not None:
            deadline.saw_progress()

    link.handle("session.output", on_output)
    link.handle("session.exit", lambda payload: None)


def describe(pod: dict) -> str:
    machine = pod.get("machine") or {}
    gpu = pod.get("gpuTypeId") or machine.get("gpuTypeId") or "?"
    return (
        f"{pod.get('id')}  {pod.get('name'):<20} {gpu:<28} "
        f"{pod.get('desiredStatus', '?'):<10} {pod.get('publicIp') or '-'}"
    )


def workspace_hint(remote_dir: str) -> str:
    files = sync.build_manifest(remote_dir)
    total = sum(size for size, _ in files.values())
    return f"{len(files)} files, {total / (1 << 20):.1f} MiB"


def output_dir_for(project) -> str:
    return os.path.abspath(project.output_path)
