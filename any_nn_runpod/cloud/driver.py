"""
Driving a pod from here: connect, sync, build, run, wind down.

This is the pod half of what ``cli.command_local`` does with a subprocess.  The
steps are deliberately the same steps in the same order, because that is the
claim ``run.py local`` is making -- if it worked there it works here, and what
is left to go wrong is only the part that needs a pod.
"""

from __future__ import annotations

import time

from tqdm.auto import tqdm

from any_nn_runpod.cloud import sync
from any_nn_runpod.cloud.api import RunPodError, endpoint
from any_nn_runpod.link import Link, LinkError
from any_nn_runpod.wire.protocol import Channel
from any_nn_runpod.wire.transport import TcpTransport


def _dial_once(host, mapped, role, info, handshake_timeout=30.0):
    """One connect-and-handshake attempt, cleaned up if it does not finish.

    Connecting is not the same as being ready. RunPod publishes the port
    mapping as soon as the pod exists, so the TCP connect succeeds against its
    edge long before anything inside the container is listening -- and the
    handshake then hangs until it times out. Treating only connection refusal
    as "not yet" fails every cold start that gets that far.
    """
    channel = Channel(TcpTransport.connect(host, mapped, timeout=15))
    try:
        # Bounded, so an accepted-but-silent connection becomes a retry rather
        # than a wait forever. Link.start clears it before the reader begins --
        # a session goes quiet for perfectly good reasons.
        link = Link(channel, role=role)
        link.start(initiate=True, info=info, handshake_timeout=handshake_timeout)
        return link
    except BaseException:
        try:
            channel.close()
        except Exception:  # noqa: BLE001
            pass
        raise


def connect_supervisor(pod: dict, port: int, token=None, timeout=420.0, say=print):
    """Dial the supervisor, waiting out the pod's boot and pip install.

    A freshly created pod is reachable long before the supervisor is: the
    container has to pull the image, then install the library from git.
    """
    host, mapped = endpoint(pod, port)
    say(f"Connecting to the supervisor at {host}:{mapped}...")
    deadline = time.monotonic() + timeout
    last, attempts = None, 0
    while time.monotonic() < deadline:
        attempts += 1
        try:
            return _dial_once(
                host, mapped, "launcher", {"token": token} if token else None
            )
        except (OSError, EOFError, LinkError) as exc:
            last = exc
            if attempts % 6 == 0:
                say(
                    f"  still waiting for the supervisor "
                    f"({(time.monotonic() - deadline + timeout) / 60:.1f} min; "
                    f"the pod is installing any_nn_runpod)"
                )
            time.sleep(5.0)
    raise ConnectionError(
        f"the supervisor on {host}:{mapped} never answered ({timeout:.0f}s, "
        f"last error: {last}).\n"
        "The pod is up but its start command may have failed -- check the pod's "
        "log in the RunPod console. The command installs any_nn_runpod from git; "
        "a private repo needs pod.library_source in anr.toml to point somewhere "
        "the pod can actually reach."
    )


#: Room to leave beyond the upload itself -- pip's cache, a checkpoint being
#: written, the odd log.  A sync that fills the disk to the last byte is a pod
#: that fails at the next thing it tries instead of at the sync.
HEADROOM_GB = 5.0


def _check_room(link, wanted_bytes: int, say=print):
    """Refuse an upload the pod has nowhere to put, and say so in numbers.

    Without this the failure is an OSError from a half-written ``.part`` file
    somewhere inside a progress bar, which tells you nothing about which disk
    ran out or how much it had.
    """
    info = link.call("info", timeout=60) or {}
    free = info.get("disk_free_gb")
    if free is None:
        return  # an older supervisor; not a reason to refuse

    wanted = wanted_bytes / (1 << 30)
    if wanted + HEADROOM_GB <= free:
        return
    raise RunPodError(
        f"the pod has {free:.1f} GB free where remote/ goes "
        f"({info.get('workspace', '?')}), and this sync needs "
        f"{wanted:.1f} GB.\n"
        "Raise [pod] container_disk_gb in remote/anr.toml and make a new pod "
        "-- disk size is fixed when a pod is created.\n"
        "If that path is under /workspace it is a volume, sized by "
        "[pod] volume_gb (RunPod's default is 20 GB), not by container_disk_gb."
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
        _check_room(link, total, say)
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
            # Only meaningful without a link; with one, output goes home and
            # the pod writes nothing.
            "output_dir": recipe.output_dir if no_link else None,
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
            link = _dial_once(host, mapped, "local", None)
        except (OSError, EOFError, LinkError) as exc:
            # Same as the supervisor: the port answers before the session has
            # imported torch and started listening behind it.
            last = exc
            time.sleep(2.0)
            continue
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
    # Every field defaulted and stringified: this is used by `ps` and by the
    # confirmation before `down`, and a pod with an unexpected shape must still
    # be printable. Failing to format a line is a poor reason not to show
    # someone what is about to be terminated.
    return (
        f"{pod.get('id') or '?':<16}  {str(pod.get('name') or '?'):<20} "
        f"{str(gpu):<28} {str(pod.get('desiredStatus') or '?'):<10} "
        f"{pod.get('publicIp') or '-'}"
    )


def workspace_hint(remote_dir: str) -> str:
    files = sync.build_manifest(remote_dir)
    total = sum(size for size, _ in files.values())
    return f"{len(files)} files, {total / (1 << 20):.1f} MiB"

