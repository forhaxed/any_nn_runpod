"""
Renting a pod, and giving it back.

**Only pods this tool created are ever touched.**  Every pod it makes carries
``ANR_MANAGED=1`` in its environment, and nothing here will stop, terminate or
even list a pod without that marker.  Your own pods are invisible to it, by
construction rather than by care -- ``managed_pods`` filters before anything
else sees the list, and ``require_managed`` refuses an id that is not in it.

Ending a pod is the launcher's job, because the launcher is the only side with
an API key.  Three things end one:

* the session reports ``finished`` or ``failed`` -- ``on_finish`` decides what
  that means;
* you interrupt the launcher, and it asks;
* ``max_hours`` runs out, or the session stops answering.

The gap this leaves is honest: if your machine dies, the pod keeps running.
``run.py ps`` shows what is up and what it is costing, and ``run.py down --all``
ends all of it -- still only the managed ones.
"""

from __future__ import annotations

import time

from any_nn_runpod.cloud.api import RunPod, RunPodError, endpoint

#: Set on every pod this tool creates, and the only thing that makes a pod ours.
MARKER = "ANR_MANAGED"


def is_managed(pod: dict) -> bool:
    return str((pod.get("env") or {}).get(MARKER, "")) == "1"


def managed_pods(client: RunPod) -> tuple[list, int]:
    """``(ours, how many others)``.  The others are never returned, only counted."""
    everything = client.list_pods()
    ours = [pod for pod in everything if is_managed(pod)]
    return ours, len(everything) - len(ours)


def require_managed(client: RunPod, pod_id: str) -> dict:
    """Fetch a pod, refusing to hand back one this tool did not create."""
    pod = client.get_pod(pod_id)
    if not is_managed(pod):
        raise RunPodError(
            f"pod {pod_id} ({pod.get('name')!r}) was not created by any_nn_runpod "
            f"-- it has no {MARKER} marker. Refusing to touch it. Manage it from "
            "the RunPod console instead."
        )
    return pod


def start_command(library_source: str, ports: tuple, token: str | None) -> list:
    """What the container runs on boot.

    Installing the library from git at boot is what makes any stock image work:
    the pod needs no custom build, and it picks up whatever is on the branch.
    """
    install = library_source or "git+https://github.com/forhaxed/any_nn_runpod.git"
    environment = f"ANR_PORT={ports[0]} ANR_SESSION_PORT={ports[1]}"
    if token:
        environment += f" ANR_TOKEN={token}"
    # A `git+` source needs git, and not every stock image ships it. Installing
    # it on the second attempt turns "the pod booted and died silently" into a
    # pod that comes up anyway -- and costs nothing on images that have it.
    pip = f"pip install --no-cache-dir --upgrade '{install}'"
    return [
        "bash",
        "-lc",
        f"echo '[anr] installing any_nn_runpod from {install}'; "
        f"({pip} || (apt-get update -qq && apt-get install -y -qq git && {pip})) && "
        f"echo '[anr] starting supervisor' && "
        f"{environment} python -m any_nn_runpod.agent",
    ]


def create(
    client: RunPod,
    project,
    recipe,
    token: str | None = None,
    gpu_types: list | None = None,
    say=print,
) -> dict:
    """Create a pod for this project and wait until it can be reached."""
    gpus = list(gpu_types or recipe.gpu)
    if not gpus:
        raise RunPodError(
            "no GPU chosen. Put one in remote/anr.toml under [pod] gpu = [...], "
            "or pass --gpu. `run.py gpus` lists what is available."
        )

    ports = tuple(project.ports)
    say(
        f"Creating pod {project.pod_name!r} ({recipe.image}) on "
        + (gpus[0] if len(gpus) == 1 else f"{len(gpus)} candidate GPUs: {gpus[0]}, ...")
        + f" [{recipe.cloud_type}"
        + (f", {', '.join(recipe.data_centers)}" if recipe.data_centers else "")
        + "]..."
    )
    pod = client.create_pod(
        name=project.pod_name,
        image=recipe.image,
        gpu_types=gpus,
        gpu_count=recipe.gpu_count,
        ports=[f"{port}/tcp" for port in ports] + ["22/tcp"],
        start_command=start_command(project.library_source, ports, token),
        # The marker is what makes this pod ours, and nothing else here will
        # act on a pod without it. No API key: see the module docstring.
        env={MARKER: "1", "ANR_PROJECT": project.pod_name, **recipe.env},
        container_disk_gb=recipe.container_disk_gb,
        volume_gb=recipe.volume_gb,
        network_volume_id=recipe.network_volume_id,
        cloud_type=recipe.cloud_type,
        data_centers=recipe.data_centers,
        data_center_priority=recipe.data_center_priority,
        gpu_priority=recipe.gpu_priority,
    )
    say(
        f"  pod {pod['id']} created; waiting for a machine and an address.\n"
        f"  (Cold starts pull the image first. Several minutes is normal, and "
        f"longer on the community cloud.)"
    )
    started = time.monotonic()
    try:
        pod = client.wait_until_ready(
            pod["id"], ports, on_wait=_progress(say, started)
        )
    except BaseException as problem:
        # The pod exists and is billing from the moment it is created. If it
        # never becomes reachable -- no public IP, an image that will not pull,
        # a timeout, or a Ctrl-C while waiting -- letting the exception out of
        # here would leave it running with nobody holding a reference to it.
        # This is the one window where the caller's own cleanup cannot help,
        # because it has not been handed the pod yet.
        say(f"  pod {pod['id']} never became reachable; terminating it.")
        try:
            client.terminate_pod(pod["id"])
        except Exception as cleanup:  # noqa: BLE001
            say(
                f"  WARNING: could not terminate {pod['id']}: {cleanup}. "
                f"It is still billing -- end it with `run.py down` or the "
                f"RunPod console."
            )
        raise problem

    host, mapped = endpoint(pod, ports[0])
    # What you actually rented, and what it costs. Without this the only way to
    # find out you were given a card three times the price of the one you asked
    # for first is the invoice.
    got = pod.get("gpuTypeId") or "?"
    rate = pod.get("costPerHr")
    price = f" at ${rate}/hr" if rate else ""
    say(f"  running on {got}{price}, reachable at {host}:{mapped}")
    return pod


def _progress(say, started, every=20.0):
    """Say what a pod is doing while it starts, rather than going quiet.

    Waiting for an address is the longest silent stretch in the whole flow, and
    a silent stretch is indistinguishable from a hang -- which is how a pod that
    was merely still pulling its image gets killed for looking stuck.
    """
    last = [0.0]

    def report(pod):
        elapsed = time.monotonic() - started
        if elapsed - last[0] < every:
            return
        last[0] = elapsed
        state = pod.get("desiredStatus") or "?"
        placed = "on a machine" if pod.get("machineId") else "not placed yet"
        address = pod.get("publicIp") or "no address yet"
        say(f"    {elapsed / 60:.1f} min: {state}, {placed}, {address}")

    return report


def find_or_create(client: RunPod, project, recipe, token=None, gpu_types=None, say=print):
    """Reuse this project's pod if it is up, otherwise make one."""
    ours, _others = managed_pods(client)
    for pod in ours:
        if pod.get("name") != project.pod_name:
            continue
        if pod.get("desiredStatus") == "RUNNING":
            say(f"Reusing pod {pod['id']} ({project.pod_name!r}).")
            return client.wait_until_ready(pod["id"], tuple(project.ports))
        if pod.get("desiredStatus") == "EXITED":
            say(f"Starting stopped pod {pod['id']} ({project.pod_name!r})...")
            client.start_pod(pod["id"])
            return client.wait_until_ready(pod["id"], tuple(project.ports))
    return create(client, project, recipe, token, gpu_types, say)


def finish(client: RunPod, pod_id: str, policy: str, say=print) -> str:
    """Apply ``on_finish`` to a pod we created.  Returns what was done."""
    if policy == "keep":
        say(f"Leaving pod {pod_id} running (on_finish = keep). It is still billing.")
        return "kept"

    require_managed(client, pod_id)  # refuses anything that is not ours
    if policy == "stop":
        client.stop_pod(pod_id)
        say(f"Stopped pod {pod_id}. Its disk is kept, and still charged for.")
        return "stopped"

    client.terminate_pod(pod_id)
    say(f"Terminated pod {pod_id}.")
    return "terminated"


class Deadline:
    """Watches the clock and the session's pulse so a hung run is not forever."""

    def __init__(self, max_hours: float = 0.0, idle_seconds: float = 900.0):
        self.max_seconds = max_hours * 3600.0 if max_hours else 0.0
        self.idle_seconds = idle_seconds
        self.started = time.monotonic()
        self.last_seen = self.started

    def saw_progress(self):
        self.last_seen = time.monotonic()

    def expired(self) -> str | None:
        now = time.monotonic()
        if self.max_seconds and now - self.started > self.max_seconds:
            return f"max_hours ({self.max_seconds / 3600:.1f}h) reached"
        if self.idle_seconds and now - self.last_seen > self.idle_seconds:
            return (
                f"the session said nothing for "
                f"{(now - self.last_seen) / 60:.0f} minutes"
            )
        return None

    @property
    def elapsed_hours(self) -> float:
        return (time.monotonic() - self.started) / 3600.0
