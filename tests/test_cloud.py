"""
The pod half, tested without a pod.

Everything except the RunPod REST calls themselves: the supervisor runs over
loopback exactly as it does on a pod, and the "never touch a pod we did not
create" rule is checked against pod records rather than against an account.
"""

from __future__ import annotations

import os
import threading
import time

import pytest

from any_nn_runpod.agent import Supervisor
from any_nn_runpod.cloud import driver, lifecycle, sync
from any_nn_runpod.cloud.api import RunPodError
from any_nn_runpod.link import Link
from any_nn_runpod.wire.protocol import Channel
from any_nn_runpod.wire.transport import TcpTransport


# ======================================================================
#  Not touching other people's pods
# ======================================================================
MANAGED = {"id": "a1", "name": "anr", "env": {"ANR_MANAGED": "1"}}
THEIRS = {"id": "b2", "name": "my-important-run", "env": {"JUPYTER": "1"}}
THEIRS_LOOKALIKE = {"id": "c3", "name": "anr", "env": {}}  # same name, not ours


class FakeClient:
    """Records every destructive call so the tests can assert none happened."""

    def __init__(self, pods):
        self.pods = {pod["id"]: pod for pod in pods}
        self.terminated, self.stopped = [], []

    def list_pods(self):
        return list(self.pods.values())

    def get_pod(self, pod_id):
        return self.pods[pod_id]

    def terminate_pod(self, pod_id):
        self.terminated.append(pod_id)

    def stop_pod(self, pod_id):
        self.stopped.append(pod_id)


def test_only_pods_we_created_are_ours():
    assert lifecycle.is_managed(MANAGED)
    assert not lifecycle.is_managed(THEIRS)
    # A pod that merely shares the project's name is still not ours: the marker
    # is the whole test, because a name is something anyone might reuse.
    assert not lifecycle.is_managed(THEIRS_LOOKALIKE)


def test_listing_never_returns_someone_elses_pod():
    client = FakeClient([MANAGED, THEIRS, THEIRS_LOOKALIKE])
    ours, others = lifecycle.managed_pods(client)
    assert [pod["id"] for pod in ours] == ["a1"]
    assert others == 2  # counted, never handed over


def test_an_unmanaged_pod_cannot_be_fetched_for_action():
    client = FakeClient([MANAGED, THEIRS])
    assert lifecycle.require_managed(client, "a1") is MANAGED
    with pytest.raises(RunPodError, match="Refusing to touch it"):
        lifecycle.require_managed(client, "b2")


@pytest.mark.parametrize("policy", ["terminate", "stop"])
def test_finish_refuses_an_unmanaged_pod_and_does_nothing(policy):
    client = FakeClient([THEIRS])
    with pytest.raises(RunPodError, match="Refusing to touch it"):
        lifecycle.finish(client, "b2", policy, say=lambda _t: None)
    assert client.terminated == [] and client.stopped == []


def test_finish_acts_on_our_own_pod():
    client = FakeClient([MANAGED])
    assert lifecycle.finish(client, "a1", "terminate", say=lambda _t: None) == "terminated"
    assert client.terminated == ["a1"]

    client = FakeClient([MANAGED])
    assert lifecycle.finish(client, "a1", "stop", say=lambda _t: None) == "stopped"
    assert client.stopped == ["a1"]

    client = FakeClient([MANAGED])
    assert lifecycle.finish(client, "a1", "keep", say=lambda _t: None) == "kept"
    assert not client.terminated and not client.stopped


def test_the_pod_is_created_with_the_marker_and_no_api_key():
    from any_nn_runpod.manifest import Project, Recipe

    captured = {}

    class Recorder(FakeClient):
        def create_pod(self, **kwargs):
            captured.update(kwargs)
            return {"id": "new", "env": kwargs["env"]}

        def wait_until_ready(self, pod_id, ports, **kwargs):
            return {"id": pod_id, "publicIp": "1.2.3.4", "portMappings": {"7777": 1}}

    lifecycle.create(
        Recorder([]),
        Project(pod_name="anr-test"),
        Recipe(gpu=["NVIDIA GeForce RTX 4090"]),
        say=lambda _t: None,
    )
    assert captured["env"]["ANR_MANAGED"] == "1"
    # The key that can delete pods stays on this machine. A pod is rented from
    # strangers and runs code pulled from the internet.
    assert not any("RUNPOD_API_KEY" in key for key in captured["env"])
    assert not any("api" in str(value).lower() for value in captured["env"].values())


def test_a_pod_that_never_comes_up_is_terminated_rather_than_left_billing():
    """The one window the caller's own cleanup cannot cover.

    A pod bills from the moment it is created, and ``create`` does not hand it
    back until it is reachable. If it never becomes reachable -- no public IP,
    an image that will not pull, a timeout -- and the exception simply escaped,
    nobody would be holding a reference to a pod that is now running forever.
    """
    from any_nn_runpod.manifest import Project, Recipe

    class Stalls(FakeClient):
        def create_pod(self, **kwargs):
            self.pods["new"] = {"id": "new", "env": kwargs["env"]}
            return self.pods["new"]

        def wait_until_ready(self, pod_id, ports, **kwargs):
            raise RunPodError("never got a public IP")

    client = Stalls([])
    with pytest.raises(RunPodError, match="never got a public IP"):
        lifecycle.create(
            client,
            Project(pod_name="anr-test"),
            Recipe(gpu=["NVIDIA GeForce RTX 4090"]),
            say=lambda _t: None,
        )
    assert client.terminated == ["new"], "the stranded pod was left running"


def test_a_failure_to_terminate_a_stranded_pod_is_reported_not_hidden():
    from any_nn_runpod.manifest import Project, Recipe

    class Hopeless(FakeClient):
        def create_pod(self, **kwargs):
            self.pods["new"] = {"id": "new", "env": kwargs["env"]}
            return self.pods["new"]

        def wait_until_ready(self, pod_id, ports, **kwargs):
            raise RunPodError("never got a public IP")

        def terminate_pod(self, pod_id):
            raise RunPodError("RunPod is down too")

    said = []
    with pytest.raises(RunPodError, match="never got a public IP"):
        lifecycle.create(
            Hopeless([]),
            Project(pod_name="anr-test"),
            Recipe(gpu=["NVIDIA GeForce RTX 4090"]),
            say=said.append,
        )
    # The original problem still surfaces, and the money still gets mentioned.
    assert any("still billing" in line for line in said), said


def test_the_start_command_installs_the_library_and_runs_the_agent():
    command = lifecycle.start_command("git+https://example/repo", (7777, 7778), "sec")
    script = command[-1]
    assert "pip install" in script and "git+https://example/repo" in script
    assert "ANR_PORT=7777" in script and "ANR_SESSION_PORT=7778" in script
    assert "ANR_TOKEN=sec" in script
    assert script.rstrip().endswith("python -m any_nn_runpod.agent")


# ======================================================================
#  Sync
# ======================================================================
def write(root, relative, content):
    path = os.path.join(root, *relative.split("/"))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as handle:
        handle.write(content if isinstance(content, bytes) else content.encode())
    return path


def test_a_directory_you_named_yourself_always_travels(tmp_path):
    """``remote/`` goes up whole. "out" is not a reserved word.

    A run's output was once excluded here, because it was written inside
    remote/. That made the contract a lie: a directory of yours called ``out``
    -- weights, samples, anything -- silently stayed at home. Output is written
    beside remote/ now, so nothing about the upload has to know the name.
    """
    root = str(tmp_path)
    write(root, "train.py", "print(1)")
    write(root, "out/weights.bin", b"\x00" * 128)
    write(root, "output/notes.txt", "mine")
    write(root, "data/set.bin", b"\x00" * 64)

    manifest = sync.build_manifest(root)
    assert set(manifest) == {
        "train.py",
        "out/weights.bin",
        "output/notes.txt",
        "data/set.bin",
    }


def test_only_machine_generated_directories_are_skipped(tmp_path):
    root = str(tmp_path)
    write(root, "train.py", "print(1)")
    for junk in (
        "__pycache__/train.cpython-311.pyc",
        ".git/config",
        ".venv/pyvenv.cfg",
        ".pytest_cache/x",
    ):
        write(root, junk, b"junk")
    write(root, "notes.py~", b"junk")

    assert set(sync.build_manifest(root)) == {"train.py"}


def test_a_plan_sends_only_what_differs(tmp_path):
    root = str(tmp_path)
    write(root, "a.py", "one")
    write(root, "big.bin", b"x" * 4096)
    here = sync.build_manifest(root)

    # An identical far side sends nothing at all -- the point of the whole thing.
    assert sync.plan(here, dict(here)) == ([], [])

    there = dict(here)
    there["a.py"] = (3, "different-hash")
    there["stale.py"] = (1, "whatever")
    send, delete = sync.plan(here, there)
    assert send == ["a.py"]          # big.bin is unchanged, so it stays put
    assert delete == ["stale.py"]


def test_a_touched_file_is_not_resent(tmp_path):
    """mtime changes constantly; content does not. Re-sending 8 GB for a
    touched file is precisely what this design exists to avoid."""
    root = str(tmp_path)
    path = write(root, "weights.bin", b"y" * 8192)
    before = sync.build_manifest(root)

    os.utime(path, (time.time() + 10_000, time.time() + 10_000))
    after = sync.build_manifest(root)

    assert after == before
    assert sync.plan(after, before) == ([], [])


# ======================================================================
#  The supervisor, over loopback
# ======================================================================
class Pod:
    """A supervisor running here, driven exactly as one on a pod would be."""

    def __init__(self, workspace):
        self.supervisor = Supervisor(workspace, session_port=0, token=None)
        self.listener = None
        self.link = None
        ready = threading.Event()

        def serve():
            from any_nn_runpod.wire.transport import TcpListener

            self.listener = TcpListener("127.0.0.1", 0)
            ready.set()
            while True:
                channel = self.listener.accept(timeout=1.0)
                if channel is None:
                    if self.listener.closed:
                        return
                    continue
                try:
                    self.supervisor._serve_one(channel)
                except Exception:  # noqa: BLE001
                    return

        self.thread = threading.Thread(target=serve, daemon=True)
        self.thread.start()
        assert ready.wait(10)

        transport = TcpTransport.connect("127.0.0.1", self.listener.port)
        self.link = Link(Channel(transport), role="launcher").start(initiate=True)

    def close(self):
        if self.link is not None:
            self.link.close("test over")
        if self.listener is not None:
            self.listener.close()


def test_the_session_port_is_bound_before_any_session_exists(tmp_path):
    """The public port must be listening from boot, not from the first run.

    RunPod's edge only routes a mapped port that something was listening on
    when the pod started. A session port bound minutes later is mapped and
    dead: connecting to it times out forever while the session sits there
    listening perfectly happily. So the supervisor holds it and relays.
    """
    import socket as socket_module

    from any_nn_runpod.agent import _Forwarder

    free = socket_module.socket()
    free.bind(("127.0.0.1", 0))
    public = free.getsockname()[1]
    free.close()

    forwarder = _Forwarder(public, internal_port=public + 1000)
    forwarder.start()

    # Nothing is running behind it, and it still accepts -- which is the point.
    assert settle(lambda: _accepts("127.0.0.1", public)), "the port never opened"


def _accepts(host, port):
    import socket as socket_module

    try:
        with socket_module.create_connection((host, port), timeout=1):
            return True
    except OSError:
        return False


def test_the_forwarder_relays_to_whatever_is_listening_inside(tmp_path):
    import socket as socket_module

    from any_nn_runpod.agent import _Forwarder

    probe = socket_module.socket()
    probe.bind(("127.0.0.1", 0))
    public = probe.getsockname()[1]
    probe.close()
    internal = public + 1000

    inside = socket_module.socket()
    inside.setsockopt(socket_module.SOL_SOCKET, socket_module.SO_REUSEADDR, 1)
    inside.bind(("127.0.0.1", internal))
    inside.listen(1)

    def echo():
        # A loop, not a single accept: the readiness probe below opens a
        # connection of its own, and so does every retry. A one-shot server
        # would be consumed before the real client arrived.
        while True:
            try:
                connection, _ = inside.accept()
            except OSError:
                return
            connection.sendall(b"hello from inside")
            connection.close()

    threading.Thread(target=echo, daemon=True).start()
    _Forwarder(public, internal).start()

    assert settle(lambda: _accepts("127.0.0.1", public))
    with socket_module.create_connection(("127.0.0.1", public), timeout=10) as client:
        client.settimeout(10)
        assert client.recv(64) == b"hello from inside"
    inside.close()


def settle(predicate, timeout=10.0, interval=0.05):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(interval)
    return False


@pytest.fixture
def pod(tmp_path):
    running = Pod(str(tmp_path / "workspace"))
    yield running
    running.close()


def test_a_standalone_run_writes_beside_the_workspace_not_into_it(pod, tmp_path):
    """Output must not land inside the directory that gets synced.

    remote/ is kept in step with the local copy, so a checkpoint written into
    it is a file the next sync sees as stale and deletes. Writing beside it
    also means the upload needs no exception for a directory name.
    """
    workspace = pod.supervisor.workspace
    chosen = pod.supervisor._output_dir(None)

    assert not chosen.startswith(workspace.rstrip("/\\") + os.sep)
    assert os.path.dirname(chosen) == os.path.dirname(workspace.rstrip("/\\"))
    assert os.path.basename(chosen) == "out"

    # And a recipe that names something else still lands beside, not inside.
    named = pod.supervisor._output_dir("results")
    assert os.path.basename(named) == "results"
    assert not named.startswith(workspace.rstrip("/\\") + os.sep)


def test_the_supervisor_describes_itself(pod):
    info = pod.link.call("info", timeout=15)
    assert info["workspace"].endswith("workspace")
    assert info["running"] is False
    assert info["disk_total_gb"] > 0


def test_a_first_sync_uploads_everything_and_a_second_uploads_nothing(pod, tmp_path):
    source = tmp_path / "remote"
    write(str(source), "train.py", "print('hi')")
    write(str(source), "weights/model.bin", b"\xab" * (3 << 20))  # 3 MiB, multi-chunk
    write(str(source), "nested/deep/config.json", '{"a": 1}')

    first = driver.push_remote(pod.link, str(source), say=lambda _t: None)
    assert first["sent"] == 3
    assert first["bytes"] >= 3 << 20

    landed = tmp_path / "workspace"
    assert (landed / "train.py").read_bytes() == b"print('hi')"
    assert (landed / "weights" / "model.bin").read_bytes() == b"\xab" * (3 << 20)
    assert (landed / "nested" / "deep" / "config.json").read_text() == '{"a": 1}'

    # The claim that matters: nothing moves the second time.
    second = driver.push_remote(pod.link, str(source), say=lambda _t: None)
    assert second == {"sent": 0, "bytes": 0, "deleted": 0}


def test_sync_sends_only_the_changed_file_and_removes_stale_ones(pod, tmp_path):
    source = tmp_path / "remote"
    write(str(source), "train.py", "v1")
    write(str(source), "big.bin", b"z" * (2 << 20))
    write(str(source), "doomed.py", "delete me")
    driver.push_remote(pod.link, str(source), say=lambda _t: None)

    write(str(source), "train.py", "v2 -- edited")
    os.remove(source / "doomed.py")

    result = driver.push_remote(pod.link, str(source), say=lambda _t: None)

    # If this ever fails, the counts alone say nothing useful -- so say which
    # files the two sides disagreed about. (This has flaked once, under load,
    # and was not reproducible in fifteen further runs; the diagnosis is here
    # rather than a weakened assertion, because the counts are the property.)
    def disagreement():
        here = sync.build_manifest(str(source))
        there = pod.link.call("sync.manifest", timeout=60)["manifest"]
        return "\n".join(
            f"    {name}: local={here.get(name)} pod={there.get(name)}"
            for name in sorted(set(here) | set(there))
        )

    assert result["sent"] == 1, f"resent more than the edited file:\n{disagreement()}"
    assert result["bytes"] < 1 << 20, (
        f"sent {result['bytes']} bytes -- big.bin went again:\n{disagreement()}"
    )
    assert result["deleted"] == 1, f"deletion count wrong:\n{disagreement()}"

    landed = tmp_path / "workspace"
    assert (landed / "train.py").read_text() == "v2 -- edited"
    assert not (landed / "doomed.py").exists()
    assert (landed / "big.bin").read_bytes() == b"z" * (2 << 20)


def test_no_half_written_file_is_left_behind_when_a_sync_dies(pod, tmp_path):
    """A killed upload must not leave a truncated file that looks complete."""
    workspace = tmp_path / "workspace"
    workspace.mkdir(parents=True, exist_ok=True)
    (workspace / "model.bin").write_bytes(b"the good copy")

    # The supervisor is already accepting "sync" streams, so this goes through
    # its real receive path rather than a second one set up alongside it.
    stream = pod.link.open_stream("sync", depth=4)
    stream.send({"path": "model.bin", "data": b"partial", "first": True, "last": False})
    time.sleep(0.3)
    stream.end(error="the launcher died")

    deadline = time.monotonic() + 15
    while (workspace / "model.bin.part").exists() and time.monotonic() < deadline:
        time.sleep(0.1)

    # The previous copy is untouched, and the fragment is gone -- not left
    # sitting there at the real name, looking complete.
    assert (workspace / "model.bin").read_bytes() == b"the good copy"
    assert not (workspace / "model.bin.part").exists()


def test_a_sync_is_not_finished_until_the_files_are_really_there(pod, tmp_path):
    """``sync.finish`` returning must mean remote/ is ready to be run out of.

    Files are written on the supervisor's sync thread while this call is
    answered on its handler pool, so answering immediately would report a
    workspace as ready while chunks were still landing in it -- and the very
    next thing the launcher does is start the training script from it.
    """
    from any_nn_runpod.link import RemoteError

    with pytest.raises(RemoteError, match="never arrived intact"):
        pod.link.call(
            "sync.finish",
            {
                "delete": [],
                "expect": {"ghost.bin": (1234, "a" * 64)},  # never sent
                "timeout": 2,
            },
            timeout=60,
        )


def test_a_sync_that_did_land_is_confirmed(pod, tmp_path):
    source = tmp_path / "remote"
    write(str(source), "train.py", "print('ok')")
    driver.push_remote(pod.link, str(source), say=lambda _t: None)

    here = sync.build_manifest(str(source))
    answer = pod.link.call(
        "sync.finish", {"delete": [], "expect": here, "timeout": 10}, timeout=60
    )
    assert answer["verified"] == 1


def test_the_supervisor_starts_a_session_and_reports_its_exit(pod, tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir(parents=True, exist_ok=True)
    (workspace / "train.py").write_text(
        "from any_nn_runpod import session\n"
        "print('entry ran, connected =', session.connected)\n"
    )

    exits, output = [], []
    pod.link.handle("session.exit", lambda payload: exits.append(payload["code"]))
    pod.link.handle("session.output", lambda payload: output.append(payload["text"]))

    import sys

    started = pod.link.call(
        "run.start",
        {"entry": "train.py", "python": sys.executable, "no_link": True},
        timeout=60,
    )
    assert started["pid"]

    deadline = time.monotonic() + 90
    while not exits and time.monotonic() < deadline:
        time.sleep(0.1)
    assert exits == [0], "".join(output)
    assert "entry ran, connected = False" in "".join(output)
    assert pod.link.call("run.status", timeout=15)["running"] is False


def test_starting_a_missing_entry_says_so_instead_of_failing_obscurely(pod):
    from any_nn_runpod.link import RemoteError

    with pytest.raises(RemoteError, match="has remote/ been synced"):
        pod.link.call("run.start", {"entry": "nope.py"}, timeout=30)
