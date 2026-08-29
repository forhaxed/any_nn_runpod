"""
What happens to the pod when a run ends.

The expensive mistake in one direction is a pod left billing; in the other it
is a terminated pod that was holding the only copy of a run's output.
"""

from __future__ import annotations

import argparse

import pytest

from any_nn_runpod import cli
from any_nn_runpod.cloud import lifecycle
from any_nn_runpod.cloud.api import RunPodError
from any_nn_runpod.manifest import Project

POD = {"id": "p1", "name": "anr", "env": {"ANR_MANAGED": "1"}}


class FakeClient:
    def __init__(self):
        self.terminated, self.stopped = [], []

    def get_pod(self, pod_id):
        return POD

    def terminate_pod(self, pod_id):
        self.terminated.append(pod_id)

    def stop_pod(self, pod_id):
        self.stopped.append(pod_id)


def finish(had_local, on_finish=None, project_policy="terminate", outcome="finished"):
    client = FakeClient()
    said = []
    cli._say = said.append  # the messages are part of the behaviour here
    code = cli._finish_pod(
        client,
        POD,
        Project(on_finish=project_policy),
        outcome,
        argparse.Namespace(on_finish=on_finish, yes=True),
        lifecycle.Deadline(),
        had_local=had_local,
    )
    return client, "\n".join(said), code


@pytest.fixture(autouse=True)
def restore_say():
    original = cli._say
    yield
    cli._say = original


def test_a_run_with_a_local_side_terminates_as_configured():
    """Everything it produced is already at home, so the pod is disposable."""
    client, _said, _code = finish(had_local=True)
    assert client.terminated == ["p1"]
    assert client.stopped == []


def test_a_run_with_no_local_side_is_stopped_instead_of_terminated():
    """Its output exists only on the pod. Terminating would be deleting it.

    Stopping keeps /workspace -- which is a volume -- and releases the GPU.
    """
    client, said, _code = finish(had_local=False)
    assert client.terminated == [], "the only copy of the output was destroyed"
    assert client.stopped == ["p1"]
    assert "only exists on the pod" in said
    assert "run.py down" in said  # and it says how to finish the job


def test_an_explicit_terminate_still_wins_without_a_local_side():
    """Someone passing --on-finish terminate is saying they know."""
    client, _said, _code = finish(had_local=False, on_finish="terminate")
    assert client.terminated == ["p1"]


def test_the_downgrade_only_applies_to_terminate():
    for policy in ("stop", "keep"):
        client, _said, _code = finish(had_local=False, project_policy=policy)
        assert client.terminated == []


def test_a_failure_to_end_the_pod_is_reported_loudly():
    """Silence here is a pod that bills until someone happens to look."""

    class Stubborn(FakeClient):
        def terminate_pod(self, pod_id):
            raise RunPodError("RunPod said no")

    said = []
    cli._say = said.append
    cli._finish_pod(
        Stubborn(),
        POD,
        Project(on_finish="terminate"),
        "finished",
        argparse.Namespace(on_finish=None, yes=True),
        lifecycle.Deadline(),
        had_local=True,
    )
    text = "\n".join(said)
    assert "still running and still billing" in text
    assert "run.py down" in text
