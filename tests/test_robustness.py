"""
The things that only go wrong under load, or when something dies.

A pod run is hours long over a link that can drop, feeding a GPU that must not
be starved.  These are the failure modes that would otherwise be discovered
there, at hourly rates.
"""

from __future__ import annotations

import threading
import time

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from any_nn_runpod.dataset import DatasetWrapper
from any_nn_runpod.link import RemoteError
from any_nn_runpod.local import Local
from any_nn_runpod.reporting import LoggerWrapper
from test_pipeline import TinyTrainer, Wired, build_trainer, make_loader


@pytest.fixture
def wired(tmp_path):
    def build(app=None):
        build.harness = Wired(app or Local(output_dir=str(tmp_path), console=False))
        return build.harness

    build.harness = None
    yield build
    if build.harness is not None:
        build.harness.close()


def settle(predicate, timeout=15.0, interval=0.05):
    """Wait for something that happens on another thread."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(interval)
    return False


# ======================================================================
#  Volume
# ======================================================================
def test_a_large_payload_arrives_byte_for_byte(wired, tmp_path):
    """64 MB in one message, through the real socket, unchanged.

    Weights and latent batches are this size routinely, and a framing bug that
    a 4-element tensor hides shows up here as silent corruption.
    """
    big = torch.randn(4, 1024, 1024, dtype=torch.float32)  # 16 MiB
    checksum = float(big.double().sum())

    app = Local(output_dir=str(tmp_path), console=False)
    app.dataset("train", lambda: [(big, torch.zeros(4))] * 4)
    harness = wired(app)

    data = DatasetWrapper("train", precache=2, prepare=1).bind(harness.remote)
    for tensor, _ in data:
        assert tensor.shape == big.shape
        assert float(tensor.double().sum()) == pytest.approx(checksum, rel=1e-12)
        assert torch.equal(tensor, big)


def test_every_dtype_survives_the_socket(wired, tmp_path):
    """bfloat16 in particular: numpy cannot hold it, so it must go as raw bytes."""
    payload = {
        "f32": torch.randn(64, 64),
        "f16": torch.randn(64, 64).half(),
        "bf16": torch.randn(64, 64).bfloat16(),
        "i64": torch.randint(-999, 999, (64, 64)),
        "u8": torch.randint(0, 255, (64, 64), dtype=torch.uint8),
        "bool": torch.rand(64, 64) > 0.5,
    }
    app = Local(output_dir=str(tmp_path), console=False)
    app.dataset("train", lambda: [payload])
    harness = wired(app)

    got = list(DatasetWrapper("train", precache=2).bind(harness.remote))[0]
    for key, original in payload.items():
        assert got[key].dtype == original.dtype, key
        assert torch.equal(got[key], original), key


def test_logs_and_data_share_the_link_without_tangling(wired, tmp_path):
    """Both directions at once: batches going one way, logs the other.

    There is no multiplexing by design, so this is the test that says the
    "one writer per direction" rule actually holds under concurrent use.
    """
    app = Local(output_dir=str(tmp_path), console=False)
    app.dataset("train", lambda: make_loader(samples=64 * 40, batch_size=64))
    harness = wired(app)
    logger = LoggerWrapper(harness.remote, str(tmp_path))

    data = DatasetWrapper("train", precache=8, prepare=2).bind(harness.remote)
    count = 0
    for index, _batch in enumerate(data):
        count += 1
        logger.log({"train/loss": 1.0 / (index + 1)}, index)
        logger.print(f"step {index}")
    assert count == 40


# ======================================================================
#  Things dying
# ======================================================================
def test_a_dropped_link_raises_on_the_consumer_rather_than_hanging(wired, tmp_path):
    app = Local(output_dir=str(tmp_path), console=False)
    app.dataset("train", lambda: make_loader(samples=64 * 200, batch_size=64))
    harness = wired(app)

    data = DatasetWrapper("train", precache=4, prepare=1).bind(harness.remote)
    outcome = {}

    def consume():
        try:
            for index, _batch in enumerate(data):
                if index == 2:
                    harness.local.close("the local side died")
                time.sleep(0.02)
        except BaseException as exc:  # noqa: BLE001
            outcome["error"] = exc

    thread = threading.Thread(target=consume, daemon=True)
    thread.start()
    thread.join(timeout=20)

    assert not thread.is_alive(), "the consumer hung instead of failing"
    assert isinstance(outcome.get("error"), RemoteError)


def test_a_dataset_that_raises_reports_the_reason(wired, tmp_path):
    class Exploding:
        def __len__(self):
            return 10

        def __iter__(self):
            yield (torch.zeros(2), torch.zeros(2))
            raise RuntimeError("the disk fell over")

    app = Local(output_dir=str(tmp_path), console=False)
    app.dataset("train", Exploding)
    harness = wired(app)

    data = DatasetWrapper("train", precache=4, prepare=1).bind(harness.remote)
    with pytest.raises(RemoteError, match="the disk fell over"):
        list(data)


def test_a_handler_that_raises_does_not_take_the_link_down(wired, tmp_path):
    app = Local(output_dir=str(tmp_path), console=False)

    @app.on("boom")
    def boom(payload, ctx):
        raise ValueError("handler is broken")

    harness = wired(app)
    logger = LoggerWrapper(harness.remote, str(tmp_path))

    logger.artifact("boom", {"t": torch.ones(2)}, step=1)  # notify: swallowed
    time.sleep(0.5)
    assert harness.remote.connected
    assert harness.remote.ping(timeout=5)

    # And as a call, the caller gets told rather than left waiting.
    with pytest.raises(RemoteError, match="handler is broken"):
        harness.remote.call("boom", {}, timeout=10)
    assert harness.remote.connected


# ======================================================================
#  Control
# ======================================================================
def run_in_background(trainer):
    box = {}

    def run():
        try:
            box["result"] = trainer.train()
        except BaseException as exc:  # noqa: BLE001
            box["error"] = exc

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    return thread, box


def test_stop_ends_the_run_and_says_it_was_stopped(wired, tmp_path):
    app = Local(output_dir=str(tmp_path / "home"), console=False)
    app.dataset("train", lambda: make_loader(samples=64 * 400, batch_size=64))
    harness = wired(app)

    data = DatasetWrapper("train", precache=8, prepare=2).bind(harness.remote)
    trainer = build_trainer(tmp_path / "pod", data, epochs=50)
    trainer.logger = LoggerWrapper(harness.remote, str(tmp_path / "pod"))
    trainer.init()

    thread, box = run_in_background(trainer)
    assert settle(lambda: trainer.global_step > 3)
    app.control("stop")
    thread.join(timeout=30)

    assert not thread.is_alive(), "stop did not end the run"
    assert box["result"]["stopped"] is True
    assert box["result"]["global_step"] < trainer.total_steps


def test_pause_holds_the_run_and_resume_lets_it_go(wired, tmp_path):
    app = Local(output_dir=str(tmp_path / "home"), console=False)
    app.dataset("train", lambda: make_loader(samples=64 * 400, batch_size=64))
    harness = wired(app)

    data = DatasetWrapper("train", precache=8, prepare=2).bind(harness.remote)
    trainer = build_trainer(tmp_path / "pod", data, epochs=50)
    trainer.logger = LoggerWrapper(harness.remote, str(tmp_path / "pod"))
    trainer.init()

    thread, box = run_in_background(trainer)
    assert settle(lambda: trainer.global_step > 2)
    app.control("pause")
    assert settle(lambda: trainer._paused)

    held = trainer.global_step
    time.sleep(1.0)
    assert trainer.global_step == held, "a paused run kept training"

    app.control("resume")
    assert settle(lambda: trainer.global_step > held, timeout=20)

    app.control("stop")
    thread.join(timeout=30)
    assert not thread.is_alive()


def test_an_explicit_save_writes_a_checkpoint_on_the_local_side(wired, tmp_path):
    app = Local(output_dir=str(tmp_path / "home"), console=False)
    app.dataset("train", lambda: make_loader(samples=64 * 400, batch_size=64))
    harness = wired(app)

    data = DatasetWrapper("train", precache=8, prepare=2).bind(harness.remote)
    trainer = build_trainer(tmp_path / "pod", data, epochs=50)
    trainer.logger = LoggerWrapper(harness.remote, str(tmp_path / "pod"))
    trainer.init()

    thread, _box = run_in_background(trainer)
    assert settle(lambda: trainer.global_step > 2)
    app.control("save")
    assert settle(lambda: (tmp_path / "home" / "checkpoints").is_dir())

    app.control("stop")
    thread.join(timeout=30)


# ======================================================================
#  Resume
# ======================================================================
def test_a_run_resumes_where_it_left_off(tmp_path):
    from any_nn_runpod.link import NullLink

    def fresh(epochs):
        data = DatasetWrapper(
            "train", fallback=lambda: make_loader(samples=64 * 8, batch_size=64)
        ).bind(NullLink())
        trainer = build_trainer(tmp_path, data, epochs=epochs)
        trainer.logger = LoggerWrapper(NullLink(), str(tmp_path), console=False)
        return trainer

    first = fresh(epochs=1)
    first.init()
    first.train()
    assert first.global_step == 8

    second = fresh(epochs=2)
    second.init()
    assert second.resume() is True
    assert second.global_step == 8
    assert second.epochs_trained == 1

    second.train()
    assert second.global_step == 16


def test_resume_with_nothing_to_resume_from_says_so(tmp_path):
    from any_nn_runpod.link import NullLink

    data = DatasetWrapper(
        "train", fallback=lambda: make_loader(samples=128, batch_size=64)
    ).bind(NullLink())
    trainer = build_trainer(tmp_path, data)
    trainer.logger = LoggerWrapper(NullLink(), str(tmp_path), console=False)
    trainer.init()
    assert trainer.resume() is False
    assert trainer.global_step == 0


# ======================================================================
#  Endurance
# ======================================================================
def test_many_epochs_leave_no_threads_behind(wired, tmp_path):
    """Each epoch opens a stream; ten epochs must not leave ten producers."""
    app = Local(output_dir=str(tmp_path / "home"), console=False)
    app.dataset("train", lambda: make_loader(samples=64 * 4, batch_size=64))
    harness = wired(app)
    before = {t.name for t in threading.enumerate()}

    data = DatasetWrapper("train", precache=2, prepare=1).bind(harness.remote)
    trainer = build_trainer(tmp_path / "pod", data, epochs=10)
    trainer.logger = LoggerWrapper(harness.remote, str(tmp_path / "pod"))
    trainer.init()
    result = trainer.train()

    assert result["global_step"] == 40  # 4 batches x 10 epochs
    assert settle(
        lambda: not any(
            name.startswith("anr-feed")
            for name in {t.name for t in threading.enumerate()} - before
        )
    ), [t.name for t in threading.enumerate()]


def test_the_window_stays_bounded_across_a_whole_run(wired, tmp_path):
    """precache is a promise about memory, and it has to hold for the run.

    A window that leaked even one slot per epoch would, over a real run, end up
    buffering the dataset into the training machine's RAM.
    """
    app = Local(output_dir=str(tmp_path / "home"), console=False)
    app.dataset("train", lambda: make_loader(samples=64 * 20, batch_size=64))
    harness = wired(app)

    precache = 4
    data = DatasetWrapper("train", precache=precache, prepare=1).bind(harness.remote)
    high_water = 0
    for _epoch in range(3):
        for _batch in data:
            time.sleep(0.01)  # let the producer get as far ahead as allowed
            high_water = max(high_water, data.ready)
    assert high_water <= precache, f"ran {high_water} batches ahead of {precache}"
