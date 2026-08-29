"""
The two wrappers and the trainer, over a real link.

These are the tests that would have caught the reasons the old design pinned
``batch_size`` to 1: what a "unit" of queued work is, and what happens to a
loss that is not a scalar.
"""

from __future__ import annotations

import threading

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from any_nn_runpod.dataset import DatasetWrapper
from any_nn_runpod.link import Link, NotConnected, NullLink
from any_nn_runpod.local import Local
from any_nn_runpod.reporting import LoggerWrapper
from any_nn_runpod.trainer import RunpodTrainer
from any_nn_runpod.wire.protocol import Channel
from any_nn_runpod.wire.transport import TcpListener, TcpTransport


def make_loader(samples=256, batch_size=64, features=4):
    x = torch.arange(samples * features, dtype=torch.float32).reshape(samples, features)
    y = (x.sum(1) > x.sum(1).median()).long()
    return DataLoader(TensorDataset(x, y), batch_size=batch_size, drop_last=True)


class Wired:
    """A ``Local`` app on one side, a link on the other."""

    def __init__(self, app: Local):
        self.app = app
        self.listener = TcpListener("127.0.0.1", 0)
        self.remote = None
        ready = threading.Event()

        def serve():
            channel = self.listener.accept(timeout=10)
            self.remote = Link(channel, role="remote").start(initiate=False)
            ready.set()

        threading.Thread(target=serve, daemon=True).start()
        transport = TcpTransport.connect("127.0.0.1", self.listener.port)
        self.local = Link(Channel(transport), role="local").start(initiate=True)
        assert ready.wait(10)
        app.attach(self.local)

    def close(self):
        self.app.shutdown()
        for link in (self.remote, self.local):
            if link is not None:
                link.close("test over")
        self.listener.close()


@pytest.fixture
def wired(tmp_path):
    def build(app=None):
        build.harness = Wired(app or Local(output_dir=str(tmp_path), console=False))
        return build.harness

    build.harness = None
    yield build
    if build.harness is not None:
        build.harness.close()


# ======================================================================
#  DatasetWrapper
# ======================================================================
def test_length_is_counted_in_batches_not_samples(wired, tmp_path):
    """256 samples at batch_size 64 is 4 batches, and 4 is what len() says.

    ``any_nn`` derived the step count from ``len(dataset) // (batch_size *
    grad_accum)``, which a streamed dataset cannot answer honestly -- it does
    not have a dataset, it has a flow of batches.  Counting batches is what
    makes any batch size ordinary.
    """
    app = Local(output_dir=str(tmp_path), console=False)
    app.dataset("train", lambda: make_loader(samples=256, batch_size=64))
    harness = wired(app)

    data = DatasetWrapper("train").bind(harness.remote)
    assert len(data) == 4
    assert data.batch_size == 64


def test_batches_arrive_intact_and_in_order(wired, tmp_path):
    app = Local(output_dir=str(tmp_path), console=False)
    app.dataset("train", lambda: make_loader(samples=256, batch_size=64))
    harness = wired(app)

    data = DatasetWrapper("train", precache=4, prepare=2).bind(harness.remote)
    received = list(data)
    assert len(received) == 4

    expected = list(make_loader(samples=256, batch_size=64))
    for (got_x, got_y), (want_x, want_y) in zip(received, expected):
        assert torch.equal(got_x, want_x)
        assert torch.equal(got_y, want_y)


def test_pack_and_unpack_are_a_pair(wired, tmp_path):
    """What the local side packs is exactly what the training side unpacks."""
    app = Local(output_dir=str(tmp_path), console=False)
    app.dataset(
        "train",
        lambda: make_loader(samples=128, batch_size=32),
        pack=lambda batches: [(x.half(), y) for x, y in batches],
    )
    harness = wired(app)

    data = DatasetWrapper(
        "train",
        precache=4,
        prepare=2,
        unpack=lambda batches: [(x.float(), y) for x, y in batches],
    ).bind(harness.remote)

    for x, _y in data:
        assert x.dtype == torch.float32  # unpacked back on this side


def test_iterating_twice_gives_two_full_epochs(wired, tmp_path):
    app = Local(output_dir=str(tmp_path), console=False)
    app.dataset("train", lambda: make_loader(samples=128, batch_size=32))
    harness = wired(app)

    data = DatasetWrapper("train", precache=4).bind(harness.remote)
    assert len(list(data)) == 4
    assert len(list(data)) == 4


def test_skip_drops_the_first_batches(wired, tmp_path):
    app = Local(output_dir=str(tmp_path), console=False)
    app.dataset("train", lambda: make_loader(samples=256, batch_size=64))
    harness = wired(app)

    data = DatasetWrapper("train", precache=4).bind(harness.remote)
    assert len(list(data.skip(2))) == 2
    assert len(data.skip(2)) == 2


def test_breaking_out_early_does_not_wedge_the_producer(wired, tmp_path):
    """Stopping mid-epoch must release the local side, not leave it blocked.

    Without the cancel, the producer sits waiting for credit that will never
    come and the next epoch deadlocks behind it.
    """
    app = Local(output_dir=str(tmp_path), console=False)
    app.dataset("train", lambda: make_loader(samples=1024, batch_size=32))
    harness = wired(app)

    # precache=1 and a break on the very first batch is the deterministic
    # shape: credit is returned when the *next* batch is asked for, so leaving
    # after the first returns none at all, and the producer is provably parked
    # inside send() waiting for a window that can never reopen. With a larger
    # window the producer may happen to be at the top of its loop instead,
    # notice the cancel flag, and exit on its own -- which is luck, not design.
    data = DatasetWrapper("train", precache=1, prepare=1).bind(harness.remote)
    before = {t.name for t in threading.enumerate()}

    for _batch in data:
        break
    threading.Event().wait(0.3)  # let the producer reach its blocking send

    # The next pass has to work, which it only can if the first was cleaned up.
    assert len(list(data)) == 32

    # And the abandoned producer has to be gone, not merely ignored: it holds a
    # dataloader and its worker processes for as long as it is parked on a
    # credit window nobody will ever open again.
    for _ in range(100):
        leaked = {t.name for t in threading.enumerate()} - before
        if not any(name.startswith("anr-feed") for name in leaked):
            break
        threading.Event().wait(0.05)
    assert not any(name.startswith("anr-feed") for name in leaked), leaked


def test_missing_dataset_says_which_one(wired, tmp_path):
    harness = wired()
    data = DatasetWrapper("nope").bind(harness.remote)
    with pytest.raises(NotConnected, match="nope"):
        len(data)


def test_optional_dataset_is_simply_empty(wired, tmp_path):
    harness = wired()
    data = DatasetWrapper("nope", optional=True).bind(harness.remote)
    assert len(data) == 0


def test_fallback_runs_without_a_link():
    data = DatasetWrapper(
        "train", fallback=lambda: make_loader(samples=128, batch_size=32)
    ).bind(NullLink())
    assert not data.remote
    assert len(data) == 4
    assert len(list(data)) == 4


def test_no_link_and_no_fallback_explains_the_two_ways_out():
    data = DatasetWrapper("train").bind(NullLink())
    with pytest.raises(NotConnected, match="fallback"):
        len(data)


def test_prepare_larger_than_precache_is_rejected_at_construction():
    with pytest.raises(ValueError, match="could never be sent"):
        DatasetWrapper("train", precache=4, prepare=8)


# ======================================================================
#  The trainer
# ======================================================================
class TinyTrainer(RunpodTrainer):
    def train_step(self, step, batch, device, weight_dtype):
        x, y = batch
        logits = self.models[0](x.to(device, torch.float32))
        loss = torch.nn.functional.cross_entropy(logits, y.to(device))
        # Deliberately NOT a scalar: this is the shape that made the old
        # gather(value.repeat(batch_size)) produce nonsense.
        per_sample = torch.full((x.shape[0],), 0.5, device=device)
        return loss, {"per_sample": per_sample}


def build_trainer(output_dir, dataloader, **overrides):
    torch.manual_seed(0)
    model = torch.nn.Linear(4, 2)
    trainer = TinyTrainer(output_dir=str(output_dir))
    trainer.models = [model]
    trainer.optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    trainer.train_dataloader = dataloader
    trainer.batch_size = 64
    trainer.epochs = 1
    trainer.gradient_accumulation_steps = 1
    for key, value in overrides.items():
        setattr(trainer, key, value)
    return trainer


def test_step_count_follows_batches_and_accumulation(tmp_path):
    data = DatasetWrapper(
        "train", fallback=lambda: make_loader(samples=640, batch_size=64)
    ).bind(NullLink())
    trainer = build_trainer(tmp_path, data, gradient_accumulation_steps=2, epochs=3)
    assert len(data) == 10
    assert trainer.steps_per_epoch == 5
    assert trainer.total_steps == 15


def test_a_non_scalar_loss_value_reduces_to_its_mean(tmp_path):
    """``repeat(batch_size)`` on a per-sample tensor tiles it; mean() does not.

    The old code path would report 0.5 tiled 64 times and average *that* -- the
    right answer here by luck, and the wrong one the moment the values differ.
    """
    data = DatasetWrapper(
        "train", fallback=lambda: make_loader(samples=256, batch_size=64)
    ).bind(NullLink())
    trainer = build_trainer(tmp_path, data)
    trainer.logger = LoggerWrapper(NullLink(), str(tmp_path), console=False)
    trainer.init()

    values = torch.tensor([0.0, 1.0, 2.0, 5.0])
    assert trainer._reduce(values) == pytest.approx(2.0)
    assert trainer._reduce(torch.tensor(3.0)) == pytest.approx(3.0)
    assert trainer._reduce(7) == pytest.approx(7.0)


def test_a_whole_run_standalone(tmp_path):
    data = DatasetWrapper(
        "train", fallback=lambda: make_loader(samples=256, batch_size=64)
    ).bind(NullLink())
    trainer = build_trainer(tmp_path, data, epochs=2, save_checkpoint_every_steps=4)
    trainer.logger = LoggerWrapper(NullLink(), str(tmp_path), console=False)
    trainer.init()
    result = trainer.train()

    assert result["global_step"] == 8  # 4 batches x 2 epochs
    assert not result["stopped"]
    assert (tmp_path / "checkpoints").is_dir()
    assert (tmp_path / "logs").is_dir()


def test_a_whole_run_over_the_link(wired, tmp_path):
    """The real thing: batches from the local side, output back to it."""
    app = Local(output_dir=str(tmp_path / "home"), console=False)
    app.dataset("train", lambda: make_loader(samples=512, batch_size=64))
    app.dataset("eval", lambda: make_loader(samples=128, batch_size=64))
    harness = wired(app)

    data = DatasetWrapper("train", precache=4, prepare=2).bind(harness.remote)
    evaluation = DatasetWrapper("eval", precache=2).bind(harness.remote)
    trainer = build_trainer(
        tmp_path / "pod",
        data,
        eval_dataloader=evaluation,
        eval_every_steps=4,
        save_checkpoint_every_steps=4,
    )
    trainer.logger = LoggerWrapper(harness.remote, str(tmp_path / "pod"))
    trainer.link = harness.remote
    trainer.init()
    result = trainer.train()

    assert result["global_step"] == 8
    # Everything landed on the local side, and nothing on the training side.
    assert (tmp_path / "home" / "checkpoints").is_dir()
    assert (tmp_path / "home" / "logs").is_dir()
    assert not (tmp_path / "pod" / "checkpoints").exists()
    assert app.wait(timeout=10)["global_step"] == 8


def test_artifacts_reach_a_local_handler(wired, tmp_path):
    seen = {}
    app = Local(output_dir=str(tmp_path), console=False)
    app.dataset("train", lambda: make_loader(samples=128, batch_size=64))

    @app.on("picture")
    def picture(payload, ctx):
        seen["step"] = ctx.step
        seen["tensor"] = payload["t"]

    harness = wired(app)
    logger = LoggerWrapper(harness.remote, str(tmp_path))
    logger.artifact("picture", {"t": torch.ones(2, 2)}, step=7)

    for _ in range(100):
        if seen:
            break
        threading.Event().wait(0.05)
    assert seen["step"] == 7
    assert torch.equal(seen["tensor"], torch.ones(2, 2))


def test_checkpoints_round_trip_through_the_local_side(wired, tmp_path):
    app = Local(output_dir=str(tmp_path / "home"), console=False)
    harness = wired(app)
    logger = LoggerWrapper(harness.remote, str(tmp_path / "pod"))

    logger.save_checkpoint("step_1", {"weights.pt": torch.ones(3), "note.txt": b"hi"})
    assert logger.latest_checkpoint().endswith("step_1")

    payload = logger.load_checkpoint("step_1")
    assert bytes(payload["note.txt"]) == b"hi"


def test_control_commands_reach_the_training_side(wired, tmp_path):
    app = Local(output_dir=str(tmp_path), console=False)
    harness = wired(app)
    logger = LoggerWrapper(harness.remote, str(tmp_path))

    app.control("pause")
    app.control("stop")

    # take_control drains, so polling has to accumulate -- the two commands may
    # well arrive on separate polls, and a poll that returns one of them has
    # already consumed it.
    actions = []
    for _ in range(200):
        actions.extend(logger.take_control())
        if len(actions) == 2:
            break
        threading.Event().wait(0.05)
    assert actions == ["pause", "stop"]  # order preserved


def test_lock_files_work_when_there_is_no_link(tmp_path):
    logger = LoggerWrapper(NullLink(), str(tmp_path), console=False)
    assert logger.take_control() == []

    (tmp_path / "pause.lock").write_text("")
    assert logger.take_control() == ["pause"]
    assert logger.take_control() == []  # edge-triggered, not level

    (tmp_path / "pause.lock").unlink()
    (tmp_path / "do_eval.lock").write_text("")
    assert sorted(logger.take_control()) == ["eval", "resume"]
    assert not (tmp_path / "do_eval.lock").exists()  # one shot
