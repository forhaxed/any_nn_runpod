"""
The training loop.

This is ``any_nn.AnyTrainer`` with four changes, and nothing else:

1. Everything it emits goes through ``self.logger`` (a ``LoggerWrapper``)
   instead of straight to accelerate's tracker and the local disk.  That is the
   whole reason a run on a rented GPU writes its TensorBoard to your machine.
2. The step count is worked out from ``len(train_dataloader)`` in **batches**,
   not from ``len(dataset) // (batch_size * grad_accum)``.  Same number for a
   plain DataLoader; correct for a streamed one, where samples are somebody
   else's business.
3. Loss reduction is safe for ``batch_size > 1``.  ``any_nn`` and ``easy_nn``
   both do ``gather(value.repeat(self.batch_size))``, which on a non-scalar
   tensor tiles it instead of averaging it -- the reason the Klein trainer had
   to assert ``batch_size == 1``.
4. ``precache_size`` / ``precache_dataset()`` are gone.  ``DatasetWrapper``'s
   ``unpack`` does that job, in the same place, with the GPU already there.

Everything you override is unchanged: ``train_step``, ``eval_step``,
``eval_begin``, ``eval_end``, ``gradient_sync``.  Checkpoints are the one place
the shape differs -- ``save_checkpoint`` returns a dict of files rather than
writing a directory -- because with a link there is no directory on this side
worth writing to.
"""

from __future__ import annotations

import json
import os
import random
import shutil
import tempfile
import time
from datetime import datetime

import torch
from colorama import Fore, Style

PAUSE, RESUME, SAVE, EVAL, STOP = "pause", "resume", "save", "eval", "stop"


def seed_everything(seed: int):
    """Seed torch, numpy and python, right now.

    ``trainer.seed`` is applied by ``init()``, which is late: a model built
    before that -- and most are, since the trainer needs it -- gets its weights
    from an unseeded generator, and two "identical" runs quietly start from
    different places. Call this at the top of ``main()`` and the run is
    reproducible from its first line.
    """
    import random as _random

    _random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    try:
        import numpy

        numpy.random.seed(seed)
    except ImportError:
        pass
    return seed


class RunpodTrainer:
    """Subclass this and override ``train_step``."""

    def __init__(self, output_dir: str = "./out"):
        self.output_dir = output_dir

        # -- the knobs, same names and meanings as AnyTrainer --------
        self.batch_size = None
        self.epochs = None
        self.gradient_accumulation_steps = 1
        self.optimizer = None
        self.scheduler = None
        self.train_dataloader = None
        self.eval_dataloader = None
        self.models = []
        self.non_trainable_models = []
        self.repeats = 1
        self.max_grad_norm = None
        self.mixed_precision = "no"
        self.seed = None
        self.save_checkpoint_every_steps = 0
        self.eval_every_steps = 0
        self.allow_skip_batches_on_resume = True

        self.global_step = 0
        self.epochs_trained = 0
        self.steps_in_epoch = 0

        self.accelerator = None
        self.weight_dtype = torch.float32

        #: Set by the session.  A LoggerWrapper over a Link, or over the local
        #: disk when running standalone.
        self.logger = None
        #: The raw Link, for anything this class does not model.  Ask the local
        #: side to compute something: ``self.link.call("encode", {...})``.
        self.link = None

        self._paused = False
        self._save_requested = False
        self._eval_requested = False
        self._stop_requested = False

    # ================================================================
    #  Setup
    # ================================================================
    @property
    def device(self):
        return self.accelerator.device

    @property
    def steps_per_epoch(self) -> int:
        """Optimizer steps in one pass over the data.

        Counted in batches, because that is what the loop consumes and what a
        streamed dataset can honestly report.
        """
        try:
            batches = len(self.train_dataloader)
        except TypeError as exc:
            raise TypeError(
                f"{type(self.train_dataloader).__name__} has no length, so the "
                "number of steps in a run cannot be worked out.\n"
                "An IterableDataset does this. Either give its DataLoader a "
                "__len__, or wrap it: DatasetWrapper takes its length from the "
                "machine that owns the data, which usually does know."
            ) from exc
        # Multiply before dividing. The loop yields every batch ``repeats``
        # times, so the micro-batches in an epoch are batches * repeats and the
        # optimizer steps are that, floored by the accumulation. Flooring first
        # and multiplying after loses up to repeats-1 steps an epoch -- and on
        # a dataset smaller than one accumulation group it reports zero steps
        # for a pass that does real work.
        return max(1, (batches * self.repeats) // self.gradient_accumulation_steps)

    @property
    def total_steps(self) -> int:
        return self.steps_per_epoch * self.epochs

    def _ensure_logger(self):
        if self.logger is None:
            from any_nn_runpod.link import NullLink
            from any_nn_runpod.reporting import LoggerWrapper

            self.link = self.link or NullLink()
            self.logger = LoggerWrapper(self.link, self.output_dir)
        return self.logger

    def init(self):
        from accelerate import Accelerator

        self._ensure_logger()

        if self.seed is not None:
            from accelerate.utils import set_seed

            set_seed(self.seed)

        # No project_dir and no trackers: TensorBoard is the logger's business,
        # and with a link it is not even on this machine.
        self.accelerator = Accelerator(
            gradient_accumulation_steps=self.gradient_accumulation_steps,
            mixed_precision=self.mixed_precision,
        )
        self.weight_dtype = {
            "fp16": torch.float16,
            "bf16": torch.bfloat16,
        }.get(self.accelerator.mixed_precision, torch.float32)

        self.print(f"{Fore.GREEN}any_nn_runpod training started...{Style.RESET_ALL}\n")
        self.print(f"{Fore.BLUE}Models Summary:{Style.RESET_ALL}")
        for model in self.models:
            total = sum(p.numel() for p in model.parameters())
            trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
            self.print(
                f" {model.__class__.__name__}:\n"
                f"  Total Params: {total}\n"
                f"  Trainable Params: {trainable}"
            )

        for i, model in enumerate(self.models):
            model.to(self.accelerator.device)
            for param in model.parameters():
                param.data = param.to(
                    dtype=torch.float32 if param.requires_grad else self.weight_dtype
                )
            model.train()
            self.models[i] = self.accelerator.prepare(model)

        if self.scheduler is None and self.optimizer is not None:
            self.scheduler = torch.optim.lr_scheduler.ConstantLR(
                optimizer=self.optimizer, factor=1.0
            )
        self.optimizer = self.accelerator.prepare(self.optimizer)
        if self.scheduler is not None:
            self.scheduler = self.accelerator.prepare(self.scheduler)

        # Dataloaders are deliberately not prepared: a DatasetWrapper is not a
        # DataLoader, and batches are moved to the device by train_step, which
        # is where the code that knows their shape lives.

    # ================================================================
    #  Services your code can use
    # ================================================================
    def log(self, values: dict, step: int | None = None):
        self._ensure_logger().log(
            values, self.global_step if step is None else step
        )

    def print(self, *args, **kwargs):
        self._ensure_logger().print(*args, **kwargs)

    def send_artifact(self, name: str, payload, step: int | None = None, compress=False):
        """Hand something to a handler in your ``local.py``."""
        self._ensure_logger().artifact(
            name, payload, self.global_step if step is None else step, compress
        )

    # ================================================================
    #  Hooks to override
    # ================================================================
    def train_step(self, step, batch, device, weight_dtype):
        raise NotImplementedError(f"{type(self).__name__} must define train_step()")

    def eval_begin(self, step):
        pass

    def eval_step(self, step, batch, device, weight_dtype):
        return self.train_step(step, batch, device, weight_dtype)

    def eval_end(self, step):
        pass

    def gradient_sync(self, step):
        pass

    def save_checkpoint(self, step) -> dict:
        """Return ``{filename: bytes | tensor-bearing object}`` to write.

        The default hands back a full accelerate state, which is what you need
        to resume exactly.  Override to send less -- adapter weights only, say --
        and then nothing large crosses the wire on every save.
        """
        return self._full_state_payload()

    def load_checkpoint(self, payload: dict) -> None:
        self._load_full_state(payload)

    # ================================================================
    #  The loop
    # ================================================================
    def train(self):
        missing = [
            name
            for name in ("batch_size", "epochs", "optimizer", "train_dataloader")
            if getattr(self, name) is None
        ]
        if missing:
            raise ValueError(
                f"trainer is missing {', '.join(missing)} -- set them before "
                "calling train()"
            )
        if self.accelerator is None:
            raise RuntimeError("call init() before train()")

        logger = self._ensure_logger()
        effective_batch_size = self.batch_size * self.gradient_accumulation_steps
        total_steps = self.total_steps
        dataset = self.train_dataloader

        self.print(f"\n{Fore.BLUE}Training Configuration:{Style.RESET_ALL}")
        self.print(f" Device: {self.accelerator.device}")
        self.print(f" Mixed Precision: {self.accelerator.mixed_precision}")
        self.print(f" Batches per epoch: {len(dataset)}")
        self.print(f" Total Epochs: {self.epochs}")
        self.print(f" Batch Size: {self.batch_size}")
        self.print(f" Gradient Accumulation Steps: {self.gradient_accumulation_steps}")
        self.print(f" Effective Batch Size: {effective_batch_size}")
        self.print(f" Repeats: {self.repeats}")
        if getattr(dataset, "remote", False):
            self.print(f" Batches kept ready here: {dataset.precache}")
        self.print(f" Total Training Steps: {total_steps}\n")

        for model in self.non_trainable_models:
            model.to(self.accelerator.device, dtype=self.weight_dtype)
            model.eval()

        logger.progress(total=total_steps, step=self.global_step, start=True)
        logger.status("running", total_steps=total_steps, step=self.global_step)

        for _epoch in range(self.epochs_trained, self.epochs):
            over = self._run_epoch(dataset, total_steps)

            # An epoch counts as trained when it has produced its full share of
            # optimizer steps -- not when the loop happens to fall out the
            # bottom. The last epoch of a run leaves through the step budget,
            # and counting only the bottom exit would record it as never having
            # happened; a later resume would then repeat it.
            if self.steps_in_epoch >= self.steps_per_epoch:
                self.epochs_trained += 1
                self.steps_in_epoch = 0
            if over:
                break

        # Reaching total_steps is finishing, not stopping. Only a stop request
        # -- a lock file, a control message -- counts as the latter.
        stopped = self._stop_requested
        self._emit_checkpoint(f"step_{self.global_step}_final")
        self.print(
            f"{Fore.YELLOW}Stopped at step {self.global_step}.{Style.RESET_ALL}\n"
            if stopped
            else f"{Fore.GREEN}Training complete!{Style.RESET_ALL}\n"
        )
        logger.status("finished", step=self.global_step, stopped=stopped)
        return {"global_step": self.global_step, "stopped": stopped}

    def _run_epoch(self, dataset, total_steps) -> bool:
        """One pass over the data.  True if the run is over -- done or stopped."""
        source = dataset
        if self.steps_in_epoch > 0 and self.allow_skip_batches_on_resume:
            to_skip = (
                self.steps_in_epoch * self.gradient_accumulation_steps
            ) // max(1, self.repeats)
            source = dataset.skip(to_skip) if hasattr(dataset, "skip") else source
            self.print(f"Resuming: skipping {to_skip} batches.")

        accumulated = {}
        step_started = time.perf_counter()
        previous_wait = _wait_seconds(dataset)

        for batch in self._iterate(source):
            with self.accelerator.accumulate(self.models):
                loss, loss_dict = self.train_step(
                    self.global_step,
                    batch,
                    device=self.accelerator.device,
                    weight_dtype=self.weight_dtype,
                )
                merged = dict(loss_dict or {})
                merged["loss"] = loss
                for key, value in merged.items():
                    accumulated[key] = (
                        accumulated.get(key, 0.0)
                        + self._reduce(value) / self.gradient_accumulation_steps
                    )

                self.accelerator.backward(loss)

                grad_norm = None
                if self.accelerator.sync_gradients:
                    params = [
                        p
                        for group in self.optimizer.param_groups
                        for p in group["params"]
                        if p.grad is not None
                    ]
                    if params:
                        grad_norm = self.accelerator.clip_grad_norm_(
                            params,
                            float("inf")
                            if self.max_grad_norm is None
                            else self.max_grad_norm,
                        )

                self.optimizer.step()
                if self.accelerator.sync_gradients and self.scheduler is not None:
                    self.scheduler.step()
                self.optimizer.zero_grad()

            if not self.accelerator.sync_gradients:
                continue

            self.gradient_sync(self.global_step)

            now = time.perf_counter()
            wait_now = _wait_seconds(dataset)
            log_dict = {
                "time/step_s": now - step_started,
                "time/data_wait_s": wait_now - previous_wait,
                "queue/ready": float(getattr(dataset, "ready", 0)),
            }
            step_started, previous_wait = now, wait_now

            if self.scheduler is not None:
                log_dict["train/lr"] = self.scheduler.get_last_lr()[0]
            if grad_norm is not None:
                log_dict["train/grad_norm"] = float(grad_norm)

            from any_nn_runpod.reporting import gpu_stats

            log_dict.update(gpu_stats())

            for model in self.models:
                unwrapped = self.accelerator.unwrap_model(model)
                magnitude, count = 0.0, 0
                for param in unwrapped.parameters():
                    if param.requires_grad:
                        magnitude += param.data.abs().sum().item()
                        count += param.numel()
                if count:
                    name = unwrapped.__class__.__name__
                    log_dict[f"train/avg_magnitude/{name}"] = magnitude / count

            for key, value in accumulated.items():
                log_dict[f"train/{key}"] = value

            self.log(log_dict, self.global_step)
            self.logger.progress(
                step=self.global_step + 1,
                total=total_steps,
                postfix={
                    "loss": round(accumulated.get("loss", 0.0), 4),
                    "wait": f"{log_dict['time/data_wait_s']:.2f}s",
                },
            )
            accumulated = {}

            self.global_step += 1
            self.steps_in_epoch += 1

            self._pump_controls()

            if self._save_requested or (
                self.save_checkpoint_every_steps > 0
                and self.global_step % self.save_checkpoint_every_steps == 0
            ):
                self._save_requested = False
                self._emit_checkpoint(f"step_{self.global_step}")

            if self._eval_requested or (
                self.eval_every_steps > 0
                and self.global_step % self.eval_every_steps == 0
            ):
                self._eval_requested = False
                self._run_eval()

            if self._paused:
                self._wait_while_paused()

            if self._stop_requested:
                return True
            if self.global_step >= total_steps:
                return True

            step_started = time.perf_counter()
        return False

    def _iterate(self, source):
        """Batches, with ``repeats`` applied and a hash stamped on each."""
        for batch in source:
            if isinstance(batch, dict):
                batch.setdefault("batch_hash", random.randint(0, 2**63))
            for _ in range(self.repeats):
                yield batch

    def _reduce(self, value) -> float:
        """One number out of whatever ``train_step`` reported.

        ``any_nn`` does ``gather(value.repeat(self.batch_size)).mean()``, which
        assumes the value is a scalar -- ``repeat`` on anything else tiles it,
        and the average comes out of the wrong shape.  Reducing to a scalar
        first makes the batch size irrelevant, which it always should have been.
        """
        if not torch.is_tensor(value):
            return float(value)
        scalar = value.detach().float().mean()
        gathered = self.accelerator.gather(scalar.reshape(1))
        return float(gathered.mean())

    # ================================================================
    #  Control, eval, checkpoints
    # ================================================================
    def _pump_controls(self):
        for command in self._ensure_logger().take_control():
            if command == PAUSE:
                self._paused = True
            elif command == RESUME:
                self._paused = False
            elif command == SAVE:
                self._save_requested = True
            elif command == EVAL:
                self._eval_requested = True
            elif command == STOP:
                self._stop_requested = True

    def _wait_while_paused(self):
        self.print(
            f"{Fore.YELLOW}Pausing at step {self.global_step}.{Style.RESET_ALL}"
        )
        for model in list(self.models) + list(self.non_trainable_models):
            model.to("cpu")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        while self._paused and not self._stop_requested:
            time.sleep(0.25)
            self._pump_controls()

        for model in list(self.models) + list(self.non_trainable_models):
            model.to(self.accelerator.device)
        self.print(
            f"{Fore.GREEN}Resuming at step {self.global_step}.{Style.RESET_ALL}"
        )

    def _run_eval(self):
        self.eval_begin(self.global_step)
        if self.eval_dataloader is None or len(self.eval_dataloader) == 0:
            self.eval_end(self.global_step)
            return

        self.print(f"\nStarting evaluation at step {self.global_step}...")
        totals, count = {}, 0
        wrapped = list(self.models)
        for i, model in enumerate(wrapped):
            self.models[i] = self.accelerator.unwrap_model(model)
            self.models[i].eval()

        try:
            with torch.no_grad():
                for batch in self.eval_dataloader:
                    loss, loss_dict = self.eval_step(
                        self.global_step,
                        batch,
                        device=self.accelerator.device,
                        weight_dtype=self.weight_dtype,
                    )
                    merged = dict(loss_dict or {})
                    if loss is not None:
                        merged["loss"] = loss
                    for key, value in merged.items():
                        totals[key] = totals.get(key, 0.0) + (
                            float(value.detach().float().mean())
                            if torch.is_tensor(value)
                            else float(value)
                        )
                    count += 1
        finally:
            self.models = wrapped
            for model in self.models:
                model.train()

        if count and totals:
            log_dict = {f"eval/{k}": v / count for k, v in totals.items()}
            self.log(log_dict, self.global_step)
            self.print(f"Evaluation at step {self.global_step}: {log_dict}\n")

        self.eval_end(self.global_step)

    def _emit_checkpoint(self, name):
        payload = dict(self.save_checkpoint(self.global_step) or {})
        payload["trainer_metadata.json"] = json.dumps(
            {
                "global_step": self.global_step,
                "epochs_trained": self.epochs_trained,
                "steps_in_epoch": self.steps_in_epoch,
                "saved_at": datetime.now().isoformat(timespec="seconds"),
            },
            indent=4,
        ).encode()
        path = self._ensure_logger().save_checkpoint(name, payload)
        self.print(f"Saved checkpoint to {path}")
        return path

    def resume(self, name: str | None = None) -> bool:
        """Restore a checkpoint by name, or the most recent one.  Call after init()."""
        logger = self._ensure_logger()
        if name is None:
            name = logger.latest_checkpoint()
            if name is None:
                return False
        payload = logger.load_checkpoint(name)
        if not payload:
            return False

        metadata = payload.pop("trainer_metadata.json", None)
        if metadata:
            meta = json.loads(bytes(metadata))
            self.global_step = meta.get("global_step", 0)
            self.epochs_trained = meta.get("epochs_trained", 0)
            self.steps_in_epoch = meta.get("steps_in_epoch", 0)
        self.load_checkpoint(payload)
        self.print(f"Resumed from {name} at step {self.global_step}.")
        return True

    # -- default full-state implementation ---------------------------
    @staticmethod
    def _scratch_dir():
        """Somewhere to let accelerate write.  RAM-backed when we can."""
        if os.path.isdir("/dev/shm"):
            try:
                return tempfile.mkdtemp(prefix="anr_ckpt_", dir="/dev/shm")
            except OSError:
                pass
        return tempfile.mkdtemp(prefix="anr_ckpt_")

    def _full_state_payload(self) -> dict:
        directory = self._scratch_dir()
        try:
            self.accelerator.save_state(directory)
            payload = {}
            for root, _, files in os.walk(directory):
                for filename in files:
                    path = os.path.join(root, filename)
                    relative = os.path.relpath(path, directory).replace(os.sep, "/")
                    with open(path, "rb") as handle:
                        payload[relative] = handle.read()
            return payload
        finally:
            shutil.rmtree(directory, ignore_errors=True)

    def _load_full_state(self, payload: dict) -> None:
        directory = self._scratch_dir()
        try:
            for relative, data in payload.items():
                path = os.path.join(directory, *str(relative).split("/"))
                os.makedirs(os.path.dirname(path), exist_ok=True)
                with open(path, "wb") as handle:
                    handle.write(bytes(data))
            self.accelerator.load_state(directory)
        finally:
            shutil.rmtree(directory, ignore_errors=True)


def _wait_seconds(dataset) -> float:
    return float(getattr(dataset, "wait_seconds", 0.0) or 0.0)
