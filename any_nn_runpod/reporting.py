"""
Everything the training loop emits, and where it goes.

``LoggerWrapper`` is the other half of the pair with ``DatasetWrapper``: one
brings data in, this one sends results out.  The training loop only ever talks
to this object, so the same ``remote/train.py`` writes its TensorBoard to your
machine when there is a link and to its own disk when there is not.

    logger.log({"train/loss": 0.31}, step)
    logger.print("epoch done")
    logger.save_checkpoint("step_100", {"model.safetensors": blob})
    logger.artifact("validation", {"latents": t}, step)   # -> @app.on("validation")

The last one is the interesting one.  ``artifact`` is for work the training
machine should *not* finish: it hands a tensor-bearing payload to a handler in
your ``local.py``, where your dataset, your VAE and your fonts are.
"""

from __future__ import annotations

import collections
import os
import time

CONTROLS = ("pause", "resume", "save", "eval", "stop")


class LoggerWrapper:
    """Routes a run's output to the local side, or to disk when there is none.

    Args:
        link: a ``Link`` or ``NullLink``.
        output_dir: where to write when there is no link.
        console: mirror printing and the progress bar on this machine too.
            Off by default with a link, because the local side already draws
            them and two bars fighting over one terminal helps nobody.
    """

    def __init__(self, link, output_dir: str = "./out", console: bool | None = None):
        self.link = link
        self.output_dir = output_dir
        self.connected = bool(getattr(link, "connected", False))
        self._console_enabled = (not self.connected) if console is None else console
        self._controls: collections.deque = collections.deque()
        self._board = None
        self._console = None
        self._checkpoints = None
        self._watcher = None

        if self.connected:
            link.handle("control", self._on_control)
        else:
            os.makedirs(output_dir, exist_ok=True)

    # -- lazily built local sinks ------------------------------------
    @property
    def board(self):
        if self._board is None:
            from any_nn_runpod.sinks import TensorBoardSink

            self._board = TensorBoardSink(self.output_dir)
        return self._board

    @property
    def console(self):
        if self._console is None:
            from any_nn_runpod.sinks import ConsoleSink

            self._console = ConsoleSink(enabled=self._console_enabled)
        return self._console

    @property
    def checkpoints(self):
        if self._checkpoints is None:
            from any_nn_runpod.sinks import CheckpointSink

            self._checkpoints = CheckpointSink(self.output_dir)
        return self._checkpoints

    # -- what the training loop calls --------------------------------
    def log(self, values: dict, step: int):
        clean = {key: _scalar(value) for key, value in values.items()}
        if self.connected:
            self.link.notify("log", {"values": clean, "step": int(step)})
        else:
            self.board.log(clean, int(step))

    def log_image(self, tag: str, image, step: int):
        if self.connected:
            self.link.notify(
                "log_image", {"tag": tag, "image": _as_tensor(image), "step": int(step)}
            )
        else:
            self.board.log_image(tag, image, int(step))

    def print(self, *args, sep=" ", end="\n"):
        text = sep.join(str(a) for a in args) + end
        if self.connected:
            self.link.notify("print", {"text": text})
        if self._console_enabled or not self.connected:
            self.console.text(text)

    def progress(self, **fields):
        if self.connected:
            self.link.notify("progress", fields)
        if self._console_enabled or not self.connected:
            self.console.progress(**fields)

    def save_checkpoint(self, name: str, payload: dict) -> str:
        """Write a checkpoint wherever checkpoints live.  Returns its path.

        A call rather than a notify: a run that is told a checkpoint was saved
        and then dies should be telling the truth.
        """
        if self.connected:
            answer = self.link.call(
                "checkpoint.save", {"name": name, "payload": payload}, timeout=3600
            )
            return (answer or {}).get("path", name)
        return self.checkpoints.save(name, payload)

    def load_checkpoint(self, name: str) -> dict | None:
        if self.connected:
            return self.link.call("checkpoint.load", {"name": name}, timeout=3600)
        return self.checkpoints.load(name)

    def latest_checkpoint(self) -> str | None:
        if self.connected:
            answer = self.link.call("checkpoint.latest", timeout=120)
            return (answer or {}).get("name")
        return self.checkpoints.latest()

    def artifact(self, name: str, payload, step: int, compress: bool = False):
        """Hand something to a handler in ``local.py``.  Does nothing without one."""
        if self.connected:
            self.link.notify(
                "artifact",
                {"name": name, "payload": payload, "step": int(step)},
                compress=compress,
            )
        else:
            self.print(
                f"[artifact {name!r} at step {step} dropped: no local side to "
                "handle it]"
            )

    def status(self, state: str, **fields):
        """Tell the local side how the run is doing: running/finished/failed."""
        if self.connected:
            self.link.notify("status", {"state": state, **fields})

    # -- control -----------------------------------------------------
    def take_control(self) -> list[str]:
        """Drain pending pause/resume/save/eval/stop requests."""
        if not self.connected:
            self._poll_locks()
        # Popped one at a time rather than list()-then-clear(): the deque is
        # filled from the link's handler thread, and anything appended between
        # those two calls would be silently dropped. Losing a "stop" that way
        # would be a run that ignores you.
        drained = []
        while True:
            try:
                drained.append(self._controls.popleft())
            except IndexError:
                return drained

    def _on_control(self, payload):
        action = (payload or {}).get("action")
        if action in CONTROLS:
            self._controls.append(action)

    def _poll_locks(self):
        """Without a link the lock files are here, so read them here."""
        from any_nn_runpod.locks import poll_locks

        self._controls.extend(poll_locks(self.output_dir, self))

    def close(self):
        if self._board is not None:
            self._board.close()
        if self._console is not None:
            self._console.close()


class Reporter:
    """What an ``@app.on(...)`` artifact handler is given on the local side.

    A narrow view of the local sinks, so a handler can decode an image and put
    it on TensorBoard without knowing anything about how the session is wired.
    """

    def __init__(self, board, console, step: int = 0):
        self._board = board
        self._console = console
        self.step = step

    def log(self, values: dict, step: int | None = None):
        self._board.log(values, self.step if step is None else step)

    def log_image(self, tag: str, image, step: int | None = None):
        self._board.log_image(tag, image, self.step if step is None else step)

    def print(self, text: str):
        self._console.text(text if text.endswith("\n") else text + "\n")


class NetworkMeter:
    """Turns a transport's byte counters into rates.

    Measured over a window, not between samples: batches arrive in bursts, so
    one step in eight carries a whole message and the rest carry nothing.
    Instantaneous rates would read zero almost always and spike absurdly once --
    useless for telling whether the link is the bottleneck.
    """

    def __init__(self, transport, window: float = 20.0):
        self.transport = transport
        self.window = window
        self._history: collections.deque = collections.deque()
        self._record()

    def _read(self, attribute):
        return float(getattr(self.transport, attribute, 0) or 0)

    def _record(self):
        now = time.perf_counter()
        self._history.append(
            (now, self._read("bytes_sent"), self._read("bytes_received"))
        )
        while len(self._history) > 2 and now - self._history[0][0] > self.window:
            self._history.popleft()
        return self._history[-1]

    def sample(self) -> dict:
        now, sent, received = self._record()
        then, was_sent, was_received = self._history[0]
        span = max(now - then, 1e-6)
        return {
            "net/up_MBps": (sent - was_sent) / span / 1e6,
            "net/down_MBps": (received - was_received) / span / 1e6,
            "net/up_total_GB": sent / (1 << 30),
            "net/down_total_GB": received / (1 << 30),
        }


def gpu_stats() -> dict:
    """VRAM figures for the log.  Empty when there is no CUDA device."""
    import torch

    if not torch.cuda.is_available():
        return {}
    free, total = torch.cuda.mem_get_info()
    giga = float(1 << 30)
    return {
        "gpu/allocated_GB": torch.cuda.memory_allocated() / giga,
        "gpu/reserved_GB": torch.cuda.memory_reserved() / giga,
        "gpu/peak_GB": torch.cuda.max_memory_allocated() / giga,
        "gpu/free_GB": free / giga,
        "gpu/total_GB": total / giga,
    }


def _scalar(value):
    import torch

    if torch.is_tensor(value):
        return float(value.detach().float().mean())
    return float(value)


def _as_tensor(image):
    """Images travel as tensors so the codec can send them as raw parts."""
    import numpy as np
    import torch

    if torch.is_tensor(image):
        return image.detach().cpu()
    return torch.from_numpy(np.ascontiguousarray(image))
