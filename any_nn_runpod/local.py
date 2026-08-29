"""
The local side.

``local/local.py`` describes what this machine offers a run: the datasets it
owns and the handlers it answers with.  It never imports anything from
``remote/`` and it is never uploaded anywhere.

    from any_nn_runpod import Local

    app = Local(output_dir="out")

    app.dataset("train", lambda: DataLoader(ds, batch_size=64, num_workers=8))

    @app.on("validation")
    def validation(payload, ctx):
        ctx.log_image("validation", make_grid(payload), ctx.step)

Handlers all look the same -- ``(payload, ctx)`` -- whether the other side sent
them with ``send_artifact`` (fire and forget) or ``link.call`` (waiting for the
return value).  ``ctx`` is a ``Reporter``: TensorBoard and the console, plus the
step the message was sent at.

The launcher (``run.py``) imports this module, finds ``app``, and drives it.
"""

from __future__ import annotations

import itertools
import os
import threading
import traceback

from colorama import Fore, Style

from any_nn_runpod.locks import LockWatcher
from any_nn_runpod.reporting import NetworkMeter, Reporter
from any_nn_runpod.sinks import CheckpointSink, ConsoleSink, TensorBoardSink


class Local:
    """What this machine offers a training run happening somewhere else."""

    def __init__(self, output_dir: str = "./out", console: bool = True):
        # Pinned now, not resolved later: the sinks below keep the path they
        # are built with, and a relative one would quietly follow the process
        # around if anything changed directory afterwards.
        self.output_dir = os.path.abspath(output_dir)
        output_dir = self.output_dir
        self.link = None
        self.console = ConsoleSink(enabled=console)
        self.board = TensorBoardSink(output_dir)
        self.checkpoints = CheckpointSink(output_dir)

        self._datasets: dict = {}
        self._loaders: dict = {}
        self._handlers: dict = {}
        self._producers: dict = {}
        self._watcher = None
        self._meter = None
        self._ids = itertools.count(1)
        self._finished = threading.Event()
        self.result: dict = {}
        self.failure: str | None = None

        os.makedirs(output_dir, exist_ok=True)

    # ================================================================
    #  What you declare
    # ================================================================
    def dataset(self, name: str, factory, pack=None, compress: bool = False):
        """Offer a dataset under ``name``.

        ``factory`` takes no arguments and returns a DataLoader (or any
        iterable with a length).  It is called once, the first time the run
        asks -- so an expensive dataset costs nothing until it is wanted.

        ``pack`` is the counterpart of ``DatasetWrapper``'s ``unpack``: it takes
        the list of batches about to be sent and returns what should actually
        cross the wire.  Use it to drop columns nothing reads, cast to fp16, or
        run an encoder that lives on this machine.
        """
        self._datasets[name] = {"factory": factory, "pack": pack, "compress": compress}
        return self

    def on(self, name: str):
        """Register a handler for ``name``.  Signature ``(payload, ctx)``.

        It answers both ``send_artifact(name, ...)`` and ``link.call(name, ...)``
        from the training side; return a value and a ``call`` gets it back.
        """

        def register(function):
            self._handlers[name] = function
            return function

        return register

    def call(self, name: str, payload=None, timeout=None):
        """Ask the training side for something.  Needs a live link."""
        return self.link.call(name, payload, timeout=timeout)

    def control(self, action: str):
        """pause / resume / save / eval / stop, right now."""
        if self.link is not None and self.link.connected:
            self.link.notify("control", {"action": action})

    # ================================================================
    #  Driven by the launcher
    # ================================================================
    def attach(self, link):
        """Wire this app to a live link and start serving."""
        self.link = link
        self._meter = NetworkMeter(getattr(link.channel, "transport", None))

        link.handle("dataset.info", self._on_dataset_info)
        link.handle("dataset.begin", self._on_dataset_begin)
        link.handle("dataset.cancel", self._on_dataset_cancel)
        link.handle("log", self._on_log)
        link.handle("log_image", self._on_log_image)
        link.handle("print", self._on_print)
        link.handle("progress", self._on_progress)
        link.handle("checkpoint.save", self._on_checkpoint_save)
        link.handle("checkpoint.load", self._on_checkpoint_load)
        link.handle("checkpoint.latest", self._on_checkpoint_latest)
        link.handle("artifact", self._on_artifact)
        link.handle("status", self._on_status)

        # User handlers answer by name, for both notify and call.
        for name in self._handlers:
            link.handle(name, self._make_user_handler(name))

        self._watcher = LockWatcher(self.output_dir, self.control)
        self._watcher.start()

        peer = link.peer
        self.console.text(
            f"{Fore.BLUE}Training on:{Style.RESET_ALL} {peer.get('host', '?')} "
            f"| python {peer.get('python', '?')} | torch {peer.get('torch', '?')} "
            f"| {peer.get('device', '?')}\n"
        )
        return self

    def wait(self, timeout=None) -> dict:
        """Block until the run reports finished or the link dies."""
        while not self._finished.wait(0.5):
            if self.link is None or not self.link.connected:
                if self.failure is None and not self.result:
                    self.failure = (
                        f"the training side went away ({self.link.close_reason})"
                        if self.link is not None
                        else "no link"
                    )
                break
            if timeout is not None:
                timeout -= 0.5
                if timeout <= 0:
                    break
        return self.result

    def shutdown(self):
        for name in list(self._producers):
            self._cancel(name)
        if self._watcher is not None:
            self._watcher.stop()
        self.console.close()
        self.board.close()

    @property
    def log_dir(self) -> str:
        return self.board.log_dir

    # ================================================================
    #  Handlers
    # ================================================================
    def _loader(self, name: str):
        if name not in self._datasets:
            return None
        if name not in self._loaders:
            self._loaders[name] = self._datasets[name]["factory"]()
        return self._loaders[name]

    def _on_dataset_info(self, payload):
        name = payload["name"]
        loader = self._loader(name)
        if loader is None:
            return None
        return {
            "batches": len(loader),
            "batch_size": getattr(loader, "batch_size", None),
        }

    def _on_dataset_begin(self, payload):
        name = payload["name"]
        if self._loader(name) is None:
            raise KeyError(f"this machine has no dataset named {name!r}")

        # A pass that is still running is cancelled first: the training side
        # only starts a new one after abandoning the old, but the old
        # producer may be blocked on a credit window that will never reopen.
        self._cancel(name)

        state = {"cancelled": False, "epoch": payload.get("epoch", 0), "stream": None}
        self._producers[name] = state
        thread = threading.Thread(
            target=self._produce,
            args=(name, payload, state),
            name=f"anr-feed-{name}",
            daemon=True,
        )
        thread.start()
        return {"started": True}

    def _on_dataset_cancel(self, payload):
        self._cancel(payload.get("name"))

    def _cancel(self, name):
        """Stop feeding ``name``, even if the producer is blocked mid-send.

        Setting the flag alone is not enough. A producer waiting for credit is
        inside ``Stream.send``, and the credit it is waiting for can only come
        from a consumer that has already stopped consuming -- so the stream has
        to be abandoned to release it. Without this every early break (a stop
        request, an exception, an epoch cut short) leaks a thread holding a
        dataloader and its worker processes.
        """
        state = self._producers.get(name)
        if state is None:
            return
        state["cancelled"] = True
        stream = state.get("stream")
        if stream is not None:
            stream.abandon()

    def _produce(self, name, request, state):
        """Feed one pass over ``name`` to the training side."""
        spec = self._datasets[name]
        loader = self._loader(name)
        pack = spec["pack"]
        prepare = max(1, int(request.get("prepare", 1)))
        skip = int(request.get("skip", 0))

        stream = self.link.open_stream(
            f"data:{name}",
            depth=max(1, int(request.get("precache", 16))),
            info={"batches": len(loader), "name": name},
            compress=spec["compress"],
        )
        state["stream"] = stream
        error = None
        try:
            source = iter(loader)
            if skip:
                source = itertools.islice(source, skip, None)
            group = []
            for batch in source:
                if state["cancelled"]:
                    break
                group.append(batch)
                if len(group) >= prepare:
                    if not stream.send(_pack(pack, group), units=len(group)):
                        break
                    group = []
            if group and not state["cancelled"]:
                stream.send(_pack(pack, group), units=len(group))
        except Exception as exc:  # noqa: BLE001 -- reported down the stream
            error = f"{type(exc).__name__}: {exc}"
            traceback.print_exc()
        finally:
            stream.end(error=error)

    # -- output ------------------------------------------------------
    def _on_log(self, payload):
        step, values = payload["step"], payload["values"]
        self.board.log(values, step)
        if self._meter is not None:
            extra = self._meter.sample()
            self.board.log(extra, step)
            self.console.set_extra(up=f"{extra['net/up_MBps']:.1f}MB/s")

    def _on_log_image(self, payload):
        self.board.log_image(payload["tag"], payload["image"], payload["step"])

    def _on_print(self, payload):
        self.console.text(payload["text"])

    def _on_progress(self, payload):
        self.console.progress(**payload)

    def _on_checkpoint_save(self, payload):
        # Deliberately silent: the training side prints the path it gets back,
        # and announcing it here as well says everything twice.
        return {"path": self.checkpoints.save(payload["name"], payload["payload"])}

    def _on_checkpoint_load(self, payload):
        return self.checkpoints.load(payload["name"])

    def _on_checkpoint_latest(self, _payload=None):
        return {"name": self.checkpoints.latest()}

    def _on_artifact(self, payload):
        name = payload["name"]
        handler = self._handlers.get(name)
        if handler is None:
            self.console.text(
                f"{Fore.YELLOW}No handler for artifact {name!r}; add "
                f"@app.on({name!r}) to local.py.{Style.RESET_ALL}\n"
            )
            return
        self._invoke(name, handler, payload["payload"], payload.get("step", 0))

    def _on_status(self, payload):
        state = payload.get("state")
        if state == "finished":
            self.result = {
                "global_step": payload.get("step"),
                "stopped": payload.get("stopped", False),
                "log_dir": self.board.log_dir,
            }
            self._finished.set()
        elif state == "failed":
            self.failure = payload.get("traceback") or payload.get("error")
            self.console.text(
                f"\n{Fore.RED}The training side failed:{Style.RESET_ALL}\n"
                f"{self.failure}\n"
            )
            self._finished.set()

    def _make_user_handler(self, name):
        def handler(payload):
            return self._invoke(name, self._handlers[name], payload, 0)

        return handler

    def _invoke(self, name, handler, payload, step):
        try:
            return handler(payload, Reporter(self.board, self.console, step))
        except Exception:
            self.console.text(
                f"{Fore.RED}Handler {name!r} raised:{Style.RESET_ALL}\n"
                f"{traceback.format_exc()}"
            )
            raise


def _pack(pack, batches):
    return pack(batches) if pack is not None else batches
