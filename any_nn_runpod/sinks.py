"""
Where everything a run produces actually lands.

These are the four things a training loop emits -- scalars, images, text and
files -- and they are written by whoever is holding the sinks.  With a link that
is the local machine; without one it is the machine doing the training.  Nothing
above this module knows which.

Each sink serializes its own writes.  On the local side they are driven from the
link's handler pool, which runs several handlers at once, and none of what they
wrap is thread-safe: two ``SummaryWriter`` calls can interleave into a corrupt
event file, and a progress bar redrawing while ``print`` writes turns the
terminal into confetti.  Locking here rather than around the callers means a
slow artifact handler does not block logging while it computes -- only while it
actually writes.
"""

from __future__ import annotations

import os
import sys
import threading
from datetime import datetime

import torch
from tqdm.auto import tqdm


class ConsoleSink:
    """Terminal output and the progress bar."""

    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self.bar = None
        self.extra: dict = {}
        self._lock = threading.RLock()

    def text(self, text: str):
        if not self.enabled:
            return
        with self._lock:
            if self.bar is not None:
                # Goes above the bar instead of through it.
                self.bar.write(text.rstrip("\n"))
            else:
                sys.stdout.write(text)
                sys.stdout.flush()

    def progress(self, step=None, total=None, postfix=None, start=False, desc="Steps"):
        if not self.enabled:
            return
        with self._lock:
            if start or self.bar is None:
                if self.bar is not None:
                    self.bar.close()
                self.bar = tqdm(total=total, initial=step or 0, desc=desc)
                return
            if step is not None:
                self.bar.update(max(0, step - self.bar.n))
            if postfix:
                merged = dict(self.extra)
                merged.update(postfix)
                self.bar.set_postfix(**merged)

    def set_extra(self, **fields):
        """Fields the holder of the sinks owns -- bandwidth, queue depth."""
        with self._lock:
            self.extra.update(fields)

    def close(self):
        with self._lock:
            if self.bar is not None:
                self.bar.close()
                self.bar = None


class TensorBoardSink:
    """The only TensorBoard writer in the system."""

    def __init__(self, output_dir: str, run_name: str | None = None):
        stamp = run_name or datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.log_dir = os.path.join(output_dir, "logs", stamp)
        self._writer = None
        self._lock = threading.RLock()

    @property
    def writer(self):
        with self._lock:
            if self._writer is None:
                from torch.utils.tensorboard import SummaryWriter

                os.makedirs(self.log_dir, exist_ok=True)
                self._writer = SummaryWriter(self.log_dir)
            return self._writer

    def log(self, values: dict, step: int):
        with self._lock:
            writer = self.writer
            for key, value in values.items():
                writer.add_scalar(key, value, step)
            writer.flush()

    def log_image(self, tag: str, image, step: int):
        """``image`` is HWC uint8, or CHW -- whatever TensorBoard accepts."""
        import numpy as np

        array = np.asarray(image)
        fmt = "HWC" if array.ndim == 3 and array.shape[-1] in (1, 3, 4) else "CHW"
        with self._lock:
            self.writer.add_image(tag, array, step, dataformats=fmt)
            self.writer.flush()

    def close(self):
        with self._lock:
            if self._writer is not None:
                self._writer.close()
                self._writer = None


class CheckpointSink:
    """Writes a ``{relative path: bytes | tensor-bearing object}`` payload."""

    def __init__(self, output_dir: str):
        self.root = os.path.join(output_dir, "checkpoints")
        self._lock = threading.RLock()

    def path_for(self, name: str) -> str:
        return os.path.join(self.root, name)

    def save(self, name: str, payload: dict) -> str:
        directory = self.path_for(name)
        with self._lock:
            os.makedirs(directory, exist_ok=True)
            for relative, value in payload.items():
                path = os.path.join(directory, *str(relative).split("/"))
                os.makedirs(os.path.dirname(path), exist_ok=True)
                if isinstance(value, (bytes, bytearray, memoryview)):
                    with open(path, "wb") as handle:
                        handle.write(value)
                else:
                    torch.save(value, path)
        return directory

    def load(self, name: str) -> dict | None:
        directory = name if os.path.isabs(name) else self.path_for(name)
        if not os.path.isdir(directory):
            return None
        payload = {}
        for root, _, files in os.walk(directory):
            for filename in files:
                path = os.path.join(root, filename)
                relative = os.path.relpath(path, directory).replace(os.sep, "/")
                with open(path, "rb") as handle:
                    payload[relative] = handle.read()
        return payload

    def latest(self) -> str | None:
        """The most recently written checkpoint directory, if any."""
        if not os.path.isdir(self.root):
            return None
        entries = [
            os.path.join(self.root, name)
            for name in os.listdir(self.root)
            if os.path.isdir(os.path.join(self.root, name))
        ]
        if not entries:
            return None
        return max(entries, key=os.path.getmtime)
