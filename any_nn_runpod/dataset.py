"""
One dataset, two sides.

``DatasetWrapper`` is what ``remote/train.py`` puts where a ``DataLoader`` would
go.  It looks like one -- it iterates and it has a length -- but the samples are
built on whichever machine owns the data:

    # remote/train.py
    trainer.train_dataloader = DatasetWrapper("train", precache=32)

    # local/local.py
    app.dataset("train", lambda: DataLoader(ds, batch_size=64, num_workers=8))

The unit of exchange is **one collated batch**, whatever its batch size.  That
is the whole reason ``batch_size > 1`` is unremarkable here: the queue is
counted in the same thing the training loop consumes, so nothing has to divide
sample counts by anything to work out how many steps a run takes.

``precache=N`` is the promise: N batches are kept sitting on the training
machine at all times.  Below that the local side tops it up.  It is a credit
window, so it is a real bound in both directions -- the GPU never waits for the
network, and a fast local machine cannot buffer an entire epoch into the pod's
RAM.

With no link and a ``fallback``, the very same object builds the data locally
and the training loop cannot tell the difference.  That is what lets ``local/``
be empty.
"""

from __future__ import annotations

import itertools

from any_nn_runpod.link import NotConnected


class DatasetWrapper:
    """A dataloader whose batches may be coming from the other machine.

    Args:
        name: what the local side registered this dataset as.
        precache: batches kept ready on the training machine.  Bigger hides
            more network jitter and costs more RAM.
        prepare: batches packed into one message.  Amortizes per-message
            overhead; must not exceed ``precache``.
        fallback: called with no arguments to build a real dataloader when
            there is no link.  Without it, a linkless run has no data.
        optional: an absent dataset is empty rather than an error.  Use it for
            ``eval_dataloader``.
        unpack: runs on the training machine with the GPU available.  Takes the
            list of batches the local side packed, returns a list of batches.
            This is where you finish work you deliberately did not send over
            the wire.
        compress: zstd the payloads.  Worth it for masks and labels, not for
            image tensors.
    """

    def __init__(
        self,
        name: str,
        precache: int = 16,
        prepare: int = 1,
        fallback=None,
        optional: bool = False,
        unpack=None,
        compress: bool = False,
    ):
        if precache < 1:
            raise ValueError("precache must be at least 1 batch")
        if prepare < 1:
            raise ValueError("prepare must be at least 1 batch")
        if prepare > precache:
            raise ValueError(
                f"prepare={prepare} cannot exceed precache={precache}: a message "
                "bigger than the window could never be sent"
            )
        self.name = name
        self.precache = precache
        self.prepare = prepare
        self.optional = optional
        self.compress = compress
        self._fallback_factory = fallback
        self._unpack = unpack
        self._link = None
        #: Whether a live link was ever handed to this wrapper.  The difference
        #: between "there is no local side" and "there was one and it went
        #: away" is the difference between two completely unrelated fixes.
        self._had_link = False
        self._fallback = None
        self._length = None
        self._batch_size = None
        self._epoch = 0
        #: Seconds the training loop spent blocked waiting for a batch, and how
        #: many batches were sitting ready when it last looked.  Logged every
        #: step, because together they are the answer to "is the link the
        #: bottleneck".
        self.wait_seconds = 0.0
        self.ready = 0

    # -- wiring ------------------------------------------------------
    def bind(self, link):
        """Called by the session once the link (or NullLink) exists."""
        self._link = link
        self._had_link = self._had_link or bool(link is not None and link.connected)
        return self

    @property
    def remote(self) -> bool:
        """True when the batches are coming from the other machine."""
        return self._link is not None and self._link.connected

    # -- the DataLoader surface --------------------------------------
    def __len__(self) -> int:
        if self._length is None:
            self._length = self._resolve_length()
        return self._length

    @property
    def batch_size(self):
        len(self)  # resolving the length also learns the batch size
        return self._batch_size

    def __iter__(self):
        yield from self._pass(skip=0)

    def _pass(self, skip: int):
        """One pass over the data, optionally dropping the first ``skip``."""
        if self.remote:
            yield from self._iter_remote(skip)
            return

        loader = self._local_loader()
        if loader is None:
            return
        yield from (itertools.islice(iter(loader), skip, None) if skip else loader)

    # -- local fallback ----------------------------------------------
    def _local_loader(self):
        if self._fallback is None and self._fallback_factory is not None:
            self._fallback = self._fallback_factory()
        if self._fallback is None and not self.optional:
            if self._had_link:
                # Telling someone to add a fallback here would send them off to
                # fix the wrong thing entirely: they had a local side, and it
                # was serving this dataset until the connection died.
                reason = getattr(self._link, "close_reason", None) or "no reason given"
                raise NotConnected(
                    f"the link to the local side dropped before {self.name!r} "
                    f"could be read ({reason}).\n"
                    "This is not a missing fallback -- the local side was there. "
                    "Look at what happened to the connection."
                )
            raise NotConnected(
                f"dataset {self.name!r} has no link and no fallback. Either run "
                f"with a local/ script that registers {self.name!r}, or give the "
                "DatasetWrapper a fallback= that builds the data here."
            )
        return self._fallback

    def _resolve_length(self) -> int:
        if self.remote:
            info = self._link.call("dataset.info", {"name": self.name}, timeout=600)
            if info is None:
                if self.optional:
                    return 0
                raise NotConnected(
                    f"the local side has no dataset named {self.name!r}. "
                    f"Register it with app.dataset({self.name!r}, ...)."
                )
            self._batch_size = info.get("batch_size")
            return int(info["batches"])

        loader = self._local_loader()
        if loader is None:
            return 0
        self._batch_size = getattr(loader, "batch_size", None)
        return len(loader)

    # -- streamed from the other side --------------------------------
    def _iter_remote(self, skip: int = 0):
        self._epoch += 1
        self._link.call(
            "dataset.begin",
            {
                "name": self.name,
                "precache": self.precache,
                "prepare": self.prepare,
                "epoch": self._epoch,
                "skip": skip,
            },
            timeout=600,
        )
        reader = self._link.accept_stream(f"data:{self.name}", timeout=600)
        finished = False
        try:
            for payload, _units in reader:
                self.wait_seconds = reader.wait_seconds
                self.ready = reader.depth
                batches = self._unpack(payload) if self._unpack else payload
                yield from batches
            finished = True
        finally:
            reader.close()
            if not finished:
                # The loop stopped early -- a break, an exception, a stop
                # request. Tell the producer, or it sits blocked on a window
                # that will never open again.
                self._link.notify(
                    "dataset.cancel", {"name": self.name, "epoch": self._epoch}
                )

    def skip(self, batches: int):
        """Resume support: drop the first ``batches`` of the next pass."""
        if batches <= 0:
            return self
        return _SkippingView(self, batches)


class _SkippingView:
    """``DatasetWrapper`` with the first N batches of one pass dropped.

    Resuming mid-epoch needs this, and ``accelerate.skip_first_batches`` cannot
    provide it -- it wants a real ``DataLoader``.  When the batches come from
    the other machine the skip is done there, so the skipped ones are never
    built, let alone sent.
    """

    def __init__(self, source: DatasetWrapper, batches: int):
        self._source = source
        self._skip = batches

    def __len__(self):
        return max(0, len(self._source) - self._skip)

    @property
    def batch_size(self):
        return self._source.batch_size

    def __iter__(self):
        yield from self._source._pass(skip=self._skip)
