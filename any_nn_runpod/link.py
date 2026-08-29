"""
The link between local and remote.

One symmetric object, the same on both sides, with three ways to move
something across:

    link.notify("log", {...})            fire and forget
    link.call("encode", {...})           request, block, get an answer
    link.open_stream("data", depth=32)   a flow of payloads with back-pressure

Everything else in this library is built on it -- the dataset stream, the log
sink, checkpoints, artifacts, control commands.  Adding a new kind of traffic
between your ``local.py`` and your ``remote.py`` is a new name, not a new
protocol version:

    # remote.py
    embeds = link.call("encode_prompts", {"prompts": [...]})

    # local.py
    @app.on("encode_prompts")
    def encode_prompts(payload):
        return {"embeds": text_encoder(payload["prompts"])}

Only plain data crosses: dict, list, tuple, str, bytes, numbers, None and
tensors.  See ``wire/codec.py`` for why.

Threading.  One reader thread owns ``channel.recv()`` and does as little as
possible -- results resolve futures, stream payloads go on a queue *undecoded*,
so a multi-gigabyte tensor is rebuilt by whoever consumes it rather than
stalling the socket.  Handlers run on a small pool, because a handler that
blocks (a dataloader waking up its workers, say) must not stop the link.
"""

from __future__ import annotations

import itertools
import platform
import queue
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor

from any_nn_runpod.wire import codec, protocol


class LinkError(RuntimeError):
    """Something went wrong on the link itself."""


class NotConnected(LinkError):
    """Asked for something that needs the other side, and there isn't one."""


class RemoteError(RuntimeError):
    """The other side raised.  The traceback below happened over there."""


class StreamClosed(LinkError):
    pass


# ======================================================================
#  Streams
# ======================================================================
class Stream:
    """Producer end.  ``send`` blocks once ``depth`` units are outstanding.

    That block is the whole point: it is what keeps exactly ``depth`` units
    sitting on the consumer -- enough that it never waits for the network, few
    enough that memory stays bounded.  Credit comes back per unit *consumed*,
    not per message received, so the producer's picture of the far side is
    honest no matter how units were grouped for transport.
    """

    def __init__(self, link, sid: str, depth: int, compress: bool = False):
        self.sid = sid
        self.depth = depth
        self._link = link
        self._compress = compress
        self._room = threading.Semaphore(depth)
        #: "open" -> "ended" (we said so) or "abandoned" (the consumer stopped
        #: caring).  The two are not the same: ending is an event the consumer
        #: is told about, abandoning is one it caused.
        self._state = "open"
        self.units_sent = 0

    @property
    def _closed(self) -> bool:
        return self._state != "open"

    def send(self, payload, units: int = 1) -> bool:
        """Send one message worth ``units`` of work.

        False means stop feeding: the consumer went away, or the link did.
        """
        if self._state == "abandoned":
            return False
        if self._state == "ended":
            raise StreamClosed(f"stream {self.sid} has already ended")
        if units > self.depth:
            # Otherwise send() would wait for credit that cannot arrive until
            # this very message is delivered.
            raise ValueError(
                f"a message of {units} units cannot fit a window of "
                f"{self.depth}: raise precache, or send fewer units at once"
            )
        for _ in range(units):
            while not self._room.acquire(timeout=0.25):
                if self._closed or self._link.closed:
                    return False
        if self._link.closed:
            return False
        self._link._send(
            protocol.STREAM_DATA,
            {"sid": self.sid, "units": units},
            codec.encode(payload, compress=self._compress),
        )
        self.units_sent += units
        return True

    def end(self, error: str | None = None):
        """Tell the consumer the flow is over -- cleanly, or because of ``error``."""
        if self._closed:
            return
        self._state = "ended"
        self._link._forget_stream(self.sid)
        if not self._link.closed:
            self._link._send(protocol.STREAM_END, {"sid": self.sid, "error": error})

    def _credit(self, n: int):
        for _ in range(n):
            self._room.release()

    def abandon(self):
        """The consumer stopped reading.  Release a blocked ``send`` and go quiet.

        No STREAM_END: the consumer already knows, and it may have forgotten
        this stream entirely. Releasing the window is the point -- a producer
        parked on credit that can no longer arrive would wait forever.
        """
        if self._state == "abandoned":
            return
        self._state = "abandoned"
        self._link._forget_stream(self.sid)
        self._room.release(self.depth)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.end(error=None if exc is None else f"{exc_type.__name__}: {exc}")
        return False


class StreamReader:
    """Consumer end.  Iterates ``(payload, units)`` and returns credit.

    Credit for a message is returned when the *next* one is asked for, so it
    reflects work actually finished rather than work merely delivered.
    """

    def __init__(self, link, sid: str, info: dict):
        self.sid = sid
        self.info = info or {}
        self._link = link
        self._queue: queue.Queue = queue.Queue()
        self._done = False
        self._error = None
        self._pending_credit = 0
        #: Seconds the consumer spent blocked waiting for a payload.  The one
        #: number that says whether the link is starving the GPU.
        self.wait_seconds = 0.0
        self.units_received = 0
        self.units_consumed = 0

    # -- filled by the reader thread ---------------------------------
    def _put(self, message):
        self._queue.put(("data", message))

    def _finish(self, error=None):
        self._queue.put(("end", error))

    def _fail(self, error):
        self._queue.put(("end", error))

    # -- read by whoever is training ---------------------------------
    def __iter__(self):
        while True:
            self._flush_credit()
            started = time.perf_counter()
            kind, item = self._queue.get()
            self.wait_seconds += time.perf_counter() - started

            if kind == "end":
                self._done = True
                if item:
                    raise RemoteError(f"stream {self.sid} failed: {item}")
                return

            units = int(item.meta.get("units", 1))
            self.units_received += units
            # Decoded here, on the consumer's thread -- never on the reader's.
            yield item.body, units
            self._pending_credit += units
            self.units_consumed += units

    def _flush_credit(self):
        if self._pending_credit and not self._link.closed:
            self._link._send(
                protocol.STREAM_CREDIT, {"sid": self.sid, "n": self._pending_credit}
            )
            self._pending_credit = 0

    @property
    def depth(self) -> int:
        """Units delivered but not yet consumed."""
        return self.units_received - self.units_consumed

    def close(self):
        self._done = True
        self._link._forget_reader(self.sid)


# ======================================================================
#  The link
# ======================================================================
class Link:
    """A live connection.  Symmetric: both sides can notify, call and stream."""

    def __init__(self, channel, role: str, workers: int = 4):
        self.channel = channel
        self.role = role
        self.peer: dict = {}
        self.closed = False
        self.close_reason: str | None = None

        self._handlers: dict = {}
        self._pending: dict = {}  # cid -> (Event, box)
        self._streams: dict = {}  # sid -> Stream (we produce)
        self._readers: dict = {}  # sid -> StreamReader (we consume)
        self._offered: dict = {}  # name -> [StreamReader waiting to be claimed]
        self._offered_event = threading.Condition()
        self._ids = itertools.count(1)
        self._lock = threading.Lock()
        #: Calls run on a pool -- they are independent request/response pairs,
        #: and one slow answer should not hold up another.
        self._pool = ThreadPoolExecutor(
            max_workers=workers, thread_name_prefix="anr-call"
        )
        #: Notifications run on ONE thread, in arrival order. They are a stream
        #: of events where order is the meaning: prints, progress, logs and the
        #: final status. Dispatched to a pool they interleave, and a run ends up
        #: announcing that it finished before it says what it was doing.
        self._notices: queue.Queue = queue.Queue()
        self._notice_thread = None
        self._reader_thread = None
        self._closed_event = threading.Event()

    # -- lifecycle ---------------------------------------------------
    @property
    def connected(self) -> bool:
        return not self.closed

    def start(
        self,
        initiate: bool,
        info: dict | None = None,
        handshake_timeout: float | None = None,
    ):
        """Shake hands, then run the reader thread.

        ``initiate`` decides who speaks first; it has no other meaning.  The
        side that dialled out says HELLO, the side that was listening replies.

        ``handshake_timeout`` bounds the greeting only, and is cleared before
        the reader starts.  It has to be: a socket timeout applies to whatever
        read is in progress, so a reader thread that inherited one would treat
        the first quiet stretch of a real session -- a model being built, an
        epoch boundary -- as a dead connection.
        """
        mine = {"role": self.role, "proto": protocol.PROTO_VERSION, **_describe()}
        mine.update(info or {})
        socket_ = getattr(getattr(self.channel, "transport", None), "socket", None)

        try:
            if handshake_timeout and socket_ is not None:
                socket_.settimeout(handshake_timeout)

            if initiate:
                self.channel.send(protocol.HELLO, mine)
                reply = self.channel.recv()
                if reply.type != protocol.WELCOME:
                    raise LinkError(f"expected WELCOME, got {reply!r}")
                self.peer = reply.meta
            else:
                hello = self.channel.recv()
                if hello.type != protocol.HELLO:
                    raise LinkError(f"expected HELLO, got {hello!r}")
                self.peer = hello.meta
                self.channel.send(protocol.WELCOME, mine)
        finally:
            # Before the reader exists, and on the failure path too.
            if handshake_timeout and socket_ is not None:
                try:
                    socket_.settimeout(None)
                except OSError:
                    pass

        if self.peer.get("proto") != protocol.PROTO_VERSION:
            raise LinkError(
                f"protocol mismatch: this side speaks {protocol.PROTO_VERSION}, "
                f"the other speaks {self.peer.get('proto')}. Reinstall "
                "any_nn_runpod on whichever side is behind."
            )

        self._notice_thread = threading.Thread(
            target=self._notice_loop, name="anr-link-notices", daemon=True
        )
        self._notice_thread.start()
        self._reader_thread = threading.Thread(
            target=self._read_loop, name="anr-link-reader", daemon=True
        )
        self._reader_thread.start()
        return self

    def close(self, reason: str = "closed"):
        if self.closed:
            return
        self.closed = True
        self.close_reason = reason
        try:
            self.channel.send(protocol.BYE, {"reason": reason})
        except Exception:
            pass
        self._teardown(reason)

        # Break the reader out of its blocking recv and let it unwind before
        # tearing the streams down underneath it.
        transport = getattr(self.channel, "transport", None)
        if transport is not None and hasattr(transport, "unblock"):
            transport.unblock()
        if self._reader_thread is not None and self._reader_thread is not threading.current_thread():
            self._reader_thread.join(timeout=5)

        try:
            self.channel.close()
        except Exception:
            pass

    def _teardown(self, reason):
        """Wake everyone who is blocked, so nothing hangs on a dead link."""
        self._closed_event.set()
        for stream in list(self._streams.values()):
            stream.abandon()
        for reader in list(self._readers.values()):
            reader._fail(reason)
        for event, box in list(self._pending.values()):
            box["error"] = reason
            event.set()
        with self._offered_event:
            self._offered_event.notify_all()
        # Drained rather than dropped: the last notifications before a close
        # are usually the ones that say why, and the sentinel goes in behind
        # them so the worker finishes the backlog before it stops.
        self._notices.put(None)
        self._pool.shutdown(wait=False)

    def wait_closed(self, timeout=None) -> bool:
        return self._closed_event.wait(timeout)

    # -- handlers ----------------------------------------------------
    def on(self, name: str):
        """Register a handler.  Its return value answers a ``call``."""

        def register(function):
            self._handlers[name] = function
            return function

        return register

    def handle(self, name: str, function):
        self._handlers[name] = function

    # -- messaging ---------------------------------------------------
    def notify(self, name: str, payload=None, compress: bool = False):
        if self.closed:
            return
        body = None if payload is None else codec.encode(payload, compress=compress)
        self._send(protocol.NOTIFY, {"name": name}, body)

    def call(self, name: str, payload=None, timeout: float | None = None):
        if self.closed:
            raise NotConnected(f"call({name!r}): the link is closed")
        cid = next(self._ids)
        event, box = threading.Event(), {}
        with self._lock:
            self._pending[cid] = (event, box)
        try:
            body = None if payload is None else codec.encode(payload)
            self._send(protocol.CALL, {"name": name, "cid": cid}, body)
            if not event.wait(timeout):
                raise TimeoutError(f"call({name!r}) timed out after {timeout}s")
        finally:
            with self._lock:
                self._pending.pop(cid, None)
        if "error" in box:
            raise RemoteError(box.get("traceback") or box["error"])
        return box.get("value")

    # -- streams -----------------------------------------------------
    def open_stream(
        self, name: str, depth: int, info: dict | None = None, compress: bool = False
    ) -> Stream:
        """Start producing a stream the other side will accept by ``name``."""
        if self.closed:
            raise NotConnected(f"open_stream({name!r}): the link is closed")
        sid = f"{name}#{next(self._ids)}"
        stream = Stream(self, sid, depth, compress=compress)
        with self._lock:
            self._streams[sid] = stream
        self._send(
            protocol.STREAM_OPEN, {"sid": sid, "name": name, "info": info or {}}
        )
        return stream

    def accept_stream(self, name: str, timeout: float | None = None) -> StreamReader:
        """Wait for the other side to open a stream called ``name``."""
        deadline = None if timeout is None else time.monotonic() + timeout
        with self._offered_event:
            while True:
                waiting = self._offered.get(name)
                if waiting:
                    return waiting.pop(0)
                if self.closed:
                    raise NotConnected(
                        f"accept_stream({name!r}): the link closed "
                        f"({self.close_reason})"
                    )
                remaining = None if deadline is None else deadline - time.monotonic()
                if remaining is not None and remaining <= 0:
                    raise TimeoutError(
                        f"no stream named {name!r} was opened within {timeout}s"
                    )
                self._offered_event.wait(remaining if remaining else 0.25)

    def ping(self, timeout: float = 10.0) -> bool:
        try:
            self.call("__ping__", timeout=timeout)
            return True
        except (LinkError, TimeoutError, RemoteError):
            return False

    def flush(self, timeout: float = 60.0) -> bool:
        """Wait until the other side has *handled* everything sent so far.

        A ping does not do this. Pings are answered the moment they arrive,
        while notifications queue up to be handled in order -- so a ping can
        come back with a thousand log messages still waiting their turn.

        This goes through that queue instead of past it, so when it returns,
        every notification sent before it has been dealt with. Which is what
        makes it safe to hang up: a run that finishes, says so, and closes
        immediately would otherwise have its "finished" still sitting in the
        queue when the far side notices the socket is gone -- and a successful
        run gets reported as a run that died.
        """
        try:
            self.call("__flush__", timeout=timeout)
            return True
        except (LinkError, TimeoutError, RemoteError):
            return False

    # -- plumbing ----------------------------------------------------
    def _send(self, mtype, meta, body=None):
        try:
            self.channel.send(mtype, meta, body)
        except (OSError, ValueError, EOFError) as exc:
            if not self.closed:
                self.closed = True
                self.close_reason = f"send failed: {exc}"
                self._teardown(self.close_reason)

    def _forget_stream(self, sid):
        with self._lock:
            self._streams.pop(sid, None)

    def _forget_reader(self, sid):
        with self._lock:
            self._readers.pop(sid, None)

    def _read_loop(self):
        try:
            while not self.closed:
                self._dispatch(self.channel.recv())
        except (EOFError, OSError) as exc:
            reason = f"the other side went away ({type(exc).__name__}: {exc})"
        except protocol.ProtocolError as exc:
            reason = f"protocol error: {exc}"
        except Exception as exc:  # noqa: BLE001 -- surfaced to whoever is blocked
            reason = f"link reader crashed: {exc!r}"
        else:
            reason = self.close_reason or "closed"
        if not self.closed:
            self.closed = True
            self.close_reason = reason
        self._teardown(reason)

    def _dispatch(self, message):
        kind = message.type

        # -- cheap, done inline on the reader thread --------------------
        if kind == protocol.STREAM_DATA:
            reader = self._readers.get(message.meta["sid"])
            if reader is not None:
                reader._put(message)
            return

        if kind == protocol.STREAM_CREDIT:
            stream = self._streams.get(message.meta["sid"])
            if stream is not None:
                stream._credit(int(message.meta.get("n", 1)))
            return

        if kind == protocol.RESULT or kind == protocol.FAILED:
            entry = self._pending.get(message.meta["cid"])
            if entry is None:
                return
            event, box = entry
            if kind == protocol.FAILED:
                box["error"] = message.meta.get("error", "remote call failed")
                box["traceback"] = message.meta.get("traceback")
            else:
                box["value"] = message.body
            event.set()
            return

        if kind == protocol.STREAM_OPEN:
            sid, name = message.meta["sid"], message.meta["name"]
            reader = StreamReader(self, sid, message.meta.get("info"))
            with self._lock:
                self._readers[sid] = reader
            with self._offered_event:
                self._offered.setdefault(name, []).append(reader)
                self._offered_event.notify_all()
            return

        if kind == protocol.STREAM_END:
            reader = self._readers.pop(message.meta["sid"], None)
            if reader is not None:
                reader._finish(message.meta.get("error"))
            return

        if kind == protocol.PING:
            self._send(protocol.PONG, {})
            return

        if kind == protocol.PONG:
            return

        if kind == protocol.BYE:
            self.closed = True
            self.close_reason = message.meta.get("reason", "the other side said bye")
            return

        # -- may block, so it leaves the reader thread -------------------
        if kind == protocol.NOTIFY:
            self._notices.put(message)  # one ordered queue, see __init__
            return

        if kind == protocol.CALL:
            if message.meta["name"] == "__flush__":
                # Deliberately queued rather than dispatched: answering it is
                # the proof that everything queued ahead of it has been handled.
                self._notices.put(message)
            else:
                self._pool.submit(self._run_call, message)
            return

    def _notice_loop(self):
        """Run notification handlers one at a time, in the order they arrived."""
        while True:
            message = self._notices.get()
            if message is None:  # the sentinel from _teardown
                return
            if message.type == protocol.CALL:  # a flush barrier; see flush()
                self._send(protocol.RESULT, {"cid": message.meta["cid"]})
                continue
            name = message.meta["name"]
            handler = self._handlers.get(name)
            if handler is None:
                continue
            try:
                handler(message.body)
            except Exception:  # noqa: BLE001 -- nothing is waiting for this
                traceback.print_exc()

    def _run_call(self, message):
        name, cid = message.meta["name"], message.meta["cid"]
        if name == "__ping__":
            self._send(protocol.RESULT, {"cid": cid})
            return
        handler = self._handlers.get(name)
        if handler is None:
            self._send(
                protocol.FAILED,
                {
                    "cid": cid,
                    "error": f"no handler named {name!r} on the {self.role} side",
                },
            )
            return
        try:
            value = handler(message.body)
        except Exception as exc:  # noqa: BLE001 -- reported to the caller
            self._send(
                protocol.FAILED,
                {
                    "cid": cid,
                    "error": f"{type(exc).__name__}: {exc}",
                    "traceback": traceback.format_exc(),
                },
            )
            return
        body = None if value is None else codec.encode(value)
        self._send(protocol.RESULT, {"cid": cid}, body)


# ======================================================================
#  No other side
# ======================================================================
class NullLink:
    """The same surface, with nobody on the far end.

    This is what makes ``local/`` optional.  ``notify`` goes nowhere,
    ``accept_stream`` never yields, and ``call`` falls back to a handler
    registered on this side if there is one -- so a script that asks the local
    machine to encode something can still be made to work standalone by
    registering the same name here.
    """

    connected = False
    closed = True
    close_reason = "no link"
    role = "standalone"
    peer: dict = {}

    def __init__(self):
        self._handlers = {}

    def on(self, name):
        def register(function):
            self._handlers[name] = function
            return function

        return register

    def handle(self, name, function):
        self._handlers[name] = function

    def notify(self, name, payload=None, compress=False):
        handler = self._handlers.get(name)
        if handler is not None:
            handler(payload)

    def call(self, name, payload=None, timeout=None):
        handler = self._handlers.get(name)
        if handler is None:
            raise NotConnected(
                f"call({name!r}) needs the local side, and this run has none. "
                "Start with a local/ script, or register a handler of that "
                "name on the remote side."
            )
        return handler(payload)

    def open_stream(self, name, depth, info=None, compress=False):
        raise NotConnected(f"open_stream({name!r}) needs the other side")

    def accept_stream(self, name, timeout=None):
        raise NotConnected(f"accept_stream({name!r}) needs the other side")

    def ping(self, timeout=10.0):
        return False

    def wait_closed(self, timeout=None):
        return True

    def close(self, reason="closed"):
        pass


def _describe() -> dict:
    """What each side tells the other about itself."""
    info = {"python": platform.python_version(), "host": platform.node()}
    try:
        import torch

        info["torch"] = torch.__version__
        info["cuda"] = torch.cuda.is_available()
        info["device"] = (
            torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
        )
    except Exception:
        pass
    return info
