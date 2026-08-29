"""The wire and the link, exercised over a real loopback socket pair."""

from __future__ import annotations

import threading
import time

import pytest
import torch

from any_nn_runpod.link import Link, NotConnected, NullLink, RemoteError
from any_nn_runpod.wire import codec
from any_nn_runpod.wire.transport import TcpListener, TcpTransport
from any_nn_runpod.wire.protocol import Channel


# ======================================================================
#  codec
# ======================================================================
def test_codec_round_trips_plain_data():
    payload = {"a": 1, "b": [1.5, "two", None], "c": (True, b"raw")}
    encoded = codec.encode(payload)
    assert codec.decode(encoded.header, encoded.parts) == payload


@pytest.mark.parametrize(
    "dtype", [torch.float32, torch.float16, torch.bfloat16, torch.int64, torch.uint8]
)
def test_codec_round_trips_every_dtype(dtype):
    if dtype.is_floating_point:
        tensor = torch.randn(4, 5).to(dtype)
    else:
        tensor = torch.randint(0, 100, (4, 5), dtype=dtype)
    encoded = codec.encode({"t": tensor})
    back = codec.decode(encoded.header, encoded.parts)["t"]
    assert back.dtype == dtype and back.shape == tensor.shape
    assert torch.equal(back, tensor)


def test_codec_keeps_tensor_identity():
    """One tensor met twice must come back as one object, not two copies.

    This is what keeps an optimizer's param_groups pointing at the very
    parameters a model holds; if it broke, a checkpoint would round-trip into
    an optimizer quietly updating detached copies.
    """
    shared = torch.randn(3, 3)
    encoded = codec.encode({"x": shared, "y": shared, "nested": [shared]})
    assert len(encoded.parts) == 1  # sent once, not three times

    back = codec.decode(encoded.header, encoded.parts)
    assert back["x"] is back["y"] is back["nested"][0]


def test_codec_handles_empty_and_noncontiguous():
    empty = torch.empty(0, 4)
    view = torch.randn(4, 6).t()  # non-contiguous
    encoded = codec.encode({"e": empty, "v": view})
    back = codec.decode(encoded.header, encoded.parts)
    assert back["e"].shape == (0, 4)
    assert torch.equal(back["v"], view)


def test_codec_compression_is_transparent():
    tensor = torch.zeros(512, 512)  # compresses well
    encoded = codec.encode({"t": tensor}, compress=True)
    back = codec.decode(encoded.header, encoded.parts)["t"]
    assert torch.equal(back, tensor)


def test_codec_rejects_a_truncated_body():
    encoded = codec.encode({"t": torch.randn(2, 2)})
    with pytest.raises(ValueError, match="tensor parts"):
        codec.decode(encoded.header, [])


# ======================================================================
#  a live pair of links
# ======================================================================
class Pair:
    """A connected (local, remote) pair over loopback, for tests."""

    def __init__(self):
        self.listener = TcpListener("127.0.0.1", 0)
        self.remote = None
        accepted = threading.Event()

        def serve():
            channel = self.listener.accept(timeout=10)
            self.remote = Link(channel, role="remote").start(initiate=False)
            accepted.set()

        thread = threading.Thread(target=serve, daemon=True)
        thread.start()

        transport = TcpTransport.connect("127.0.0.1", self.listener.port)
        self.local = Link(Channel(transport), role="local").start(initiate=True)
        assert accepted.wait(10), "the remote side never accepted"

    def close(self):
        for link in (self.local, self.remote):
            if link is not None:
                link.close("test over")
        self.listener.close()


@pytest.fixture
def pair():
    connected = Pair()
    yield connected
    connected.close()


def test_handshake_exchanges_descriptions(pair):
    assert pair.local.peer["role"] == "remote"
    assert pair.remote.peer["role"] == "local"
    assert pair.local.connected and pair.remote.connected


def test_notify_reaches_a_handler(pair):
    seen = []
    arrived = threading.Event()

    @pair.local.on("log")
    def _(payload):
        seen.append(payload)
        arrived.set()

    pair.remote.notify("log", {"loss": 0.5})
    assert arrived.wait(5)
    assert seen == [{"loss": 0.5}]


def test_notifications_are_handled_in_the_order_they_were_sent(pair):
    """Notifications are a stream of events, and the order is the meaning.

    Prints, progress and the final status all arrive this way. Dispatched to a
    thread pool they interleave, and a run announces that it finished before it
    has said what it was doing -- or a log lands under the wrong step.
    """
    seen = []
    done = threading.Event()

    @pair.local.on("tick")
    def _(payload):
        # Uneven work, so a pool would reorder these almost every time.
        time.sleep(0.02 if payload["i"] % 2 == 0 else 0.001)
        seen.append(payload["i"])
        if payload["i"] == 29:
            done.set()

    for i in range(30):
        pair.remote.notify("tick", {"i": i})
    assert done.wait(20)
    assert seen == list(range(30))


def test_a_handshake_timeout_does_not_outlive_the_handshake():
    """A bounded greeting must not leave the session on a stopwatch.

    A socket timeout applies to whatever read is in progress, so a reader
    thread that inherited the handshake's would treat the first quiet stretch
    of a real run -- a model being built, an epoch boundary -- as a dead link.
    """
    listener = TcpListener("127.0.0.1", 0)
    remote_box = {}
    accepted = threading.Event()

    def serve():
        channel = listener.accept(timeout=10)
        remote_box["link"] = Link(channel, role="remote").start(initiate=False)
        accepted.set()

    threading.Thread(target=serve, daemon=True).start()
    transport = TcpTransport.connect("127.0.0.1", listener.port)
    local = Link(Channel(transport), role="local").start(
        initiate=True, handshake_timeout=2.0
    )
    assert accepted.wait(10)

    try:
        # Longer than the handshake timeout, and completely silent.
        time.sleep(3.0)
        assert local.connected, f"the link died while idle: {local.close_reason}"
        assert local.ping(timeout=5)
    finally:
        local.close("test over")
        remote_box["link"].close("test over")
        listener.close()


def test_flush_waits_for_queued_notifications_to_be_handled(pair):
    """The barrier that makes it safe to hang up.

    A run that finishes, says so, and closes immediately would otherwise lose
    its own "finished": notifications are handled in order on one thread, so
    the socket can go while the last one is still queued behind a thousand log
    messages -- and a successful run gets reported as one that died.
    """
    handled = []

    @pair.local.on("log")
    def _(payload):
        time.sleep(0.003)
        handled.append(payload["i"])

    for i in range(200):
        pair.remote.notify("log", {"i": i})

    # A ping is answered on arrival, so it proves nothing about the backlog.
    assert pair.remote.ping(timeout=10)
    assert len(handled) < 200, "the backlog was already drained; test is not testing"

    assert pair.remote.flush(timeout=30)
    assert handled == list(range(200))


def test_flush_on_a_dead_link_reports_failure_rather_than_hanging(pair):
    pair.remote.close("gone")
    assert pair.remote.flush(timeout=5) is False


def test_a_notification_handler_that_raises_does_not_stop_the_rest(pair):
    seen = []
    done = threading.Event()

    @pair.local.on("tick")
    def _(payload):
        if payload["i"] == 1:
            raise ValueError("this one is broken")
        seen.append(payload["i"])
        if payload["i"] == 3:
            done.set()

    for i in range(4):
        pair.remote.notify("tick", {"i": i})
    assert done.wait(10)
    assert seen == [0, 2, 3]


def test_call_returns_a_value_with_tensors(pair):
    @pair.local.on("encode")
    def _(payload):
        return {"embeds": torch.ones(2, 3) * len(payload["prompts"])}

    answer = pair.remote.call("encode", {"prompts": ["a", "b"]}, timeout=10)
    assert torch.equal(answer["embeds"], torch.full((2, 3), 2.0))


def test_call_surfaces_the_other_sides_traceback(pair):
    @pair.local.on("boom")
    def _(payload):
        raise ValueError("this happened over there")

    with pytest.raises(RemoteError, match="this happened over there"):
        pair.remote.call("boom", timeout=10)


def test_call_without_a_handler_says_so(pair):
    with pytest.raises(RemoteError, match="no handler"):
        pair.remote.call("nobody_home", timeout=10)


def test_call_times_out_rather_than_hanging(pair):
    @pair.local.on("slow")
    def _(payload):
        time.sleep(5)

    with pytest.raises(TimeoutError):
        pair.remote.call("slow", timeout=0.5)


def test_stream_delivers_in_order_with_units(pair):
    def produce():
        stream = pair.local.open_stream("data", depth=4, info={"batches": 10})
        for i in range(10):
            stream.send({"i": i, "t": torch.tensor([float(i)])})
        stream.end()

    threading.Thread(target=produce, daemon=True).start()

    reader = pair.remote.accept_stream("data", timeout=10)
    assert reader.info["batches"] == 10
    got = [(payload["i"], int(payload["t"].item())) for payload, _ in reader]
    assert got == [(i, i) for i in range(10)]


def test_stream_window_never_exceeds_depth(pair):
    """The producer must not run ahead of ``depth`` unconsumed units.

    This is the property the whole precache knob rests on: without it a fast
    local machine would happily buffer an entire epoch into the pod's RAM.
    """
    depth = 3
    high_water = [0]
    ready = threading.Event()

    def produce():
        stream = pair.local.open_stream("data", depth=depth)
        ready.wait(5)
        for i in range(20):
            stream.send({"i": i})
        stream.end()

    threading.Thread(target=produce, daemon=True).start()
    reader = pair.remote.accept_stream("data", timeout=10)
    ready.set()

    seen = 0
    for _payload, _units in reader:
        seen += 1
        # Let the producer run as far ahead as it is allowed to.
        time.sleep(0.02)
        high_water[0] = max(high_water[0], reader.depth)
    assert seen == 20
    assert high_water[0] <= depth, f"producer ran {high_water[0]} units ahead"


def test_stream_end_with_an_error_raises_on_the_consumer(pair):
    def produce():
        stream = pair.local.open_stream("data", depth=4)
        stream.send({"i": 0})
        stream.end(error="the dataloader died")

    threading.Thread(target=produce, daemon=True).start()
    reader = pair.remote.accept_stream("data", timeout=10)
    with pytest.raises(RemoteError, match="the dataloader died"):
        list(reader)


def test_closing_wakes_a_blocked_consumer(pair):
    reader_box = {}

    def consume():
        reader = pair.remote.accept_stream("data", timeout=10)
        try:
            list(reader)
        except Exception as exc:  # noqa: BLE001
            reader_box["error"] = exc

    thread = threading.Thread(target=consume, daemon=True)
    thread.start()
    stream = pair.local.open_stream("data", depth=4)
    stream.send({"i": 0})
    time.sleep(0.3)
    pair.remote.close("hung up")
    thread.join(timeout=5)
    assert not thread.is_alive(), "the consumer never woke up"


def test_ping(pair):
    assert pair.remote.ping(timeout=5)


# ======================================================================
#  NullLink
# ======================================================================
def test_null_link_swallows_notifications():
    link = NullLink()
    link.notify("log", {"loss": 1.0})  # must not raise
    assert not link.connected


def test_null_link_call_falls_back_to_a_local_handler():
    link = NullLink()
    link.handle("encode", lambda payload: {"n": len(payload["prompts"])})
    assert link.call("encode", {"prompts": ["a"]}) == {"n": 1}


def test_null_link_call_without_a_handler_explains_itself():
    with pytest.raises(NotConnected, match="local side"):
        NullLink().call("encode", {})
