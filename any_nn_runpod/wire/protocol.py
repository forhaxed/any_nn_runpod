"""
Wire protocol.

A frame is ``[type u8][len u64][payload]``.  A *message* is one header frame
optionally followed by raw part frames -- the header says how many to expect.

There is deliberately no multiplexing.  TCP is already full duplex and each
direction has exactly one writer, so messages are strictly sequential per
direction: one side can send a control message while the other streams batches
back, and neither has to interleave.

Where easy_nn had one message type per concept (LOG, PRINT, PROGRESS,
CHECKPOINT, ARTIFACT...), this has four generic ones -- NOTIFY, CALL, RESULT,
STREAM_* -- and the concept lives in ``meta["name"]``.  Adding a new kind of
traffic between local and remote is then a new name, not a new protocol
version.
"""

from __future__ import annotations

import pickle
import struct
import threading

PROTO_VERSION = 1

# -- session ---------------------------------------------------------------
HELLO = 1
WELCOME = 2
BYE = 3

# -- generic messaging -----------------------------------------------------
NOTIFY = 10  # fire and forget:  meta={"name"}, optional body
CALL = 11  # request:          meta={"name", "cid"}, optional body
RESULT = 12  # reply:            meta={"cid"}, optional body
FAILED = 13  # reply that raised: meta={"cid", "error", "traceback"}

# -- streams ---------------------------------------------------------------
STREAM_OPEN = 20  # meta={"sid", "name", "info"}
STREAM_DATA = 21  # meta={"sid", "units"}, body
STREAM_CREDIT = 22  # meta={"sid", "n"}
STREAM_END = 23  # meta={"sid", "error"?}

# -- liveness --------------------------------------------------------------
PING = 30
PONG = 31

#: Raw payload frame, always preceded by a header frame that announces it.
PART = 0xF0

NAMES = {
    value: name
    for name, value in list(globals().items())
    if isinstance(value, int) and name.isupper() and name != "PROTO_VERSION"
}

_HEADER = struct.Struct("!BQ")
HEADER_SIZE = _HEADER.size

MAX_FRAME = 8 << 30  # 8 GiB, a sanity bound rather than a real limit


class ProtocolError(Exception):
    pass


def pack_header(ftype: int, length: int) -> bytes:
    return _HEADER.pack(ftype, length)


def unpack_header(raw: bytes) -> tuple[int, int]:
    return _HEADER.unpack(raw)


class Message:
    """A decoded header frame plus its raw parts."""

    __slots__ = ("type", "meta", "header", "parts")

    def __init__(self, mtype: int, meta: dict, header: bytes | None = None, parts=None):
        self.type = mtype
        self.meta = meta
        self.header = header  # codec structure, when the message carries a body
        self.parts = parts or []

    @property
    def has_body(self) -> bool:
        return self.header is not None

    @property
    def body(self):
        """Decode the carried object, or None if the message has no body."""
        if self.header is None:
            return None
        from any_nn_runpod.wire import codec

        return codec.decode(self.header, self.parts)

    @property
    def nbytes(self) -> int:
        return sum(len(part) for part in self.parts)

    def __repr__(self):
        return (
            f"<Message {NAMES.get(self.type, self.type)} "
            f"meta={self.meta} parts={len(self.parts)}>"
        )


class Channel:
    """Message-level view of a transport.  Sends are serialized by a lock."""

    def __init__(self, transport):
        self.transport = transport
        self._send_lock = threading.Lock()

    # -- sending ---------------------------------------------------------
    def send(self, mtype: int, meta: dict | None = None, body=None, on_progress=None):
        """Send one message.  ``body`` is an ``Encoded`` from the codec."""
        meta = meta or {}
        if body is None:
            head = {"meta": meta, "n_parts": None}
            with self._send_lock:
                self.transport.send_frame(mtype, pickle.dumps(head))
            return

        head = {"meta": meta, "n_parts": len(body.parts)}
        with self._send_lock:
            self.transport.send_frame(mtype, pickle.dumps(head))
            self.transport.send_frame(PART, body.header)
            sent = 0
            for part in body.parts:
                self.transport.send_frame(PART, part)
                sent += len(part)
                if on_progress is not None:
                    on_progress(sent)

    # -- receiving -------------------------------------------------------
    def recv(self) -> Message:
        """Read one whole message, blocking.  Raises EOFError on clean close."""
        mtype, payload = self.transport.recv_frame()
        if mtype == PART:
            raise ProtocolError("unexpected part frame outside a message")
        head = pickle.loads(payload)
        n_parts = head["n_parts"]
        if n_parts is None:
            return Message(mtype, head["meta"])

        ftype, header = self.transport.recv_frame()
        if ftype != PART:
            raise ProtocolError("expected body header frame")
        parts = []
        for _ in range(n_parts):
            ftype, part = self.transport.recv_frame()
            if ftype != PART:
                raise ProtocolError("expected body part frame")
            parts.append(part)
        return Message(mtype, head["meta"], header, parts)

    def close(self):
        self.transport.close()
