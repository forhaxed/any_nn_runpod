"""
Transports: framed bytes in, framed bytes out.

Byte counters live here because the one number that says whether the link is
the bottleneck -- upload MB/s -- can only be measured at this level.
"""

from __future__ import annotations

import socket

from any_nn_runpod.wire import protocol

#: Chosen to keep the syscall count low on large tensor frames.
_BUFFER = 1 << 20


class Transport:
    def send_frame(self, ftype: int, payload) -> None:
        raise NotImplementedError

    def recv_frame(self) -> tuple[int, bytes]:
        raise NotImplementedError

    def close(self) -> None:
        raise NotImplementedError


class StreamTransport(Transport):
    """Framing over any pair of binary streams."""

    def __init__(self, reader, writer):
        self._reader = reader
        self._writer = writer
        self.bytes_sent = 0
        self.bytes_received = 0

    def send_frame(self, ftype: int, payload) -> None:
        view = payload if isinstance(payload, memoryview) else memoryview(payload)
        self._writer.write(protocol.pack_header(ftype, len(view)))
        if len(view):
            self._writer.write(view)
        self._writer.flush()
        self.bytes_sent += protocol.HEADER_SIZE + len(view)

    def recv_frame(self) -> tuple[int, bytes]:
        head = self._read_exactly(protocol.HEADER_SIZE)
        ftype, length = protocol.unpack_header(head)
        if length > protocol.MAX_FRAME:
            raise protocol.ProtocolError(f"frame of {length} bytes is out of bounds")
        payload = self._read_exactly(length)
        self.bytes_received += protocol.HEADER_SIZE + length
        return ftype, payload

    def _read_exactly(self, n: int) -> bytes:
        if n == 0:
            return b""
        chunks = []
        remaining = n
        while remaining:
            chunk = self._reader.read(remaining)
            if not chunk:
                raise EOFError("connection closed mid-frame")
            chunks.append(chunk)
            remaining -= len(chunk)
        return b"".join(chunks) if len(chunks) > 1 else chunks[0]

    def unblock(self) -> None:
        """Make a reader blocked in ``recv_frame`` return.  Called before close.

        Closing a ``BufferedReader`` that another thread is sitting inside
        deadlocks: ``close`` wants the buffer lock, and the blocked ``readinto``
        is holding it until bytes arrive that never will.  Transports that can
        break the read from underneath -- a socket can, via ``shutdown`` --
        override this.
        """

    def close(self) -> None:
        self.unblock()
        for stream in (self._writer, self._reader):
            try:
                stream.close()
            except Exception:
                pass


class TcpTransport(Transport):
    """Framing straight on the socket -- deliberately without ``makefile``.

    Two reasons, and the first one is a bug we hit:

    ``socket.makefile`` bumps the socket's ``_io_refs``, so ``sock.close()``
    stops closing the real handle until every file object is closed too.  That
    makes it impossible to break a reader blocked in ``recv`` from another
    thread: ``shutdown`` does not abort a pending recv on Windows, ``close`` has
    been turned into a no-op, and closing the ``BufferedReader`` itself
    deadlocks against the buffer lock the blocked read is holding.  Closing a
    session then hangs for as long as anyone is patient.

    Second, ``recv_into`` reads a multi-gigabyte tensor part straight into the
    buffer that becomes the payload, where a ``BufferedReader`` would copy it
    through its own buffer first.
    """

    def __init__(self, sock: socket.socket):
        self.socket = sock
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, _BUFFER)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, _BUFFER)
        _keepalive(sock)
        self.bytes_sent = 0
        self.bytes_received = 0
        self._send_closed = False

    @classmethod
    def connect(cls, host: str, port: int, timeout: float = 30.0):
        sock = socket.create_connection((host, port), timeout=timeout)
        sock.settimeout(None)  # blocking for the life of the session
        return cls(sock)

    # -- keepalive ----------------------------------------------------
    #: Probe an idle connection after this long, then every ``KEEPALIVE_INTERVAL``
    #: until ``KEEPALIVE_COUNT`` go unanswered.  Dead in roughly 90 seconds.
    KEEPALIVE_IDLE = 60
    KEEPALIVE_INTERVAL = 10
    KEEPALIVE_COUNT = 3

    def send_frame(self, ftype: int, payload) -> None:
        view = payload if isinstance(payload, memoryview) else memoryview(payload)
        self.socket.sendall(protocol.pack_header(ftype, len(view)))
        if len(view):
            self.socket.sendall(view)
        self.bytes_sent += protocol.HEADER_SIZE + len(view)

    def recv_frame(self) -> tuple[int, bytes]:
        head = self._read_exactly(protocol.HEADER_SIZE)
        ftype, length = protocol.unpack_header(bytes(head))
        if length > protocol.MAX_FRAME:
            raise protocol.ProtocolError(f"frame of {length} bytes is out of bounds")
        payload = self._read_exactly(length)
        self.bytes_received += protocol.HEADER_SIZE + length
        return ftype, payload

    def _read_exactly(self, n: int) -> bytearray:
        buffer = bytearray(n)
        if n == 0:
            return buffer
        view = memoryview(buffer)
        filled = 0
        while filled < n:
            try:
                got = self.socket.recv_into(view[filled:], n - filled)
            except OSError as exc:
                raise EOFError(f"connection lost mid-frame: {exc}") from exc
            if not got:
                raise EOFError("connection closed mid-frame")
            filled += got
        return buffer

    def unblock(self) -> None:
        """Abort whatever the reader thread is blocked on."""
        try:
            self.socket.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass  # already gone, which is what we wanted anyway
        try:
            self.socket.close()
        except OSError:
            pass

    def close(self) -> None:
        self.unblock()


class TcpListener:
    """Accepts connections one at a time."""

    def __init__(self, host: str, port: int, backlog: int = 4):
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.socket.bind((host, port))
        self.socket.listen(backlog)
        self.closed = False
        self._port = self.socket.getsockname()[1]

    @property
    def port(self) -> int:
        return self._port

    def accept(self, timeout: float | None = None):
        """Wait for one client.

        Returns None on timeout *and* once the listener has been closed, so an
        accept loop shutting down ends rather than raising: closing the socket
        from another thread is how a supervisor is stopped, and on Windows that
        surfaces here as "not a socket" from the very first call.
        """
        if self.closed:
            return None
        try:
            self.socket.settimeout(timeout)
            sock, _ = self.socket.accept()
        except (socket.timeout, TimeoutError, OSError):
            return None
        finally:
            try:
                self.socket.settimeout(None)
            except OSError:
                pass
        return protocol.Channel(TcpTransport(sock))

    def close(self):
        self.closed = True
        try:
            self.socket.close()
        except OSError:
            pass


def _keepalive(sock: socket.socket):
    """Make the kernel notice a peer that vanished without saying goodbye.

    Without this a half-open connection is invisible: the socket stays
    readable-forever from the application's point of view, and whoever is
    blocked on it waits for the rest of the pod's life.  On a supervisor that
    means a launcher whose machine died leaves the pod accepting TCP
    connections it will never answer -- alive, billing, and deaf.

    Default idle time on Linux is two hours, which is not a timeout so much as
    a rumour, so the intervals are set explicitly where the platform allows it.
    """
    try:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
    except OSError:
        return
    for option, value in (
        ("TCP_KEEPIDLE", TcpTransport.KEEPALIVE_IDLE),
        ("TCP_KEEPINTVL", TcpTransport.KEEPALIVE_INTERVAL),
        ("TCP_KEEPCNT", TcpTransport.KEEPALIVE_COUNT),
    ):
        number = getattr(socket, option, None)
        if number is None:
            continue  # Windows has no per-socket knobs; SO_KEEPALIVE it is
        try:
            sock.setsockopt(socket.IPPROTO_TCP, number, value)
        except OSError:
            pass
