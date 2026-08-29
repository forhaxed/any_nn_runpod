"""
Tensor-aware serialization.

One codec serves everything that crosses the wire: batches of training data,
checkpoints, artifacts and plain RPC payloads.

The structure of an object is pickled; every ``torch.Tensor`` is pulled out of
the pickle stream and sent as a separate raw part, so nothing is copied twice
into RAM and the transport can report progress while a multi-gigabyte payload
is uploading.

Tensors are hooked through ``persistent_id``.  The pickler consults it before
the memo table, so object identity is decided here and here only: the same
tensor met twice encodes to one part and decodes back to one object.

Unlike easy_nn this uses plain ``pickle``, not ``cloudpickle``.  Code no longer
travels -- ``remote/`` is deployed as files and imported normally -- so there is
nothing to send by value, and dropping cloudpickle drops the requirement that
both sides run the same Python and torch minor versions.  The price is a rule:
**only plain data crosses the wire** -- dict, list, tuple, str, bytes, numbers,
None, and tensors.  Classes defined in your own files will not unpickle on the
other side, because that side has never imported them.
"""

from __future__ import annotations

import ctypes
import io
import pickle
from dataclasses import dataclass, field

import torch

try:
    import zstandard
except ImportError:  # optional
    zstandard = None


TENSOR_TYPES = (torch.Tensor, torch.nn.Parameter)

#: Below this, framing overhead dominates and compression is not worth a pass.
_COMPRESS_MIN = 1 << 16


@dataclass
class Encoded:
    """An object split into a pickled structure plus raw tensor parts."""

    header: bytes = b""
    parts: list = field(default_factory=list)
    descriptors: list = field(default_factory=list, repr=False)
    # Contiguous CPU tensors backing the zero-copy memoryviews in ``parts``.
    # They must outlive the send, hence the reference.
    keepalive: list = field(default_factory=list, repr=False)

    @property
    def nbytes(self) -> int:
        return len(self.header) + sum(len(part) for part in self.parts)


def _raw_view(tensor: torch.Tensor) -> tuple[memoryview, torch.Tensor]:
    """Zero-copy bytes of a tensor, dtype-agnostic (bfloat16 included)."""
    contiguous = tensor.detach().to("cpu", copy=False).contiguous()
    size = contiguous.numel() * contiguous.element_size()
    if size == 0:
        return memoryview(b""), contiguous
    buffer = (ctypes.c_char * size).from_address(contiguous.data_ptr())
    return memoryview(buffer), contiguous


def _from_raw(data, dtype: torch.dtype, shape) -> torch.Tensor:
    tensor = torch.empty(tuple(shape), dtype=dtype)
    size = tensor.numel() * tensor.element_size()
    if size:
        source = (ctypes.c_char * size).from_buffer_copy(data)
        ctypes.memmove(tensor.data_ptr(), ctypes.addressof(source), size)
    return tensor


class _Pickler(pickle.Pickler):
    def __init__(self, file, encoded: Encoded, compress: bool):
        super().__init__(file, protocol=pickle.HIGHEST_PROTOCOL)
        self._encoded = encoded
        self._compress = compress
        self._seen: dict[int, int] = {}

    def persistent_id(self, obj):
        if type(obj) not in TENSOR_TYPES:
            return None

        key = id(obj)
        index = self._seen.get(key)
        if index is not None:
            return ("t", index)

        view, keep = _raw_view(obj)
        index = len(self._encoded.parts)
        self._seen[key] = index

        raw_len = len(view)
        compression = None
        if self._compress and raw_len >= _COMPRESS_MIN and zstandard is not None:
            packed = zstandard.ZstdCompressor(level=1).compress(view)
            if len(packed) < raw_len * 0.9:
                view, compression = memoryview(packed), "zstd"
                keep = None

        self._encoded.parts.append(view)
        self._encoded.keepalive.append(keep)
        self._encoded.descriptors.append(
            {
                "param": type(obj) is torch.nn.Parameter,
                "dtype": str(obj.dtype).removeprefix("torch."),
                "shape": tuple(obj.shape),
                "requires_grad": bool(obj.requires_grad),
                "comp": compression,
                "raw_len": raw_len,
            }
        )
        return ("t", index)


class _Unpickler(pickle.Unpickler):
    def __init__(self, file, descriptors, parts):
        super().__init__(file)
        self._descriptors = descriptors
        self._parts = parts
        self._cache: dict[int, torch.Tensor] = {}

    def persistent_load(self, pid):
        tag, index = pid
        if tag != "t":
            raise pickle.UnpicklingError(f"unknown persistent id {pid!r}")

        obj = self._cache.get(index)
        if obj is not None:
            return obj

        descriptor = self._descriptors[index]
        data = self._parts[index]
        if descriptor["comp"] == "zstd":
            if zstandard is None:
                raise RuntimeError(
                    "payload is zstd-compressed but zstandard is not installed"
                )
            data = zstandard.ZstdDecompressor().decompress(
                bytes(data), max_output_size=descriptor["raw_len"]
            )
        obj = _from_raw(data, getattr(torch, descriptor["dtype"]), descriptor["shape"])
        if descriptor["param"]:
            obj = torch.nn.Parameter(obj, requires_grad=descriptor["requires_grad"])
        else:
            obj.requires_grad_(descriptor["requires_grad"])
        self._cache[index] = obj
        return obj


def encode(obj, compress: bool = False) -> Encoded:
    """Split ``obj`` into a pickled structure and a list of raw tensor parts."""
    encoded = Encoded()
    buffer = io.BytesIO()
    _Pickler(buffer, encoded, compress).dump(obj)
    encoded.header = pickle.dumps(
        {"descriptors": encoded.descriptors, "structure": buffer.getvalue()},
        protocol=pickle.HIGHEST_PROTOCOL,
    )
    return encoded


def decode(header: bytes, parts) -> object:
    head = pickle.loads(header)
    descriptors = head["descriptors"]
    if len(parts) != len(descriptors):
        raise ValueError(f"expected {len(descriptors)} tensor parts, got {len(parts)}")
    return _Unpickler(io.BytesIO(head["structure"]), descriptors, parts).load()


def has_zstd() -> bool:
    return zstandard is not None
