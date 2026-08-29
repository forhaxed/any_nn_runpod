"""
Getting ``remote/`` onto the pod, and not doing it twice.

``remote/`` is uploaded whole -- that is the contract, and it is what lets you
keep weights in there next to the script that loads them.  Uploading it whole
*every run* would be intolerable, so what actually crosses is the difference:
both sides hash their copy, and only files that differ are sent.

Hashing, not timestamps.  A checkout, a copy, a fresh clone all disturb mtimes
without changing a byte, and re-sending 8 GB because a file was touched is
exactly the failure this is here to avoid.  Size is checked first, so the hash
is only computed when it might matter.
"""

from __future__ import annotations

import hashlib
import os

#: Never uploaded: caches, outputs, and the environment's own droppings.
#: Deliberately short. ``remote/`` goes up whole -- that is the contract, and
#: it is what lets weights sit beside the script that loads them. Anything
#: cleverer here would eventually refuse to upload the very file the run needs.
IGNORE_DIRS = {
    "__pycache__",
    ".git",
    ".venv",
    "venv",
    ".pytest_cache",
    ".mypy_cache",
    ".ipynb_checkpoints",
    "out",
}
IGNORE_SUFFIXES = (".pyc", ".pyo", ".swp", "~")

#: Big enough that a multi-gigabyte file is not thousands of messages, small
#: enough that progress moves and neither side holds much in memory.
CHUNK = 8 << 20


def build_manifest(root: str) -> dict:
    """``{relative path: (size, sha256)}`` for everything worth uploading."""
    manifest = {}
    for directory, subdirectories, files in os.walk(root):
        subdirectories[:] = [
            name
            for name in subdirectories
            if name not in IGNORE_DIRS and not name.startswith(".")
        ]
        for filename in files:
            if filename.endswith(IGNORE_SUFFIXES):
                continue
            path = os.path.join(directory, filename)
            relative = os.path.relpath(path, root).replace(os.sep, "/")
            try:
                manifest[relative] = (os.path.getsize(path), _digest(path))
            except OSError:
                continue  # vanished mid-walk; it simply is not part of this sync
    return manifest


def _digest(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            block = handle.read(1 << 20)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def plan(here: dict, there: dict) -> tuple[list, list]:
    """What to send and what to delete, to make ``there`` match ``here``."""
    send = sorted(
        relative
        for relative, fingerprint in here.items()
        if there.get(relative) != fingerprint
    )
    delete = sorted(set(there) - set(here))
    return send, delete


def total_bytes(root: str, relatives) -> int:
    total = 0
    for relative in relatives:
        try:
            total += os.path.getsize(os.path.join(root, *relative.split("/")))
        except OSError:
            pass
    return total


def send_files(stream, root: str, relatives, on_progress=None):
    """Push each file down ``stream`` in chunks.  Returns bytes sent."""
    sent = 0
    for relative in relatives:
        path = os.path.join(root, *relative.split("/"))
        try:
            size = os.path.getsize(path)
        except OSError:
            continue
        with open(path, "rb") as handle:
            first, written = True, 0
            while True:
                block = handle.read(CHUNK)
                if not block and not first:
                    break
                written += len(block)
                delivered = stream.send(
                    {
                        "path": relative,
                        "data": block,
                        "first": first,
                        "last": written >= size,
                        "mode": os.stat(path).st_mode & 0o777,
                    }
                )
                if not delivered:
                    return sent
                first = False
                sent += len(block)
                if on_progress is not None:
                    on_progress(sent)
                if written >= size:
                    break
    return sent


def receive_files(reader, root: str) -> dict:
    """Write what comes down ``reader`` into ``root``.  Returns a summary.

    Files land under a ``.part`` name and are moved into place only once their
    last chunk has arrived, so a sync that dies halfway leaves the previous
    copy intact rather than a truncated one that looks complete.
    """
    written, handles, count = 0, {}, 0
    try:
        for payload, _units in reader:
            relative = payload["path"]
            path = os.path.join(root, *relative.split("/"))
            if payload["first"]:
                os.makedirs(os.path.dirname(path) or root, exist_ok=True)
                handles[relative] = open(path + ".part", "wb")
            handle = handles.get(relative)
            if handle is None:
                continue  # a continuation whose start we never saw
            handle.write(payload["data"])
            written += len(payload["data"])
            if payload["last"]:
                handle.close()
                del handles[relative]
                os.replace(path + ".part", path)
                mode = payload.get("mode")
                if mode:
                    try:
                        os.chmod(path, mode)
                    except OSError:
                        pass
                count += 1
    finally:
        for relative, handle in handles.items():
            handle.close()
            _discard(os.path.join(root, *relative.split("/")) + ".part")
    return {"files": count, "bytes": written}


def apply_deletions(root: str, relatives) -> int:
    removed = 0
    for relative in relatives:
        path = os.path.join(root, *relative.split("/"))
        if _discard(path):
            removed += 1
    _prune_empty(root)
    return removed


def _discard(path: str) -> bool:
    try:
        os.remove(path)
        return True
    except OSError:
        return False


def _prune_empty(root: str):
    for directory, _subdirectories, _files in sorted(os.walk(root), reverse=True):
        if directory == root:
            continue
        try:
            if not os.listdir(directory):
                os.rmdir(directory)
        except OSError:
            pass
