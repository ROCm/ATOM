"""Opt-in per-request host timeline: wall-clock stage marks, one line each.

``ATOM_REQ_TIMELINE=<prefix>`` makes every process that marks a stage append
``<key> <stage> <unix seconds>`` lines to ``<prefix>.<pid>.txt``. Keys are a
request id or a sequence id; the API process writes one ``map`` line joining
the two, so a reader can follow a request across the API and engine-core
processes. Lines are buffered and written in blocks: a killed process loses its
last block, nothing else.
"""

from __future__ import annotations

import atexit
import os
import threading
import time

_PREFIX = os.getenv("ATOM_REQ_TIMELINE")
_lock = threading.Lock()
_buf: list[str] = []


def enabled() -> bool:
    return bool(_PREFIX)


def mark(key, stage: str, t: float | None = None) -> None:
    if not _PREFIX:
        return
    line = f"{key} {stage} {time.time() if t is None else t:.6f}\n"
    with _lock:
        _buf.append(line)
        if len(_buf) >= 256:
            _flush_locked()


def _flush_locked() -> None:
    with open(f"{_PREFIX}.{os.getpid()}.txt", "a") as fh:
        fh.writelines(_buf)
    _buf.clear()


def flush() -> None:
    if not _PREFIX:
        return
    with _lock:
        if _buf:
            _flush_locked()


atexit.register(flush)
