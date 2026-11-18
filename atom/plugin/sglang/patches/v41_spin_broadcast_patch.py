# SPDX-License-Identifier: MIT
"""DeepSeek-V4.1 on the ATOM plugin, SGLang 0.5.22: the scheduler's request
broadcast waits by polling instead of sleeping.

Every scheduler loop iteration, TP rank 0 broadcasts the requests it pulled
(often none) to the other ranks over the TP group's gloo CPU group
(``request_receiver.broadcast_pyobj``). ``torch.distributed.broadcast``
blocks in ``Work.wait()``, a futex sleep, so an idle or lightly loaded
scheduler main thread sleeps and wakes a few thousand times a second. On a
host whose cores scale frequency on load (intel_pstate/HWP, C1E), such a core
sits near its minimum clock, and the first steps of a request then run their
host Python up to ~3x slower until the core speeds up again. Native ATOM's
model runner busy-polls its input and keeps its core at full clock.

This issues the same broadcasts with ``async_op=True`` and polls
``Work.is_completed()`` before the ``wait()``: the same bytes, order and
group, the same protocol as SGLang's ``broadcast_pyobj`` (a rank on either
version talks to the other), only the waiting rank stays on its CPU. Each
scheduler main thread then uses a full core, as native's runners do. Between
polls it releases the GIL with ``os.sched_yield()``, so other Python threads
in the scheduler process are not held off by the busy wait.

Scope: the DeepSeek-V4.1 ATOM model path (installed when the V4.1 model is
prepared) and gloo groups only; anything else goes to SGLang's function.
Kill switch: ``ATOM_SGLANG_V41_SPIN_BROADCAST=0``. Version guard: SGLang
0.5.22 with ``request_receiver.broadcast_pyobj``; any other version keeps
SGLang's blocking broadcast and logs why.
"""

from __future__ import annotations

import inspect
import logging
import os
import pickle

logger = logging.getLogger("atom")

SUPPORTED_SGLANG_VERSIONS = ("0.5.22",)
V41_SPIN_BROADCAST_ENV = "ATOM_SGLANG_V41_SPIN_BROADCAST"
_EXPECTED_PARAMS = ["data", "rank", "dist_group", "src", "force_cpu_device"]


def v41_spin_broadcast_enabled() -> bool:
    return os.environ.get(V41_SPIN_BROADCAST_ENV, "1") != "0"


def _sglang_version() -> str:
    try:
        import sglang

        return str(getattr(sglang, "__version__", "?"))
    except Exception:  # noqa: BLE001 - a version string
        return "?"


def _upstream_shape_error(module) -> str | None:
    version = _sglang_version()
    if version not in SUPPORTED_SGLANG_VERSIONS:
        return f"sglang {version} not in {SUPPORTED_SGLANG_VERSIONS}"
    function = getattr(module, "broadcast_pyobj", None)
    if function is None:
        return "request_receiver.broadcast_pyobj is missing"
    params = list(inspect.signature(inspect.unwrap(function)).parameters)
    if params != _EXPECTED_PARAMS:
        return f"broadcast_pyobj takes {params}, not {_EXPECTED_PARAMS}"
    return None


# ``os.sched_yield`` releases the GIL around the syscall and returns at once
# when nothing else is runnable on the core, so the thread keeps its core busy
# while another Python thread of the scheduler process (e.g. HiCache's cache
# controller) can take the GIL between polls instead of waiting for the
# interpreter's switch interval. ``time.sleep(0)`` is not a substitute: it is a
# timed sleep (timer slack, ~50 us), which idles the core and lets it clock down.
_yield = getattr(os, "sched_yield", None) or (lambda: None)


def _broadcast_polling(dist, tensor, src, group) -> None:
    """``dist.broadcast`` that waits on the CPU instead of in a futex,
    yielding the GIL (not the core) between polls."""
    work = dist.broadcast(tensor, src=src, group=group, async_op=True)
    while not work.is_completed():
        _yield()
    # Raises what the collective raised (a timeout, a peer gone).
    work.wait()


def make_spin_broadcast_pyobj(original):
    """SGLang's ``broadcast_pyobj`` (sglang/srt/utils/common.py, 0.5.22) with
    polling waits on a gloo group; ``original`` serves every other call."""
    import numpy as np
    import torch
    import torch.distributed as dist

    def broadcast_pyobj(data, rank, dist_group=None, src=0, force_cpu_device=True):
        if (
            not force_cpu_device
            or dist_group is None
            or dist.get_backend(dist_group) != "gloo"
        ):
            return original(data, rank, dist_group, src, force_cpu_device)
        if rank == src:
            if len(data) == 0:
                size = torch.tensor([0], dtype=torch.long)
                _broadcast_polling(dist, size, src, dist_group)
            else:
                serialized = pickle.dumps(data)
                size = torch.tensor([len(serialized)], dtype=torch.long)
                payload = torch.ByteTensor(np.frombuffer(serialized, dtype=np.uint8))
                _broadcast_polling(dist, size, src, dist_group)
                _broadcast_polling(dist, payload, src, dist_group)
            return data
        size = torch.tensor([0], dtype=torch.long)
        _broadcast_polling(dist, size, src, dist_group)
        n = size.item()
        if n == 0:
            return []
        payload = torch.empty(n, dtype=torch.uint8)
        _broadcast_polling(dist, payload, src, dist_group)
        return pickle.loads(bytes(payload.numpy()))

    broadcast_pyobj._atom_v41_spin_broadcast = True
    broadcast_pyobj.__wrapped__ = original
    return broadcast_pyobj


def install_v41_spin_broadcast_patch() -> bool:
    """Swap the scheduler's request broadcast for the polling one
    (idempotent). Returns whether it is active."""
    if not v41_spin_broadcast_enabled():
        logger.info(
            "ATOM DeepSeek-V4.1: scheduler request broadcast blocks (%s=0)",
            V41_SPIN_BROADCAST_ENV,
        )
        return False
    from sglang.srt.managers.scheduler_components import request_receiver

    current = request_receiver.broadcast_pyobj
    if getattr(current, "_atom_v41_spin_broadcast", False):
        return True
    reason = _upstream_shape_error(request_receiver)
    if reason is not None:
        logger.warning(
            "ATOM DeepSeek-V4.1: scheduler request broadcast not patched (%s); "
            "it keeps blocking",
            reason,
        )
        return False
    request_receiver.broadcast_pyobj = make_spin_broadcast_pyobj(current)
    logger.info(
        "ATOM DeepSeek-V4.1: scheduler request broadcast polls its gloo "
        "collective instead of sleeping (%s=0 turns it off)",
        V41_SPIN_BROADCAST_ENV,
    )
    return True
