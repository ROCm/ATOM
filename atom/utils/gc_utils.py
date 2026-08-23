# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Garbage-collector policy for the serving processes.

CPython's gen-2 pass is stop-the-world and traverses every tracked container,
so its cost tracks the live heap -- which here is almost all startup state
(model, compiled graph, tokenizer, KV block pool) that is never garbage.
Measured on DeepSeek-V4-Flash-DSpark tp1: 268 ms in the EngineCore, 979 ms in a
ModelRunner worker, 265 ms in the API server, each reclaiming 0 objects.

That is the argument for freezing rather than tuning thresholds: the collector
is not merely slow here, it finds nothing. Everything the hot loop allocates is
acyclic, so reference counting takes it.
"""

import gc
import logging
import time

logger = logging.getLogger("atom")


def tune_gc() -> None:
    """Raise the collection thresholds from ``ATOM_GC_THRESHOLD``.

    Per-interpreter, so every process calls it for itself. This spaces passes
    out; `freeze_gc_heap` removes what one costs and is on by default, so with
    `ATOM_GC_FREEZE=1` there is little left for this to do.
    """
    from atom.utils import envs

    thresholds = envs.ATOM_GC_THRESHOLD
    if not thresholds:
        return
    try:
        t = tuple(int(x) for x in thresholds.split(","))
        old = gc.get_threshold()
        gc.set_threshold(*t)
        logger.info("[gc] thresholds %s -> %s", old, t)
    except (ValueError, TypeError):
        logger.warning("[gc] bad ATOM_GC_THRESHOLD=%r, ignored", thresholds)


def freeze_gc_heap(context: str) -> int:
    """Move everything alive now into the permanent generation, which
    collections skip. Call once, between "startup done" and "traffic starts".
    Returns the total frozen count.

    Collecting first is not optional: `gc.freeze()` takes every generation as
    it finds it, so current garbage would be made permanently unreclaimable.
    One full pass suffices -- `gc.collect()` covers all three generations, and
    freezing does not care which one an object ended up in.

    Objects created afterwards are still tracked and collected, so this is not
    `gc.disable()` -- a cycle written by later code is still caught. What it
    forfeits is anything alive *now* that later becomes garbage, which in these
    processes outlives the process anyway. `unfreeze_gc_heap` covers the one
    case where that is false: tearing an engine down in-process.
    """
    from atom.utils import envs

    if not envs.ATOM_GC_FREEZE:
        return 0
    gc.collect()
    before = gc.get_freeze_count()
    gc.freeze()
    total = gc.get_freeze_count()
    logger.info(
        "[gc] %s: froze %d objects (%d already frozen)", context, total - before, before
    )
    return total


def unfreeze_gc_heap() -> None:
    """Hand the permanent generation back. Required on engine shutdown: a
    frozen object is invisible to the collector, so an engine destroyed inside
    a live interpreter would leave its weights and KV cache unreachable *and*
    uncollectable, which presents as a GPU memory leak."""
    frozen = gc.get_freeze_count()
    if frozen:
        logger.info("[gc] unfroze %d objects", frozen)
    gc.unfreeze()


def maybe_attach_gc_debug_callback(context: str) -> None:
    """Under ``ATOM_GC_DEBUG``, log every collection.

    Deliberately expensive: it counts the tracked set on each pass, which cost
    ~90s of extra startup on a V4-Flash tp1. It is also the only way to see
    these pauses -- a stall in the EngineCore idles the workers, and an idle
    worker emits no trace event at all.
    """
    from atom.utils import envs

    if not envs.ATOM_GC_DEBUG:
        return

    started: dict[str, float | int] = {"at": 0.0, "tracked": 0}

    def _log(phase: str, info: dict) -> None:
        gen = info.get("generation")
        if gen is None:
            return
        if phase == "start":
            started["at"] = time.perf_counter()
            started["tracked"] = len(gc.get_objects(gen))
            return
        logger.info(
            "[gc] %s: gen-%d took %.2f ms, reclaimed %s of %d tracked",
            context,
            gen,
            (time.perf_counter() - started["at"]) * 1e3,
            info.get("collected", "?"),
            started["tracked"],
        )

    gc.callbacks.append(_log)
    logger.info("[gc] %s: debug callback attached", context)
