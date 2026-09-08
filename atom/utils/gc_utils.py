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
import types
from itertools import islice

logger = logging.getLogger("atom")


# The API server's default, and only its own. At concurrency 4096 its collector
# ran 13,956 times in twenty minutes over a set that grew to 688,646 objects and
# reclaimed *zero*; each ModelRunner worker reclaims thousands per pass, so
# raising it there would defer real work. `reclaim_watch` is what keeps the
# "reclaims nothing" premise honest now that it is a default.
#
# The triple is the one measured end-to-end (+20% on an EPLB c=4096 benchmark);
# the split between t0 and t1/t2 was not measured separately.
FRONTEND_GC_THRESHOLD = (20000, 50, 50)


def tune_gc(default: tuple[int, int, int] | None = None) -> None:
    """Raise this interpreter's collection thresholds.

    `ATOM_GC_THRESHOLD` wins if set; otherwise `default`, which callers pass
    only where they can say why. Per-interpreter, so every process that serves
    calls it for itself -- an enumeration here has gone stale twice, so the rule
    is the documentation and `tests/test_gc_utils.py` is what checks it.

    Spacing passes out is a different lever from `freeze_gc_heap`, not a
    fallback for it: freezing removes the startup heap from every pass, and
    what is left is per-request state that a serving frontend rebuilds faster
    than any freeze can help with.
    """
    from atom.utils import envs

    thresholds = envs.ATOM_GC_THRESHOLD
    if thresholds:
        try:
            default = tuple(int(x) for x in thresholds.split(","))
        except (ValueError, TypeError):
            logger.warning("[gc] bad ATOM_GC_THRESHOLD=%r, ignored", thresholds)
            return
    if default is None:
        return
    old = gc.get_threshold()
    gc.set_threshold(*default)
    logger.info("[gc] thresholds %s -> %s", old, default)


# The premise the raised thresholds rest on, and the only part of it that can
# go stale: they are free exactly while the collector finds nothing. Baseline
# after the freeze, because everything before it reclaimed plenty.
_reclaim_baseline: tuple[int, ...] | None = None
_reclaim_warned = False


def arm_reclaim_watch() -> None:
    """Record what has been reclaimed so far, so later growth is visible.

    Called once, after `freeze_gc_heap`. Before that point the counters carry
    startup's collections -- 18,826 objects in one measured run -- and a watch
    armed there would report that as steady-state growth on its first check.
    """
    global _reclaim_baseline, _reclaim_warned
    _reclaim_baseline = tuple(s["collected"] for s in gc.get_stats())
    _reclaim_warned = False


def reclaim_watch(context: str) -> int:
    """How many generations have reclaimed something since the watch was armed.

    Zero is the expected answer and the one the raised thresholds assume. A
    non-zero answer is not yet a fault: it says this process builds reference
    cycles, so spacing collections out is no longer free and is deferring real
    work into a growing heap.

    Warns once -- at four thousand streams a repeating line buries the log.
    `atom:gc_collected` is the continuous signal; this only makes someone look.
    """
    global _reclaim_warned
    if _reclaim_baseline is None:
        return 0
    now = tuple(s["collected"] for s in gc.get_stats())
    grew = [g for g, (was, is_) in enumerate(zip(_reclaim_baseline, now)) if is_ > was]
    if grew and not _reclaim_warned:
        _reclaim_warned = True
        logger.warning(
            "[gc] %s: generations %s reclaimed objects since startup (%s -> %s). "
            "Raised thresholds assume this process builds no cycles; it does, "
            "so collections are now deferring real work. See "
            "atom:gc_collected_total.",
            context,
            grew,
            _reclaim_baseline,
            now,
        )
    return len(grew)


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


def _most_common(counts: dict[str, int], n: int) -> list[str]:
    """The `n` most frequent keys, rendered `key xCOUNT`."""
    return [f"{k} x{c}" for k, c in sorted(counts.items(), key=lambda kv: -kv[1])[:n]]


def _shape(obj: object, fields: int) -> str:
    """A one-line fingerprint of a plain container, naming it by its contents.

    A type census answers "300k dicts", which is not yet an answer; keys tell
    you *which* dicts. Exact types only -- a subclass may compute `keys` or
    `__getitem__`, and a census must not run the code it is measuring.
    """
    if type(obj) is dict:
        # Empty gets its own bucket: "a million empty dicts" is an answer, and
        # folding it in with the unnameable ones buries whatever is second.
        if not obj:
            return "{}"
        keys = [k for k in islice(obj, fields) if isinstance(k, str)]
        return "{" + ",".join(keys) + "}" if keys else "{non-str keys}"
    if type(obj) in (list, tuple):
        return "[" + ",".join(type(x).__name__ for x in islice(obj, fields)) + "]"
    return ""


# (label, path marker, module prefix). Ordered: the first hit wins, so a more
# specific entry must come before any prefix of it.
_OWNERS = tuple(
    (name, f"/{name}/", name)
    for name in (
        "atom",
        "fastapi",
        "starlette",
        "uvicorn",
        "h11",
        "anyio",
        "pydantic",
        "msgspec",
        "transformers",
    )
)

# Exact type -> where its code object lives. Keyed on `type(obj)` rather than
# probed with `getattr`, which would run the object's `__getattr__` and its
# descriptors -- for a census, executing the code it is measuring.
_CODE_ATTR = {
    types.CoroutineType: "cr_code",
    types.AsyncGeneratorType: "ag_code",
    types.GeneratorType: "gi_code",
    types.FrameType: "f_code",
    types.FunctionType: "__code__",
}
_PLAIN_CONTAINERS = frozenset({tuple, list, dict, set, frozenset})


def _owner(obj: object) -> str:
    """Which library this object came from, or why it cannot be said.

    Code-carrying objects go by the file their code compiled from, instances by
    their class's module.

    Plain containers have no owner: a `tuple` of two cells belongs to whoever
    built the closure, and that is not recoverable from the tuple. They are
    reported as `unattributable` rather than dropped -- a breakdown missing a
    third of the set would read as precision it does not have.
    """
    try:
        kind = type(obj)
        if kind is types.MethodType:
            obj, kind = obj.__func__, types.FunctionType  # code lives on the function
        attr = _CODE_ATTR.get(kind)
        if attr is not None:
            path = getattr(getattr(obj, attr, None), "co_filename", "")
            for name, marker, _ in _OWNERS:
                if marker in path:
                    return name
            return "site-other" if "/site-packages/" in path else "stdlib"
        if kind in _PLAIN_CONTAINERS:
            return "unattributable"
        module = kind.__module__ or ""
        root = module.split(".", 1)[0]  # not startswith: "atom" must not eat "atomic"
        for name, _, prefix in _OWNERS:
            if root == prefix:
                return name
        return f"module:{root}" if root else "unattributable"
    except Exception:  # noqa: BLE001 - a census must not fail on one odd object
        return "error"


def gc_census(top: int = 30, samples: int = 2, fields: int = 8) -> dict:
    """What the collector actually rescans, broken down by type.

    The generation sizes say the collector is expensive; this says what it is
    expensive *on*, which is the only form of the answer you can act on. Read
    against `freeze_gc_heap`: everything counted here was allocated after the
    freeze, so at a serving concurrency it is per-request state, and dividing
    by the in-flight request count gives the per-stream object cost that
    thresholds do not change.

    On demand and never on a timer: it walks every tracked object and holds
    the event loop for roughly a second at a million of them. That is also why
    `gc.get_stats()` is included -- those counters are free, so a caller that
    only needs "is anything being reclaimed" should read them and not this.

    What is absent is as informative as what is here. A dict or tuple whose
    contents are all atomic is *untracked* by the collector on its first pass,
    so it never appears and never costs a scan; one holding a list or an
    instance stays. Per-request state built out of scalars is therefore free,
    and the same state with one list in it is not -- which is the difference
    this census exists to find.
    """
    gen2 = gc.get_objects(2)

    by_type: dict[str, int] = {}
    by_owner: dict[str, int] = {}
    owner_types: dict[str, dict[str, int]] = {}
    for obj in gen2:
        name = type(obj).__qualname__
        by_type[name] = by_type.get(name, 0) + 1
        who = _owner(obj)
        by_owner[who] = by_owner.get(who, 0) + 1
        per_owner = owner_types.setdefault(who, {})
        per_owner[name] = per_owner.get(name, 0) + 1
    ranked = sorted(by_type.items(), key=lambda kv: -kv[1])[:top]

    # A second walk, because a shape is only worth tallying for a type that
    # ranked and ranking needs the first walk finished. Tallied and not
    # sampled: which shape you meet first is an artifact of allocation order,
    # and the question is which one there are a million of.
    shapes: dict[str, list[str]] = {}
    if samples:
        wanted = {name for name, _ in ranked}
        tallies: dict[str, dict[str, int]] = {}
        for obj in gen2:
            name = type(obj).__qualname__
            if name in wanted and (shape := _shape(obj, fields)):
                per_shape = tallies.setdefault(name, {})
                per_shape[shape] = per_shape.get(shape, 0) + 1
        shapes = {name: _most_common(per, samples) for name, per in tallies.items()}

    return {
        # `len(gen2)` rather than a third `get_objects(2)`: that call
        # materialises the whole generation, and this one is already held.
        "generations": {
            "0": len(gc.get_objects(0)),
            "1": len(gc.get_objects(1)),
            "2": len(gen2),
        },
        "frozen": gc.get_freeze_count(),
        "thresholds": list(gc.get_threshold()),
        # `collected` is the one that decides whether raising thresholds is
        # free here or is deferring real work.
        "stats": gc.get_stats(),
        "by_type": [{"type": n, "count": c} for n, c in ranked],
        "shapes": shapes,
        "by_owner": [
            {"owner": who, "count": c, "types": _most_common(owner_types[who], samples)}
            for who, c in sorted(by_owner.items(), key=lambda kv: -kv[1])
        ],
    }
