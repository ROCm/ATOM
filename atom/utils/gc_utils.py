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

logger = logging.getLogger("atom")


def tune_gc(thresholds: str | None = None) -> None:
    """Raise this interpreter's collection thresholds from `ATOM_GC_THRESHOLD`,
    or from `thresholds` when a caller has its own setting.

    Per-interpreter, so every process that serves calls it for itself -- an
    enumeration here has gone stale twice, so the rule is the documentation and
    `tests/test_gc_utils.py` is what checks it.

    No default: raising these spaces passes out without making one cheaper, so
    the same total scan lands in fewer, longer stop-the-world pauses -- a trade
    against tail latency that nothing here has measured. `freeze_gc_heap` is
    the lever that removes work rather than rescheduling it.
    """
    from atom.utils import envs

    if thresholds is None:
        thresholds = envs.ATOM_GC_THRESHOLD
    if not thresholds:
        return
    old = gc.get_threshold()
    # Ignoring a bad value has to cover applying it too: this runs unguarded in
    # four processes, and in a ModelRunner worker a raise here lands between
    # "load model runner success" and "ready", where nothing reports it.
    # `OverflowError` is the one `set_threshold` actually raises, and it is an
    # ArithmeticError, so the other two do not cover it.
    try:
        t = tuple(int(x) for x in thresholds.split(","))
        if len(t) != len(old):
            raise ValueError(f"want {len(old)} values, got {len(t)}")
        # CPython accepts both of these and neither means what it looks like.
        # `t[0] == 0` stops automatic collection (measured: nothing reclaimed
        # over 50k cycles) while `reclaim_watch` and `atom:gc_collected` read
        # like the healthy case. The other two are ratios, where zero means
        # *more* collection, so only a negative one is meaningless.
        if t[0] < 1 or min(t) < 0:
            raise ValueError(f"t0 must be >= 1 and none negative, got {t}")
        gc.set_threshold(*t)
    except (ValueError, TypeError, OverflowError) as exc:
        # `set_threshold` converts and stores one argument at a time, so an
        # overflow on the second or third leaves the earlier ones applied:
        # `1,2**63,10` lands on `(1, 10, 10)` and logs that it ignored the
        # value. Restoring is what makes "ignored" true, whatever the reason --
        # and it is a no-op on the paths that never reached the call.
        gc.set_threshold(*old)
        logger.warning("[gc] bad GC threshold %r (%s), ignored", thresholds, exc)
        return
    logger.info("[gc] thresholds %s -> %s", old, gc.get_threshold())


# Whether this process's collector finds anything at all -- the number that
# decides whether spacing its passes out would be free, and the one thing about
# it that can change without anyone noticing. Baseline after the freeze,
# because everything before it reclaimed plenty.
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

    Zero is the expected answer for the API server, measured over a full GSM8K
    run at its thresholds. A non-zero answer is not a fault: it says this
    process builds reference cycles, which is what would make spacing its
    collections out cost something rather than nothing.

    Blind to one case, and it is the one that matters most: a cycle promoted to
    gen-2 before it becomes garbage is reclaimed only by a gen-2 pass, so where
    those are rare it is neither collected nor counted here. `/debug/gc_census`
    reports the gen-2 size, which is what would show it.

    Warns once -- at four thousand streams a repeating line buries the log.
    `atom:gc_collected_total` is the continuous signal; this only makes someone
    look.
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
            "This process builds reference cycles, so raising its "
            "ATOM_GC_THRESHOLD would defer real work rather than nothing. See "
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

    global _heap_frozen

    if not envs.ATOM_GC_FREEZE:
        return 0
    gc.collect()
    before = gc.get_freeze_count()
    gc.freeze()
    _heap_frozen = True
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
    global _heap_frozen

    frozen = gc.get_freeze_count()
    if frozen:
        logger.info("[gc] unfroze %d objects", frozen)
    gc.unfreeze()
    _heap_frozen = False


# Set by `freeze_gc_heap`. A slow gen-2 pass before it is startup state, which
# the freeze is about to take out of reach, so only later ones are censused.
_heap_frozen = False
CENSUS_GAP_S = 900.0


def _type_name(tp: type) -> str:
    if tp.__module__ == "builtins":
        return tp.__qualname__
    return f"{'.'.join(tp.__module__.split('.')[-2:])}.{tp.__qualname__}"


def _holder(r: object, skip: tuple, depth: int = 1) -> str:
    """Who `r` is, for a referrer line: code for frames and coroutines, the
    class for instances, and one level further up for a bare container."""
    kind = type(r)
    if kind is types.FunctionType:
        return f"func:{r.__qualname__}"
    if kind in _CODE_ATTR:
        return f"{kind.__name__}:{getattr(r, _CODE_ATTR[kind]).co_qualname}"
    name = _type_name(kind)
    if depth > 0 and type(r) in (dict, list, tuple):
        up = []
        for x in gc.get_referrers(r):
            if not any(x is s for s in skip):
                up.append(_holder(x, skip, depth - 1))
            if len(up) == 3:
                break
        name += "<" + "|".join(up)
    return name


def log_gc_heap_census(
    context: str, pass_ms: float, top: int = 15, big: int = 10_000, chase: int = 3
) -> None:
    """Log what the oldest generation holds: the commonest types, the element
    count every full pass walks, and the largest containers with who holds
    them. Stalls this process for about one full pass per container chased.

    Referrers are found among the collected generations only: a holder that
    was frozen at startup shows as its container's other holders, or not at
    all, never as itself.
    """
    t = time.perf_counter()
    objs = gc.get_objects(generation=2)
    by_type: dict[type, int] = {}
    elems = 0
    bigs: list[tuple[int, int]] = []
    for i, o in enumerate(objs):
        tp = type(o)
        by_type[tp] = by_type.get(tp, 0) + 1
        if tp in (list, tuple, dict, set):
            k = len(o)
            elems += k
            if k >= big:
                bigs.append((k, i))
    ranked = sorted(by_type.items(), key=lambda kv: -kv[1])[:top]
    bigs.sort(reverse=True)
    big_s = []
    for rank, (k, i) in enumerate(bigs[:8]):
        o = objs[i]
        if type(o) is dict:
            key, value = next(iter(o.items()), (None, None))
            inner = f"{_type_name(type(key))}->{_type_name(type(value))}"
        else:
            inner = _type_name(type(next(iter(o), None)))
        desc = f"{type(o).__name__}[{inner}]x{k}"
        if rank < chase:
            refs = [r for r in gc.get_referrers(o) if r is not objs]
            desc += "<-" + "|".join(_holder(r, (objs, refs)) for r in refs[:4])
            refs = None
        big_s.append(desc)
    o = None
    total = len(objs)
    objs = None
    logger.info(
        "[gc-census] ctx=%s pass_ms=%.0f census_ms=%.0f gen2=%d elems=%d nbig=%d "
        "top=%s big=%s",
        context,
        pass_ms,
        (time.perf_counter() - t) * 1e3,
        total,
        elems,
        len(bigs),
        ",".join(f"{_type_name(tp)}:{c}" for tp, c in ranked),
        ";".join(big_s) or "-",
    )


def _attach_gc_refreeze(context: str, slow_ms: int) -> None:
    """After a gen-2 pass at least `slow_ms` long, freeze what survived it.

    For a process whose long-lived heap keeps growing after startup: a
    prefill worker's LMCache CPU cache holds ~7 tracked objects per cached
    chunk and fills for hours, and every full pass rescans all of it. Frozen
    objects are still freed by reference counting when evicted. What is given
    up is a reference cycle among them that dies later, which no pass will
    reclaim -- hence the reclaimed count on every line.
    """
    started = [0.0]

    def _refreeze(phase: str, info: dict) -> None:
        if info.get("generation") != 2:
            return
        if phase == "start":
            started[0] = time.perf_counter()
            return
        ms = (time.perf_counter() - started[0]) * 1e3
        if not _heap_frozen or ms < slow_ms:
            return
        before = gc.get_freeze_count()
        gc.freeze()
        logger.info(
            "[gc] %s: refroze %d objects after a %.0f ms gen-2 pass "
            "that reclaimed %s",
            context,
            gc.get_freeze_count() - before,
            ms,
            info.get("collected", "?"),
        )

    gc.callbacks.append(_refreeze)


def _attach_gc_timing(context: str, interval_s: float, census_ms: int = 0) -> None:
    """Log, every `interval_s`, how many collections each generation ran here,
    their total and their longest pause; with `census_ms`, also a census of
    the oldest generation after a gen-2 pass that slow, at most once every
    `CENSUS_GAP_S`.

    Cheap enough to leave on under load, unlike `ATOM_GC_DEBUG`: two clock
    reads per pass and no heap walk. Reported from the callback itself, so an
    idle process says nothing rather than needing a timer thread.
    """
    state = {"t0": 0.0, "since": time.perf_counter(), "census": None}
    n, total, longest = [0, 0, 0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]

    def _time(phase: str, info: dict) -> None:
        now = time.perf_counter()
        if phase == "start":
            state["t0"] = now
            return
        gen = info.get("generation")
        if gen not in (0, 1, 2):
            return
        dt = now - state["t0"]
        n[gen] += 1
        total[gen] += dt
        longest[gen] = max(longest[gen], dt)
        if (
            census_ms
            and gen == 2
            and _heap_frozen
            and dt * 1e3 >= census_ms
            and (state["census"] is None or now - state["census"] >= CENSUS_GAP_S)
        ):
            state["census"] = now
            log_gc_heap_census(context, dt * 1e3)
        if now - state["since"] < interval_s:
            return
        logger.info(
            "[gc-timing] ctx=%s s=%.1f n=%s ms=%s max_ms=%s",
            context,
            now - state["since"],
            "/".join(map(str, n)),
            "/".join(f"{x * 1e3:.1f}" for x in total),
            "/".join(f"{x * 1e3:.1f}" for x in longest),
        )
        state["since"] = now
        n[:], total[:], longest[:] = [0, 0, 0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]

    gc.callbacks.append(_time)


def maybe_attach_gc_debug_callback(context: str) -> None:
    """Under ``ATOM_GC_DEBUG``, log every collection; under
    ``ATOM_GC_TIMING_S``, a per-generation summary every that many seconds.

    The first is deliberately expensive: it counts the tracked set on each
    pass, which cost ~90s of extra startup on a V4-Flash tp1. Both are the only
    way to see these pauses -- a stall in the EngineCore idles the workers, and
    an idle worker emits no trace event at all.
    """
    from atom.utils import envs

    if envs.ATOM_GC_TIMING_S > 0:
        _attach_gc_timing(context, envs.ATOM_GC_TIMING_S, envs.ATOM_GC_CENSUS_SLOW_MS)
    # After the timing callback, so a census sees the generation it reports.
    if envs.ATOM_GC_REFREEZE_MS > 0:
        _attach_gc_refreeze(context, envs.ATOM_GC_REFREEZE_MS)
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


def gc_census(top: int = 30, types_per_owner: int = 2) -> dict:
    """What the collector actually rescans, broken down by type.

    The generation sizes say the collector is expensive; this says what it is
    expensive *on*, which is the only form of the answer you can act on. Read
    against `freeze_gc_heap`: everything counted here was allocated after the
    freeze, so at a serving concurrency it is per-request state, and dividing
    by the in-flight request count gives the per-stream object cost that
    thresholds do not change.

    On demand and never on a timer: it walks every tracked object, which takes
    roughly a second at a million of them, so the caller runs it off the event
    loop. That is also why `gc.get_stats()` is included -- those counters are
    free, so a caller that only needs "is anything being reclaimed" should read
    them and not this.

    Types and owners only. An earlier version also fingerprinted plain
    containers by their contents, which on this process meant serialising the
    keys of parsed request bodies into the response; naming what a dict holds
    is exactly the thing that cannot be reported from a live serving heap.

    What is absent is as informative as what is here. A dict or tuple whose
    contents are all atomic is *untracked* by the collector on its first pass,
    so it never appears and never costs a scan; one holding a list or an
    instance stays. Per-request state built out of scalars is therefore free,
    and the same state with one list in it is not -- which is the difference
    this census exists to find.
    """
    # Slicing is silent about a nonsensical bound: `[:0]` returns nothing and
    # `[:-1]` drops the smallest row, both of which read as a real answer.
    top = max(1, top)
    types_per_owner = max(0, types_per_owner)
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

    return {
        # `len(gen2)` rather than a third `get_objects(2)`: that call
        # materialises the whole generation, and this one is already held.
        "generations": {
            "0": len(gc.get_objects(0)),
            "1": len(gc.get_objects(1)),
            "2": len(gen2),
        },
        # Walks the permanent generation, so it belongs to a caller that is
        # already walking. Nothing on a scrape path may ask for it.
        "frozen": gc.get_freeze_count(),
        "thresholds": list(gc.get_threshold()),
        # `collected` is the one that decides whether raising thresholds is
        # free here or is deferring real work.
        "stats": gc.get_stats(),
        "by_type": [{"type": n, "count": c} for n, c in ranked],
        "by_owner": [
            {
                "owner": who,
                "count": c,
                "types": _most_common(owner_types[who], types_per_owner),
            }
            for who, c in sorted(by_owner.items(), key=lambda kv: -kv[1])
        ],
    }
