# SPDX-License-Identifier: MIT
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

"""Pick the fastest MXFP4 GEMM backend per shape, by measuring rather than guessing.

Lumen can reach three AITER MXFP4 GEMM kernels — the plain Triton one, its
shuffled-layout sibling, and the prebuilt A4W4 ASM/CK kernels. Which wins depends
on the shape, and the two fast ones carry a layout prologue that only pays off
once the problem is large enough.

A hand-measured byte threshold does not survive a change of model: the constant
tuned on Llama 3.1 8B (28 MiB MLP weights) excludes Qwen3-8B (24 MiB) entirely.
So the first call for a shape times the legal backends and remembers the winner;
a model issues only a few dozen shapes, so this costs a second once.

Safe because the three backends are bit-for-bit identical — they differ in memory
layout, not arithmetic.

Environment:
    ``LUMEN_MXFP4_AUTOTUNE=0``       fall back to the static byte thresholds
    ``LUMEN_MXFP4_AUTOTUNE_CACHE``   JSON file to persist and reuse decisions
    ``LUMEN_MXFP4_GEMM_SHAPE_LOG``   CSV to record every shape the model issues
"""

import atexit
import hashlib
import json
import logging as _logging
import os
import threading
from collections.abc import Callable, Sequence

import torch
from aiter.jit.core import AITER_CONFIG_GEMM_A4W4, AITER_CONFIGS
from aiter.ops.triton.utils._triton.arch_info import get_arch

_logger = _logging.getLogger(__name__)

AUTOTUNE_ENABLED = os.environ.get("LUMEN_MXFP4_AUTOTUNE", "1") == "1"
_CACHE_PATH = os.environ.get("LUMEN_MXFP4_AUTOTUNE_CACHE", "")
_SHAPE_LOG_PATH = os.environ.get("LUMEN_MXFP4_GEMM_SHAPE_LOG", "")

# AITER reads this when it first looks up a tuned A4W4 config, and colon-joins
# multiple paths into one merged table.
AITER_TUNED_CONFIG_ENV = "AITER_CONFIG_GEMM_A4W4"

# Past the first-call JIT and cache warmup without the measurement itself
# stalling. The median of the timed iterations decides, so a stray slow one does
# not.
_WARMUP_ITERS = 3
_TIMED_ITERS = 11

# How much faster a layout-rewriting backend has to be to be worth leaving the
# plain kernel. Run-to-run spread reaches a few percent -- enough to rank the
# backends differently on two measurements of the same shape -- and the fast
# paths cost extra host work and transient memory.
_SWITCH_MARGIN = 1.05

ShapeKey = tuple[int, int, int]
Candidate = tuple[str, Callable[[], torch.Tensor]]

_lock = threading.Lock()
_choice: dict[ShapeKey, str] = {}
_shape_log: dict[ShapeKey, dict[str, object]] = {}
_cache_loaded = False
_cache_dirty = False
_hooks_registered = False


def _arch() -> str:
    try:
        return get_arch()
    except (OSError, RuntimeError):
        return "unknown"


# ---------------------------------------------------------------------------
# Persisted decisions
# ---------------------------------------------------------------------------


def _cache_key(key: ShapeKey) -> str:
    return "{},{},{}".format(*key)


def _tuned_table_paths() -> list[str]:
    """The A4W4 tuned tables AITER will actually consult, in its merge order.

    AITER reads the env var if set and its own bundled table otherwise, so the
    two cases are not additive -- ``configure`` merges the default in by hand
    precisely because setting the variable replaces it.
    """
    env = os.environ.get(AITER_TUNED_CONFIG_ENV, "")
    if env:
        return [p for p in env.split(":") if p]
    default = _aiter_default_tuned_config()
    return [default] if default else []


def _tuned_table_fingerprint() -> str:
    """Identity of the tuned tables a decision was measured against.

    A choice recorded as ``asm`` means "AITER had a tuned kernel for this shape
    when we measured it". Replaying that against a different table dispatches
    the ASM kernel to a shape it has no config for, where ``gemm_a4w4`` picks an
    unvalidated default kernel and returns garbage -- 0.6 dB, no error. The
    ``arch`` check does not catch it because the GPU has not changed.

    Contents rather than paths and mtimes: the same table arrives at a new path
    with a new timestamp on every container rebuild and bind mount, and
    invalidating on that would throw away good measurements constantly.
    """
    h = hashlib.sha256()
    for p in _tuned_table_paths():
        # Basename, not the full path, for the same bind-mount reason.
        h.update(os.path.basename(p).encode())
        try:
            with open(p, "rb") as f:
                h.update(hashlib.sha256(f.read()).digest())
        except OSError:
            h.update(b"<unreadable>")
    return h.hexdigest()[:16]


def _load_cache() -> None:
    """Read previously measured decisions, if they still apply to this process."""
    global _cache_loaded
    _cache_loaded = True
    if not _CACHE_PATH or not os.path.exists(_CACHE_PATH):
        return
    try:
        with open(_CACHE_PATH) as f:
            blob = json.load(f)
    except (OSError, ValueError) as e:
        _logger.warning("MXFP4 autotune cache %s unreadable: %s", _CACHE_PATH, e)
        return
    # A decision measured on another GPU says nothing about this one.
    if blob.get("arch") != _arch():
        _logger.info(
            "MXFP4 autotune cache %s was measured on %s, ignoring on %s",
            _CACHE_PATH,
            blob.get("arch"),
            _arch(),
        )
        return
    # Nor does one measured against a different tuned table. Caches written
    # before this field existed carry None and are discarded for the same
    # reason: there is no way to tell what they were measured against.
    live_tables = _tuned_table_fingerprint()
    if blob.get("tuned_tables") != live_tables:
        _logger.info(
            "MXFP4 autotune cache %s was measured against tuned tables %s, "
            "ignoring against %s -- decisions will be re-measured",
            _CACHE_PATH,
            blob.get("tuned_tables"),
            live_tables,
        )
        return
    for k, name in blob.get("choices", {}).items():
        try:
            m, n, kk = (int(x) for x in k.split(","))
        except ValueError:
            continue
        _choice.setdefault((m, n, kk), name)
    _logger.info(
        "MXFP4 autotune: loaded %d cached decisions from %s", len(_choice), _CACHE_PATH
    )


def _save_cache() -> None:
    if not _CACHE_PATH or not _cache_dirty:
        return
    blob = {
        "arch": _arch(),
        "tuned_tables": _tuned_table_fingerprint(),
        "choices": {_cache_key(k): v for k, v in sorted(_choice.items())},
    }
    # Every rank runs this atexit, and the path is usually a shared results
    # directory, so a plain open(path, "w") lets one rank truncate the file
    # another is mid-write in. Write private, then rename: the reader either
    # sees the old file or the new one.
    tmp = f"{_CACHE_PATH}.{os.getpid()}.tmp"
    try:
        os.makedirs(os.path.dirname(_CACHE_PATH) or ".", exist_ok=True)
        with open(tmp, "w") as f:
            json.dump(blob, f, indent=2, sort_keys=True)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, _CACHE_PATH)
    except OSError as e:
        _logger.warning("could not write MXFP4 autotune cache %s: %s", _CACHE_PATH, e)
        try:
            os.unlink(tmp)
        except OSError:
            pass


# ---------------------------------------------------------------------------
# Shape collection
# ---------------------------------------------------------------------------


def record_shape(key: ShapeKey, tuned: bool, backend: str) -> None:
    """Note that the model issued this GEMM. Off unless the env var is set.

    Feeds ``scripts/mxfp4_tune_shapes.py``, which turns the log into the untuned
    CSV that AITER's tuner consumes. Collecting beats deriving the shapes by hand:
    every linear issues three GEMMs, and the backward pair permutes the dims in
    ways that are easy to get wrong (a wgrad's M is the output width and its K is
    the token count).
    """
    if not _SHAPE_LOG_PATH:
        return
    with _lock:
        entry = _shape_log.get(key)
        if entry is None:
            _shape_log[key] = {"tuned": tuned, "backend": backend, "calls": 1}
        else:
            entry["calls"] = int(entry["calls"]) + 1
            entry["backend"] = backend


def _save_shape_log() -> None:
    if not _SHAPE_LOG_PATH or not _shape_log:
        return
    try:
        os.makedirs(os.path.dirname(_SHAPE_LOG_PATH) or ".", exist_ok=True)
        with open(_SHAPE_LOG_PATH, "w") as f:
            f.write("M,N,K,tuned,backend,calls\n")
            f.writelines(
                f"{m},{n},{k},{int(bool(e['tuned']))},{e['backend']},{e['calls']}\n"
                for (m, n, k), e in sorted(_shape_log.items())
            )
        _logger.info(
            "MXFP4 shape log: wrote %d distinct shapes to %s",
            len(_shape_log),
            _SHAPE_LOG_PATH,
        )
    except OSError as e:
        _logger.warning("could not write MXFP4 shape log %s: %s", _SHAPE_LOG_PATH, e)


def _register_hooks() -> None:
    global _hooks_registered
    if _hooks_registered:
        return
    _hooks_registered = True
    atexit.register(_save_cache)
    atexit.register(_save_shape_log)


# ---------------------------------------------------------------------------
# Measurement
# ---------------------------------------------------------------------------


def _measure(key: ShapeKey, candidates: Sequence[Candidate]) -> str:
    """Time every candidate and return the winner, favouring the last one on ties.

    ``candidates`` ends with the backend that needs no layout rewrite, which is
    the one to stay on unless something is clearly faster.

    The candidates are timed round-robin rather than one after another, and every
    iteration is synchronised. Interleaving matters because the decision is cached
    for the life of the process: measured back to back, whichever backend went
    first would absorb the cold caches and clock ramp-up and could lose a contest
    it deserves to win, permanently. Syncing each call charges every backend for
    its own launch chain rather than letting the CPU run ahead and hide it, which
    is the honest comparison for a training step issuing hundreds of GEMMs.
    """
    baseline_name = candidates[-1][0]
    live = list(candidates)
    samples: dict[str, list[float]] = {name: [] for name, _ in live}

    for name, fn in live:
        try:
            for _ in range(_WARMUP_ITERS):
                fn()
        except (AssertionError, RuntimeError) as e:
            # A backend that cannot run this shape simply loses the contest.
            _logger.debug("MXFP4 autotune: %s failed on %s: %s", name, key, e)
            samples.pop(name, None)
    live = [(n, f) for n, f in live if n in samples]
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    for _ in range(_TIMED_ITERS):
        for name, fn in live:
            start.record()
            fn()
            end.record()
            torch.cuda.synchronize()
            samples[name].append(start.elapsed_time(end))

    timings: dict[str, float] = {}
    for name, xs in samples.items():
        if xs:
            xs.sort()
            timings[name] = xs[len(xs) // 2]
    if not timings:
        return baseline_name

    best = min(timings, key=timings.get)
    baseline = timings.get(baseline_name)
    if baseline is not None and timings[best] * _SWITCH_MARGIN > baseline:
        best = baseline_name

    _logger.info(
        "MXFP4 autotune %dx%dx%d: %s -> %s",
        *key,
        ", ".join(f"{n}={timings[n]:.3f}ms" for n in sorted(timings, key=timings.get)),
        best,
    )
    return best


def _capturing() -> bool:
    """True inside a CUDA graph capture, where extra launches are not allowed."""
    try:
        return torch.cuda.is_current_stream_capturing()
    except RuntimeError:
        return False


def pick_backend(
    key: ShapeKey,
    candidates: Sequence[Candidate],
    fallback: str | None = None,
) -> str:
    """Return the name of the backend to use for this shape.

    ``candidates`` are the backends that can legally run the shape, cheapest
    fallback last. The first call for a shape measures them; later calls reuse
    the answer. ``fallback`` is the static policy's pick, used when autotune is
    off or cannot run.
    """
    _register_hooks()
    global _cache_dirty

    if not _cache_loaded:
        with _lock:
            if not _cache_loaded:
                _load_cache()

    name = _choice.get(key)
    if name is not None:
        return name

    if not candidates:
        return fallback or ""
    if len(candidates) == 1:
        return candidates[0][0]
    if not AUTOTUNE_ENABLED or _capturing():
        return fallback or candidates[0][0]

    name = _measure(key, candidates)
    with _lock:
        _choice[key] = name
        _cache_dirty = True
    return name


def _aiter_default_tuned_config() -> str | None:
    return AITER_CONFIG_GEMM_A4W4


def _aiter_config_already_resolved() -> bool:
    """True once AITER has looked up its config file and cached the answer."""
    try:
        return AITER_CONFIGS.get_config_file.cache_info().currsize > 0
    except (AttributeError, RuntimeError):
        return False


def configure(
    tuned_config: str | Sequence[str] | None = None,
    autotune_cache: str | None = None,
    merge_aiter_default: bool = True,
) -> dict[str, str]:
    """Point AITER at extra tuned A4W4 rows, and persist autotune decisions.

    ``tuned_config`` takes one path or several, highest priority first; a model's
    own table has to come ahead of the generic one, since AITER keeps the first
    row it finds for a shape.

    Call once at process start. It does not have to precede ``import aiter`` --
    AITER reads the variable when it first looks up a config, not at import --
    but it must precede the first MXFP4 GEMM, after which both AITER's lookup and
    Lumen's backend decisions are cached.

    A tuned table only widens which shapes can reach the ASM kernels; it cannot
    change results, since every backend is bit-for-bit identical.

    Environment variables already set win, so a job can override without edits.
    Returns what ended up in effect, for logging.
    """
    global _CACHE_PATH

    if tuned_config and not os.environ.get(AITER_TUNED_CONFIG_ENV):
        requested = (
            [tuned_config] if isinstance(tuned_config, str) else list(tuned_config)
        )
        paths = []
        for path in requested:
            if not os.path.exists(path):
                _logger.warning("MXFP4 tuned config %s does not exist, ignoring", path)
            else:
                paths.append(os.path.abspath(path))
        if paths:
            # Keep AITER's own table; the tables cover different shapes.
            default = _aiter_default_tuned_config() if merge_aiter_default else None
            if default and os.path.exists(default):
                paths.append(default)
            os.environ[AITER_TUNED_CONFIG_ENV] = ":".join(paths)
            if _aiter_config_already_resolved():
                _logger.warning(
                    "%s set after AITER already resolved its tuned config; the new "
                    "rows will be ignored in this process",
                    AITER_TUNED_CONFIG_ENV,
                )

    if autotune_cache and not _CACHE_PATH:
        _CACHE_PATH = os.path.abspath(autotune_cache)
        _register_hooks()
    if _CACHE_PATH and not _cache_loaded:
        # Load now rather than on the first GEMM, so the decisions are visible
        # (and logged) at startup instead of appearing mid-step.
        with _lock:
            if not _cache_loaded:
                _load_cache()

    return {
        "tuned_config": os.environ.get(AITER_TUNED_CONFIG_ENV, ""),
        "autotune_cache": _CACHE_PATH,
    }


def cached(key: ShapeKey) -> str | None:
    """The decision for this shape if one is already in memory, else None.

    Lets the dispatcher skip building the candidate list on the hot path.
    """
    return _choice.get(key)


def forget(key: ShapeKey) -> None:
    """Drop one shape's decision so the next call re-measures it.

    The dispatcher calls this when a loaded decision names a backend that is
    not legal for these operands -- a cache written against a tuned table this
    process does not have, say.
    """
    global _cache_dirty
    with _lock:
        if _choice.pop(key, None) is not None:
            _cache_dirty = True


def clear() -> None:
    """Drop measured decisions. For tests."""
    global _cache_loaded
    with _lock:
        _choice.clear()
        _shape_log.clear()
        _cache_loaded = False
