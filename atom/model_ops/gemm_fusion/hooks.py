# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Model-runner entry points for GEMM fusion. Default OFF.

Called from ``ModelRunner`` (never from model files). With no profile set the
runner calls none of them, so OFF behavior is unchanged.

Lifecycle (every step is collective and runs identically on every SP rank):
  init_gemm_fusion   -> distributed up, before model build. phase "pre".
  install (adapters) -> after model build, before compile/trace.
  bind_kv_cache      -> after KV allocation: resolve every profile bucket
                        (match + consensus + window ensure), finalize, prepare
                        plans, quiesce. phase "ready".
  before/after_graph_capture -> coverage log + clean epoch (no new resolution).
  check_step         -> end of every model step, outside capture: the periodic device error
                        check; returns the request ids to abort when a rank reported an error.
  tick               -> the first registered site advances the device STEP counter once per forward.
  recover / shutdown.

Phases make every per-call decision rank-uniform without a hot-path collective:
  "pre"   (before bind_kv_cache): every call is native, reason ``not_prepared``.
  "ready" (after prepare):        a bucket not resolved during prepare is native,
                                  reason ``late_bucket`` -- by construction it is not
                                  in the exact-row profile, so this equals ``uncovered``
                                  without running a collective inside the forward.
"""

import logging
import threading
from collections import deque

logger = logging.getLogger("atom.gemm_fusion")

RECOVER_TIMEOUT_S = 120.0  # recover(): bound on the collective quiesce/reset
SHUTDOWN_TIMEOUT_S = 30.0  # shutdown(): bound on release() (a peer may already be gone)

_SITES: dict = {}
_RES = None  # FusionResourcesWrapper | None
_PROFILE: str = ""
_KV = None  # bound kv_cache_data
_PHASE = "pre"  # "pre" | "preparing" | "ready"
_BROKEN = None  # reason string once recovery failed; fused path must not be used
_PLANS: dict = {}  # owner -> plan (chosen by consensus => identical on all ranks)
_TOKEN_IDS: dict = (
    {}
)  # (tokens, mode) -> _TokenIds: the call-id verdicts of ONE token count
# -- device error check (see ``check_step``). All rank-uniform: every rank passes exactly one
# ``check_step`` per model forward, so the counters below advance identically on every rank.
_PERIOD = (
    14  # steps between error snapshots; 1 = synchronous same-step check (KV publishers)
)
_STEP_IDX = 0  # number of ``check_step`` calls that ran in the ready phase
_PENDING = None  # step index of the error snapshot in flight (async mode), else None
_WINDOW: deque = (
    deque()
)  # (step_idx, req_ids) of every step since the last snapshot proven clean
_STEP_T = None  # device view of the STEP counter, created when the phase becomes ready
# Evaluate a snapshot this many steps after it was enqueued: the host's per-step sync is on the
# PREVIOUS step's token copy, and the snapshot sits after the step's own events in stream order, so
# only two steps later is it certainly complete. The abort window is therefore period + this delay.
SNAPSHOT_EVAL_DELAY = 2


class GemmFusionDeviceError(RuntimeError):
    """The device error word was set (or fused execution was disabled by a failed recovery) at a
    point where no request can be aborted; the worker must be treated as failed."""


class GemmFusionRecoveryTimeout(RuntimeError):
    """recover() did not complete in time (a peer rank never arrived)."""


# --------------------------------------------------------------------- state
def profile_enabled(config=None) -> bool:
    if config is None:
        try:
            from atom.config import get_current_atom_config

            config = get_current_atom_config()
        except Exception:  # noqa: BLE001 - no current config: treat as OFF
            return False
    gf = getattr(config, "gemm_fusion", None)
    return bool(gf is not None and gf.profile)


def register_site(site) -> None:
    """Register a site. The FIRST site registered (``install_m3_sites`` order, identical on every
    rank) is the one that advances the device STEP counter, once per forward."""
    if site.name not in _SITES:
        site.ticks_step = not _SITES
    _SITES[site.name] = site


def resources():
    return _RES


def profile() -> str:
    """The profile name/path from ``config.gemm_fusion.profile``, set by init.

    Every ATOM call into aiter (``load_profile``, ``match_prefix``) passes THIS value
    explicitly. aiter's own fallback to the ``AITER_GEMM_FUSION_PROFILE`` environment variable
    is never relied on: a spawned worker has the pickled Config, not the launcher's env, and
    an env-only profile would hide a broken CLI -> runtime path. Raises if none is set.
    """
    if not _PROFILE:
        raise RuntimeError(
            "gemm_fusion: no profile configured (config.gemm_fusion.profile / "
            "--gemm-fusion-profile); refusing to fall back to AITER_GEMM_FUSION_PROFILE"
        )
    return _PROFILE


def phase() -> str:
    return _PHASE


def world_or_1() -> int:
    return _RES.world if _RES is not None else 1


def register_plan(plan) -> None:
    owner = getattr(plan, "owner", None) or getattr(plan, "vid", "plan")
    _PLANS[owner] = plan


# ----------------------------------------------------------------- lifecycle
def init_gemm_fusion(config, sp_group, kv_publisher: bool = False) -> None:
    """After distributed + SP group init, before model construction/KV alloc.

    ``kv_publisher``: this rank publishes KV to another instance (a P/D producer). It then checks
    for a device error synchronously in every step, before the step's KV is announced.
    """
    global _RES, _PROFILE, _PHASE, _PERIOD
    if not profile_enabled(config):
        return
    from .resources import FusionResourcesWrapper
    from .stages import aiter_gf, sp_world

    gf = aiter_gf()
    name = config.gemm_fusion.profile
    # Fail fast, BEFORE any CCO resource exists: a missing/invalid profile must stop start-up
    # here with aiter's ProfileError, not surface later inside bind_kv_cache.
    gf.load_profile(name)
    _PROFILE = name
    # the same profile feeds the resources (window budget / VMM heap), not only the matcher
    _RES = FusionResourcesWrapper(sp_group, sp_world(), profile=name)
    _PHASE = "pre"
    _PERIOD = (
        1 if kv_publisher else int(getattr(config.gemm_fusion, "check_period", 14))
    )
    _reset_check_state()
    inner = getattr(_RES, "inner", None)
    logger.info(
        "gemm_fusion ON profile=%s world=%d window_bytes=%s max_windows=%s "
        "check_period=%d%s",
        _PROFILE,
        _RES.world,
        getattr(inner, "window_bytes", None),
        getattr(inner, "max_windows", None),
        _PERIOD,
        " (KV publisher: synchronous)" if kv_publisher else "",
    )
    _prebuild_profile(gf, name, _RES)


def _prebuild_profile(gf, name, res) -> None:
    """Compile every kernel variant the profile can select, once, before any forward.

    Without it the first fused call of each variant JIT-builds inside a step, on whichever rank
    reaches it first, which stalls that rank while its peers wait in a collective (and can blow the
    process-group timeout). ``aiter.gemm_fusion.prebuild_profile(profile, *, arch, cu, world)`` is
    idempotent and single-builder inside aiter (lock + cache), so EVERY rank calls it.

    Collective-safe: each rank reports success with a MIN all-reduce, so when any rank failed every
    rank raises (none proceeds to a forward while a peer died building) and all ranks leave this
    function together. An aiter without the function is logged loudly and skipped (variants then
    JIT-build on first use); every rank has the same install, so the skip is uniform and needs no
    collective."""
    fn = getattr(gf, "prebuild_profile", None)
    if fn is None:
        logger.warning(
            "gemm_fusion: the installed aiter has no prebuild_profile(); kernel variants will be "
            "JIT-built on first use, inside a step (slow first forward, rank-skewed)"
        )
        return
    err = None
    try:
        built = fn(name, arch=res.arch, cu=res.cu, world=res.world)
        logger.info(
            "gemm_fusion prebuild_profile(%s): %s",
            name,
            built if built is not None else "done",
        )
    except Exception as e:  # noqa: BLE001 - reported to every rank below
        err = e
        logger.error(
            "gemm_fusion prebuild_profile(%s) failed on this rank: %r", name, e
        )
    ok = 0 if err is not None else 1
    reduce = getattr(res, "_reduce_min", None)
    all_ok = ok if reduce is None else int(reduce([ok])[0])
    if err is not None:
        raise err
    if not all_ok:
        raise RuntimeError(
            f"gemm_fusion: prebuild_profile({name}) failed on a peer rank; refusing to start "
            "with an incomplete kernel build"
        )


def profile_tokens(world: int, arch: str | None = None, cu: int | None = None):
    """Padded token counts (m * world) of every exact profile row for this
    world/arch/cu: the buckets that are PREPARED (plan, binding, windows). Any other token count is
    served, when it can be, by the prepared bucket aiter's ``bucket_for_m`` names for its real rows
    (R16 pad-M, ``GemmFusionSite.resolve``); it never gets a bucket of its own."""
    from .stages import aiter_gf

    gf = aiter_gf()
    prof = gf.load_profile(profile())
    toks = set()
    for row in prof.rows():
        if row.world != world:
            continue
        if arch is not None and row.arch != arch:
            continue
        if cu is not None and row.cu != cu:
            continue
        toks.add(int(row.m) * world)
    return sorted(toks)


def prepare_buckets(tokens_list) -> None:
    """Collective, outside capture, on every rank in the same order:
    resolve (match + consensus + ensure) every (site, bucket, mode) ->
    ``finalize`` -> ``plan.prepare`` -> clean epoch -> STEP view -> phase "ready".
    """
    global _PHASE, _STEP_T
    if _RES is None:
        return
    _PHASE = "preparing"
    for s in _SITES.values():
        for t in tokens_list:
            for mode in ("eager", "capture"):
                s.resolve(t, graph_mode=mode)
    _refuse_colliding_call_ids()
    _RES.finalize()
    if _KV is not None:
        import torch

        from atom.config import get_current_atom_config

        device = torch.device("cuda", torch.cuda.current_device())
        out_dtype = get_current_atom_config().torch_dtype
        for s in _SITES.values():
            for t in tokens_list:
                prepared = s.prepare_bucket(t, device, _KV, out_dtype)
                s.agree_prepared(t, prepared)  # collective: uniform prepared state
    _clean_epoch("prepare")
    _STEP_T = _RES.step_tensor()  # created here, outside any capture
    _reset_check_state()
    _PHASE = "ready"


def _refuse_colliding_call_ids() -> None:
    """Every fused launch carries ``call_id = decision_key(site, tokens, vid, n_fused)`` and the
    kernels' handshake tags rely on two DIFFERENT launches never sharing one. The key is a 31-bit
    hash, so uniqueness is checked here, once, over every fused resolution (aiter's static check),
    before any launch. A colliding or zero key makes its bucket native on every rank: the keys, the
    check and therefore the verdict are rank-uniform."""
    keys = {
        (s.name, tokens, r.vid, r.n_fused)
        for s in _SITES.values()
        for (tokens, _mode), r in s._by_tokens.items()
        if getattr(r, "plan", None) is not None  # an entry can also be an Exception
    }
    report = _RES.check_call_ids(keys)
    if report is None or report.ok:
        return
    refused = report.refused_keys
    n = sum(s.refuse_call_ids(refused) for s in _SITES.values())
    logger.error(
        "gemm_fusion: %d fused bucket(s) refused for call_id collisions; they run native on every rank",
        n,
    )


class _TokenIds:
    """The call-id verdicts of one token count: ``ids`` = the ids aiter assigned to the launches whose
    tag carries only the low bits of ``call_id`` (large r13), ``refused`` = the keys that must run
    native (a collision among the launches of this token count, or no id could be assigned).
    """

    __slots__ = ("ids", "refused")

    def __init__(self, ids, refused):
        self.ids, self.refused = ids, frozenset(refused)


def token_call_ids(tokens, mode) -> _TokenIds:
    """The call-id verdicts for one token count, built once and cached. Only the launches of ONE
    token count (= one forward pass) can meet in a handshake, so that is the scope of the check: over
    every registered site's resolution of ``tokens``. A pure function of (sites, profile, the
    prepared buckets), no collective: every rank builds the same answer. Used for pad-M (R16), where
    the key carries the REAL token count and the universe of keys is every m a bucket can serve.
    """
    got = _TOKEN_IDS.get((tokens, mode))
    if got is None:
        got = _TOKEN_IDS[(tokens, mode)] = _build_token_call_ids(tokens, mode)
    return got


def _build_token_call_ids(tokens, mode) -> _TokenIds:
    from . import consensus

    keys = {}
    for s in _SITES.values():
        r = s.raw_resolution(tokens, mode)
        if r is not None and getattr(r, "plan", None) is not None:
            keys[(s.name, tokens, r.vid, r.n_fused)] = r
    if not keys or _RES is None:
        return _TokenIds({}, ())
    bits_of = getattr(_RES, "plan_tag_bits", None)
    tagged = [k for k, r in keys.items() if bits_of is not None and bits_of(r.plan)]
    ids, refused = {}, set()
    if tagged:
        try:
            ids = _RES.tag_unique_ids(tagged) or {}
        except ValueError:  # more (site, vid, n_fused) triples than tag ids
            ids = {}
        if not ids:
            refused.update(tagged)  # no id can be assigned: those launches run native
    bits = max((bits_of(keys[k].plan) for k in tagged), default=None)

    def key_fn(site, tok, vid, n):
        got = ids.get((site, tok, vid, n))
        return consensus.decision_key(site, tok, vid, n) if got is None else got

    report = _RES.check_call_ids(
        list(keys),
        key_fn=key_fn,
        tag_bits=bits if ids else None,
        tag_vids={k[2] for k in ids} or None,
        per_tokens=True,
    )
    if report is not None:
        refused.update(report.refused_keys)
    padded = sorted(
        {
            (r.bucket_tokens // _RES.world, r.vid)
            for r in keys.values()
            if r.bucket_tokens
        }
    )
    if padded:
        logger.info(
            "gemm_fusion pad-M: tokens=%d (m_real=%d) %s -> bucket m / variant %s; %d call_id "
            "key(s) refused",
            tokens,
            tokens // _RES.world,
            mode,
            padded,
            len(refused),
        )
    return _TokenIds(ids, refused)


def bind_kv_cache(kv_cache_data) -> None:
    """After KV allocation. Resolves/prepares every profile bucket."""
    global _KV
    if _KV is not None and _RES is not None:
        # RE-bind. Plans hold raw pointers to the KV caches they were prepared on; a
        # re-allocated pool would be silently ignored by them. Agree across ranks (this call is
        # made on every rank, so the collective is uniform) and fail on ALL ranks before any
        # state changes. Not a per-call check, and absent on the first bind.
        moved = int(any(s.kv_moved(kv_cache_data) for s in _SITES.values()))
        reduce = getattr(_RES, "_reduce_min", None)
        any_moved = bool(moved) if reduce is None else reduce([-moved, 0, 0])[0] != 0
        if any_moved:
            from .site import GemmFusionBindError

            raise GemmFusionBindError(
                "gemm_fusion: bind_kv_cache called again with KV caches at different addresses "
                "than the prepared buckets were bound to; re-binding a live pool is unsupported"
            )
    _KV = kv_cache_data
    for s in _SITES.values():
        s.bind_caches(kv_cache_data)
    if _RES is not None:
        prepare_buckets(profile_tokens(_RES.world))


def before_graph_capture(buckets) -> None:
    """Coverage log + clean epoch. The exact bucket rows were resolved and prepared in
    ``bind_kv_cache``; a capture size that is not one resolves, without a collective and identically
    on every rank, to the prepared bucket that serves its real rows (``pad_hit``) or to native. The
    graph is then captured with the real ``m`` (a by-value kernel argument), which is the m its
    replays run with.
    ``buckets``: padded token counts or ``(padded, real)`` pairs."""
    if _RES is None:
        return
    pairs = [b if isinstance(b, tuple) else (b, b) for b in buckets]
    for s in _SITES.values():
        for padded, _real in pairs:
            r = s.resolve(padded, graph_mode="capture")
            logger.debug(
                "gemm_fusion capture bucket %s %d -> %s/%s (bucket tokens %d)",
                s.name,
                padded,
                r.vid,
                r.reason,
                r.bucket_tokens or padded,
            )
    _raise_on_device_error("before graph capture")
    _clean_epoch("before graph capture")


def after_graph_capture() -> None:
    if _RES is not None:
        _raise_on_device_error("during graph capture")
        _clean_epoch("after graph capture")


# ----------------------------------------------------------- per-step check
def _reset_check_state() -> None:
    global _STEP_IDX, _PENDING
    _STEP_IDX = 0
    _PENDING = None
    _WINDOW.clear()


def tick() -> None:
    """Advance the device STEP counter by one. Called by the first registered site at the top of its
    ``run`` (inside the custom op, so it is captured into every graph and executed in eager), once per
    forward per rank. Not before the phase is ready: the counter does not exist earlier, and the
    phase is rank-uniform."""
    if _STEP_T is not None and _PHASE == "ready" and not _RES.launch_gate:
        _STEP_T.add_(1)  # not while a reset is zeroing the page this counter lives in


def check_launch_gate() -> None:
    """Refuse to start a forward or a graph replay while aiter's collective reset is in progress.

    ``recover()`` blocks the forward thread for the whole reset, so the gate can only be seen set
    when a reset outlived its caller (``GemmFusionRecoveryTimeout``: the reset thread is still
    running) or is run from another thread. A fused launch in that window could write the error
    word, STEP and the control windows after they were zeroed, and a CAPTURED GRAPH REPLAY runs no
    Python, so neither aiter's own ``Plan.run`` check nor anything else would stop it: this one bool
    read, made on the forward thread before the replay, is the only guard. The worker is failed.
    """
    res = _RES
    if res is not None and res.launch_gate:
        raise GemmFusionDeviceError(
            "gemm_fusion: a collective reset is in progress (launch gate set); refusing to start "
            "a forward / graph replay, the worker must be treated as failed"
        )


def _any_rank_error(word: int) -> bool:
    return int(_RES.reduce_max([int(word != 0)])[0]) != 0


def _device_error_now(step: int) -> bool:
    """Synchronous check: snapshot the error word under ``step``, wait for it, agree across ranks."""
    _RES.error_readback_async(step)
    return _any_rank_error(_RES.error_word(step))


def _trim_window(upto: int) -> None:
    while _WINDOW and _WINDOW[0][0] <= upto:
        _WINDOW.popleft()


def _verdict() -> list:
    """Some rank's error word was set: collective reset, then the ids of every request that ran in a
    step not yet proven clean. ``recover`` may raise (timeout): the worker is then failed.
    """
    global _PENDING
    ids = sorted({rid for _step, rids in _WINDOW for rid in rids})
    logger.error(
        "gemm_fusion: device error word set on some rank (this rank: %s); collective reset; %d "
        "request(s) of the last %d unchecked step(s) must be aborted",
        _RES.error_description(),
        len(ids),
        len(_WINDOW),
    )
    recover()
    # The step counter keeps counting (aiter keys its readback ring by step): only the window and
    # the snapshot in flight are dropped. Every rank does the same, so it stays rank-uniform.
    _WINDOW.clear()
    _PENDING = None
    return ids


def check_step(req_ids=()):
    """Periodic device error check. Call once at the END of every model step on EVERY rank, outside
    capture, after ``postprocess`` and BEFORE the step's KV is announced as ready.

    Returns None when nothing is wrong. When some rank reported an error, runs the collective reset
    and returns the ids of the requests that ran in the steps since the last snapshot proven clean
    (the caller must abort them and must not publish this step's KV); the list may be empty.

    Asynchronous mode (``_PERIOD`` >= 3, the default 14): at every ``_PERIOD``-th step enqueue a
    snapshot of the device error word under that step index (no sync); ``SNAPSHOT_EVAL_DELAY``
    steps later consume it (complete by then; the adapter falls back to a blocking wait, never to a
    skip), agree across ranks with ONE CPU-group MAX all-reduce, and either drop the
    now-proven-clean part of the window or report. An error in a step is therefore reported at most
    ``_PERIOD`` + ``SNAPSHOT_EVAL_DELAY`` steps later (16 for the default) and the window never holds
    more steps than that.

    Synchronous mode (``_PERIOD`` == 1, KV publishers): snapshot, wait and agree in the same step, so
    the check precedes the KV announcement (the same ordering the former per-step check kept).

    The step counter is rank-uniform because every rank passes exactly one call per forward; a
    forward that raised on one rank is a fail-stop anyway.
    """
    global _STEP_IDX, _PENDING
    if _RES is None or _PHASE != "ready":
        return None
    if _BROKEN is not None:
        raise GemmFusionDeviceError(
            f"gemm_fusion disabled after failed recovery: {_BROKEN}"
        )
    _STEP_IDX += 1
    step = _STEP_IDX
    _WINDOW.append((step, tuple(req_ids)))
    if _PERIOD == 1:
        if _device_error_now(step):
            return _verdict()
        _WINDOW.clear()
        return None
    # ">=": a check_now() in between advances the counter, which must not strand the snapshot
    if _PENDING is not None and step >= _PENDING + SNAPSHOT_EVAL_DELAY:
        if _any_rank_error(_RES.error_word(_PENDING)):
            return _verdict()
        _trim_window(_PENDING)
        _PENDING = None
    if step % _PERIOD == 0:
        _RES.error_readback_async(step)
        _PENDING = step
    return None


def check_now():
    """Synchronous check outside the step loop (graph-capture points, test harness). Same return
    contract as ``check_step``: None when clean, else the reset has run and the ids are returned.
    """
    global _STEP_IDX
    if _RES is None or _PHASE != "ready":
        return None
    if _BROKEN is not None:
        raise GemmFusionDeviceError(
            f"gemm_fusion disabled after failed recovery: {_BROKEN}"
        )
    # a fresh snapshot index (never one a pending snapshot owns); called at uniform points only
    _STEP_IDX += 1
    if _device_error_now(_STEP_IDX):
        return _verdict()
    return None


def _raise_on_device_error(where: str) -> None:
    """At a point with no request to abort (graph capture) an error is fatal."""
    if check_now() is not None:
        raise GemmFusionDeviceError(
            f"gemm_fusion device error word set {where}; plans re-initialised"
        )


# ---------------------------------------------------------------- recovery
def _fatal_reset(exc, where: str) -> "GemmFusionDeviceError":
    """A collective reset FAILED (aiter's ``ResetError``: the error word could not be cleared after
    its bounded rounds; or an unreadable error word, a HIP error, a barrier error). The communicator
    cannot be trusted, so the fused path is marked broken for good and the caller raises the returned
    error: it leaves ``forward`` and ends the worker (nothing in the worker loop catches it), so the
    engine shuts down instead of serving from a state that was not proven clean. Every rank gets the
    same ``ResetError`` (the verification is an all-gather), so every rank fails together.
    """
    global _BROKEN
    words = _RES.reset_error_words(exc) if _RES is not None else None
    if words is not None:
        bad = {r: hex(w) for r, w in sorted(words.items()) if w}
        _BROKEN = f"{where}: reset left a non-zero error word on rank(s) {bad}"
        logger.critical(
            "gemm_fusion %s: the collective reset could not clear the device error "
            "(non-zero ERR words %s); failing the worker: %s",
            where,
            bad,
            exc,
        )
        return GemmFusionDeviceError(
            f"gemm_fusion {where}: collective reset could not clear the error word "
            f"({exc}); the worker must be treated as failed"
        )
    _BROKEN = f"{where} failed: {exc!r}"
    logger.critical("gemm_fusion %s: the collective reset failed: %r", where, exc)
    return GemmFusionDeviceError(
        f"gemm_fusion {where}: collective reset failed ({exc!r}); the worker must be "
        "treated as failed"
    )


def _clean_epoch(where: str) -> None:
    """Collective epoch reset at a clean point. aiter's reset also verifies the error word, so it
    can raise too; that is as fatal as a failed reset after a verdict."""
    try:
        _RES.clean_epoch()
    except Exception as e:
        raise _fatal_reset(e, where) from e


def recover(timeout_s: float | None = None) -> None:
    """Collective recovery: device sync -> drain -> barrier -> re-init -> barrier.

    EVERY SP rank must call it (the error check makes the decision collective).
    A rank that never arrives would block the others in the barrier, so the call
    is bounded: after ``timeout_s`` (default ``RECOVER_TIMEOUT_S``) it marks the fused path
    broken and raises ``GemmFusionRecoveryTimeout``; the worker must then be treated as failed.
    """
    global _BROKEN
    if _RES is None:
        return
    if timeout_s is None:
        timeout_s = RECOVER_TIMEOUT_S
    err = []

    def _run():
        try:
            _RES.reset_after_error()
        except BaseException as e:  # noqa: BLE001
            err.append(e)

    t = threading.Thread(target=_run, name="gemm_fusion_recover", daemon=True)
    t.start()
    t.join(timeout_s)
    if t.is_alive():
        _BROKEN = f"recover timeout after {timeout_s}s"
        raise GemmFusionRecoveryTimeout(_BROKEN)
    if err:
        raise _fatal_reset(err[0], "recover") from err[0]


def shutdown(timeout_s: float | None = None) -> None:
    """Release the CCO resources, best effort and BOUNDED.

    ``release()`` does a device sync and a barrier. At exit after a failed start-up (a peer
    rank already dead, a HIP error, the runner never built) that barrier could block process
    exit or raise and hide the original failure, so it runs in a daemon thread with a timeout
    and never raises: problems are logged. Normal shutdown (all ranks alive) is unchanged.
    """
    global _RES, _PHASE, _STEP_T
    res, _RES = _RES, None
    _STEP_T = None
    _reset_check_state()
    if res is not None:
        if timeout_s is None:
            timeout_s = SHUTDOWN_TIMEOUT_S
        err = []

        def _run():
            try:
                res.release()
            except BaseException as e:  # noqa: BLE001
                err.append(e)

        t = threading.Thread(target=_run, name="gemm_fusion_release", daemon=True)
        t.start()
        t.join(timeout_s)
        if t.is_alive():
            logger.error(
                "gemm_fusion shutdown: release() did not finish in %.0fs (peer gone?); "
                "continuing",
                timeout_s,
            )
        elif err:
            logger.error(
                "gemm_fusion shutdown: release() failed: %r (continuing)", err[0]
            )
    _PLANS.clear()
    _TOKEN_IDS.clear()
    _SITES.clear()
    _PHASE = "pre"
