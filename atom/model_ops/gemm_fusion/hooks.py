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
  check_step         -> end of every model step, outside capture.
  recover / shutdown.

Phases make every per-call decision rank-uniform without a hot-path collective:
  "pre"   (before bind_kv_cache): every call is native, reason ``not_prepared``.
  "ready" (after prepare):        a bucket not resolved during prepare is native,
                                  reason ``late_bucket`` -- by construction it is not
                                  in the exact-row profile, so this equals ``uncovered``
                                  without running a collective inside the forward.
"""

import hashlib
import logging
import threading
from typing import Optional

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
_STEP_DIGEST = hashlib.blake2b(digest_size=4)
_STEP_CALLS = 0


class GemmFusionPoisonError(RuntimeError):
    """A fused plan reported device poison (or ranks disagreed) during a step."""


class GemmFusionRecoveryTimeout(RuntimeError):
    """recover() did not complete in time (a peer rank never arrived)."""


# --------------------------------------------------------------------- state
def profile_enabled(config=None) -> bool:
    if config is None:
        try:
            from atom.config import get_current_atom_config

            config = get_current_atom_config()
        except Exception:
            return False
    gf = getattr(config, "gemm_fusion", None)
    return bool(gf is not None and gf.profile)


def register_site(site) -> None:
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


def record_decision(d) -> None:
    """Fold the rank-uniform part of a decision into the step digest.

    Digest = (site, tokens, vid, n_fused, path) and, for native paths, the
    reason. ``bind_mode`` is rank-local and excluded.
    """
    global _STEP_CALLS
    reason = d.reason if d.path == "native" else ""
    _STEP_DIGEST.update(f"{d.site}|{d.tokens}|{d.vid}|{d.n_fused}|{d.path}|{reason};".encode())
    _STEP_CALLS += 1


# ----------------------------------------------------------------- lifecycle
def init_gemm_fusion(config, sp_group) -> None:
    """After distributed + SP group init, before model construction/KV alloc."""
    global _RES, _PROFILE, _PHASE
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
    inner = getattr(_RES, "inner", None)
    logger.info(
        "gemm_fusion ON profile=%s world=%d window_bytes=%s max_windows=%s",
        _PROFILE, _RES.world, getattr(inner, "window_bytes", None), getattr(inner, "max_windows", None),
    )


def profile_tokens(world: int, arch: Optional[str] = None, cu: Optional[int] = None):
    """Padded token counts (m * world) of every exact profile row for this
    world/arch/cu. These are the only buckets the profile can ever fuse."""
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
    ``finalize`` -> ``plan.prepare`` -> ``quiesce_and_reset`` -> phase "ready".
    """
    global _PHASE
    if _RES is None:
        return
    _PHASE = "preparing"
    for s in _SITES.values():
        for t in tokens_list:
            for mode in ("eager", "capture"):
                s.resolve(t, graph_mode=mode)
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
    _RES.quiesce_and_reset()
    _PHASE = "ready"


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
    """Coverage log + clean epoch. Everything the profile can fuse was resolved
    and prepared in ``bind_kv_cache``; a capture bucket outside it is native
    (``late_bucket``) on every rank, decided without a collective.
    ``buckets``: padded token counts or ``(padded, real)`` pairs."""
    if _RES is None:
        return
    pairs = [b if isinstance(b, tuple) else (b, b) for b in buckets]
    for s in _SITES.values():
        for padded, _real in pairs:
            r = s.resolve(padded, graph_mode="capture")
            logger.debug("gemm_fusion capture bucket %s %d -> %s/%s", s.name, padded, r.vid, r.reason)
    _RES.quiesce_and_reset()


def after_graph_capture() -> None:
    if _RES is not None:
        _RES.quiesce_and_reset()


# ----------------------------------------------------------- per-step check
def _poison_local() -> int:
    return int(_RES.inner.poison_local())


def check_step() -> None:
    """Per-step host poison + uniformity check. Call at the END of every model
    step, outside capture, after the step's sync point, on EVERY rank.

    Reads every fused plan's device poison word (aiter ``poison_local``: 4-byte
    D2H per plan), and all-reduces ``[-poison, digest, -digest]`` (MIN) so all
    ranks agree on (a) any poison anywhere, (b) that every rank made identical
    fused/native decisions this step. On either, runs the collective recovery and
    raises. A step that made no fused call and has no prepared plan does nothing
    (uniform across ranks).

    LIMIT: this is a collective at the END of the step, so it only sees steps that
    COMPLETE on every rank. A non-uniform decision whose collectives do not match (one rank in
    the native RCCL all_to_all while its peers wait in the fused CCO kernel) blocks the step
    mid-way and never reaches this call: that is FAIL-STOP at the process-group timeout, not a
    recovered error. The digest only catches decision differences among ranks that still
    finish the step (and device poison on completing ranks).
    """
    global _STEP_DIGEST, _STEP_CALLS
    if _RES is None or _PHASE != "ready":
        return
    if _BROKEN is not None:
        raise GemmFusionPoisonError(f"gemm_fusion disabled after failed recovery: {_BROKEN}")
    if not _PLANS and _STEP_CALLS == 0:
        return
    p = _poison_local()
    digest = int.from_bytes(_STEP_DIGEST.digest(), "little") & 0x7FFFFFFF
    reduce = getattr(_RES, "_reduce_min", None)
    if reduce is None:  # single rank
        any_poison, uniform = p != 0, True
    else:
        out = reduce([-int(p != 0), digest, -digest])
        any_poison, uniform = out[0] != 0, out[1] == -out[2]
    _reset_step()
    if any_poison or not uniform:
        what = "device poison" if any_poison else "cross-rank decision mismatch"
        logger.error("gemm_fusion check_step: %s (local poison=%d) -> recover", what, p)
        recover()
        raise GemmFusionPoisonError(
            f"gemm_fusion {what}; outputs of this step are invalid; plans re-initialised"
        )


def _reset_step() -> None:
    global _STEP_DIGEST, _STEP_CALLS
    _STEP_DIGEST = hashlib.blake2b(digest_size=4)
    _STEP_CALLS = 0


# ---------------------------------------------------------------- recovery
def recover(timeout_s: Optional[float] = None) -> None:
    """Collective recovery: device sync -> drain -> barrier -> re-init -> barrier.

    EVERY SP rank must call it (``check_step`` makes the decision collective).
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
            _RES.quiesce_and_reset()
        except BaseException as e:  # noqa: BLE001
            err.append(e)

    t = threading.Thread(target=_run, name="gemm_fusion_recover", daemon=True)
    t.start()
    t.join(timeout_s)
    if t.is_alive():
        _BROKEN = f"recover timeout after {timeout_s}s"
        raise GemmFusionRecoveryTimeout(_BROKEN)
    if err:
        _BROKEN = f"recover failed: {err[0]!r}"
        raise err[0]


def shutdown(timeout_s: Optional[float] = None) -> None:
    """Release the CCO resources, best effort and BOUNDED.

    ``release()`` does a device sync and a barrier. At exit after a failed start-up (a peer
    rank already dead, a HIP error, the runner never built) that barrier could block process
    exit or raise and hide the original failure, so it runs in a daemon thread with a timeout
    and never raises: problems are logged. Normal shutdown (all ranks alive) is unchanged.
    """
    global _RES, _PHASE
    res, _RES = _RES, None
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
            logger.error("gemm_fusion shutdown: release() did not finish in %.0fs (peer gone?); "
                         "continuing", timeout_s)
        elif err:
            logger.error("gemm_fusion shutdown: release() failed: %r (continuing)", err[0])
    _PLANS.clear()
    _SITES.clear()
    _PHASE = "pre"
