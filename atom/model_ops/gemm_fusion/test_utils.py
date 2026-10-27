# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Test and evidence utilities for ``atom.model_ops.gemm_fusion``.

NEVER imported by a runtime module. The runtime package keeps only what serving needs; everything
that exists only for CPU unit tests or GPU evidence runs lives here:

  1. test state helpers      reset_for_tests, set_phase, inject_aiter, init_with_resources,
                             InjectedResourcesWrapper, decision log (DECISIONS)
  2. fault injection         install_faults / install_from_env / uninstall_faults: E-1, E-3, E-4,
                             E-5, E-7 seams, applied by MONKEYPATCHING the runtime (the runtime has
                             no fault branches)
  3. debug helpers           binding_key, fallback_reason, shared-window alias check
  4. direct-bind site        DirectGemmFusionSite (binds the caller's own tensors; the model path
                             cannot use it, its per-call tensors are not address-stable)
  5. golden-harness adapter  ``run_cases --candidate atom.model_ops.gemm_fusion.test_utils``
  6. MiniMax-M3 smoke model  m3_config / m3_vl_config / make_model_dir / serve_args / serve_env

FAULT INJECTION IN SPAWNED WORKERS. The seams patch module attributes of the process that calls
``install_faults``. ATOM's ModelRunner workers are spawned processes, so the launch script must
call ``install_from_env()`` in them: put the call at the TOP LEVEL of the run script (a spawned child
re-imports ``__main__``), or in a ``sitecustomize.py`` on ``PYTHONPATH``. Setting
``ATOM_GEMM_FUSION_FAULT`` alone does nothing any more.
"""

import argparse
import contextlib
import json
import logging
import os
import sys
from types import SimpleNamespace

import torch

from . import consensus, hooks
from . import stages as S
from .context import AttnFusionCtx
from .resources import FusionResourcesWrapper, fusion_resources_cls
from .site import (
    GemmFusionBindError,
    GemmFusionSite,
    Resolution,
)

logger = logging.getLogger("atom.gemm_fusion")

# =============================================================================
# 1. test state helpers
# =============================================================================
DECISIONS: list = (
    []
)  # bounded dispatch evidence, newest last (the runtime keeps no decision history)
_MAX_DECISIONS = 4096


def reset_for_tests(res=None, profile="test-profile"):
    """Reset the runtime's module state. ``profile`` is what ``hooks.profile()`` returns while
    ``res`` is set (the harness passes the one it loaded; unit tests with fakes keep the default).
    """
    uninstall_faults()
    hooks._SITES.clear()
    hooks._PLANS.clear()
    hooks._TOKEN_IDS.clear()
    DECISIONS.clear()
    hooks._KV = None
    hooks._RES = res
    hooks._PROFILE = profile if res is not None else ""
    hooks._PHASE = "ready" if res is not None else "pre"
    hooks._BROKEN = None
    hooks._PERIOD = 14
    hooks._STEP_T = None
    hooks._reset_check_state()
    install_decision_log()
    install_from_env()  # env-selected seams are re-applied, as the old env-driven seams were


def set_phase(p):
    hooks._PHASE = p


def inject_aiter(monkeypatch, fake):
    """Make ``aiter.gemm_fusion`` resolve to ``fake`` (a stand-in module/namespace)."""
    monkeypatch.setitem(sys.modules, "aiter.gemm_fusion", fake)


def init_with_resources(config, res, sp_group=None, kv_publisher=False):
    """``hooks.init_gemm_fusion`` with ``res`` as the resources object (the runtime builds its own
    ``FusionResourcesWrapper`` and takes no injected one)."""
    from . import resources as R

    orig_cls, orig_world = R.FusionResourcesWrapper, S.sp_world
    R.FusionResourcesWrapper = lambda *a, **k: res
    S.sp_world = lambda: getattr(
        res, "world", 1
    )  # the real one needs the distributed runtime
    try:
        hooks.init_gemm_fusion(config, sp_group, kv_publisher=kv_publisher)
    finally:
        R.FusionResourcesWrapper, S.sp_world = orig_cls, orig_world


class InjectedResourcesWrapper(FusionResourcesWrapper):
    """``FusionResourcesWrapper`` whose pieces can be injected (``inner``, ``reduce_min``,
    ``reduce_max``, ``rank``, ``pg``, ``arch``, ``cu``). The runtime constructor takes only
    ``(sp_group, world, profile)``.
    """

    def __init__(
        self,
        sp_group=None,
        world=1,
        *,
        rank=None,
        pg=None,
        inner=None,
        reduce_min=None,
        reduce_max=None,
        device=None,
        arch=None,
        cu=None,
        profile=None,
    ):
        gf = S.aiter_gf()
        self.group = sp_group
        self.world = world
        if rank is None:
            rank = sp_group.rank_in_group
        if pg is None and sp_group is not None:
            pg = getattr(sp_group, "cpu_group", None) or getattr(
                sp_group, "device_group", None
            )
        self.rank, self.pg = rank, pg
        self.cpu_pg = getattr(sp_group, "cpu_group", None)
        if reduce_max is not None:
            self.reduce_max = reduce_max  # instance attribute shadows the method
        if inner is None:
            device = torch.cuda.current_device() if device is None else device
            caps = gf.detect_capabilities(sp_group)
            if profile is None:
                profile = hooks.profile()
            inner = fusion_resources_cls(gf)(
                rank=rank,
                world=world,
                group=pg,
                device=device,
                caps=caps,
                profile=profile,
            )
        self.inner = inner
        self.caps = frozenset(getattr(inner, "caps", ()))
        self._arch = arch or gf.detect_arch()
        self._cu = cu if cu is not None else gf.detect_cu()
        if reduce_min is None and world > 1:
            reduce_min = consensus.torch_min_reducer(
                pg, device=torch.device("cuda", torch.cuda.current_device())
            )
        self._reduce_min = reduce_min


def install_decision_log():
    """Keep the last ``_MAX_DECISIONS`` ``Decision`` records in ``DECISIONS`` (the runtime keeps
    no decision history: ``GemmFusionSite._decision`` only builds the record)."""
    if any(p[1] == "_decision" for p in _PATCHES):
        return
    orig = GemmFusionSite._decision

    def decision(self, *a, **kw):
        d = orig(self, *a, **kw)
        DECISIONS.append(d)
        if len(DECISIONS) > _MAX_DECISIONS:
            del DECISIONS[: len(DECISIONS) - _MAX_DECISIONS]
        return d

    _patch(GemmFusionSite, "_decision", decision)


# =============================================================================
# 2. fault injection (monkeypatching; the runtime has no fault branches)
# =============================================================================
FAULT_ENV = "ATOM_GEMM_FUSION_FAULT"
RECOVER_TIMEOUT_ENV = "ATOM_GEMM_FUSION_RECOVER_TIMEOUT_S"
SHUTDOWN_TIMEOUT_ENV = "ATOM_GEMM_FUSION_SHUTDOWN_TIMEOUT_S"
CHECK_ALIAS_ENV = "ATOM_GEMM_FUSION_CHECK_ALIAS"
"""Kinds (``kind[:sp_rank[:arg]]``, comma separated; empty or ``*`` rank = every rank):

  error_word[:r]       rank r's device error word reads as set (exercises the HOST agree / reset /
                       abort-ids path of check_step; a real error word needs a kernel-side forced
                       timeout or tag-mismatch build).
  op_raise[:r]         the fused op raises on rank r (E-4: one-rank exception).
  rank_native[:r]      rank r takes a native decision where others fuse. This DELIBERATELY violates
                       cross-rank uniformity (E-3). Expected outcome: FAIL-STOP. Rank r enters the
                       native RCCL all_to_all while its peers wait in the fused CCO kernel, so the
                       step cannot complete and the step-end check in check_step is never reached;
                       the run ends at the process-group timeout. E-3 passes when the failure is
                       bounded, every rank exits non-zero, and no wrong output is returned silently.
  slow_remainder[:r:n] after the native remainder of a 1-stage (large-M) call, rank r spins the GPU
                       ~n microseconds (E-5: WAR window).
  recover_skip[:r]     rank r skips recover() (exercises the recovery timeout).
  no_error_check       disable the device error check (E-1 NEGATIVE CONTROL: proves the check has
                       power; never set in production)."""

_PATCHES: list = []  # (obj, attr, original, had_own_attr)
_FAULTS: dict = {}  # kind -> (rank or None, arg or None)
_FIRED: set = set()  # fault kinds that have taken effect at least once in this process
_STATS = {"alias_checks": 0}


def stats():
    """What this process's seams did: installed fault specs, kinds that fired, alias checks run.
    A spawned worker can log this to PROVE the seams were installed and executed."""
    return {
        "pid": os.getpid(),
        "faults": dict(_FAULTS),
        "fired": sorted(_FIRED),
        "alias_checks": _STATS["alias_checks"],
        "patches": [
            f"{p[0].__name__}.{p[1]}" if hasattr(p[0], "__name__") else p[1]
            for p in _PATCHES
        ],
    }


def _patch(obj, attr, value):
    had = attr in getattr(obj, "__dict__", {})
    orig = obj.__dict__[attr] if had else getattr(obj, attr, None)
    _PATCHES.append((obj, attr, orig, had))
    setattr(obj, attr, value)


def parse_faults(spec):
    out = {}
    for item in filter(None, (spec or "").split(",")):
        parts = item.strip().split(":")
        rank = parts[1] if len(parts) > 1 and parts[1] not in ("", "*") else None
        out[parts[0]] = (
            None if rank is None else int(rank),
            parts[2] if len(parts) > 2 else None,
        )
    return out


def _my_rank():
    return getattr(hooks._RES, "rank", 0) if hooks._RES is not None else 0


def fault_active(kind):
    """-> arg string ('' if none) when the fault applies to this process's SP rank, else None."""
    ent = _FAULTS.get(kind)
    if ent is None:
        return None
    rank, arg = ent
    if rank is not None and rank != _my_rank():
        return None
    if kind not in _FIRED:
        _FIRED.add(kind)
        logger.warning(
            "gemm_fusion test_utils: fault %r FIRED (pid=%d sp_rank=%d arg=%r)",
            kind,
            os.getpid(),
            _my_rank(),
            arg,
        )
    return arg or ""


def _spin_gpu_us(us):
    """Keep the current stream busy ~``us`` microseconds (E-5)."""
    try:
        torch.cuda._sleep(int(us * 1000))  # cycles; order-of-magnitude is enough
    except Exception:  # noqa: BLE001 - no _sleep on this torch: sync instead
        torch.cuda.synchronize()


def _install_error_word():
    orig = FusionResourcesWrapper.error_word

    def error_word(self, step_idx):
        if fault_active("error_word") is not None:
            return 1
        return orig(self, step_idx)

    _patch(FusionResourcesWrapper, "error_word", error_word)


def _install_no_error_check():
    from atom.model_ops import gemm_fusion as pkg

    orig = hooks.check_step

    def check_step(req_ids=()):
        if fault_active("no_error_check") is not None:
            return None
        return orig(req_ids)

    _patch(hooks, "check_step", check_step)
    if hasattr(pkg, "check_step"):  # the runner calls it through the package
        _patch(pkg, "check_step", check_step)


def _install_recover_skip():
    orig = hooks.recover

    def recover(timeout_s=None):
        if fault_active("recover_skip") is not None:
            return None
        return orig(timeout_s)

    _patch(hooks, "recover", recover)


def _install_op_raise():
    from . import ops

    orig = ops.run_qkv_attention

    def run_qkv_attention(attn, x, x_scale, positions):
        if fault_active("op_raise") is not None:
            try:
                raise RuntimeError("gemm_fusion fault seam: op_raise")  # E-4
            except Exception:
                hooks.recover()  # what the real op does on any host exception
                raise
        return orig(attn, x, x_scale, positions)

    _patch(ops, "run_qkv_attention", run_qkv_attention)


def _install_rank_native():
    orig = GemmFusionSite.__dict__["decide_path"].__func__

    def decide_path(**kw):
        path, why = orig(**kw)
        if path == "fused" and fault_active("rank_native") is not None:
            return "native", "fault_rank_native"  # deliberately non-uniform (E-3)
        return path, why

    _patch(GemmFusionSite, "decide_path", staticmethod(decide_path))


def _install_slow_remainder():
    orig = GemmFusionSite._remainder_after_exchange

    def remainder(self, r, outs, ctx, consumer):
        out = orig(self, r, outs, ctx, consumer)
        delay = fault_active("slow_remainder")
        if delay is not None:  # widen the WAR window on this rank
            _spin_gpu_us(int(delay or "2000"))
        return out

    _patch(GemmFusionSite, "_remainder_after_exchange", remainder)


_FAULT_INSTALLERS = {
    "error_word": _install_error_word,
    "no_error_check": _install_no_error_check,
    "recover_skip": _install_recover_skip,
    "op_raise": _install_op_raise,
    "rank_native": _install_rank_native,
    "slow_remainder": _install_slow_remainder,
}


def install_faults(spec=None):
    """Apply the seams named in ``spec`` (default: the ``ATOM_GEMM_FUSION_FAULT`` env var).
    Idempotent per kind; undone by ``uninstall_faults``. Returns the parsed spec."""
    spec = os.environ.get(FAULT_ENV, "") if spec is None else spec
    parsed = parse_faults(spec)
    unknown = [k for k in parsed if k not in _FAULT_INSTALLERS]
    if unknown:
        raise ValueError(
            f"unknown gemm_fusion fault kind(s) {unknown}; known: {sorted(_FAULT_INSTALLERS)}"
        )
    for kind, ent in parsed.items():
        if kind not in _FAULTS:
            _FAULT_INSTALLERS[kind]()
            logger.warning(
                "gemm_fusion test_utils: fault %r installed (pid=%d rank=%s arg=%r)",
                kind,
                os.getpid(),
                "*" if ent[0] is None else ent[0],
                ent[1],
            )
        _FAULTS[kind] = ent
    return parsed


def install_from_env():
    """Apply env-selected seams: ``ATOM_GEMM_FUSION_FAULT``, the recover/shutdown timeouts, and the
    shared-window alias check (``ATOM_GEMM_FUSION_CHECK_ALIAS=1``). Call it at the top of a launch
    script so spawned workers get it (see the module docstring)."""
    installed = []
    if os.environ.get(FAULT_ENV):
        install_faults()
        installed.append(f"faults={os.environ[FAULT_ENV]}")
    for env, attr in (
        (RECOVER_TIMEOUT_ENV, "RECOVER_TIMEOUT_S"),
        (SHUTDOWN_TIMEOUT_ENV, "SHUTDOWN_TIMEOUT_S"),
    ):
        if os.environ.get(env):
            _patch(hooks, attr, float(os.environ[env]))
            installed.append(f"{attr}={os.environ[env]}")
    if os.environ.get(CHECK_ALIAS_ENV) == "1":
        install_alias_check()
        installed.append("alias_check")
    logger.warning(
        "gemm_fusion test_utils: install_from_env ran (pid=%d): %s",
        os.getpid(),
        ", ".join(installed) or "nothing selected by the environment",
    )


def uninstall_faults():
    """Undo every patch made by this module (faults, decision log, timeouts, alias check)."""
    while _PATCHES:
        obj, attr, orig, had = _PATCHES.pop()
        if had:
            setattr(obj, attr, orig)
        else:
            try:
                delattr(obj, attr)
            except AttributeError:
                pass
    _FAULTS.clear()
    _FIRED.clear()
    _STATS["alias_checks"] = 0
    global _ALIAS_INSTALLED
    _ALIAS_INSTALLED = False


# =============================================================================
# 3. debug helpers
# =============================================================================
def fallback_reason(site, ctx, x, x_scale, n_fused=2):
    """Why the fused prefix cannot run for this call (None = it can): a view of
    ``GemmFusionSite.decide_path`` over the call's tensors (``_prepare`` uses the same function).
    """
    fp8, ok = site.input_facts(x, x_scale)
    path, why = site.decide_path(
        resolution=Resolution(object(), n_fused, "", ""),
        is_dummy_run=ctx.is_dummy_run,
        x_dtype_is_fp8=fp8,
        x_scale_ok=ok,
        x_k_ok=x.shape[-1] == site.linear.input_size,
    )
    return None if path == "fused" else why


def binding_key(site, st, state, x_scale=None):
    """Named data_ptr of every tensor aiter's small plan keys its binding on (x_scale, w_scale,
    norms, cos/sin, positions, slot_mapping, q out, caches, k/v/index_q outs, index_cache) plus the
    row count. Diffing the key at prepare against the key at run names the tensor whose address
    moved."""
    meta = site.build_meta(st, state)
    ptr = site._ptr
    d = {
        "x_scale": ptr(x_scale if x_scale is not None else st.xs),
        "w_scale": ptr(site.linear.weight_scale),
    }
    for k in (
        "q_norm_w",
        "k_norm_w",
        "index_q_norm_w",
        "index_k_norm_w",
        "cos_cache",
        "sin_cache",
        "positions",
        "slot_mapping",
        "k_cache",
        "v_cache",
        "k_scale",
        "v_scale",
        "index_cache",
    ):
        if k in meta:
            d[k] = ptr(meta[k])
    for k, v in st.outs.items():
        d["out_" + k] = ptr(v)
    d["rows"] = st.m
    return d


def check_no_window_alias(named_tensors):
    """C-S4: after the large-M consumers have run INSIDE the op, no tensor the op returns, and none
    it leaves stored on ``attn.impl`` (e.g. ``_index_q``), may lie inside a window: it would be
    overwritten by the next large launch of ANY plan on this rank (under the shared arena layout
    every large plan's ``recv_view`` aliases the same R window).

    ``named_tensors``: iterable of ``(label, tensor-or-None)``. Delegates to aiter's own helper,
    ``aiter.ops.gemm_fusion.sp_qkv.test_utils.assert_no_window_alias(res, obj, what=, force=True)``,
    which walks nested dicts/lists/tuples, honours strides, and raises ``WindowAliasError``; that is
    re-raised as ``GemmFusionBindError``."""
    import importlib

    aiter_test_utils = importlib.import_module(
        "aiter.ops.gemm_fusion.sp_qkv.test_utils"
    )

    items = {n: t for n, t in named_tensors if t is not None and hasattr(t, "data_ptr")}
    try:
        aiter_test_utils.assert_no_window_alias(
            hooks._RES.inner, items, what="gemm_fusion_qkv_attention", force=True
        )
    except Exception as e:
        raise GemmFusionBindError(f"gemm_fusion: tensor aliases a window: {e}") from e


_ALIAS_INSTALLED = False


def install_alias_check():
    """Run ``check_no_window_alias`` after every fused op call: the op's output, its scale and
    ``attn.impl._index_q``. Debug only (patches ``ops.run_qkv_attention``)."""
    global _ALIAS_INSTALLED
    if _ALIAS_INSTALLED:
        return
    from . import ops

    orig = ops.run_qkv_attention

    def run_qkv_attention(attn, x, x_scale, positions):
        result = orig(attn, x, x_scale, positions)
        (out, out_scale), _decision = result
        _STATS["alias_checks"] += 1
        if _STATS["alias_checks"] == 1:
            logger.warning(
                "gemm_fusion test_utils: alias check EXECUTED for the first time (pid=%d)",
                os.getpid(),
            )
        check_no_window_alias(
            [
                ("out", out),
                ("scale", out_scale),
                ("impl._index_q", getattr(attn.impl, "_index_q", None)),
            ]
        )
        return result

    _patch(ops, "run_qkv_attention", run_qkv_attention)
    _ALIAS_INSTALLED = True


# =============================================================================
# 4. direct-bind site (golden harness only)
# =============================================================================
class DirectGemmFusionSite(GemmFusionSite):
    """A site whose r17/r18 plans bind the CALLER's own x_scale / positions / slot_mapping / cache
    tensors at prepare time, with no per-call staging copies. Valid only where the caller guarantees
    stable addresses (the golden harness cases; graph static buffers). The model path cannot use it:
    ATOM's per-call tensors are not address-stable, which is why the runtime site stages.

    A moved x_scale / positions / slot_mapping is refreshed by a device-side copy (decision label
    ``direct+copy``); moved KV caches are reported as "not prepared" at prepare time so
    ``agree_prepared`` downgrades the bucket to native on ALL ranks.

    An r19 plan (``plan.abi >= 3``) binds none of x_scale / positions / slot_mapping: they are
    per-launch kernel arguments, so the golden case's tensors are simply passed on every launch
    (decision label ``per_launch``, no copy) and only the KV caches of ``direct["state"]`` are bound.
    """

    bind_mode = "direct"

    def __init__(self, *args, direct_provider=None, register=True, **kw):
        super().__init__(*args, **kw)
        # optional callable(tokens) -> {"x_scale","positions","slot_mapping","state"}
        self.direct_provider = direct_provider
        if not register:
            hooks._SITES.pop(self.name, None)

    def _stage(self, tokens, device, dtype):
        st = super()._stage(tokens, device, dtype)
        if not hasattr(st, "ptrs"):
            st.ptrs, st.rebound, st.own_token_arrays = None, False, False
        return st

    def _live_ptrs(self, x_scale, positions, state):
        """Addresses the plan binds (small-M); compared per call to detect moved inputs."""
        ptr = self._ptr
        return (
            ptr(x_scale),
            ptr(positions),
            ptr(state["slot_mapping"]),
            ptr(state["k_cache"]),
            ptr(state.get("v_cache_shuffle", state["v_cache"])),
            ptr(state.get("k_scale_view", state.get("k_scale"))),
            ptr(state.get("v_scale_view", state.get("v_scale"))),
        )

    def prepare_bucket(self, tokens, device, kv_cache_data, out_dtype, direct=None):
        """Bind the plan for one bucket against the caller's tensors OUTSIDE graph capture.
        ``direct`` (else ``direct_provider(tokens)``): ``{"x_scale","positions","slot_mapping",
        "state"}``, the caller's stable tensors, bound as-is.

        PADDED BUCKETS (the caller's positions / slot_mapping are shorter than ``tokens`` = T): the
        small-M kernel reads T entries, so the site's own T-length arrays (position 0 / slot -1 in the
        pad rows) are bound instead and refreshed by copy on every call; x_scale stays direct.
        """
        r = self.resolve(tokens, "eager")
        if r.plan is None:
            return False
        if r.n_fused < 2:
            return self._prepare_one_stage(r)
        st = self._stage(tokens, device, out_dtype)
        if direct is None and self.direct_provider is not None:
            direct = self.direct_provider(tokens)
        if direct is None:
            return (
                st.prepared
            )  # no stable tensors to bind; agree_prepared() downgrades uniformly
        if st.prepared:
            # Already bound. The KV cache/scale addresses are fixed at KV allocation: if the
            # caller's differ from the ones bound, the plan would write a stale cache.
            live = self._live_ptrs(
                direct["x_scale"], direct["positions"], direct["state"]
            )
            return live[3:] == st.ptrs[3:]
        state = direct["state"]
        if self.per_launch(r.plan):
            # r19: nothing of x_scale / positions / slot_mapping is bound (per-launch kernel
            # arguments): keep the site's own conforming arrays for prepare(), pass the caller's
            # on every launch.
            st.own_token_arrays = False
        else:
            st.xs = direct["x_scale"]
            st.own_token_arrays = (
                int(direct["positions"].shape[-1]) < tokens
                or int(direct["slot_mapping"].shape[-1]) < tokens
            )
            if not st.own_token_arrays:
                st.pos, st.slot = direct["positions"], direct["slot_mapping"]
            # else: keep the T-length arrays _stage allocated (the caller's would be read OOB)
        ptrs = list(self._live_ptrs(st.xs, st.pos, state))
        ptrs[2] = self._ptr(st.slot)  # the slot array the plan is actually bound to
        st.ptrs = tuple(ptrs)
        return self._bind_plan(r, tokens, st, state, device, kv_cache_data)

    def _prepare(self, ctx, x, x_scale):
        r, reason, state, st = super()._prepare(ctx, x, x_scale)
        if st is not None:
            if self.per_launch(r.plan):
                st.rebound = False  # nothing is bound on the caller's arrays: they are per launch
            else:
                # x_scale / positions / slot_mapping may have moved since prepare: refreshed by
                # copy in _stage_inputs. The KV addresses were checked at prepare, not here.
                live = self._live_ptrs(x_scale, ctx.positions, state)
                st.rebound = (
                    bool(getattr(st, "own_token_arrays", False))
                    or live[:3] != st.ptrs[:3]
                )
        return r, reason, state, st

    def _stage_inputs(self, x_scale, ctx, state, st):
        if getattr(
            st, "rebound", False
        ):  # unmoved: st.xs/pos/slot ARE the caller's tensors
            super()._stage_inputs(x_scale, ctx, state, st)

    def _bind_label(self, r, path, st):
        if path == "fused" and r.n_fused < 2:
            return "raw"
        if path == "fused" and self.per_launch(r.plan):
            return "per_launch"
        bm = "direct"
        if path == "fused" and st is not None and getattr(st, "rebound", False):
            bm += "+copy"  # rank-local
        return bm


def run_prologue(site, x, x_scale, ctx, native_full, native_after_exchange, out_dtype):
    """GEMM + exchange + norm/rope/cache only (no attention), for the golden harness.

    Returns ``(kind, payload, Decision)``: ``("outs", dict)`` when the fused prefix covered the
    norm/rope/cache stage, else ``("native", result)`` from ``native_full(x, x_scale)`` /
    ``native_after_exchange(packed)``."""
    r, reason, state, st = site._prepare(ctx, x, x_scale)
    if r.plan is None:
        return (
            "native",
            native_full(x, x_scale),
            site._decision(ctx, r, reason, "native"),
        )
    outs = site._execute(r, x, x_scale, ctx, state, st, out_dtype)
    if r.n_fused >= 2:
        return "outs", outs, site._decision(ctx, r, reason, "fused", st)
    out = site._remainder_after_exchange(r, outs, ctx, native_after_exchange)
    return "native", out, site._decision(ctx, r, reason, "fused", st)


# =============================================================================
# 5. golden-harness candidate adapter
# =============================================================================
# Use as ``--candidate`` module: it exposes ``qkv_proj_all2all_norm_rope_cache(case, comm)`` and
# drives the REAL site layer (resolve -> consensus -> aiter ``match_prefix`` -> ``Plan.prepare/run``)
# through ``run_prologue`` for the golden case shapes. No MiniMax-M3 weights are loaded.
#
# Scope (state it when citing results): this exercises the site/plan layer on the GEMM -> exchange
# -> norm/rope/cache prologue, which is exactly what the golden case compares. It does NOT run
# attention, the custom op registry, or ``MiniMaxM3*Attention.forward``; those need model weights.
#
# Profile OFF (no resources configured) => every call takes the site's native branch, which here is
# the golden harness's own baseline operators. Profile ON: call ``configure(...)`` once per process,
# then ``prepare_cases(cases, comm)`` for every case BEFORE the first measured call (all plans must be
# resolved on all ranks before ``finalize``).
#
# RESULT CONTRACT: the result is returned as the consumer produced it, with no copy, matching the
# golden harness's native behaviour and timing. On a 1-stage large-M call a returned tensor may
# therefore be a BORROWED view of the shared R window, valid only until the next large launch of ANY
# plan on this rank. Compare or clone each large-M result before the next large launch. The model
# path's aliasing check (C-S4) does not apply here.
#
# PAD-M (R16): ``prepare_cases`` prepares the BUCKET cases only (a case passed to it is prepared at its
# own token count, as an exact bucket, and never pads). A later call for a case whose token count was
# NOT prepared is served the way the engine serves it: through the site of the prepared bucket that
# aiter's ``bucket_for_m`` names for its real rows (one layer = one site, every m runs through it),
# with the kernel handed the case's real rows only. ``padded_bucket(case)`` says which bucket that is;
# the decision records ``m_real`` and ``bucket_m``. A case's own weights / norms / rope tables / index
# cache are what run (the bucket's site is re-pointed at them for the call; aiter then makes a second
# binding for those pointers, which is fine eagerly and refused inside a graph capture).
#
# Evidence: ``last_decision()``, ``decisions()`` (selected vid, fused prefix, fallback reason).
WORLD = 4
_SITES: dict = {}
_LAST = None


class _HarnessCtx(AttnFusionCtx):
    state = None

    @property
    def is_dummy_run(self):
        return False

    def prologue_state(self):
        return self.state


def configure(profile, *, rank, world=WORLD, pg=None, resources=None):
    """Turn fusion ON for this process (collective: builds the comm)."""
    gf = S.aiter_gf()
    gf.load_profile(profile)
    reset_for_tests(
        # profile reaches FusionResources too (window budget / VMM heap), not only the matcher
        resources
        or InjectedResourcesWrapper(None, world, rank=rank, pg=pg, profile=profile),
        profile=profile,
    )
    _SITES.clear()


def _maybe_configure_from_env(comm):
    prof = os.environ.get("GEMM_FUSION_HARNESS_PROFILE", "")
    if prof and hooks.resources() is None:
        configure(prof, rank=comm.rank)


def _mode(case):
    m = case.spec.mode
    return (
        "dense" if m == "qkv_dense" else ("indexer" if m == "qkv_indexer" else "reuse")
    )


def _prologue_supported(mode, impl):
    """The answer the REAL impl gives for this mode.

    * indexer / reuse -> ``SparseMHAPagedAttentionImpl.gemm_fusion_prologue_supported`` is
      unconditionally True.
    * dense -> ``PagedAttentionImpl.gemm_fusion_prologue_supported``: a rotary embedding and a
      Gemma-style q_norm / a k_norm, and nothing else. It does NOT read ``ATOM_FORCE_ATTN_TRITON`` /
      sliding window / head_dim: those only pick the decode kernel that follows, and the Gemma
      norm-rope-cache branch writes the same SHUFFLE cache either way. The real method cannot be
      called here (it imports the model stack, and its ``isinstance(q_norm, GemmaRMSNorm)`` rejects
      the golden case's plain norm objects), so the same conditions are evaluated on the stand-in; a
      test pins them to the source text. The golden dense case is, by construction, the Gemma/Triton
      chain (``triton_norm_rope_cache_v1``), so the norm-type condition holds.
    """
    if mode != "dense":
        return True
    return bool(
        impl.rotary_emb is not None
        and impl.q_norm is not None
        and impl.k_norm is not None
    )


def _caches_state(case, slot_mapping):
    c = case.caches
    return {
        "k_cache": c["k_cache"],
        "v_cache": c["v_cache"],  # golden v_cache is already the SHUFFLE view
        "k_scale": c["k_scale"],
        "v_scale": c["v_scale"],
        "slot_mapping": slot_mapping,
    }


@contextlib.contextmanager
def _kv_page(page):
    """The dense layers' KV page comes from ``config.kv_cache_block_size``; the golden case carries
    its own page, so the stage constructor reads it from here while a site is built."""
    orig = S._kv_block_size
    S._kv_block_size = lambda: int(page)
    try:
        yield
    finally:
        S._kv_block_size = orig


class _GoldenSite(DirectGemmFusionSite):
    """Golden-harness site. The harness's weight IS the layout the plan consumes (16x16-preshuffled),
    but golden tensors carry no ``is_shuffled`` marker (ATOM's loader sets it), so the layout verdict
    is taken from the harness contract while the shape checks (K, N, scale rows) still run.
    """

    def _static_weight_reason(self):
        lin = self.linear
        proxy = SimpleNamespace(
            output_size=lin.output_size,
            input_size=lin.input_size,
            weight=SimpleNamespace(is_shuffled=True, shape=lin.weight.shape),
            weight_scale=lin.weight_scale,
        )
        saved, self.linear = lin, proxy
        try:
            return super()._static_weight_reason()
        finally:
            self.linear = saved


def _pad_entry(case, mode):
    """The ``_SITES`` entry of the PREPARED bucket whose plan serves this case's rows (its padded
    token count is that bucket's own, or pad-M maps it there: aiter's pure ``bucket_for_m``, through
    that site's own prepared buckets), or None: the phase is not ready, nothing prepared serves it, or
    the plan cannot launch fewer rows.
    """
    t = case.spec.padded_tokens
    for (m, _tokens), ent in list(_SITES.items()):
        if m != mode:
            continue
        r = ent[0].raw_resolution(t, "eager")
        if r is not None and r.plan is not None:
            return ent
    return None


def padded_bucket(case):
    """``(bucket_m, site_name)`` of the prepared bucket this case would run in (pad-M, or the bucket
    whose padded token count it shares), or None: the case has a site of its own, nothing prepared
    serves it, or the phase is not ready. Pure: no collective.
    """
    mode = _mode(case)
    if (mode, case.spec.tokens) in _SITES:
        return None
    ent = _pad_entry(case, mode) if hooks.phase() == "ready" else None
    if ent is None:
        return None
    t = case.spec.padded_tokens
    r = ent[0].raw_resolution(t, "eager")
    return (r.bucket_tokens or t) // WORLD, ent[0].name


def padded_call_audit(case):
    """Host-only (no launch): what a padded call hands aiter's plan for this case, and whose tensors
    they are. ``None`` when the case is not served by a prepared bucket's site; else
    ``{"site": name, "bucket_m": m, "tensors": {name: "case" | "other"}}`` over the KV cache, the
    scales and the index cache the plan WRITES: "case" = that exact tensor is the case's own, "other"
    = the plan would write somewhere else. Run it on the node to see which tensor is wrong without
    guessing; it applies the same adoption the call does."""
    mode = _mode(case)
    if (mode, case.spec.tokens) in _SITES or hooks.phase() != "ready":
        return None
    ent = _pad_entry(case, mode)
    if ent is None:
        return None
    site, _holder, attn = ent
    _adopt_case_tensors(site, attn, case, mode)
    ctx = _ctx(case, attn)
    r, _reason, state, st = site._prepare(ctx, case.x, case.x_scale)
    if r.plan is None or st is None:
        return {"site": site.name, "bucket_m": 0, "tensors": {}, "native": r.reason}
    meta = site.build_meta(st, state, ctx)
    own = case.caches or {}
    tensors = {}
    for name in ("k_cache", "v_cache", "k_scale", "v_scale", "index_cache"):
        got = meta.get(name)
        if got is None:
            continue
        ref = own.get(name)
        tensors[name] = (
            "case" if ref is not None and got.data_ptr() == ref.data_ptr() else "other"
        )
    return {
        "site": site.name,
        "bucket_m": r.bucket_tokens // WORLD,
        "tensors": tensors,
    }


def _adopt_case_tensors(site, attn, case, mode):
    """Point the bucket's site at THIS case's layer tensors (weight, scales, norms, rope tables, index
    cache): the golden cases are independent tensors, whereas a layer in the engine has one set. The
    plan's pointer binding is keyed on them, so aiter makes a new binding for tensors it has not seen.
    """
    impl = attn.impl
    impl.rotary_emb, impl.q_norm, impl.k_norm = case.rotary, case.q_norm, case.k_norm
    if mode != "dense":
        impl.index_q_norm, impl.index_k_norm = case.index_q_norm, case.index_k_norm
        impl.index_cache = (case.caches or {}).get("index_cache")
    lin = site.linear
    site.linear = SimpleNamespace(
        output_size=lin.output_size,
        input_size=lin.input_size,
        weight=case.weight,
        weight_scale=case.weight_scale,
    )


def _site_for(case, comm):
    mode = _mode(case)
    key = (mode, case.spec.tokens)
    ent = _SITES.get(key)
    if ent is None and hooks.phase() == "ready":
        # a token count nobody prepared: the engine serves it through the site of the prepared bucket
        # that serves its real rows (pad-M), so does the adapter. Not registered under this key: the
        # case has no site of its own.
        ent = _pad_entry(case, mode)
        if ent is not None:
            _adopt_case_tensors(ent[0], ent[2], case, mode)
            ent[1].case = case
            return ent
    if ent is None:
        holder = SimpleNamespace(case=case)
        sparse = mode != "dense"
        impl = SimpleNamespace(
            num_heads=16,
            num_kv_heads=1,
            head_dim=128,
            rotary_emb=case.rotary,
            q_norm=case.q_norm,
            k_norm=case.k_norm,
            kv_cache_dtype="fp8",
            layer_num=0,
            sliding_window=-1,
            gemm_fusion_prologue_state=lambda fwd: _caches_state(
                holder.case, fwd.attn_metadata.slot_mapping
            ),
        )
        # the site's static eligibility check fails safe without this method
        impl.gemm_fusion_prologue_supported = lambda: _prologue_supported(mode, impl)
        if sparse:
            impl.index_head_dim, impl.index_q_size = 128, 128
            impl.index_q_norm, impl.index_k_norm = case.index_q_norm, case.index_k_norm
            impl.index_cache = (case.caches or {}).get("index_cache")
        attn = SimpleNamespace(impl=impl)
        linear = SimpleNamespace(
            output_size=case.spec.gemm_nk[0],
            input_size=case.spec.gemm_nk[1],
            weight=case.weight,
            weight_scale=case.weight_scale,
        )
        with _kv_page(case.spec.cache_page_size):
            fields = S.qkv_fields(
                q=8192,
                kv=512,
                head_dim=128,
                world=WORLD,
                **({"iq": 512, "ik": 128, "idx_head_dim": 128} if sparse else {}),
            )
            stages = [
                S.sp_exchange(
                    fields, drop=("iq", "ik") if mode == "reuse" else (), world=WORLD
                ),
                S.head_norm_rope_cache_from_attention(
                    attn, index="fp8_unit_page128" if mode == "indexer" else None
                ),
            ]
        site = (
            _GoldenSite(  # case tensors are address-stable (refresh_input keeps them)
                f"golden.{mode}.T{case.spec.tokens}",
                linear,
                stages,
                None,
                attn=attn,
                q_size=8192 // WORLD,
                kv_size=512 // WORLD,
                index_q_size=128 if sparse else 0,
                index_k_size=128 if sparse else 0,
                sparse=sparse,
                skip_index=(mode == "reuse"),
            )
        )
        ent = _SITES[key] = (site, holder, attn)
    ent[1].case = case
    return ent


def _ctx(case, attn):
    sp = case.spec
    ctx = _HarnessCtx(
        attn=attn,
        positions=case.positions,
        fwd_ctx=None,
        world=WORLD,
        real_tokens=sp.tokens,
        padded_tokens=sp.padded_tokens,
    )
    ctx.state = _caches_state(case, case.slot_mapping)
    return ctx


def prepare_cases(cases, comm):
    """Resolve every case on every rank (same order), finalize, prepare plans."""
    _maybe_configure_from_env(comm)
    if hooks.resources() is None:
        return
    ents = [_site_for(c, comm) for c in cases]
    prev = hooks.phase()
    set_phase("preparing")  # resolve (match + consensus + ensure) is allowed now
    try:
        for (site, _, _), c in zip(ents, cases):
            for mode in ("eager", "capture"):
                site.resolve(c.spec.padded_tokens, graph_mode=mode)
        hooks.resources().finalize()
        for (site, _, _), c in zip(ents, cases):
            sp = c.spec
            prepared = site.prepare_bucket(
                sp.padded_tokens,
                c.x.device,
                None,
                torch.bfloat16,
                # direct bind: the case's own tensors, no staging copies
                direct={
                    "x_scale": c.x_scale,
                    "positions": c.positions,
                    "slot_mapping": c.slot_mapping,
                    "state": _caches_state(c, c.slot_mapping),
                },
            )
            site.agree_prepared(sp.padded_tokens, prepared)  # collective, uniform
        hooks.resources().clean_epoch()
    finally:
        set_phase("ready" if prev != "pre" or hooks.resources() else prev)


def check_error():
    """Per-sample device error check for the harness: synchronous, agreed across ranks. On an error
    the collective reset has run and this raises ``GemmFusionDeviceError``. Call after every
    measured/checked sample, on every rank."""
    if hooks.check_now() is not None:
        raise hooks.GemmFusionDeviceError(
            "gemm_fusion device error word set; plans re-initialised"
        )


def decisions():
    return list(DECISIONS)


def last_decision():
    return _LAST


def qkv_proj_all2all_norm_rope_cache(case, comm):
    """Golden candidate entry: FP8 QKV GEMM -> exchange -> norm/RoPE/cache.

    RESULT CONTRACT (lead decision): the result is returned as the consumer produced it, with no
    copy, matching the golden harness's native behaviour and timing. On a 1-stage large-M call a
    returned tensor may therefore be a BORROWED view of the shared R window, valid only until the
    next large launch of ANY plan on this rank. Compare or clone each large-M result before the
    next large launch. The model path's C-S4 aliasing check does not apply here."""
    global _LAST
    import operators  # golden module (on sys.path under run_cases)

    _maybe_configure_from_env(comm)
    site, _, attn = _site_for(case, comm)
    ctx = _ctx(case, attn)
    mode = case.spec.mode

    def native_full(x, xs):
        raw = operators.project(x, xs, case)
        return operators.norm_rope_cache(comm.qkv_exchange(raw, mode), case)

    kind, payload, d = run_prologue(
        site,
        case.x,
        case.x_scale,
        ctx,
        native_full,
        lambda packed: operators.norm_rope_cache(packed, case),
        torch.bfloat16,
    )
    _LAST = d
    if kind == "native":
        return payload
    real = case.spec.tokens
    # The fused outputs are the BUCKET's staging buffers: ``T_bucket`` rows, whose tail is the SP
    # divisibility pad (position 0, slot -1) and, under pad-M, the rows between the call's padded T
    # and the bucket's. The baseline returns only the real rows, so return the same rows (a view,
    # no copy) whenever there are more: not only when the call itself was SP-padded, because a case
    # whose tokens are a multiple of 4 (real == padded) still gets a bigger bucket's buffers.
    payload = {
        k: (
            v[:real]
            if real < case.spec.padded_tokens
            or (hasattr(v, "shape") and v.shape[0] > real)
            else v
        )
        for k, v in payload.items()
    }
    return {**payload, **case.caches}


def quant_all2all_o_proj(case, comm, o_path="current"):
    """Not fused by this module: the baseline implementation."""
    import operators

    return operators.quant_all2all_o_proj(case, comm, o_path)


# =============================================================================
# 6. MiniMax-M3 smoke model (E-2 / A-4 helper)
# =============================================================================
def m3_config(layers=4, experts=8, vocab=32000, index_topk_freq=2, moe=False):
    """HF-style config dict (no weights). Geometry matches the golden cases.

    ``moe=False`` (default): every FFN is the dense MLP (``moe_layer_freq`` all 0).
    WHY (history): at the golden pin 8f9258a76 the decoder layer called ``ffn(hidden_states,
    x_scale=hidden_states_scale)`` for BOTH FFN kinds (model.py:776) but
    ``MiniMaxM3MoE.forward(self, hidden_states)`` takes no ``x_scale``, so ANY MoE layer raised
    ``TypeError`` at the first forward (E2 OFF arm, job-e2_off03, with no gemm_fusion involved).
    The SP series had collapsed main's two calls into one; the R12 branch restores main's dispatch
    (the MoE is called without a scale: its activation is BF16 by construction), so ``moe=True``
    now runs. The default stays dense-only so existing evidence stays comparable; the attention
    geometry and the three attention layer types are the same either way.

    MoE EXPERT QUANTIZATION (``moe=True``): do NOT online-quantize the experts to ``ptpc_fp8``.
    That makes ``Fp8MoEMethod`` (per_Token, Swiglu) run aiter's generic 2-stage CK kernels, a
    combination no M3 recipe uses, and on gfx950 ``ck_moe_stage2`` rejects it at warmup
    ("device_gemm with the specified compilation parameters does not support this GEMM problem").
    Use ``m3_online_quant_config("mxfp4")`` (the production M3 expert path) or ``("bf16")`` (ATOM's
    own recipe exclusion); ``serve_args(..., online_quant=...)`` passes it."""
    layer_ffn = [0] * layers if not moe else [0] + [1] * (layers - 1)
    return {
        "model_type": "minimax_m3",
        "architectures": ["MiniMaxM3SparseForCausalLM"],
        "hidden_size": 6144,
        "num_attention_heads": 64,
        "num_key_value_heads": 4,
        "head_dim": 128,
        "num_hidden_layers": layers,
        "vocab_size": vocab,
        "max_position_embeddings": 8192,
        "rms_norm_eps": 1e-6,
        "rope_theta": 5000000.0,
        "partial_rotary_factor": 0.5,
        "hidden_act": "swigluoai",
        "intermediate_size": 1536,
        "dense_intermediate_size": 6144,
        "num_local_experts": experts,
        "num_experts_per_tok": 2,
        "n_shared_experts": 1,
        # 0 = dense MLP, nonzero = MoE (see the docstring: MoE hits a stock x_scale TypeError)
        "moe_layer_freq": layer_ffn,
        # ATOM's _should_skip_minimax_m3_index_topk returns "no skip" for EVERY layer unless
        # use_index_cache is true (model.py:88), whatever index_topk_freq says. The recipes turn it
        # on with --hf-overrides '{"use_index_cache": true, "index_topk_freq": 4}'. Without this
        # line the earlier smoke config had NO skip-index layer at all (found by E-2/A-4 level 0).
        "use_index_cache": True,
        "index_topk_freq": index_topk_freq,
        "index_skip_topk_offset": 0,
        "sparse_attention_config": {
            "sparse_block_size": 128,  # must equal SPARSE_BLOCK_SIZE (checked at init)
            "sparse_num_index_heads": 4,
            "sparse_index_dim": 128,
            "sparse_topk_blocks": 8,
            "sparse_init_block": 1,
            "sparse_local_block": 2,
            "sparse_score_type": "max",
            # 0 = dense attention layer, nonzero = sparse (first layer dense)
            "sparse_attention_freq": [0] + [1] * (layers - 1),
        },
        "torch_dtype": "bfloat16",
    }


def register_hf_config(model_type="minimax_m3"):
    """Make ``AutoConfig.from_pretrained`` accept this model_type WITHOUT a transformers
    upgrade or trust_remote_code. Call it in the process that builds ``Config`` (the one
    that calls ``atom.config.get_hf_config``) BEFORE ``create_engine``.

    Why it is needed: ATOM's native path registers no config class for minimax_m3 /
    minimax_m3_vl (only the vLLM plugin does, via the same kind of shim), so
    ``get_hf_config`` falls through to ``AutoConfig.from_pretrained`` and raises for a
    model_type the installed transformers does not know. The shim is a bare
    ``PretrainedConfig``: it keeps every field of the (flat, text-only) config.json as an
    attribute, and ``architectures`` still selects ``MiniMaxM3SparseForCausalLM``.

    UNVERIFIED against real transformers 5.16.1 (not installed on the authoring host).
    Idempotent: a model_type transformers already knows is left alone.
    """
    from transformers import AutoConfig, PretrainedConfig

    cls = type("MiniMaxM3SmokeConfig", (PretrainedConfig,), {"model_type": model_type})
    try:
        AutoConfig.register(model_type, cls)
    except ValueError as exc:  # already registered (by transformers or an earlier call)
        if "already" not in str(exc):
            raise
    return cls


VL_MODEL_TYPE = (
    "minimax_m3_vl"  # outer type of the released checkpoint (repo tests + vLLM shim)
)
TEXT_MODEL_TYPE = "minimax_m3_text"  # nested text_config type (repo tests)
VL_ARCH = (
    "MiniMaxM3SparseForConditionalGeneration"  # what the recipes' VL checkpoint selects
)


def m3_vl_config(layers=4, experts=8, vocab=32000, index_topk_freq=2, moe=False):
    """The released checkpoint's LAYOUT: an outer ``minimax_m3_vl`` config whose text model
    is nested under ``text_config`` (``minimax_m3_text``). ATOM reads the text fields from
    ``text_config`` (``_get_text_config``) and copies the index-cache overrides across
    (``_normalize_minimax_m3_text_config``). ``auto_map`` points at a config class that
    ships NEXT TO config.json, which is how ATOM's own recipes load M3 (``--trust-remote-
    code``); ATOM's native path registers no class for it. Attention geometry is unchanged.
    """
    text = m3_config(
        layers=layers,
        experts=experts,
        vocab=vocab,
        index_topk_freq=index_topk_freq,
        moe=moe,
    )
    for k in ("architectures", "torch_dtype"):
        text.pop(k, None)
    text["model_type"] = TEXT_MODEL_TYPE
    # The index-cache controls live on the OUTER config (the recipes set them with
    # --hf-overrides, which updates the outer config) and ATOM copies them DOWN into
    # text_config (_normalize_minimax_m3_text_config). Keep them out of text_config here so
    # the run exercises that same copy, which is what decides skip-index layers.
    outer_index = {
        k: text.pop(k)
        for k in ("use_index_cache", "index_topk_freq", "index_skip_topk_offset")
        if k in text
    }
    return {
        "model_type": VL_MODEL_TYPE,
        "architectures": [VL_ARCH],
        "auto_map": {"AutoConfig": "configuration_minimax_m3.MiniMaxM3Config"},
        **outer_index,
        "text_config": text,
        # vision tower is never built on the text-only native path (model.py skips
        # vision_tower./multi_modal_projector./patch_merge_mlp. weights); kept minimal
        "vision_config": {"model_type": "minimax_m3_vision"},
        "torch_dtype": "bfloat16",
    }


# Shipped next to config.json. A bare PretrainedConfig keeps every field as an attribute
# (nested dicts become sub-configs), mirroring atom/plugin/vllm/register.py MiniMaxM3Config.
CONFIGURATION_PY = """\
from transformers import PretrainedConfig


class MiniMaxM3TextConfig(PretrainedConfig):
    model_type = "minimax_m3_text"


class MiniMaxM3Config(PretrainedConfig):
    model_type = "minimax_m3_vl"
    sub_configs = {"text_config": MiniMaxM3TextConfig}

    def __init__(self, text_config=None, vision_config=None, **kwargs):
        if isinstance(text_config, dict):
            text_config = MiniMaxM3TextConfig(**text_config)
        self.text_config = text_config
        self.vision_config = vision_config
        self.hidden_size = getattr(text_config, "hidden_size", None)
        super().__init__(**kwargs)
"""


def make_model_dir(path, *, layout="vl", **kw):
    """Write config.json (+ configuration_minimax_m3.py for the ``vl`` layout).

    layout="vl"   (default) the released checkpoint's layout; load with --trust-remote-code.
    layout="flat" the old single-level ``minimax_m3`` config; needs ``register_hf_config()``.
    """
    os.makedirs(path, exist_ok=True)
    cfg = m3_vl_config(**kw) if layout == "vl" else m3_config(**kw)
    with open(os.path.join(path, "config.json"), "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)
    if layout == "vl":
        with open(
            os.path.join(path, "configuration_minimax_m3.py"), "w", encoding="utf-8"
        ) as f:
            f.write(CONFIGURATION_PY)
    return path


SPARSE_BLOCK_SIZE = 128  # atom.model_ops.minimax_m3.sparse_attn.SPARSE_BLOCK_SIZE


# The exclusions of ATOM's own M3 launch (recipes/MiniMax-M3.md): these never get online quant.
M3_ONLINE_QUANT_EXCLUDE = (
    "lm_head",
    "model.embed_tokens",
    "vision_tower",
    "multi_modal_projector",
    "patch_merge_mlp",
)
M3_MOE_EXPERT_QUANT = ("mxfp4", "bf16")


def m3_online_quant_config(moe_experts="mxfp4"):
    """The ``--online_quant_config`` JSON for an M3 smoke model: ptpc_fp8 for every linear the fusion
    path consumes (attention, dense MLP), and a deliberate choice for the MoE block.

    ``moe_experts="mxfp4"`` (default): routed experts become MXFP4 online (``per_1x32`` / fp4), the
    production M3 expert path (``Mxfp4MoEMethod``; MiniMax-M3-MXFP4 is the primary M3 checkpoint;
    1536 meets its 256 pad alignment). The shared experts stay unquantized.
    ``moe_experts="bf16"``: the ``*block_sparse_moe`` exclusion of ATOM's own M3 recipe, so the whole
    MoE block (experts, shared experts) keeps the source dtype and runs ``UnquantizedFusedMoEMethod``.

    ``ptpc_fp8`` experts are not offered: they select the CK 2-stage per-token path that gfx950
    rejects for these shapes (see ``m3_config``). Not verified on a GPU by this module.
    """
    if moe_experts not in M3_MOE_EXPERT_QUANT:
        raise ValueError(
            f"moe_experts must be one of {M3_MOE_EXPERT_QUANT}, got {moe_experts!r} "
            "(ptpc_fp8 experts fail in ck_moe_stage2 on gfx950)"
        )
    exclude = list(M3_ONLINE_QUANT_EXCLUDE)
    cfg = {"global_quant_config": "ptpc_fp8"}
    if moe_experts == "bf16":
        exclude.append(
            "*block_sparse_moe"
        )  # also matches its children (.experts, .shared_*)
    else:
        cfg["layer_quant_config"] = {"*block_sparse_moe.experts*": "mxfp4"}
        exclude.append("*block_sparse_moe.shared_experts*")
    cfg["exclude_layer"] = exclude
    return json.dumps(cfg, separators=(",", ":"))


def serve_args(
    model_dir,
    *,
    level=3,
    profile="",
    enforce_eager=False,
    sp=4,
    extra=(),
    trust_remote_code=True,
    block_size=SPARSE_BLOCK_SIZE,
    online_quant=None,
):
    """ATOM CLI argv. ``--block-size`` must be a multiple of the sparse block (128):
    the attention metadata builder asserts ``model_runner.block_size % self.block_size == 0``
    (attentions/backends.py:511) and ATOM's CLI default of 16 fails it for M3's sparse layers
    (found by the E2 OFF arm, job-e2_off01). The same value also fixes the KV page size the
    dense layers' fusion signature derives from ``config.kv_cache_block_size``.

    ``online_quant`` (None: no flag, argv unchanged): ``"mxfp4"`` or ``"bf16"`` adds
    ``--online_quant_config`` from ``m3_online_quant_config`` (how the MoE experts are treated).
    """
    if block_size % SPARSE_BLOCK_SIZE:
        raise ValueError(
            f"block_size must be a multiple of {SPARSE_BLOCK_SIZE}, got {block_size}"
        )
    argv = [
        # xavier: finite, roughly real-scale random weights (arg_utils --load_dummy)
        "--model",
        model_dir,
        "--load_dummy",
        "xavier",
        "-tp",
        "1",
        "-sp",
        str(sp),
        "--kv-cache-dtype",
        "fp8",
        "--level",
        str(level),
        "--block-size",
        str(block_size),
    ]
    if trust_remote_code:  # the vl layout resolves its config class through auto_map
        argv.append("--trust-remote-code")
    if enforce_eager:
        argv.append("--enforce-eager")
    if profile:
        argv += ["--gemm-fusion-profile", profile]
    if online_quant is not None:
        argv += ["--online_quant_config", m3_online_quant_config(online_quant)]
    return argv + list(extra)


def serve_env(extra=None):
    """Process environment the M3 launch needs, as ATOM's own M3 recipes set it.

    ``ATOM_FORCE_ATTN_TRITON=1`` is REQUIRED for M3 decode (recipes/MiniMax-M3-Agentic-
    InferenceX.md: "Without it decode routes to the ASM paged-attention kernel ... and the
    server dies at startup"). E2 job-e2_new_on03 launched without it and aborted (exit -6) in
    aiter asm_pa.cu:162 ``get_heuristic_kernel: cannot get heuristic kernel! q_type:bf16
    kv_type:fp8 gqa:16 ... block_size:128`` during the bs=1 decode CUDA-graph capture. That is
    the stock dense-layer decode, with gemm_fusion not involved. ``--enforce-eager`` would NOT
    help: eager decode calls the same kernel.

    COVERAGE: dense layers ARE eligible to fuse under this flag. The flag only selects the
    DECODE kernel (Triton) that reads the cache later; the dense native ``rope_cache`` takes
    its GemmaRMSNorm norm-rope-cache branch and writes the same SHUFFLE fp8 per-token cache
    whatever the flag says, so ``PagedAttentionImpl.gemm_fusion_prologue_supported`` does not
    read it, and the fused prologue sets ``use_triton_attn`` exactly as native does
    (``_native_use_triton_attn``) so Triton decode runs on the fused-written cache. (An earlier
    revision made dense layers ineligible under the flag; that was wrong and is reverted.)
    Whether a dense layer actually HITS a profile row then depends on the cache page in its
    signature (``kv_cache``) matching a row; the check is in ``Decision`` / the install log.
    """
    env = {"ATOM_FORCE_ATTN_TRITON": "1"}
    env.update(extra or {})
    return env


def m3_smoke_main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--level", type=int, default=3)
    ap.add_argument("--profile", default="")
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--enforce-eager", action="store_true")
    ap.add_argument("--moe", action="store_true", help="MoE FFN layers (see m3_config)")
    ap.add_argument(
        "--online-quant",
        choices=M3_MOE_EXPERT_QUANT,
        default=None,
        help="add --online_quant_config; how the MoE experts are treated",
    )
    a = ap.parse_args()
    make_model_dir(a.dir, layers=a.layers, moe=a.moe)
    argv = serve_args(
        a.dir,
        level=a.level,
        profile=a.profile,
        enforce_eager=a.enforce_eager,
        online_quant=a.online_quant,
    )
    print("argv:", " ".join(f"'{x}'" if x.startswith("{") else x for x in argv))
    print("env:", " ".join(f"{k}={v}" for k, v in serve_env().items()))
    print(
        "fault env:",
        FAULT_ENV,
        "(applied by install_from_env() in the process that runs the model)",
    )


__all__ = ["check_error", "qkv_proj_all2all_norm_rope_cache", "quant_all2all_o_proj"]


if __name__ == "__main__":
    m3_smoke_main()
