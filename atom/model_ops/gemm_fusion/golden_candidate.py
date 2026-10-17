# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Candidate module for the golden ``run_cases`` harness (evidence adapter).

Use as ``--candidate`` module: it exposes ``qkv_proj_all2all_norm_rope_cache(
case, comm)`` and drives the REAL ``GemmFusionSite`` (resolve -> consensus ->
aiter ``match_prefix`` -> ``Plan.prepare/run``) through ``site.run_prologue``
for the golden case shapes. No MiniMax-M3 weights are loaded.

Scope (state it when citing results): this exercises the site/plan layer on the
GEMM -> exchange -> norm/rope/cache prologue, which is exactly what the golden
case compares. It does NOT run attention, the custom op registry, or
``MiniMaxM3*Attention.forward``; those need model weights/serving.

Profile OFF (no resources configured) => every call takes the site's native
branch, which here is the golden harness's own baseline operators, so OFF is
identical to baseline by construction (a plumbing-equivalence check, not a
numerical one). Profile ON: call ``configure(...)`` once per process, then
``prepare_cases(cases, comm)`` for every case BEFORE the first measured call
(all plans must be resolved on all ranks before ``finalize``).

Evidence: ``last_decision()``, ``decisions()`` (selected vid, fused prefix,
fallback reason per call).
"""

import os
from types import SimpleNamespace

import torch

from . import hooks
from . import stages as S
from .context import AttnFusionCtx
from .site import GemmFusionSite

WORLD = 4
_SITES: dict = {}
_LAST = None


class _HarnessCtx(AttnFusionCtx):
    state = None

    @property
    def is_dummy_run(self) -> bool:
        return False

    def prologue_state(self):
        return self.state


def configure(profile, *, rank, world=WORLD, pg=None, resources=None):
    """Turn fusion ON for this process (collective: builds the comm)."""
    from .resources import FusionResourcesWrapper

    gf = S.aiter_gf()
    gf.load_profile(profile)
    hooks.reset_for_tests(
        resources or FusionResourcesWrapper(None, world, rank=rank, pg=pg)
    )
    _SITES.clear()


def _maybe_configure_from_env(comm):
    prof = os.environ.get("GEMM_FUSION_HARNESS_PROFILE", "")
    if prof and hooks.resources() is None:
        configure(prof, rank=comm.rank)


def _mode(case):
    m = case.spec.mode
    return "dense" if m == "qkv_dense" else ("indexer" if m == "qkv_indexer" else "reuse")


def _caches_state(case, slot_mapping):
    c = case.caches
    return {
        "k_cache": c["k_cache"],
        "v_cache": c["v_cache"],  # golden v_cache is already the SHUFFLE view
        "k_scale": c["k_scale"],
        "v_scale": c["v_scale"],
        "slot_mapping": slot_mapping,
    }


def _site_for(case, comm):
    mode = _mode(case)
    key = (mode, case.spec.tokens)
    ent = _SITES.get(key)
    if ent is None:
        holder = SimpleNamespace(case=case)
        sparse = mode != "dense"
        impl = SimpleNamespace(
            num_heads=16, num_kv_heads=1, head_dim=128,
            rotary_emb=case.rotary, q_norm=case.q_norm, k_norm=case.k_norm,
            kv_cache_dtype="fp8", layer_num=0,
            gemm_fusion_kv_page=case.spec.cache_page_size,
            gemm_fusion_prologue_state=lambda fwd: _caches_state(
                holder.case, fwd.attn_metadata.slot_mapping
            ),
        )
        if sparse:
            impl.index_head_dim, impl.index_q_size = 128, 128
            impl.index_q_norm, impl.index_k_norm = case.index_q_norm, case.index_k_norm
            impl.index_cache = (case.caches or {}).get("index_cache")
        attn = SimpleNamespace(impl=impl)
        linear = SimpleNamespace(
            output_size=case.spec.gemm_nk[0], input_size=case.spec.gemm_nk[1],
            weight=case.weight, weight_scale=case.weight_scale,
        )
        fields = S.qkv_fields(
            q=8192, kv=512, head_dim=128, world=WORLD,
            **(dict(iq=512, ik=128, idx_head_dim=128) if sparse else {}),
        )
        stages = [
            S.sp_exchange(fields, drop=("iq", "ik") if mode == "reuse" else (), world=WORLD),
            S.head_norm_rope_cache_from_attention(
                attn, index="fp8_unit_page128" if mode == "indexer" else None
            ),
        ]
        site = GemmFusionSite(
            f"golden.{mode}.T{case.spec.tokens}", linear, stages, None, attn=attn,
            q_size=8192 // WORLD, kv_size=512 // WORLD,
            index_q_size=128 if sparse else 0, index_k_size=128 if sparse else 0,
            sparse=sparse, skip_index=(mode == "reuse"),
            bind_mode="direct",  # case tensors are address-stable (refresh_input keeps them)
        )
        ent = _SITES[key] = (site, holder, attn)
    ent[1].case = case
    return ent


def _ctx(case, attn):
    sp = case.spec
    ctx = _HarnessCtx(
        attn=attn, positions=case.positions, fwd_ctx=None, world=WORLD,
        real_tokens=sp.tokens, padded_tokens=sp.padded_tokens,
    )
    ctx.state = _caches_state(case, case.slot_mapping)
    return ctx


def prepare_cases(cases, comm):
    """Resolve every case on every rank (same order), finalize, prepare plans."""
    _maybe_configure_from_env(comm)
    if hooks.resources() is None:
        return
    ents = [_site_for(c, comm) for c in cases]
    for (site, _, _), c in zip(ents, cases):
        site.resolve(c.spec.padded_tokens, graph_mode="eager")
    hooks.resources().finalize()
    for (site, _, _), c in zip(ents, cases):
        sp = c.spec
        site.prepare_bucket(
            sp.padded_tokens, c.x.device, None, torch.bfloat16,
            pad_rows=sp.padded_tokens != sp.tokens,
            # direct bind: the case's own tensors, no staging copies
            direct={
                "x_scale": c.x_scale,
                "positions": c.positions,
                "slot_mapping": c.slot_mapping,
                "state": _caches_state(c, c.slot_mapping),
            },
        )


def decisions():
    return list(hooks.DECISIONS)


def last_decision():
    return _LAST


def qkv_proj_all2all_norm_rope_cache(case, comm):
    """Golden candidate entry: FP8 QKV GEMM -> exchange -> norm/RoPE/cache."""
    global _LAST
    import operators  # golden module (on sys.path under run_cases)

    _maybe_configure_from_env(comm)
    site, _, attn = _site_for(case, comm)
    ctx = _ctx(case, attn)
    mode = case.spec.mode

    def native_full(x, xs):
        raw = operators.project(x, xs, case)
        return operators.norm_rope_cache(comm.qkv_exchange(raw, mode), case)

    kind, payload, d = site.run_prologue(
        case.x, case.x_scale, ctx, native_full,
        lambda packed: operators.norm_rope_cache(packed, case), torch.bfloat16,
    )
    _LAST = d
    if kind == "native":
        return payload
    return {**payload, **case.caches}


def quant_all2all_o_proj(case, comm, o_path="current"):
    """Not fused by this module: the baseline implementation."""
    import operators

    return operators.quant_all2all_o_proj(case, comm, o_path)


__all__ = ["qkv_proj_all2all_norm_rope_cache", "quant_all2all_o_proj"]
