# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""The plans every site runs per bucket M, agreed on by all SP ranks, and the
AITER resources they share. A plan any rank cannot build or bind is dropped on
every rank, so the ranks always launch the same fused collectives."""

import dataclasses
import importlib
import logging
from types import SimpleNamespace

import torch

from atom.distributed.group_agreement import group_max, group_max_each

from .minimax_m3 import install_minimax_m3
from .site import cache_layout_reason

logger = logging.getLogger("atom")

# Per (site, bucket) at startup.
_NONE, _OK, _BROKEN = 0, 1, 2


def _bucket_ms(profile, spec, ctx, max_m):
    """The profile's exact-row M of ``spec`` up to the first that holds
    ``max_m`` local tokens."""
    ms = sorted(
        {
            row.m
            for n in range(len(spec.stages), 0, -1)
            for row in profile.bucket_rows(
                spec.prefix(n).signature(), ctx.arch, ctx.cu, ctx.world
            )
        }
    )
    out = []
    for m in ms:
        out.append(m)
        if m >= max_m:
            break
    return out


class GemmFusionRuntime:
    def __init__(self, gf, profile, ctx, group, world, block_size):
        self.gf = gf
        self.callid = importlib.import_module("aiter.gemm_fusion.callid")
        self.profile = profile
        self.ctx = ctx
        self.group = group
        self.world = world
        self.block_size = block_size
        self.sites = []
        self.res = None
        # Prologue outputs per mode, shared by every small plan.
        self.outs = {}
        self.ready = False
        self.failed = False
        self.step = None
        self._buckets = {}
        self._call_ids = {}
        self._bound = False
        self._steps = 0
        self._pending = None

    def bucket(self, spec, m):
        key = (id(spec), m)
        got = self._buckets.get(key)
        if got is None and key not in self._buckets:
            got = self.gf.bucket_for_m(
                spec, dataclasses.replace(self.ctx, m=m), self.profile
            )
            self._buckets[key] = got
        return got

    def call_id(self, site, tokens, plan):
        """This site's id for ``tokens``, or None when it has none."""
        ids = self._call_ids.get(tokens)
        if ids is None:
            ids = self._call_ids[tokens] = self._resolve_call_ids(tokens)
        return ids.get(site.name)

    def _resolve_call_ids(self, tokens):
        """Ids distinct within one forward. Large plans tag their handshakes
        with only the low bits of the id, so those bits are made unique."""
        from aiter.ops.gemm_fusion.sp_qkv.specs import large_tag_bits

        m = tokens // self.world
        keys, tagged, bits = [], [], []
        for s in self.sites:
            p = s.plans.get(self.bucket(s.spec, m))
            if p is None:
                continue
            key = (s.name, tokens, p.vid, p.n_stages)
            keys.append(key)
            if p.n_stages == 1:
                tagged.append(key)
                bits.append(large_tag_bits(p.src))
        tag_bits = min(bits, default=None)
        ids = {}
        if tagged:
            try:
                ids = self.callid.tag_unique_ids(tagged, tag_bits=tag_bits)
            except ValueError as e:
                logger.warning(
                    "gemm_fusion tokens=%d: large plans native (%s)", tokens, e
                )
                keys = [k for k in keys if k[3] != 1]

        def key_fn(*k):
            return ids.get(k) or self.callid.decision_key(*k)

        refused = self.callid.check_call_ids(
            keys,
            key_fn=key_fn,
            tag_bits=tag_bits if ids else None,
            tag_vids={k[2] for k in ids} or None,
            per_tokens=True,
        ).refused_keys
        if refused:
            logger.warning(
                "gemm_fusion tokens=%d: %d site(s) native, call ids collide",
                tokens,
                len(refused),
            )
        return {k[0]: key_fn(*k) for k in keys if k not in refused}

    def _match(self, site, m, large):
        try:
            plan, _ = self.gf.match_prefix(
                site.spec, dataclasses.replace(self.ctx, m=m), self.profile
            )
        except self.gf.ContractError:
            logger.exception("gemm_fusion %s: local M %d", site.name, m)
            return None, _BROKEN
        if plan is None or getattr(plan, "abi", 0) < 3:
            return None, _NONE
        if plan.n_stages == 1:
            # One large plan per bucket and signature, shared by its sites.
            sig = site.spec.prefix(1).signature()
            plan = large.setdefault((sig, m), plan)
        return plan, _OK

    def _allocate_staging(self, device):
        """bf16 prologue outputs for the largest small bucket, one set shared
        by every small plan: a plan binds these exact tensors."""
        small = [
            (s, m) for s in self.sites for m, p in s.plans.items() if p.n_stages == 2
        ]
        if not small:
            return
        rows = self.world * max(m for _, m in small)
        shapes = {
            (s.attn.impl.num_heads, s.attn.impl.num_kv_heads, s.attn.impl.head_dim)
            for s, _ in small
        }
        if len(shapes) != 1:
            raise RuntimeError(f"gemm_fusion: one staging shape expected, got {shapes}")
        heads, kv_heads, hd = shapes.pop()

        def empty(*shape):
            return torch.empty(shape, dtype=torch.bfloat16, device=device)

        q = empty(rows, heads, hd)
        qkv = {"q": q, "k": empty(rows, kv_heads, hd), "v": empty(rows, kv_heads, hd)}
        self.outs = {"dense": qkv, "reuse": qkv}
        idx = next((s.attn.impl for s, _ in small if s.mode == "indexer"), None)
        if idx is not None:
            index_q = empty(rows, idx.num_idx_heads, idx.index_head_dim)
            self.outs["indexer"] = {"q": q, "index_q": index_q}

    def _prepare_site(self, kv_cache_data, site, plans, done):
        """0 per bucket whose plan bound on this rank, else 1."""
        lin = site.linear
        dev = lin.weight.device
        k = lin.input_size
        state = layout = None
        if any(p.n_stages == 2 for p in plans.values()):
            impl = site.attn.impl
            slot = torch.full((1,), -1, dtype=torch.int64, device=dev)
            shim = SimpleNamespace(
                kv_cache_data=kv_cache_data,
                attn_metadata=SimpleNamespace(slot_mapping=slot),
            )
            state = impl.gemm_fusion_prologue_state(shim)
            layout = cache_layout_reason(
                impl,
                state,
                page=self.block_size if site.mode == "dense" else 16,
                with_index=site.mode == "indexer",
            )
            if layout is not None:
                logger.info("gemm_fusion %s: native (%s)", site.name, layout)
        codes = []
        for m, plan in plans.items():
            if plan.n_stages == 2 and layout is not None:
                codes.append(1)
                continue
            if id(plan) in done:
                codes.append(done[id(plan)])
                continue
            x = torch.zeros((m, k), dtype=torch.float8_e4m3fn, device=dev)
            x_scale = torch.zeros((m, 1), dtype=torch.float32, device=dev)
            try:
                if plan.n_stages == 1:
                    plan.prepare(
                        res=self.res, x=x, x_scale=x_scale, w_scale=lin.weight_scale
                    )
                else:
                    tokens = self.world * m
                    slot = torch.full((tokens,), -1, dtype=torch.int64, device=dev)
                    positions = torch.zeros(tokens, dtype=torch.int64, device=dev)
                    plan.prepare(
                        x=x,
                        x_scale=x_scale,
                        w_scale=lin.weight_scale,
                        outs=self.outs[site.mode],
                        meta=site.build_meta(state, positions, slot),
                        res=self.res,
                    )
                code = 0
            except Exception:
                logger.exception("gemm_fusion %s: prepare at local M %d", site.name, m)
                code = 1
            if plan.n_stages == 1:
                done[id(plan)] = code
            codes.append(code)
        return codes

    def prepare(self, kv_cache_data):
        """Bind every small plan to the KV cache and attach the large plans,
        once, before graph capture. Collective over the SP group."""
        if self._bound:
            raise RuntimeError(
                "--gemm-fusion-profile binds the KV cache once; it cannot be "
                "reallocated"
            )
        self._bound = True
        if self.res is None:
            return
        entries, codes, done = [], [], {}
        for site in self.sites:
            plans = dict(sorted(site.plans.items()))
            entries += [(site, m) for m in plans]
            codes += self._prepare_site(kv_cache_data, site, plans, done)
        failed = group_max_each(codes, self.group)
        for (site, m), bad in zip(entries, failed):
            if bad:
                del site.plans[m]
        if not any(s.plans for s in self.sites):
            logger.warning("gemm_fusion: no plan could be bound; all sites native")
            return
        self.res.reset()
        self.step = self.res.step_tensor()
        self.ready = True
        logger.info(
            "gemm_fusion: %d of %d (site, bucket) plans bound",
            len(entries) - sum(failed),
            len(entries),
        )

    def _fail(self, err):
        self.failed = True
        raise RuntimeError(
            f"gemm_fusion rank {self.res.rank}: a fused QKV launch failed "
            f"(ERR {err:#x}); the output of a recent step is invalid."
        )

    def before_step(self):
        """Raise if an earlier step's fused launch failed, read through an
        async copy of the ERR word without a host sync."""
        if not self.ready:
            return
        if self._pending is not None:
            err = self.res.readback_consume(self._pending)
            if err is None:
                return
            if err:
                self._fail(err)
        self._pending = self._steps
        self._steps += 1
        self.res.readback_enqueue(self._pending)

    def clean_point(self):
        """After graph capture: fail on an error the capture warmups left,
        then the collective reset() before serving."""
        if not self.ready:
            return
        torch.cuda.synchronize()
        err = group_max(int(self.res.error_flag().item()), self.group)
        if err:
            self._fail(err)
        self.res.reset()
        self._pending = None

    def close(self):
        """Collective over the SP group; skipped after a failure, which peers
        may not have seen."""
        if self.res is not None and not self.failed:
            self.res.release()


def initialize(model, config):
    from aiter.dist.parallel_state import get_pcp_group

    gf = importlib.import_module("aiter.gemm_fusion")
    pcp = get_pcp_group()
    world, group = pcp.world_size, pcp.cpu_group
    profile = gf.load_profile(config.gemm_fusion_profile)
    ctx = gf.MatchCtx(
        arch=gf.detect_arch(),
        cu=gf.detect_cu(),
        world=world,
        m=0,
        available=gf.detect_capabilities(),
    )
    rt = GemmFusionRuntime(gf, profile, ctx, group, world, config.kv_cache_block_size)
    rt.sites = install_minimax_m3(model, config, world, gf, rt)
    if not rt.sites:
        raise RuntimeError(
            "--gemm-fusion-profile: no attention layer of this model can be fused"
        )
    rt.sites[0].ticks = True

    failed = 0
    try:
        gf.prebuild_profile(profile, arch=ctx.arch, cu=ctx.cu, world=world)
    except Exception:
        logger.exception("gemm_fusion: building the profile's kernels")
        failed = 1
    if group_max(failed, group):
        raise RuntimeError("--gemm-fusion-profile: a rank failed to build the kernels")

    max_m = -(-config.max_num_batched_tokens // world)
    entries, codes, large = [], [], {}
    for site in rt.sites:
        for m in _bucket_ms(profile, site.spec, ctx, max_m):
            plan, code = rt._match(site, m, large)
            entries.append((site, m, plan))
            codes.append(code)
    agreed = group_max_each(codes + [-c for c in codes], group)
    n = len(codes)
    if _BROKEN in agreed[:n]:
        raise RuntimeError(
            f"--gemm-fusion-profile {config.gemm_fusion_profile}: a profile row "
            "fails its contract on a rank (see the log)"
        )
    for (site, m, plan), hi, neg_lo in zip(entries, agreed[:n], agreed[n:]):
        if hi == _OK and -neg_lo == _OK:
            site.plans[m] = plan
    plans = {}
    for site in rt.sites:
        for plan in site.plans.values():
            plans.setdefault(plan.build_key, plan)
    if plans:
        device = torch.cuda.current_device()
        rt.res = gf.FusionResources(
            rank=pcp.rank_in_group,
            world=world,
            group=group,
            device=device,
            caps=ctx.available,
            profile=profile,
        )
        for key in sorted(plans):
            rt.res.ensure(key, plans[key].resources())
        rt._allocate_staging(device)
        rt.res.finalize()
    logger.info(
        "gemm_fusion: profile %s, %d sites, %d of %d (site, bucket) plans, "
        "%d resource owners",
        config.gemm_fusion_profile,
        len(rt.sites),
        sum(len(s.plans) for s in rt.sites),
        n,
        len(plans),
    )
    return rt
