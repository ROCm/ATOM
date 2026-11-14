# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Mono K1 over NVFP4 pages (gfx950): the standalone K1 build with ``kv_nvfp4`` /
``index_nvfp4`` against the FP8 build on the same inputs, and against the reference
trees' kernels on the pages it wrote:

* the V pages it writes are vLLM's writer's bytes for the V rows the GEMV made (read
  back from K1's own scratch), byte for byte; bytes of other tokens are untouched;
* the K and index-K pages decode to what the FP8 build stored, within the two
  formats' quantization error;
* the block scores K1 makes from the NVFP4 index pages (the ``q_frag`` dim order)
  equal aiter's NVFP4 scorer (``k_round="e4m3"``, e4m3 q) on the pages K1 wrote, with K1's own index q, bit
  for bit (blocks that hold no new token: K1 standalone does not wait for them).

The reference trees are not part of this repo: ``ATOM_NVFP4_AITER_KERNELS`` (aiter's
``aiter/ops/flydsl/kernels`` of the nvfp4 index scorer) and ``ATOM_NVFP4_VLLM_REF``
(vLLM's ``tests/kernels/attention``); the test skips without them."""

import os
import sys

import pytest
import torch

aiter = pytest.importorskip("aiter")
pytest.importorskip("flydsl")
KERNELS = os.environ.get("ATOM_NVFP4_AITER_KERNELS")
VLLM_REF = os.environ.get("ATOM_NVFP4_VLLM_REF")
pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available()
    or "gfx950" not in torch.cuda.get_device_properties(0).gcnArchName
    or not KERNELS
    or not VLLM_REF,
    reason="needs gfx950 and the nvfp4 reference trees (see the module doc)",
)

from aiter import dtypes
from aiter.ops.shuffle import shuffle_weight

from atom.models.minimax_m3.mono.config import (
    HEAD_DIM,
    HIDDEN,
    LOCAL_Q_HEADS,
    MAX_INDEX_BLOCKS,
    MAX_TOKENS,
    PAGE16,
    ROTARY_DIM,
    qkv_rows,
)
from atom.models.minimax_m3.mono.kernels.common import KV_PAGE_BYTES, NVFP4_PAGE_BYTES
from atom.models.minimax_m3.mono.kernels.pre_attn import (
    K1_ABI,
    SCRATCH_BYTES,
    SCRATCH_QKV,
    V_OFF,
    build_pre_attn_kernel,
)
from atom.mono.runtime.uncached import uncached_bytes

SM_SCALE = HEAD_DIM**-0.5
ROWS = qkv_rows(1)


def _reference_modules():
    import aiter.ops.flydsl.kernels as pkg

    pkg.__path__.insert(0, KERNELS)
    from aiter.ops.flydsl.kernels import minimax_m3_index_score as scorer

    sys.path.insert(0, VLLM_REF)
    import minimax_m3_nvfp4_ref as ref

    return scorer, ref


class _Case:
    """One decode step of ``tokens`` tokens (a request each, q_len 1) over a context
    of ``ctx`` keys: the weights, the activations, the old keys' index pages and the
    new tokens' slots, shared by the FP8 and the NVFP4 build."""

    def __init__(self, ref, tokens, ctx, seed=0):
        g = torch.Generator(device="cuda").manual_seed(seed)
        dev = "cuda"
        self.tokens, self.ctx = tokens, ctx
        bf = torch.bfloat16
        self.ar = (torch.randn(tokens, HIDDEN, device=dev, generator=g) * 2).to(bf)
        self.res = torch.randn(tokens, HIDDEN, device=dev, generator=g).to(bf)
        self.g_in = (torch.randn(HIDDEN, device=dev, generator=g) * 0.1).to(bf)
        gam = lambda: (torch.randn(HEAD_DIM, device=dev, generator=g) * 0.2).to(bf)
        self.g_q, self.g_k, self.g_iq, self.g_ik = gam(), gam(), gam(), gam()
        w = (torch.randn(ROWS, HIDDEN, device=dev, generator=g) * 0.5).to(dtypes.fp8)
        self.w_qkv = shuffle_weight(w, layout=(16, 16)).contiguous()
        self.s_qkv = (torch.rand(ROWS, device=dev, generator=g) * 0.02 + 0.005).float()
        max_pos = ctx + 8
        ang = torch.rand(max_pos, ROTARY_DIM // 2, device=dev, generator=g) * 6.28
        self.cos_sin = torch.cat([ang.cos(), ang.sin()], -1).to(bf).contiguous()
        self.positions = torch.arange(ctx - 1, ctx - 1 + tokens, device=dev) * 0 + (
            ctx - 1
        )
        # each request its own blocks; the new token is the context's last key
        self.n_blk = (ctx + 127) // 128
        self.total_blocks = tokens * self.n_blk + 1
        self.block_table = (
            torch.randperm(self.total_blocks, device=dev, generator=g)[
                : tokens * self.n_blk
            ]
            .view(tokens, self.n_blk)
            .to(torch.int32)
            .contiguous()
        )
        self.seq = torch.full((tokens,), ctx, device=dev, dtype=torch.int32)
        last = self.block_table[:, (ctx - 1) // 128].long()
        self.slots = (last * 128 + (ctx - 1) % 128).to(torch.int64)
        self.old_keys = torch.randn(
            self.total_blocks * 128, HEAD_DIM, device=dev, generator=g
        ).to(bf)
        self.iq_in = None
        self.ref = ref

    def run(self, nvfp4, k_gs=0.8, v_gs=1.7, time_reps=0):
        ref, dev, t = self.ref, "cuda", self.tokens
        tot_slots = self.total_blocks * 128
        out = {}
        if nvfp4:
            # sentinel bytes so a write outside the new tokens' own bytes shows
            pages = torch.full(
                (self.total_blocks, KV_PAGE_BYTES), 0x5A, device=dev, dtype=torch.uint8
            )
            idx = torch.zeros(
                self.total_blocks, NVFP4_PAGE_BYTES, device=dev, dtype=torch.uint8
            )
            ref.insert_nvfp4_index(
                idx.view(self.total_blocks, 1, -1),
                self.old_keys,
                torch.arange(tot_slots, device=dev),
            )
            # (the new token's own key is rewritten by K1)
            gs = torch.tensor([k_gs, v_gs], device=dev, dtype=torch.float32)
            caches = {
                "k_cache": pages.data_ptr(), "v_cache": pages.data_ptr(),
                "k_scale": gs.data_ptr(), "v_scale": gs.data_ptr() + 4,
            }  # fmt: skip
            out.update(pages=pages, idx=idx, gs=gs)
            index_cache = idx.data_ptr()
        else:
            kc = torch.zeros(tot_slots * HEAD_DIM, device=dev, dtype=torch.uint8)
            vc = torch.zeros_like(kc)
            ks = torch.zeros(tot_slots, device=dev, dtype=torch.float32)
            vs = torch.zeros_like(ks)
            idx = self.old_keys.float().to(dtypes.fp8).view(torch.uint8).flatten()
            idx = idx.contiguous()
            caches = {
                "k_cache": kc.data_ptr(), "v_cache": vc.data_ptr(),
                "k_scale": ks.data_ptr(), "v_scale": vs.data_ptr(),
            }  # fmt: skip
            out.update(kc=kc, vc=vc, ks=ks, vs=vs, idx=idx)
            index_cache = idx.data_ptr()
        q_out = torch.zeros(MAX_TOKENS, LOCAL_Q_HEADS * HEAD_DIM, device=dev).bfloat16()
        iq_out = torch.zeros(MAX_TOKENS, HEAD_DIM, device=dev).bfloat16()
        h_out = torch.zeros(MAX_TOKENS, HIDDEN, device=dev).bfloat16()
        iscore = torch.zeros(
            2 * MAX_INDEX_BLOCKS * MAX_TOKENS, device=dev, dtype=torch.int32
        )
        scratch = uncached_bytes(SCRATCH_BYTES, torch.device(dev))
        scratch.zero_()
        # the index scores read per-token tables (q_len 1: a row a request)
        launch = build_pre_attn_kernel(
            1e-6, SM_SCALE, 0, 1, t, kv_nvfp4=nvfp4, index_nvfp4=nvfp4
        )
        args = K1_ABI.pack(
            dict(
                ar=self.ar.data_ptr(), res=self.res.data_ptr(), h_out=h_out.data_ptr(),
                g_in=self.g_in.data_ptr(), w_qkv=self.w_qkv.data_ptr(),
                s_qkv=self.s_qkv.data_ptr(), g_q=self.g_q.data_ptr(),
                g_k=self.g_k.data_ptr(), g_iq=self.g_iq.data_ptr(),
                g_ik=self.g_ik.data_ptr(), cos_sin=self.cos_sin.data_ptr(),
                positions=self.positions.data_ptr(), slot_mapping=self.slots.data_ptr(),
                index_cache=index_cache, q_out=q_out.data_ptr(),
                iq_out=iq_out.data_ptr(), block_table=self.block_table.data_ptr(),
                seq_lens=self.seq.data_ptr(), bt_width=self.n_blk, q_len=1,
                iscore=iscore.data_ptr(), scratch=scratch.data_ptr(), layer=3, tl=0,
                **caches,
            )
        )  # fmt: skip
        launch(*args)
        torch.cuda.synchronize()
        if time_reps:  # cold-cache launches, the kernel alone between two events
            flush = torch.empty(1 << 30, device=dev, dtype=torch.uint8)
            times = []
            for _ in range(time_reps):
                scratch.zero_()
                flush.zero_()
                a, b = torch.cuda.Event(True), torch.cuda.Event(True)
                a.record()
                launch(*args)
                b.record()
                torch.cuda.synchronize()
                times.append(a.elapsed_time(b) * 1000)
            out["times_us"] = times
        qkv = scratch[SCRATCH_QKV : SCRATCH_QKV + t * ROWS * 8].view(torch.int32)
        out["qkv"] = qkv.view(-1, 2)[:, 0].view(torch.float32).view(t, ROWS).clone()
        out["iq_out"] = iq_out[:t].clone()
        out["iscore"] = (
            iscore.view(-1, 2)[:, 0]
            .view(torch.float32)
            .view(MAX_TOKENS, MAX_INDEX_BLOCKS)[:t]
            .clone()
        )
        return out


def _f8(x):
    return x.contiguous().view(dtypes.fp8).float()


def _fp8_rows(o, case):
    """The FP8 build's K and V rows of the new tokens (dequantized) and its index-K
    row, [tokens, 128] f32."""
    k, v, ik = [], [], []
    page_bytes = PAGE16 * HEAD_DIM
    for s in case.slots.tolist():
        b, tt = s // PAGE16, s % PAGE16
        kpage = o["kc"][b * page_bytes : (b + 1) * page_bytes]
        vpage = o["vc"][b * page_bytes : (b + 1) * page_bytes]
        kk = kpage.view(HEAD_DIM // 16, PAGE16, 16)[:, tt, :].reshape(-1)
        k.append(_f8(kk) * o["ks"][s])
        v.append(_f8(vpage.view(HEAD_DIM, PAGE16)[:, tt]) * o["vs"][s])
        ik.append(_f8(o["idx"][s * HEAD_DIM : (s + 1) * HEAD_DIM]))
    return torch.stack(k), torch.stack(v), torch.stack(ik)


@pytest.mark.parametrize("tokens, ctx", [(1, 300 * 128 + 5), (4, 40 * 128 + 77)])
def test_k1_nvfp4_pages_and_scores(tokens, ctx):
    scorer, ref = _reference_modules()
    case = _Case(ref, tokens, ctx)
    k_gs, v_gs = 0.8, 1.7
    f8 = case.run(False)
    nv = case.run(True, k_gs, v_gs)
    # the GEMV and the norms are the same code in both builds
    assert torch.equal(f8["iq_out"], nv["iq_out"])
    assert torch.equal(f8["qkv"], nv["qkv"])

    assert nv["qkv"].abs().max() > 0 and torch.isfinite(nv["qkv"]).all()
    pages = nv["pages"]
    new = case.slots.tolist()
    assert (pages != 0x5A).any(), "K1 wrote no page"
    # V: vLLM's writer's bytes for the raw V rows the GEMV made
    v_rows = nv["qkv"][:, V_OFF : V_OFF + HEAD_DIM].to(torch.bfloat16)
    want = torch.full_like(pages, 0x5A)
    ref.insert_nvfp4_kv(
        torch.zeros(tokens, 1, HEAD_DIM, device="cuda", dtype=torch.bfloat16),
        v_rows.view(tokens, 1, HEAD_DIM),
        want.view(-1, 2, 128, 72),
        case.slots,
        k_gs,
        v_gs,
    )
    vside = slice(9216, 18432)  # the page's V half (data, then scales)
    assert torch.equal(pages[:, vside], want[:, vside]), "V pages differ"

    # K: decodes to the FP8 build's K within the two formats' error
    kd, ksc, vd, vsc = ref.nvfp4_kv_cache_token_major(pages.view(-1, 2, 128, 72))
    kx = ref.dequant(kd, ksc)[:, 0] * k_gs  # [blocks, 128, 128]
    f8k, f8v, f8ik = _fp8_rows(f8, case)
    got_k = torch.stack([kx[s // 128, s % 128] for s in new])
    rel = (got_k - f8k).norm(dim=-1) / f8k.norm(dim=-1)
    assert (rel < 0.15).all(), rel
    vx = ref.dequant(vd, vsc)[:, 0] * v_gs
    got_v = torch.stack([vx[s // 128, s % 128] for s in new])
    rel = (got_v - f8v).norm(dim=-1) / f8v.norm(dim=-1)
    assert (rel < 0.15).all(), rel

    # the untouched keys' K bytes: the sentinel, except the new tokens' own rows
    touched = torch.zeros(pages.shape[0], 128, dtype=torch.bool, device="cuda")
    for s in new:
        touched[s // 128, s % 128] = True
    k_data = pages[:, : 128 * 64].view(-1, 128, 64)
    assert (k_data[~touched] == 0x5A).all(), "K data written outside the new rows"

    # index K: decodes to the FP8 build's within the formats' error
    idx = nv["idx"]
    ik, isc = (
        idx[:, : 128 * 64].reshape(-1, 128, 64),
        idx[:, 128 * 64 :].reshape(-1, 128, 8),
    )
    ikx = ref.dequant(ik, isc)
    got_ik = torch.stack([ikx[s // 128, s % 128] for s in new])
    rel = (got_ik - f8ik).norm(dim=-1) / f8ik.norm(dim=-1)
    assert (rel < 0.15).all(), rel

    # scores: aiter's e4m3 NVFP4 scorer on the pages K1 wrote, with K1's own index q as e4m3
    kd_t = torch.as_strided(
        idx, (case.total_blocks, 128, 64), (NVFP4_PAGE_BYTES, 64, 1)
    )
    ks_t = torch.as_strided(
        idx, (case.total_blocks, 128, 8), (NVFP4_PAGE_BYTES, 8, 1), 128 * 64
    )
    want_s = scorer.score_flydsl_nvfp4(
        nv["iq_out"].view(tokens, 1, HEAD_DIM).float().clamp(-448, 448).to(torch.float8_e4m3fn),
        kd_t, ks_t, case.block_table,
        case.seq, 1, 1, SM_SCALE, case.n_blk, k_round="e4m3",
    )  # fmt: skip
    scored = slice(0, case.n_blk - 1)  # the last block holds the new key (a race)
    mine, theirs = nv["iscore"][:, scored], want_s[0, :, scored]
    assert torch.isfinite(mine).all()
    assert torch.equal(mine, theirs), (mine - theirs).abs().max().item()
