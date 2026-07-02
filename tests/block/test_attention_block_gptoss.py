# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Block-level test driving the REAL OAIAttention module end to end (qkv_proj ->
RoPE + paged KV write -> attention with sinks + alternating sliding window + GQA
-> o_proj) vs a torch reference. Exercises ATOM's actual module + arch dispatch
(PagedAttentionImpl), selected via --backend {asm,triton}.

Standalone (no ModelRunner): init TP=1, set a minimal current atom config,
instantiate OAIAttention, allocate the KV cache, build the ForwardContext by hand,
call forward. Run: python3 tests/block/test_attention_block_gptoss.py
"""

import argparse
import os
import sys
from types import SimpleNamespace

import torch

# The gfx1250 serving recipe sets ATOM_USE_UNIFIED_ATTN/AITER_ROPE_TRITON_BACKEND/
# ENABLE_CK; those assume the full ModelRunner and don't compose with this single-
# layer harness. The triton (unified_attention) serving path is instead validated
# by the aiter op_test op_tests/block/test_attention_block.py --backend triton.


# --------------------------------------------------------------------------- #
# TP=1 distributed init (QKV/RowParallelLinear need a model-parallel group)    #
# --------------------------------------------------------------------------- #
def _init_tp1():
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "12361")
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    os.environ.setdefault("LOCAL_RANK", "0")
    torch.cuda.set_device(0)
    from aiter.dist.parallel_state import (
        init_distributed_environment,
        initialize_model_parallel,
    )

    init_distributed_environment(world_size=1, rank=0, local_rank=0, backend="nccl")
    initialize_model_parallel(tensor_model_parallel_size=1)


# --------------------------------------------------------------------------- #
# GPT-OSS attention config (120B/20B share the attention shape)               #
# --------------------------------------------------------------------------- #
def make_hf_config():
    return SimpleNamespace(
        hidden_size=2880,
        num_attention_heads=64,
        num_key_value_heads=8,
        head_dim=64,
        max_position_embeddings=131072,
        sliding_window=128,
        rope_parameters={
            "rope_type": "yarn",
            "rope_theta": 150000.0,
            "factor": 32.0,
            "original_max_position_embeddings": 4096,
            "beta_fast": 32.0,
            "beta_slow": 1.0,
        },
    )


def set_atom_config(hf_config, block_size=16, kv_cache_dtype="bf16"):
    from atom.config import set_current_atom_config

    cfg = SimpleNamespace(
        torch_dtype=torch.bfloat16,
        kv_cache_dtype=kv_cache_dtype,
        kv_cache_block_size=block_size,
        max_model_len=hf_config.max_position_embeddings,
        hf_config=hf_config,
        speculative_config=None,
        compilation_config=SimpleNamespace(static_forward_context={}),
        parallel_config=SimpleNamespace(data_parallel_size=1),
    )
    set_current_atom_config(cfg)
    return cfg


def init_weights(attn):
    """Fill the real layer params with small random values (loader-free)."""
    with torch.no_grad():
        for lin in (attn.qkv_proj, attn.o_proj):
            lin.weight.normal_(0, 0.05)
            if getattr(lin, "bias", None) is not None:
                lin.bias.normal_(0, 0.05)
        attn.sinks.normal_(0, 1.0)


def alloc_kv_cache(cfg, hf, batch, seq_len, layer_num, flash_layout=False):
    """Allocate the paged KV cache and bind it via kv_cache_data (keyed
    'layer_{layer_num}'). asm/AiterBackend uses SHUFFLE/NHD layout; the
    triton/use_flash_layout path uses flash layout [nb, block, nkv, hd].
    bf16 -> no scales."""
    from atom.config import KVCacheTensor
    from atom.utils.forward_context import set_kv_cache_data

    from aiter import dtypes

    nkv = hf.num_key_value_heads
    hd = hf.head_dim
    bs = cfg.kv_cache_block_size
    kvdt = dtypes.fp8 if cfg.kv_cache_dtype == "fp8" else torch.bfloat16
    blocks_per_seq = (seq_len + bs - 1) // bs
    nb = batch * blocks_per_seq + 1
    if flash_layout:
        k_cache = torch.zeros(nb, bs, nkv, hd, dtype=kvdt, device="cuda")
        v_cache = torch.zeros(nb, bs, nkv, hd, dtype=kvdt, device="cuda")
    else:
        x = 16 // torch.empty(0, dtype=kvdt).element_size()  # 8 bf16 / 16 fp8
        k_cache = torch.zeros(nb, nkv, hd // x, bs, x, dtype=kvdt, device="cuda")
        v_cache = torch.zeros(nb, nkv, hd, bs, dtype=kvdt, device="cuda")
    kct = KVCacheTensor(
        layer_num=layer_num,
        k_cache=k_cache,
        v_cache=v_cache,
        k_scale=None,
        v_scale=None,
    )
    set_kv_cache_data({f"layer_{layer_num}": kct})
    return k_cache, v_cache, blocks_per_seq


def _fused_rope(attn, q, k, v, positions):
    """RoPE q,k via the SAME fused kernel the module uses (fused vs forward_native
    differ ~1e-2 on gfx1250), so the reference isolates the attention op, not RoPE.
    Returns roped (q, k); cache writes go to a throwaway buffer."""
    from aiter.ops.triton.fusions.fused_kv_cache import fused_qk_rope_reshape_and_cache

    nq = attn.num_local_attention_heads
    nkv = attn.num_local_key_value_heads
    hd = attn.head_dim
    t = q.shape[0]
    nb = (t + 16 - 1) // 16 + 1
    kc = torch.zeros(nb, 16, nkv, hd, dtype=torch.bfloat16, device="cuda")
    vc = torch.zeros(nb, 16, nkv, hd, dtype=torch.bfloat16, device="cuda")
    sm = torch.arange(t, dtype=torch.int64, device="cuda")
    q2 = q.view(t, nq, hd).contiguous()
    k2 = k.view(t, nkv, hd).contiguous()
    v2 = v.view(t, nkv, hd).contiguous()
    q_r, k_r, _, _ = fused_qk_rope_reshape_and_cache(
        q2,
        k2,
        v2,
        kc,
        vc,
        sm,
        positions.to(torch.int64),
        attn.rotary_emb.cos_cache,
        attn.rotary_emb.sin_cache,
        None,
        None,
        is_neox=attn.rotary_emb.is_neox_style,
        flash_layout=True,
        apply_scale=False,
        output_zeros=False,
    )
    return q_r.view(t, nq, hd), k_r.view(t, nkv, hd)


def _ref_one_seq(q, k, v, sinks, scale, sliding_window, q_offset):
    """Reference GQA attention with per-head sink for one sequence (RoPE'd q,k)."""
    sq, nq, hd = q.shape
    sk, nkv, _ = k.shape
    group = nq // nkv
    qf, kf, vf = q.float(), k.float(), v.float()
    kf = kf.repeat_interleave(group, dim=1)
    vf = vf.repeat_interleave(group, dim=1)
    scores = torch.einsum("qhd,khd->hqk", qf, kf) * scale
    qi = torch.arange(sq, device=q.device).view(sq, 1) + q_offset
    kj = torch.arange(sk, device=q.device).view(1, sk)
    mask = kj <= qi
    if sliding_window > 0:
        mask = mask & (kj > qi - sliding_window)
    scores = scores.masked_fill(~mask.view(1, sq, sk), float("-inf"))
    m = scores.max(dim=-1, keepdim=True).values
    sink = sinks.view(nq, 1, 1).float()
    m = torch.maximum(m, sink)
    p = torch.exp(scores - m)
    denom = p.sum(dim=-1, keepdim=True) + torch.exp(sink - m)
    p = p / denom
    return torch.einsum("hqk,khd->qhd", p, vf)


def ref_prefill(attn, hidden, cu, positions, sliding_window):
    """Golden: reuse the real qkv_proj/o_proj + rotary_emb, torch attention core."""
    hd = attn.head_dim
    nq, nkv = attn.num_local_attention_heads, attn.num_local_key_value_heads
    qkv = (
        attn.qkv_proj(hidden)[0]
        if isinstance(attn.qkv_proj(hidden), tuple)
        else attn.qkv_proj(hidden)
    )
    q, k, v = torch.split(qkv, [attn.q_size, attn.kv_size, attn.kv_size], dim=-1)
    q = q.view(-1, nq, hd).contiguous()
    k = k.view(-1, nkv, hd).contiguous()
    v = v.view(-1, nkv, hd).contiguous()
    q, k = _fused_rope(attn, q, k, v, positions)
    outs = []
    cu = cu.tolist()
    for b in range(len(cu) - 1):
        s, e = cu[b], cu[b + 1]
        outs.append(
            _ref_one_seq(
                q[s:e], k[s:e], v[s:e], attn.sinks, attn.scaling, sliding_window, 0
            )
        )
    o = torch.cat(outs, dim=0).to(hidden.dtype).view(-1, nq * hd)
    out = attn.o_proj(o)
    return out[0] if isinstance(out, tuple) else out


def run_prefill(attn, hf, cfg, batch, seq_len, layer_num, args, flash=False):
    from aiter.test_common import checkAllclose
    from atom.utils.forward_context import (
        AttentionMetaData,
        Context,
        set_forward_context,
    )

    sliding_window = hf.sliding_window if layer_num % 2 == 0 else -1
    _, _, bps = alloc_kv_cache(cfg, hf, batch, seq_len, layer_num, flash_layout=flash)
    total = batch * seq_len
    hidden = (torch.randn(total, hf.hidden_size, device="cuda") * 0.1).to(
        torch.bfloat16
    )
    cu = torch.arange(
        0, (batch + 1) * seq_len, seq_len, dtype=torch.int32, device="cuda"
    )
    positions = torch.arange(seq_len, dtype=torch.int64, device="cuda").repeat(batch)

    paged_bt = torch.arange(batch * bps, dtype=torch.int32, device="cuda").view(
        batch, bps
    )
    seq_ids = torch.arange(batch, device="cuda").repeat_interleave(seq_len)
    in_seq = torch.arange(seq_len, device="cuda").repeat(batch)
    bsz = cfg.kv_cache_block_size
    slot_mapping = paged_bt[seq_ids, in_seq // bsz].to(torch.int64) * bsz + (
        in_seq % bsz
    )
    context_lens = torch.full((batch,), seq_len, dtype=torch.int32, device="cuda")
    # triton pure-prefill treats raw K/V as a block_size=1 cache + a fake per-token
    # table (cu[i]+j); asm reads K/V directly (block_tables unused).
    if flash:
        block_tables = torch.arange(total, dtype=torch.int32, device="cuda").view(
            batch, seq_len
        )
    else:
        block_tables = paged_bt

    md = AttentionMetaData(
        cu_seqlens_q=cu,
        cu_seqlens_k=cu,
        max_seqlen_q=seq_len,
        max_seqlen_k=seq_len,
        min_seqlen_q=0,
        slot_mapping=slot_mapping,
        context_lens=context_lens,
        block_tables=block_tables,
        dropout_p=0.0,
    )
    ctx = Context(
        positions=positions, is_prefill=True, is_dummy_run=False, batch_size=batch
    )
    set_forward_context(md, cfg, ctx)

    out = attn(hidden, positions)
    ref = ref_prefill(attn, hidden, cu, positions, sliding_window)
    err = checkAllclose(
        ref,
        out,
        rtol=args.rtol,
        atol=args.atol,
        msg=f"prefill b{batch} s{seq_len} layer{layer_num} win{sliding_window}",
    )
    return err


def ref_decode(attn, hidden_ctx, ctx_len, batch, sliding_window, fp8=False):
    """Golden decode: last-token query attends full context (real qkv/o + rope).
    With fp8 KV, round-trip post-RoPE K and V through e4m3 with the impl's
    per-tensor scale so the reference sees the same quantized cache the kernel
    reads back."""
    hd = attn.head_dim
    nq, nkv = attn.num_local_attention_heads, attn.num_local_key_value_heads
    qkv = attn.qkv_proj(hidden_ctx)
    qkv = qkv[0] if isinstance(qkv, tuple) else qkv
    q, k, v = torch.split(qkv, [attn.q_size, attn.kv_size, attn.kv_size], dim=-1)
    q = q.view(-1, nq, hd).contiguous()
    k = k.view(-1, nkv, hd).contiguous()
    v = v.view(-1, nkv, hd).contiguous()
    positions = torch.arange(ctx_len, device="cuda").repeat(batch)
    q, k = attn.rotary_emb.forward_native(positions.long(), q, k)
    if fp8:
        from aiter import dtypes

        s = attn.attn.impl.kv_scale.to("cuda").float()
        fmax = torch.finfo(dtypes.fp8).max

        def rt(t):
            return ((t.float() / s).clamp(-fmax, fmax).to(dtypes.fp8).float() * s).to(
                t.dtype
            )

        k, v = rt(k), rt(v)
    q = q.view(batch, ctx_len, nq, hd)
    k = k.view(batch, ctx_len, nkv, hd)
    v = v.view(batch, ctx_len, nkv, hd)
    outs = []
    for b in range(batch):
        outs.append(
            _ref_one_seq(
                q[b, -1:],
                k[b],
                v[b],
                attn.sinks,
                attn.scaling,
                sliding_window,
                ctx_len - 1,
            )
        )
    o = torch.cat(outs, dim=0).to(hidden_ctx.dtype).view(batch, nq * hd)
    out = attn.o_proj(o)
    return out[0] if isinstance(out, tuple) else out


def run_decode(attn, hf, cfg, batch, ctx_len, layer_num, args, flash=False):
    from aiter.test_common import checkAllclose
    from atom.utils.forward_context import (
        AttentionMetaData,
        Context,
        set_forward_context,
    )

    sliding_window = hf.sliding_window if layer_num % 2 == 0 else -1
    _, _, bps = alloc_kv_cache(cfg, hf, batch, ctx_len, layer_num, flash_layout=flash)
    bsz = cfg.kv_cache_block_size
    block_tables = torch.arange(batch * bps, dtype=torch.int32, device="cuda").view(
        batch, bps
    )
    hidden_ctx = (torch.randn(batch * ctx_len, hf.hidden_size, device="cuda") * 0.1).to(
        torch.bfloat16
    )

    # ---- 1) Pre-fill the cache with the full context (prefill pass; untimed) ----
    cu = torch.arange(
        0, (batch + 1) * ctx_len, ctx_len, dtype=torch.int32, device="cuda"
    )
    pos_ctx = torch.arange(ctx_len, dtype=torch.int64, device="cuda").repeat(batch)
    seq_ids = torch.arange(batch, device="cuda").repeat_interleave(ctx_len)
    in_seq = torch.arange(ctx_len, device="cuda").repeat(batch)
    slot_ctx = block_tables[seq_ids, in_seq // bsz].to(torch.int64) * bsz + (
        in_seq % bsz
    )
    ctx_lens_full = torch.full((batch,), ctx_len, dtype=torch.int32, device="cuda")
    # The fill pass attention output is discarded (rope_cache writes the paged
    # cache via slot_mapping); the triton prefill path still needs the fake
    # per-token block table to avoid OOB.
    fill_bt = (
        torch.arange(batch * ctx_len, dtype=torch.int32, device="cuda").view(
            batch, ctx_len
        )
        if flash
        else block_tables
    )
    md_pf = AttentionMetaData(
        cu_seqlens_q=cu,
        cu_seqlens_k=cu,
        max_seqlen_q=ctx_len,
        max_seqlen_k=ctx_len,
        min_seqlen_q=0,
        slot_mapping=slot_ctx,
        context_lens=ctx_lens_full,
        block_tables=fill_bt,
        dropout_p=0.0,
    )
    set_forward_context(
        md_pf, cfg, Context(positions=pos_ctx, is_prefill=True, batch_size=batch)
    )
    _ = attn(hidden_ctx, pos_ctx)  # writes roped K/V for all ctx tokens into cache

    # ---- 2) Decode step: last token per seq, attends over full context ----
    hidden_last = hidden_ctx.view(batch, ctx_len, hf.hidden_size)[:, -1].contiguous()
    pos_last = torch.full((batch,), ctx_len - 1, dtype=torch.int64, device="cuda")
    last = ctx_len - 1
    slot_last = block_tables[:, last // bsz].to(torch.int64) * bsz + (last % bsz)
    cu_q = torch.arange(batch + 1, dtype=torch.int32, device="cuda")
    md_dec = AttentionMetaData(
        cu_seqlens_q=cu_q,
        cu_seqlens_k=cu,
        max_seqlen_q=1,
        max_seqlen_k=ctx_len,
        min_seqlen_q=0,
        slot_mapping=slot_last,
        context_lens=ctx_lens_full,
        block_tables=block_tables,
        dropout_p=0.0,
    )
    set_forward_context(
        md_dec, cfg, Context(positions=pos_last, is_prefill=False, batch_size=batch)
    )
    out = attn(hidden_last, pos_last)

    is_fp8 = cfg.kv_cache_dtype == "fp8"
    ref = ref_decode(attn, hidden_ctx, ctx_len, batch, sliding_window, fp8=is_fp8)
    rtol = max(args.rtol, 6e-2) if is_fp8 else args.rtol
    atol = max(args.atol, 6e-2) if is_fp8 else args.atol
    err = checkAllclose(
        ref,
        out,
        rtol=rtol,
        atol=atol,
        msg=f"decode  b{batch} ctx{ctx_len} layer{layer_num} win{sliding_window}",
    )
    return err


def make_attn(hf, layer_num, flash, kv_cache_dtype="bf16"):
    """Build a fresh real OAIAttention for one layer; set the arch backend."""
    from atom.models.gpt_oss import OAIAttention

    attn = (
        OAIAttention(
            hf,
            quant_config=None,
            cache_config=kv_cache_dtype,
            prefix="model.layers.%d.self_attn" % layer_num,
            layer_num=layer_num,
        )
        .cuda()
        .to(torch.bfloat16)
    )
    init_weights(attn)
    attn.rotary_emb.cos_cache = attn.rotary_emb.cos_cache.cuda()
    attn.rotary_emb.sin_cache = attn.rotary_emb.sin_cache.cuda()
    # use_flash_layout is normally set by the backend's build_kv_cache_tensor,
    # which we bypass; set it here to pick asm (False) vs triton (True).
    attn.attn.impl.use_flash_layout = flash
    return attn


def run_sweep(hf, cfg, backend, flash, args):
    """Serving-scenario sweep (input/output = --sweep-io, default 8192/1024) on the
    real OAIAttention: prefill (batch 1, latency + correctness) and a per-step decode
    at each --conc over a pre-filled ctx=input+output cache (latency only)."""
    import pandas as pd
    from aiter.test_common import run_perftest
    from atom.utils.forward_context import (
        AttentionMetaData,
        Context,
        set_forward_context,
    )

    inp, out_len = args.sweep_io
    ctx_len = inp + out_len
    bsz = cfg.kv_cache_block_size
    print(
        f"sweep: input={inp} output={out_len} (decode ctx={ctx_len}) conc={args.conc}"
    )
    rows = []
    for ln in (0, 1):
        tag = "swa" if ln % 2 == 0 else "causal"
        attn = make_attn(hf, ln, flash, kv_cache_dtype=args.kv_cache_dtype)
        # prefill: correctness (sets cache + fwd ctx) then latency reusing that ctx.
        err = run_prefill(attn, hf, cfg, 1, inp, ln, args, flash=flash)
        ok = err == 0 or (isinstance(err, float) and err < 0.02)
        hid = (torch.randn(inp, hf.hidden_size, device="cuda") * 0.1).to(torch.bfloat16)
        pos = torch.arange(inp, dtype=torch.int64, device="cuda")
        _, us = run_perftest(
            attn, hid, pos, num_iters=args.iters, num_warmup=3, num_rotate_args=1
        )
        rows.append(
            {
                "phase": "prefill",
                "layer": tag,
                "conc": 1,
                "us": round(us, 1),
                "pass": ok,
            }
        )
        # decode sweep reuses the SAME attn (re-creating would duplicate the
        # static_forward_context registration for this layer).
        for conc in args.conc:
            _, _, bps = alloc_kv_cache(cfg, hf, conc, ctx_len, ln, flash_layout=flash)
            bt = torch.arange(conc * bps, dtype=torch.int32, device="cuda").view(
                conc, bps
            )
            cl = torch.full((conc,), ctx_len, dtype=torch.int32, device="cuda")
            last = ctx_len - 1
            slot = bt[:, last // bsz].to(torch.int64) * bsz + (last % bsz)
            cu_q = torch.arange(conc + 1, dtype=torch.int32, device="cuda")
            cu_k = torch.arange(
                0, (conc + 1) * ctx_len, ctx_len, dtype=torch.int32, device="cuda"
            )
            pos = torch.full((conc,), last, dtype=torch.int64, device="cuda")
            md = AttentionMetaData(
                cu_seqlens_q=cu_q,
                cu_seqlens_k=cu_k,
                max_seqlen_q=1,
                max_seqlen_k=ctx_len,
                min_seqlen_q=0,
                slot_mapping=slot,
                context_lens=cl,
                block_tables=bt,
                dropout_p=0.0,
            )
            set_forward_context(
                md, cfg, Context(positions=pos, is_prefill=False, batch_size=conc)
            )
            hs = (torch.randn(conc, hf.hidden_size, device="cuda") * 0.1).to(
                torch.bfloat16
            )
            _, us = run_perftest(
                attn, hs, pos, num_iters=args.iters, num_warmup=3, num_rotate_args=1
            )
            rows.append(
                {
                    "phase": "decode",
                    "layer": tag,
                    "conc": conc,
                    "us": round(us, 1),
                    "pass": "(perf)",
                }
            )

    print("\n" + pd.DataFrame(rows).to_string(index=False))
    fail = sum(1 for r in rows if r["pass"] is False)
    if fail:
        print(f"\n{fail} prefill check(s) FAILED")
        sys.exit(1)
    print("\nSweep done (decode rows are latency-only).")


def main():
    import aiter

    p = argparse.ArgumentParser()
    p.add_argument("--phase", default="both", choices=["prefill", "decode", "both"])
    p.add_argument(
        "--layer",
        default="both",
        choices=["0", "1", "both"],
        help="0=even/sliding-window, 1=odd/causal",
    )
    p.add_argument(
        "--backend",
        default="auto",
        choices=["auto", "asm", "triton"],
        help="auto = triton on gfx1250, else asm",
    )
    p.add_argument("--kv-cache-dtype", default="bf16", choices=["bf16", "fp8"])
    p.add_argument(
        "--block-size",
        type=int,
        default=16,
        help="KV cache page size (serving uses 128)",
    )
    p.add_argument("--batch", type=int, default=2)
    p.add_argument("--seqlen", type=int, default=256)
    p.add_argument("--ctx-len", type=int, default=300)
    p.add_argument("--atol", type=float, default=2e-2)
    p.add_argument("--rtol", type=float, default=2e-2)
    p.add_argument("--iters", type=int, default=10)
    p.add_argument(
        "--sweep",
        action="store_true",
        help="serving-scenario sweep: prefill + per-step decode over --conc at "
        "input/output = --sweep-io (default 8192/1024)",
    )
    p.add_argument(
        "--sweep-io", type=int, nargs=2, default=[8192, 1024], metavar=("IN", "OUT")
    )
    p.add_argument("--conc", type=int, nargs="+", default=[1, 16, 64, 128, 256, 512])
    args = p.parse_args()

    _init_tp1()
    torch.set_default_device("cuda")
    torch.manual_seed(0)
    gfx = aiter.get_gfx()
    backend = (
        ("triton" if gfx == "gfx1250" else "asm")
        if args.backend == "auto"
        else args.backend
    )
    flash = backend == "triton"
    print(f"gfx={gfx} backend={backend} (use_flash_layout={flash})")

    hf = make_hf_config()
    cfg = set_atom_config(
        hf, block_size=args.block_size, kv_cache_dtype=args.kv_cache_dtype
    )
    print(f"kv_cache_dtype={args.kv_cache_dtype} block_size={args.block_size}")
    if args.sweep:
        run_sweep(hf, cfg, backend, flash, args)
        return
    layers = [0, 1] if args.layer == "both" else [int(args.layer)]
    phases = ["prefill", "decode"] if args.phase == "both" else [args.phase]

    rows, fail = [], 0
    for ln in layers:
        attn = make_attn(hf, ln, flash, kv_cache_dtype=args.kv_cache_dtype)
        for ph in phases:
            if ph == "prefill":
                err = run_prefill(
                    attn, hf, cfg, args.batch, args.seqlen, ln, args, flash=flash
                )
                sz = args.seqlen
            else:
                err = run_decode(
                    attn, hf, cfg, args.batch, args.ctx_len, ln, args, flash=flash
                )
                sz = args.ctx_len
            ok = err == 0 or (isinstance(err, float) and err < 0.02)
            fail += 0 if ok else 1
            rows.append((ph, "swa128" if ln % 2 == 0 else "causal", sz, ok))

    print(f"\n{'phase':8} {'layer':8} {'size':>6} {'pass':>6}")
    for ph, lt, sz, ok in rows:
        print(f"{ph:8} {lt:8} {sz:>6} {str(ok):>6}")
    print("\nAll passed." if not fail else f"\n{fail} FAILED")
    sys.exit(0 if not fail else 1)


if __name__ == "__main__":
    main()
