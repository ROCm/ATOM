# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Block-level test driving the REAL DeepseekV4Attention module end to end
(wqkv_a -> q_norm -> wq_b -> qk_norm+RoPE -> SWA write -> Compressor/Indexer ->
sparse paged attention -> inverse RoPE -> grouped output LoRA -> wo_b) vs a torch
reference. DSV4 sibling of tests/block/test_attention_block_gptoss.py (PR #1402).

Covers all three V4 layer types, selected by compress_ratio:
    0   Dense  sliding-window only, no compressor/indexer
    4   CSA    compressor + indexer top-k
    128 HCA    compressor only (deterministic compress plan)

Standalone (no ModelRunner): init TP=1, set a minimal current atom config, build
the REAL DeepseekV4AttentionMetadataBuilder against a duck-typed model_runner,
let it allocate + bind the KV/state caches, drive prepare_prefill/prepare_decode
with a synthetic batch, then call the module.

Reusing the real metadata builder is deliberate: AttentionMetaData_DSV4 carries
compress plans, per-seq state slots, batch_id_per_token and three ragged paged
index sets, so hand-rolling it would mean reimplementing ~2700 lines. This way
the test also covers production metadata construction.

What the reference does and does not prove
------------------------------------------
The attention CORE is recomputed independently in torch (`ref_sparse_attn`);
everything around it -- projections, q/kv norm + RoPE, inverse RoPE, grouped
output LoRA -- is the real module, so a mismatch localises to the sparse
attention rather than to a projection. Same split as the GPT-OSS test.

For Dense the KV set is analytic, and `check_dense_window` verifies the
builder's gather list against the sliding window independently, so that case is
end-to-end independent. For CSA/HCA the reference gathers the indices the module
actually selected: it validates the gather, the attention math, inverse RoPE and
the output LoRA, but takes the indexer's CHOICE of top-k as given. Reproducing
the DSA indexer scoring in torch is deliberately out of scope here.

Each config prints what it exercised (per-token KV-set sizes, whether the CSA
top-k truncated) -- CSA only truncates once ctx//4 > index_topk, so a short-
context run silently covers only the easy case.

Run: python3 tests/block/test_attention_block_dsv4.py
     python3 tests/block/test_attention_block_dsv4.py --layer csa --seqlen 2560
"""

import argparse
import json
import os
import sys
from types import SimpleNamespace

import numpy as np
import torch

DEFAULT_CONFIG = "/data/DeepSeek-V4-Flash/config.json"

# layer_id -> compress_ratio for the synthetic 3-layer model: one of each type.
LAYER_KINDS = {"dense": (0, 0), "csa": (4, 1), "hca": (128, 2)}


# --------------------------------------------------------------------------- #
# TP=1 distributed init (wq_b / wo_a are Column/RowParallelLinear)             #
# --------------------------------------------------------------------------- #
def _init_tp1():
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "12377")
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
# Config                                                                        #
# --------------------------------------------------------------------------- #
def make_hf_config(path, max_model_len):
    """Real V4 attention geometry from config.json, shrunk to 3 layers (one per
    compress_ratio kind) so a single harness covers Dense / CSA / HCA."""
    with open(path) as f:
        raw = json.load(f)
    raw["compress_ratios"] = [0, 4, 128]
    raw["num_hidden_layers"] = 3
    raw["num_nextn_predict_layers"] = 0
    raw["max_position_embeddings"] = max_model_len
    hf = SimpleNamespace(**raw)
    # The metadata builder reads kv_head_dim; config.json spells it head_dim.
    hf.kv_head_dim = raw["head_dim"]
    return hf


def _dspark_config():
    """Real DSparkConfig if this tree has one (prepare_decode reads
    config.dspark.ragged unguarded), else an all-defaults stand-in."""
    try:
        from atom.config import DSparkConfig

        return DSparkConfig()
    except ImportError:
        return SimpleNamespace(
            confidence_schedule=False,
            ragged=False,
            ragged_graph_sizes="",
            q_buckets="",
            disable_sps_calib=False,
        )


def set_atom_config(hf, block_size, kv_cache_dtype, max_model_len):
    from atom.config import set_current_atom_config

    cfg = SimpleNamespace(
        torch_dtype=torch.bfloat16,
        kv_cache_dtype=kv_cache_dtype,
        # Real Config defaults this to kv_cache_dtype in __post_init__; the
        # indexer reads it unguarded (deepseek_v4.py: get_current_atom_config()
        # .index_cache_dtype), so the stub has to mirror the default.
        index_cache_dtype=kv_cache_dtype,
        kv_cache_block_size=block_size,
        max_model_len=max_model_len,
        hf_config=hf,
        speculative_config=None,
        kv_transfer_config=None,
        enable_tbo_decode=False,
        dspark=_dspark_config(),
        # cudagraph_mode=None keeps DeepseekV4Attention.forward on the WIDE
        # split (-> forward_impl) instead of the PIECEWISE custom-op path.
        compilation_config=SimpleNamespace(
            static_forward_context={}, cudagraph_mode=None
        ),
        parallel_config=SimpleNamespace(data_parallel_size=1),
    )
    set_current_atom_config(cfg)
    return cfg


class StubRunner:
    """The slice of ModelRunner the V4 metadata builder actually touches.

    Deliberately does NOT define `drafter`: the builder does
    `int(model_runner.drafter.mtp_k) if hasattr(...)`, so any attribute here
    (even None) would break it. Absent => max_spec_steps = 0.
    """

    def __init__(self, cfg, max_bs, max_num_batched_tokens, num_blocks, num_swa_blocks):
        from atom.utils import CpuGpuBuffer

        self.config = cfg
        self.device = torch.device("cuda")
        # Must be a multiple of the builder's own block_size (128).
        self.block_size = cfg.kv_cache_block_size
        self.max_bs = max_bs
        self.max_num_batched_tokens = max_num_batched_tokens
        self.num_physical_kvcache_blocks = num_blocks
        self.num_swa_blocks = num_swa_blocks
        # Non-zero or _build_paged_prefill_meta early-returns (its warmup
        # guard), leaving kv_indices_prefix_* as None and the prefill kernel
        # dereferencing None. Must match allocate_per_req_cache's num_slots.
        self.max_per_req_cache_slots = max_bs
        self.kv_cache_dtype = cfg.kv_cache_dtype
        self.enforce_eager = True
        self.capture_cudagraph = False
        self.warmup_model = False
        # prepare_decode overlaps H2D on this stream (prep_stream.wait_stream).
        self.async_execute_stream = torch.cuda.Stream()
        self.tokenID_processor = None
        # ModelRunner owns "positions"; the builder's base __init__ adds the
        # rest (slot_mapping / context_lens / block_tables / cu_seqlens_*).
        self.forward_vars = {
            "positions": CpuGpuBuffer(
                max_num_batched_tokens, dtype=torch.int64, device="cuda"
            ),
        }

    def get_num_blocks(self, *a, **k):
        return self.num_physical_kvcache_blocks


# --------------------------------------------------------------------------- #
# Harness construction                                                          #
# --------------------------------------------------------------------------- #
def build_layers(hf, cfg, runner, kinds):
    """Instantiate the real attention modules + bind caches via the real builder."""
    from atom.model_ops.attentions.deepseek_v4_attn import (
        DeepseekV4AttentionMetadataBuilder,
    )
    from atom.models.deepseek_v4 import (
        DeepseekV4Args,
        DeepseekV4Attention,
        make_v4_quant_config,
    )

    args = DeepseekV4Args.from_hf_config(hf)
    # DeepseekV4ForCausalLM normally does this; without it q_norm's
    # fused_quant path is inactive and _attn_pre's `qr, qr_scale =
    # self.q_norm(...)` unpack fails.
    args.quant_config = make_v4_quant_config(hf, model_path=None)
    cfg.quant_config = args.quant_config
    builder = DeepseekV4AttentionMetadataBuilder(runner)

    # KV pools first: allocate_* setattr onto the runner, build_kv_cache_tensor
    # then reads them back to bind per-module views.
    for name, tensor in builder.allocate_kv_cache_tensors(1, 0).items():
        setattr(runner, name, tensor)
    for name, tensor in builder.allocate_per_req_cache(runner.max_bs).items():
        setattr(runner, name, tensor)

    layers = {}
    # ModelRunner builds the model under a bf16 default dtype; without this the
    # implicitly-typed norm weights come out fp32 and the flydsl qk_norm_rope
    # kernel rejects them. Explicitly-fp32 params (attn_sink) stay fp32.
    prev_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    try:
        for kind in kinds:
            _, layer_id = LAYER_KINDS[kind]
            attn = DeepseekV4Attention(
                layer_id, args, prefix=f"layers.{layer_id}.attn"
            ).cuda()
            init_weights(attn)
            # ModelRunner.allocate_kv_cache() walks every nn.Module; the builder
            # dispatches on type (Attention / Indexer / Compressor).
            for m in attn.modules():
                builder.build_kv_cache_tensor(layer_id, m)
            layers[kind] = attn
    finally:
        torch.set_default_dtype(prev_dtype)
    return args, builder, layers


def init_weights(attn):
    """Small random values for every real parameter (loader-free).

    Quantized (fp8/fp4) weights can't take normal_ directly, so they're filled
    via a bf16 draw and cast; their scale params are set to 1."""
    lowp = (torch.float8_e4m3fn, torch.float8_e4m3fnuz, torch.uint8)
    with torch.no_grad():
        for name, p in attn.named_parameters():
            if p.dtype in lowp:
                if "scale" in name:
                    (
                        p.fill_(1)
                        if p.dtype == torch.uint8
                        else p.copy_(
                            torch.ones_like(p, dtype=torch.float32).to(p.dtype)
                        )
                    )
                else:
                    p.copy_((torch.randn(p.shape, device=p.device) * 0.05).to(p.dtype))
            elif p.dtype.is_floating_point:
                p.normal_(0, 0.05) if "scale" not in name else p.fill_(1.0)
        attn.attn_sink.normal_(0, 1.0)
    if hasattr(attn, "process_weights_after_loading"):
        attn.process_weights_after_loading()


def make_batch(ctx_lens, num_cached, block_tables, swa_block_tables, is_prefill):
    """Duck-typed ScheduledBatch: only the 9 fields the builder reads."""
    bs = len(ctx_lens)
    new_tokens = [c - n for c, n in zip(ctx_lens, num_cached)]
    total = sum(new_tokens)
    return SimpleNamespace(
        context_lens=np.asarray(ctx_lens, dtype=np.int32),
        num_cached_tokens=list(num_cached),
        block_tables=block_tables,
        swa_block_tables=swa_block_tables,
        per_req_cache_groups=list(range(bs)),
        is_dummy_run=False,
        num_spec_step=0,
        total_seqs_num_prefill=bs if is_prefill else 0,
        total_tokens_num_prefill=total if is_prefill else 0,
        total_seqs_num_decode=0 if is_prefill else bs,
        total_tokens_num_decode=0 if is_prefill else total,
    )


def paged_tables(bs, ctx_len, block_size, num_blocks_per_seq):
    """Sequential physical blocks; separate pools for compressed KV and SWA."""
    bt, swa = [], []
    for b in range(bs):
        base = b * num_blocks_per_seq
        bt.append(list(range(base, base + num_blocks_per_seq)))
        swa.append(list(range(base, base + num_blocks_per_seq)))
    return bt, swa


# --------------------------------------------------------------------------- #
# Torch reference                                                               #
# --------------------------------------------------------------------------- #
def ref_sparse_attn(q, pool, topk, attn_sink, scale, chunk=128):
    """Independent torch sparse MQA with per-head sink.

    q    [T, H, D]   pool [P, D]   topk [T, K] int64 (-1 = skip)   sink [H]
    Single shared KV head acts as both K and V (MLA latent); `attn_sink` enters
    the softmax DENOMINATOR only, never the numerator.

    Chunked over tokens: the gathered [T, K, D] block is ~1.3 GB in fp32 at
    T=1024/K=640/D=512, which is wasteful to materialise in one go.
    """
    T, H, D = q.shape
    out = torch.empty(T, H, D, dtype=torch.float32, device=q.device)
    sink = attn_sink.float().view(1, H, 1)
    for s in range(0, T, chunk):
        e = min(s + chunk, T)
        idx = topk[s:e]
        valid = idx >= 0
        kvg = pool[idx.clamp(min=0).long()].float()  # [c, K, D]
        sc = torch.einsum("thd,tkd->thk", q[s:e].float(), kvg) * scale
        sc = sc.masked_fill(~valid.unsqueeze(1), float("-inf"))
        m = torch.maximum(sc.max(dim=-1, keepdim=True).values, sink)
        p = torch.exp(sc - m)
        p = torch.where(valid.unsqueeze(1), p, torch.zeros_like(p))
        p = p / (p.sum(-1, keepdim=True) + torch.exp(sink - m))
        out[s:e] = torch.einsum("thk,tkd->thd", p, kvg)
    return out


def build_topk(rec, T):
    """prefix indices (into unified_kv) ++ extend indices (into kv, offset by
    len(unified_kv)) -> a single [T, K] gather table over cat([unified_kv, kv]).

    Decode has a single source (the paged pool; its SWA write already ran), so
    the extend section is empty there."""
    n_pages = rec["unified_kv"].shape[0]
    pi, pp = rec["kvi_p"].long(), rec["kvp_p"].long()
    if rec["kvi_e"] is None:
        ei = torch.zeros(0, dtype=torch.int64, device=pi.device)
        ep = torch.zeros(T + 1, dtype=torch.int64, device=pi.device)
    else:
        ei, ep = rec["kvi_e"].long(), rec["kvp_e"].long()
    p_len = (pp[1 : T + 1] - pp[:T]).clamp(min=0)
    e_len = (ep[1 : T + 1] - ep[:T]).clamp(min=0)
    k_dim = max(int((p_len + e_len).max().item()), 1)
    dev = pi.device
    # Vectorised ragged-to-dense: a per-token python loop costs one GPU op per
    # token, which dominates the whole test at T in the thousands.
    col = torch.arange(k_dim, device=dev).unsqueeze(0)  # [1, K]
    topk = torch.full((T, k_dim), -1, dtype=torch.int64, device=dev)

    p_mask = col < p_len.unsqueeze(1)
    if pi.numel():
        p_src = (pp[:T].unsqueeze(1) + col).clamp(max=pi.numel() - 1)
        topk = torch.where(p_mask, pi[p_src], topk)

    # Extend columns sit after the prefix ones, so shift the column index.
    e_col = col - p_len.unsqueeze(1)
    e_mask = (e_col >= 0) & (e_col < e_len.unsqueeze(1))
    if ei.numel():
        e_src = (ep[:T].unsqueeze(1) + e_col.clamp(min=0)).clamp(max=ei.numel() - 1)
        e_val = ei[e_src]
        # extend indices address `kv`, which sits after unified_kv in the pool;
        # -1 sentinels must stay -1.
        e_val = torch.where(e_val >= 0, e_val + n_pages, e_val)
        topk = torch.where(e_mask, e_val, topk)
    return topk


def ref_block_output(attn, rec, positions):
    """Torch attention core -> the module's own inverse-RoPE + output LoRA.

    Mirrors the GPT-OSS test's split: everything except the attention core is
    the real module (projections, RoPE, wo_a/wo_b), so a mismatch localises to
    the sparse attention itself.
    """
    T = rec["q"].shape[0]
    pool = (
        rec["unified_kv"]
        if rec["kv"] is None
        else torch.cat([rec["unified_kv"], rec["kv"]], dim=0)
    )
    topk = build_topk(rec, T)
    o = ref_sparse_attn(rec["q"], pool, topk, rec["sink"], rec["scale"])
    o = o.to(torch.bfloat16).contiguous()
    rd = attn.rope_head_dim
    attn.rotary_emb.inverse(positions, o[..., -rd:], prefix="ref.inverse_rope")
    return attn._attn_post(o.reshape(T, -1))


def coverage(rec, T, index_topk):
    """What the config actually exercised: per-token KV-set sizes, split into
    the paged prefix (compressed entries for CSA/HCA, SWA history) and the
    per-fwd extend window. Printed so a run can't silently cover only the easy
    case -- e.g. CSA only truncates to index_topk once ctx//4 exceeds it."""
    pp = rec["kvp_p"].long()
    p_len = (pp[1 : T + 1] - pp[:T]).clamp(min=0)
    if rec["kvp_e"] is None:
        e_len = torch.zeros_like(p_len)
    else:
        ep = rec["kvp_e"].long()
        e_len = (ep[1 : T + 1] - ep[:T]).clamp(min=0)
    tot = p_len + e_len
    trunc = int(p_len.max().item()) >= index_topk
    return (
        f"kv/token max={int(tot.max())} (prefix={int(p_len.max())}, "
        f"extend={int(e_len.max())}){' topk-truncated' if trunc else ''}"
    )


def compare(ref, got, atol, rtol, msg):
    """Strict elementwise check + explicit error numbers.

    NB: aiter's checkAllclose returns the *fraction* of mismatching elements, so
    the `err < 0.05` idiom used elsewhere silently tolerates a 5%-wrong tensor.
    Here any mismatching element fails, and the actual max abs/rel deltas are
    printed so a near-miss is visible rather than hidden behind a boolean.
    """
    ref32, got32 = ref.float(), got.float()
    diff = (ref32 - got32).abs()
    max_abs = diff.max().item()
    # Relative error only makes sense against elements that carry signal;
    # floor the denominator at 1% of the tensor's scale so a near-zero
    # reference element doesn't report a meaningless 1000x ratio.
    scale = ref32.abs().max().clamp(min=1e-6)
    max_rel = (diff / ref32.abs().clamp(min=0.01 * scale)).max().item()
    bad = (~torch.isclose(ref32, got32, rtol=rtol, atol=atol)).sum().item()
    n = ref.numel()
    ok = bad == 0
    print(
        f"  {msg}: max_abs={max_abs:.4g} max_rel={max_rel:.4g} "
        f"mismatch={bad}/{n} -> {'PASS' if ok else 'FAIL'}"
    )
    return ok


class Capture:
    """Record the sparse-attn call so the reference sees exactly the tensors the
    kernel saw. q_sa must be cloned BEFORE the call: the prefill wrapper passes
    `out=q_sa`, so the kernel overwrites it in place."""

    def __init__(self):
        self.rec = None

    def wrap_prefill(self, orig):
        def f(q, unified_kv, kvi_p, kvp_p, kv, kvi_e, kvp_e, sink, scale, *a, **kw):
            self.rec = {
                "kind": "prefill",
                "q": q.detach().clone(),
                "unified_kv": unified_kv.detach().clone(),
                "kvi_p": kvi_p.detach().clone(),
                "kvp_p": kvp_p.detach().clone(),
                "kv": kv.detach().clone(),
                "kvi_e": kvi_e.detach().clone(),
                "kvp_e": kvp_e.detach().clone(),
                "sink": sink.detach().clone(),
                "scale": scale,
            }
            return orig(
                q, unified_kv, kvi_p, kvp_p, kv, kvi_e, kvp_e, sink, scale, *a, **kw
            )

        return f

    def wrap_decode(self, orig):
        def f(q, unified_kv, kv_indices, kv_indptr, sink, scale, *a, **kw):
            self.rec = {
                "kind": "decode",
                "q": q.detach().clone(),
                "unified_kv": unified_kv.detach().clone(),
                "kvi_p": kv_indices.detach().clone(),
                "kvp_p": kv_indptr.detach().clone(),
                "kv": None,
                "kvi_e": None,
                "kvp_e": None,
                "sink": sink.detach().clone(),
                "scale": scale,
            }
            return orig(q, unified_kv, kv_indices, kv_indptr, sink, scale, *a, **kw)

        return f


def check_dense_window(rec, positions, cu, win, T):
    """Dense has no compressor/indexer, so its index set is analytic: token t at
    position p attends exactly [max(0, p-win+1) .. p]. Verifying that against the
    builder makes the Dense case a fully independent check rather than one that
    trusts the builder's own gather list."""
    ep = rec["kvp_e"].long()
    got = ep[1 : T + 1] - ep[:T]
    want = (positions[:T].long() + 1).clamp(max=win)
    bad = int((got != want).sum().item())
    prefix_total = int(rec["kvp_p"][T].item())
    return bad, prefix_total


def run_prefill(attn, hf, cfg, builder, batch_size, seq_len, block_size, max_model_len):
    """Drive one prefill through the real builder + module. Returns (out, md)."""
    from atom.utils.forward_context import Context, set_forward_context

    bpr = (max_model_len + block_size - 1) // block_size
    bt, swa_bt = paged_tables(batch_size, seq_len, block_size, bpr)
    batch = make_batch(
        [seq_len] * batch_size, [0] * batch_size, bt, swa_bt, is_prefill=True
    )
    md, positions = builder.prepare_prefill(batch)

    total = batch_size * seq_len
    x = (torch.randn(total, hf.hidden_size, device="cuda") * 0.1).to(torch.bfloat16)
    set_forward_context(
        md, cfg, Context(positions=positions, is_prefill=True, batch_size=batch_size)
    )

    import atom.models.deepseek_v4 as v4

    cap = Capture()
    orig = v4.sparse_attn_v4_paged_prefill
    v4.sparse_attn_v4_paged_prefill = cap.wrap_prefill(orig)
    try:
        out = attn(x, positions)
    finally:
        v4.sparse_attn_v4_paged_prefill = orig
    return out, md, positions, cap.rec


def run_decode(attn, hf, cfg, builder, batch_size, ctx_len, block_size, max_model_len):
    """Fill the SWA ring + compressor state with a ctx_len-1 prefill, then check
    one decode step (the token at position ctx_len-1)."""
    import atom.models.deepseek_v4 as v4
    from atom.utils.forward_context import Context, set_forward_context

    bpr = (max_model_len + block_size - 1) // block_size
    bt, swa_bt = paged_tables(batch_size, ctx_len, block_size, bpr)

    # ---- 1) untimed prefill of the context prefix (output discarded) ----
    pre_len = ctx_len - 1
    pre_batch = make_batch(
        [pre_len] * batch_size, [0] * batch_size, bt, swa_bt, is_prefill=True
    )
    md_pf, pos_pf = builder.prepare_prefill(pre_batch)
    x_pf = (torch.randn(batch_size * pre_len, hf.hidden_size, device="cuda") * 0.1).to(
        torch.bfloat16
    )
    set_forward_context(
        md_pf, cfg, Context(positions=pos_pf, is_prefill=True, batch_size=batch_size)
    )
    _ = attn(x_pf, pos_pf)

    # ---- 2) one decode step ----
    dec_batch = make_batch(
        [ctx_len] * batch_size, [pre_len] * batch_size, bt, swa_bt, is_prefill=False
    )
    md, positions = builder.prepare_decode(dec_batch, batch_size)
    x = (torch.randn(batch_size, hf.hidden_size, device="cuda") * 0.1).to(
        torch.bfloat16
    )
    set_forward_context(
        md, cfg, Context(positions=positions, is_prefill=False, batch_size=batch_size)
    )
    cap = Capture()
    orig = v4.sparse_attn_v4_paged_decode
    v4.sparse_attn_v4_paged_decode = cap.wrap_decode(orig)
    try:
        out = attn(x, positions)
    finally:
        v4.sparse_attn_v4_paged_decode = orig
    return out, md, positions, cap.rec


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=DEFAULT_CONFIG)
    p.add_argument("--layer", default="all", choices=["dense", "csa", "hca", "all"])
    p.add_argument("--phase", default="both", choices=["prefill", "decode", "both"])
    p.add_argument("--batch", type=int, default=2)
    p.add_argument("--seqlen", type=int, default=512)
    p.add_argument("--ctx-len", type=int, default=512)
    p.add_argument(
        "--block-size",
        type=int,
        default=None,
        help="KV page size; default = the V4 builder's own block_size, which "
        "CommonAttentionBuilder asserts model_runner.block_size is a multiple "
        "of. Hardcoding it breaks when that constant moves (it went 128 -> 256).",
    )
    p.add_argument("--max-model-len", type=int, default=4096)
    p.add_argument("--kv-cache-dtype", default="bf16", choices=["bf16", "fp8"])
    p.add_argument("--atol", type=float, default=3e-2)
    p.add_argument("--rtol", type=float, default=3e-2)
    args_cli = p.parse_args()

    import aiter

    gfx = aiter.get_gfx()
    if args_cli.kv_cache_dtype == "fp8" and gfx == "gfx1250":
        print(
            "SKIP: --kv-cache-dtype fp8 on gfx1250. kv_fp8 sets fp8_2buff=True, "
            "which routes qk_norm_rope into aiter fused_qk_norm_rope_group_quant; "
            "that JIT module needs CK-tile and CK-tile has no gfx1250 arch."
        )
        return 0

    _init_tp1()
    torch.set_default_device("cuda")
    torch.manual_seed(0)

    if args_cli.block_size is None:
        from atom.model_ops.attentions.deepseek_v4_attn import (
            DeepseekV4AttentionMetadataBuilder,
        )

        args_cli.block_size = DeepseekV4AttentionMetadataBuilder.block_size

    hf = make_hf_config(args_cli.config, args_cli.max_model_len)
    cfg = set_atom_config(
        hf, args_cli.block_size, args_cli.kv_cache_dtype, args_cli.max_model_len
    )
    kinds = ["dense", "csa", "hca"] if args_cli.layer == "all" else [args_cli.layer]
    print(f"gfx={gfx} kv={args_cli.kv_cache_dtype} block_size={args_cli.block_size}")
    print(
        f"layers={kinds} batch={args_cli.batch} seqlen={args_cli.seqlen} "
        f"ctx_len={args_cli.ctx_len} window={hf.sliding_window} "
        f"index_topk={hf.index_topk}"
    )

    bpr = (args_cli.max_model_len + args_cli.block_size - 1) // args_cli.block_size
    num_blocks = args_cli.batch * bpr + 1
    runner = StubRunner(
        cfg,
        max_bs=max(args_cli.batch, 8),
        max_num_batched_tokens=max(args_cli.batch * args_cli.seqlen, 2048),
        num_blocks=num_blocks,
        num_swa_blocks=num_blocks,
    )
    v4args, builder, layers = build_layers(hf, cfg, runner, kinds)
    print(f"built layers: { {k: v.compress_ratio for k, v in layers.items()} }")

    phases = ["prefill", "decode"] if args_cli.phase == "both" else [args_cli.phase]
    rows, fail = [], 0
    for kind in kinds:
        attn = layers[kind]
        for phase in phases:
            if phase == "prefill":
                out, md, pos, rec = run_prefill(
                    attn,
                    hf,
                    cfg,
                    builder,
                    args_cli.batch,
                    args_cli.seqlen,
                    args_cli.block_size,
                    args_cli.max_model_len,
                )
                if kind == "dense":
                    bad, pfx = check_dense_window(
                        rec, pos, md.cu_seqlens_q, v4args.window_size, out.shape[0]
                    )
                    print(
                        f"  dense window check: {bad} token(s) with wrong extend "
                        f"length, prefix_total={pfx} (0 expected, fresh prefill)"
                    )
                    fail += 1 if (bad or pfx) else 0
            else:
                out, md, pos, rec = run_decode(
                    attn,
                    hf,
                    cfg,
                    builder,
                    args_cli.batch,
                    args_cli.ctx_len,
                    args_cli.block_size,
                    args_cli.max_model_len,
                )
            cov = coverage(rec, out.shape[0], hf.index_topk)
            ref = ref_block_output(attn, rec, pos)
            ok = compare(
                ref,
                out,
                args_cli.atol,
                args_cli.rtol,
                f"{phase:7} {kind:5} ratio={attn.compress_ratio:3} [{cov}]",
            )
            fail += 0 if ok else 1
            rows.append((phase, kind, attn.compress_ratio, ok))

    print(f"\n{'phase':8} {'layer':6} {'ratio':>5} {'pass':>6}")
    for ph, k, r, ok in rows:
        print(f"{ph:8} {k:6} {r:>5} {ok!s:>6}")
    print("\nAll passed." if not fail else f"\n{fail} FAILED")
    return 0 if not fail else 1


if __name__ == "__main__":
    sys.exit(main())
