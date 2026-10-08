"""Closed-form cost models for DSV4-Pro operators whose marker carries no shape.

Why this file exists
--------------------
`marker_roofline.py` prices an operator when the ATOM annotation carries M/N/K,
and applies one generic GEMM cost model to it. That covers 19% of a decode step
and 13% of a prefill step; everything else was left blank, including the fused
kernels that dominate decode. The blanks were not a property of those operators
-- a fused attention kernel has a perfectly well-defined FLOP and byte count --
they were a property of what the annotation happens to record.

So each operator here states its own cost from the model config instead of from
the marker. The marker is still the identity (which operator, which layer, how
many firings); the config supplies the dimensions. Keyed on the ATOM marker path
rather than on the CPU-side module name, because the marker path is what stays
stable across builds.

Every formula here must be reachable from MARKER_MAP. A cost model no kernel
routes to is not a spare part, it is a thing that looks maintained and is not --
`input_layernorm` sat here unreachable after V4 turned out to fuse every norm
with its fp8 cast (`norm_quant`). selfcheck asserts the reachability.

What a formula owes the reader
------------------------------
Return `(flops, bytes, spec)` and nothing else -- no timing, no peak, no
division. The roofline is applied in one place so there is exactly one formula
for `max(t_compute, t_memory)` in the tool. `spec` names the compute dtype (which
matrix ceiling binds), the memory dtype (already folded into `bytes`, kept for the
report), and which bandwidth applies: HBM for everything except collectives,
which are bound by the interconnect and would be flattered ~100x by an HBM roof.

It also owes its own ALGEBRA. `spec.flops_expr` and `spec.bytes_expr` carry the
symbolic form and the substituted numbers, built here at the moment the arithmetic
happens. The report shows them on hover, so a reader never has to take a number on
trust: every operator displays the expression it actually evaluated, not a generic
`2*M*N*K` that would be a lie for two thirds of this table. Recording them at the
point of computation is also what stops the display from drifting into a second,
subtly different implementation of the same maths.

Every formula is a floor, deliberately. Where a kernel fuses extra work we model
the dominant term and say so; the remainder shows up as measured-over-roofline,
which is the number the report is actually about.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field

DTYPE_BYTES = {"fp4": 0.5, "fp8": 1.0, "bf16": 2.0, "fp16": 2.0, "fp32": 4.0}

# MXFP4 (V4-Pro routed experts): fp4 weights carry one E8M0 (1-byte) scale per 32
# elements along K, so fp4 weight bytes are 0.5 + 1/32 per element (+6.25% over
# bare fp4). fp8 weights use 128x128 blocks -> scale ~1/16384, negligible.
FP4_BLOCK = 32

# Index arithmetic moves int32, which is not a compute precision and so has no
# place in DTYPE_BYTES -- pricing an index gather against a matrix ceiling would
# be the same category error the vector_tflops note in peaks/ warns about.
I32_BYTES = 4.0

# aiter's per-head-count minimum KV block for the fp8 decode path
# (`mla.py get_block_n_fp8`); it caps how finely the KV may be split.
_FP8_BLOCK_N = {8: 64, 16: 128, 24: 128, 32: 128, 48: 64, 64: 64, 128: 32, 256: 32}

_CSA_RATIO = 4  # CSA compresses 4 tokens -> 1 KV entry
_HCA_RATIO = 128  # HCA compresses 128 -> 1


# --------------------------------------------------------------------------
# dtype resolution
#
# A formula must not hard-code a precision. The benchmark catalog ships the same
# architecture at two precisions -- GLM-5.2-FP8 and GLM-5.2-MXFP4, MiniMax-M3
# MXFP8 and MXFP4, Qwen3.5 FP8 and MXFP4 -- so a literal "fp8" in a cost model is
# wrong for half the runs it will be asked about. Formulas name a ROLE; the run
# resolves the role to a dtype.
#
# Precedence, highest first. The point of the order is that the first two describe
# what RAN and the last two describe what was STORED, and those are different
# questions:
#
#   1. the marker's own a=/w=/o= tokens        what this firing actually used
#   2. the kernel name (afp8_wfp4, hgemm_bf16) what was compiled in
#   3. runtime flags (--kv-cache-dtype, ...)   caches, which no HF config covers
#   4. HF quantization_config + expert_dtype   the checkpoint baseline
#   5. a formula's own assertion               last resort, and always flagged
#
# 1 and 2 are handled where the row is priced (marker shape, DTYPE CHECK). This
# table is layers 3 and 4: the baseline a formula starts from.
#
# Why the HF config alone is not enough, on DSV4-Pro's own config:
#   * `quantization_config.quant_method: "fp8"` is ONE global value, and the
#     experts are not fp8 -- their precision lives in a separate, non-standard
#     `expert_dtype: "fp4"` field.
#   * It describes weight STORAGE. Experts store MXFP4 and compute at the FP8 peak
#     (FP4xFP8 runs at the fp8 rate; there is no fp4 matrix speedup), so storage
#     and ceiling are two different answers and the config gives one.
#   * KV cache dtype is a runtime flag (`--kv-cache-dtype fp8`), not in the config.
#   * ATOM's `_wo_a_is_bf16_on_disk` exists precisely because the same
#     quantization_config ships wo_a as fp8 on one checkpoint and bf16 on another.


@dataclass(frozen=True)
class Dtypes:
    """The precisions a run actually uses, per role."""

    linear_w: str = "fp8"  # non-expert Linear weights
    expert_w: str = "fp4"  # routed-expert weights
    act: str = "bf16"  # activations entering a quantised GEMM
    kv: str = "fp8"  # KV cache
    index: str = "fp8"  # indexer keys/queries
    source: dict = field(default_factory=dict)


_DTYPE_SPELLINGS = {
    "bfloat16": "bf16",
    "bf16": "bf16",
    "float16": "fp16",
    "half": "fp16",
    "float8_e4m3fn": "fp8",
    "float8": "fp8",
    "fp8": "fp8",
    "e4m3": "fp8",
    "float4_e2m1fn_x2": "fp4",
    "fp4": "fp4",
    "mxfp4": "fp4",
    "mxfp8": "fp8",
    "float32": "fp32",
    "fp32": "fp32",
    "auto": "bf16",
}


def _norm(text: str) -> str:
    """One spelling. Chained str.replace turned 'bfloat16' into 'bfp16'."""
    return _DTYPE_SPELLINGS.get(text.replace("torch.", "").lower(), text)


def resolve_dtypes(cfg: dict, runtime: dict | None = None) -> Dtypes:
    """Baseline dtypes for a run: HF config, overridden by runtime flags."""
    runtime = runtime or {}
    q = (cfg.get("quantization_config") or {}).get("quant_method")
    src: dict[str, str] = {}

    def pick(role: str, *candidates: tuple[str | None, str]) -> str:
        for value, where in candidates:
            if value:
                src[role] = where
                return _norm(str(value))
        src[role] = "default"
        return "fp8"

    return Dtypes(
        linear_w=pick("linear_w", (q, "quantization_config.quant_method")),
        expert_w=pick(
            "expert_w",
            (cfg.get("expert_dtype"), "config.expert_dtype"),
            (q, "quantization_config.quant_method"),
        ),
        act=pick(
            "act",
            (runtime.get("act_dtype"), "runtime"),
            (cfg.get("torch_dtype"), "config.torch_dtype"),
        ),
        kv=pick(
            "kv",
            (runtime.get("kv_cache_dtype"), "runtime --kv-cache-dtype"),
            (q, "quantization_config.quant_method"),
        ),
        index=pick(
            "index",
            (runtime.get("index_cache_dtype"), "runtime --index_cache_dtype"),
            (q, "quantization_config.quant_method"),
        ),
        source=src,
    )


def compute_dtype(mem: str) -> str:
    """Which matrix ceiling a given storage precision is actually computed at.

    fp4 weights are dequantised to fp8 losslessly and run at the FP8 peak -- there
    is no fp4 matrix speedup on CDNA4 -- so storing fp4 buys bytes, not FLOP/s.
    Everything else computes at what it stores.
    """
    return "fp8" if mem == "fp4" else mem


@dataclass(frozen=True)
class Bench:
    """The run's operating point. Everything a formula needs beyond the config.

    `kv_seq_len` is the one field with no exact answer per firing: ATOM's decode
    phase label is `decode[bs=1 tok=1 d=1]` and carries no context length, so it
    is derived (prefill ctx + half the decode steps) and carried as a stated
    assumption rather than silently baked in. It only scales attention terms.
    """

    batch: int = 1
    seq_len: int = 1
    tp: int = 1
    dp: int = 1
    dpon: bool = False
    kv_seq_len: float = 0.0
    kv_seq_note: str = ""
    dtypes: Dtypes = field(default_factory=lambda: Dtypes())
    kernel: str = ""  # the kernel this firing ran, when the trace names one
    custom_ar_max_bytes: float | None = None  # override AITER's 64 MiB default


@dataclass(frozen=True)
class Spec:
    compute: str = "bf16"  # which matrix ceiling binds
    mem: str = "bf16"  # dtype the bytes were counted in (already in `bytes`)
    bw: str = "hbm"  # "hbm" or "interconnect"
    note: str = ""
    flops_expr: str = ""  # the actual algebra, with numbers substituted
    bytes_expr: str = ""


def _n(v: float) -> str:
    """Format one substituted value: integers plainly, fractions to 4 digits."""
    if abs(v - round(v)) < 1e-9 and abs(v) < 1e15:
        return f"{round(v):,}"
    return f"{v:,.4g}"


def prod(label: str, pairs: list[tuple[str, float]], total: float) -> str:
    """A product, shown symbolically and then substituted.

    FLOPs = passes x M x n_h x c x k
          = 4 x 1 x 32 x 512 x 1,024
          = 6.711e+07
    """
    names = " x ".join(k for k, _ in pairs)
    vals = " x ".join(_n(v) for _, v in pairs)
    return f"{label} = {names}\n{' ' * len(label)} = {vals}\n{' ' * len(label)} = {total:.4g}"


def summ(label: str, parts: list[tuple[str, str]], total: float) -> str:
    """A sum of named terms, each already substituted.

    Bytes = KV(shared) + Q(per-rank)
          = 1,024 x 1,088 + 1 x 1 x 32 x 512
          = 1.130e+06
    """
    names = " + ".join(p[0] for p in parts)
    vals = " + ".join(p[1] for p in parts)
    return f"{label} = {names}\n{' ' * len(label)} = {vals}\n{' ' * len(label)} = {total:.4g}"


def w_bytes(n_elem: float, dtype: str) -> float:
    """Weight bytes including MXFP4 block-scale overhead (fp4 only)."""
    b = n_elem * DTYPE_BYTES[dtype]
    if dtype == "fp4":
        b += n_elem / FP4_BLOCK
    return b


def attn_M(b: Bench) -> float:
    """Token count per rank in the attention region.

    dp-attn on: attention is DP-partitioned, M = batch. Off: every rank runs the
    full batch, M = batch * dp. The MoE region is different (see moe_B).
    """
    return b.batch * b.seq_len * (1 if b.dpon else b.dp)


def moe_B(b: Bench) -> float:
    """Token count in the MoE region -- the GATHERED tokens of all DP ranks.

    There is no sequence parallelism: the all-gather before the experts brings
    batch*dp tokens to the expert GEMMs while attention stays at batch/dp.
    ATOM, SGLang and TRT-LLM all agree on this.
    """
    return b.batch * b.seq_len * b.dp


def gemm(
    M: float,
    K: float,
    N: float,
    w_dtype: str,
    a_dtype: str | None = None,
    o_dtype: str = "bf16",
) -> tuple[float, float, str, str]:
    """A plain GEMM. The activation dtype is NOT the weight dtype by default only
    because fp4 weights are fed fp8 activations; everything else matches.

    The preceding quant kernel is a separate row in the table, so what this GEMM
    reads is the fp8 activation that kernel produced. Charging the activation as
    bf16 regardless would overstate an fp8 GEMM's bytes by up to 1.61x once M is
    large; at decode it is invisible, weights being 99-100% of the traffic.
    """
    a_dtype = a_dtype or ("fp8" if w_dtype == "fp4" else w_dtype)
    flops = 2.0 * M * K * N
    wb = w_bytes(K * N, w_dtype)
    ab = DTYPE_BYTES[a_dtype] * M * K + DTYPE_BYTES[o_dtype] * M * N
    fe = prod("FLOPs", [("2", 2), ("M", M), ("K", K), ("N", N)], flops)
    be = summ(
        "Bytes",
        [
            (
                f"W[K x N] {w_dtype}",
                f"{_n(K)} x {_n(N)} x {DTYPE_BYTES[w_dtype]}"
                + (" x 1.0625 (MXFP4 scale)" if w_dtype == "fp4" else ""),
            ),
            (f"act in {a_dtype}", f"{DTYPE_BYTES[a_dtype]:g} x {_n(M)} x {_n(K)}"),
            (f"act out {o_dtype}", f"{DTYPE_BYTES[o_dtype]:g} x {_n(M)} x {_n(N)}"),
        ],
        wb + ab,
    )
    return flops, wb + ab, fe, be


# --------------------------------------------------------------------------
# attention -- CSA / HCA (V4 report pp.9-13)
#
# Two layer archetypes:
#   CSA  KV compressed every 4 tokens; an FP4 Lightning Indexer scores every
#        compressed block and selects top-k = index_topk; MQA core attention runs
#        over the selected blocks only.
#   HCA  KV compressed every 128; DENSE MQA over all compressed entries, no
#        indexer.
# MQA: one compressed KV entry (dim head_dim) is shared across all query heads,
# and the KV cache is stored MIXED -- rope dims bf16, the rest fp8.
# --------------------------------------------------------------------------


def kv_entries(cfg: dict, b: Bench, layer_kind: str) -> float:
    """Compressed KV entries the core attention reads per query token."""
    kv = float(b.kv_seq_len)
    if layer_kind == "hca":
        return kv / _HCA_RATIO
    return min(kv / _CSA_RATIO, float(cfg["index_topk"]))


def index_entry_bytes(cfg: dict, dtypes) -> float:
    """One entry of the Indexer's OWN compressed cache.

    Not the Main cache: `fused_compress_attn(quant=True)` writes "per-row amax ->
    ue8m0 scale -> fp8 cast -> preshuffled ... plus fp32 scale into cache_scale",
    so an entry is a flat `index_head_dim` row plus one scale -- fp32 for an fp8
    cache, uint8 for an fp4 one (`cache_scale: fp32 [NB,k] (FP8) / uint8 (FP4)`).

    Shared by the writer (`compressor_pool_index`) and the reader (`csa_indexer`)
    so the two cannot describe one cache differently, the same way
    `kv_entry_bytes` binds the Main compressor to `_core_attn`.
    """
    idx = dtypes.index
    scale = DTYPE_BYTES["fp32"] if idx != "fp4" else 1.0
    return DTYPE_BYTES[idx] * cfg["index_head_dim"] + scale


def kv_entry_bytes(cfg: dict) -> float:
    """One compressed KV entry: rope dims bf16, the rest fp8 (mixed cache)."""
    c, rope = cfg["head_dim"], cfg["qk_rope_head_dim"]
    return rope * DTYPE_BYTES["bf16"] + (c - rope) * DTYPE_BYTES["fp8"]


def _core_attn(cfg, b, layer_kind, passes, part="fused"):
    """One or both core-attention matmul passes over the compressed KV entries.

    The trace is PER-RANK and TP shards the query heads, so n_h/tp. The KV entry
    is one shared MQA head, read once per rank and NOT divided by tp. At decode
    the per-rank Q term dominates the KV read because compression makes KV tiny,
    which is exactly why the /tp on heads matters -- using full n_h overcounts
    attention bytes by about tp.
    """
    M = attn_M(b)
    n_h = cfg["num_attention_heads"] / b.tp
    c = cfg["head_dim"]
    k = kv_entries(cfg, b, layer_kind)
    # Causal masking. At decode there is one query and it sees the whole context,
    # so the factor is 1. At prefill the queries are the context: the first sees
    # nothing and the last sees everything, averaging half. Without this the
    # prefill attention floor is 2x too high, which would read as the kernel being
    # twice as efficient as it is.
    causal = 0.5 if b.seq_len > 1 else 1.0
    flops = passes * M * n_h * c * k * causal
    # Bytes, per pass: Q, the KV entries it reads, and the output it writes. When
    # QK and AV run as separate kernels Q is charged once across the pair, not once
    # per pass -- the AV pass never reads Q, it reads the scores QK wrote.
    kvb = k * kv_entry_bytes(cfg)
    qb = DTYPE_BYTES["fp8"] * M * n_h * c
    ob = DTYPE_BYTES["bf16"] * M * n_h * c  # attention output, feeds wo_a
    sb = DTYPE_BYTES["fp32"] * M * n_h * k  # the score matrix
    src = "kv/128 dense (HCA)" if layer_kind == "hca" else "min(kv/4, index_topk) (CSA)"
    fe = prod(
        "FLOPs",
        [
            ("passes", passes),
            ("M", M),
            ("n_h/tp", n_h),
            ("head_dim", c),
            (f"k [{src}]", k),
        ]
        + ([("causal 1/2", causal)] if causal != 1.0 else []),
        flops,
    )
    # KV bytes are streamed once for the whole call, not once per query, so they
    # carry neither the causal factor nor M. (At prefill a tiled kernel does re-read
    # KV per query block; here KV is a small fraction of Q, so that is left in the
    # gap rather than modelled with a guessed tile count.)
    if part == "qk":
        # QK pass: read Q and the keys, write the score matrix. Scores dominate.
        parts = [
            ("K entries (shared, MQA)", f"{_n(k)} x {_n(kv_entry_bytes(cfg))}"),
            ("Q per-rank fp8", f"{_n(M)} x {_n(n_h)} x {_n(c)}"),
            ("scores out fp32", f"4 x {_n(M)} x {_n(n_h)} x {_n(k)}"),
        ]
        byts = kvb + qb + sb
    elif part == "av":
        # AV pass: read the scores and the values, write the output. NOT Q.
        parts = [
            ("scores in fp32", f"4 x {_n(M)} x {_n(n_h)} x {_n(k)}"),
            ("V entries (shared, MQA)", f"{_n(k)} x {_n(kv_entry_bytes(cfg))}"),
            ("output bf16", f"2 x {_n(M)} x {_n(n_h)} x {_n(c)}"),
        ]
        byts = sb + kvb + ob
    else:
        # Fused: the scores never leave the chip, so they are correctly absent.
        parts = [
            ("KV entries (shared, MQA)", f"{_n(k)} x {_n(kv_entry_bytes(cfg))}"),
            ("Q per-rank fp8", f"{_n(M)} x {_n(n_h)} x {_n(c)}"),
            ("output bf16", f"2 x {_n(M)} x {_n(n_h)} x {_n(c)}"),
        ]
        byts = kvb + qb + ob
    be = summ("Bytes", parts, byts)
    return flops, byts, fe, be


def csa_core_attn(cfg, b, kind):
    """fused_compress_attn / hca_compress_forward: QK and AV fused in one kernel."""
    f, y, fe, be = _core_attn(cfg, b, kind, 4.0, part="fused")
    return (
        f,
        y,
        Spec("fp8", "fp8", "hbm", "QK+AV fused, 4 passes; scores stay on chip", fe, be),
    )


def _kv_splits(cfg, b, kind, cu_num=256.0):
    """How many KV splits aiter's decode kernel picks, from its own heuristic.

    `mla.py get_meta_param` (line numbers drift across aiter commits, so this
    cites the function): when the caller passes no `num_kv_splits` -- ATOM never
    does -- aiter maximises `bs*i/ceil(bs*i/cu)*cu * avg_kv/(avg_kv + 84.1*i)`
    over `i in 1..16`, then caps it twice for the fp8 path against a per-head
    block size (`get_block_n_fp8[nhead*max_seqlen_q]`; at decode max_seqlen_q=1). It is a runtime occupancy decision, not a shape, so it is reproduced
    here rather than inferred: it sets how many fp32 partials stage 1 writes and
    stage 2 reads, and that is the whole cost of stage 2.

    At c1 this lands on 8 splits for CSA (1024 visible entries) and 1 for HCA
    (60), which is why the merge kernel is not free on CSA layers and nearly is
    on HCA ones.
    """
    bs = max(1.0, math.floor(attn_M(b)))
    n_h = cfg["num_attention_heads"] / b.tp
    total_kv = kv_entries(cfg, b, kind) * bs
    avg_kv = total_kv / bs
    best = max(
        (
            bs
            * i
            / (math.ceil(bs * i / cu_num) * cu_num)
            * avg_kv
            / (avg_kv + 84.1 * i),
            -i,
        )
        for i in range(1, 17)
    )
    n = -best[1]
    block_n = _FP8_BLOCK_N.get(int(n_h), 32)
    n = min(n, math.ceil(total_kv / (bs * block_n)))
    if n > 1:
        n = min(n, int(abs(avg_kv - 1) // block_n) + 1)
    return float(max(n, 1))


def attn_logits(cfg, b, kind):
    """Stage 1 of the split-KV decode: the WHOLE attention over each KV split.

    Named for the logits it emits, but `mla_a8w8_..._nm` runs QK, softmax AND AV
    over its share of the KV and writes fp32 partials plus an LSE
    (`aiter/mla.py`, the `_fwd_kernel_stage2_asm` launch: "After the kernel
    returns FP32 partials, run V3's `_fwd_kernel_stage2_asm` triton kernel which
    performs the FlashAttention LSE merge across the num_kv_splits axis" -- cited
    by symbol, not line, as it drifts across aiter commits). Stage 1 is the whole
    attention, not the QK pass alone, and what it writes is fp32 partials plus an
    LSE rather than a bf16 row.

    The output is `splits * M * n_h * head_dim` fp32 partials plus an LSE per
    (split, head), not one bf16 row: that inflation is exactly what stage 2 then
    has to read back.

    The kernel name does not state this call's head count. v4 ships ONE binary
    (`mla_a8w8_qh64_qseqlen1_gqaratio64_nm`) whose 64 q-row tile satisfies
    `gqa * q_seq_logical = 64`, so it serves (gqa=16, qSeqLen=4), (gqa=64, 1) and
    (gqa=128, 1) alike (`asm_mla_v4.cu`, the CSV lookup-key normalisation). The
    head count comes from `sub_Q` and the launch grid, set per call -- which is
    why this prices `num_attention_heads / tp` (32 at TP4, and independently the
    N of `attn.wq_b` = 32 x 512) rather than the 64 in the symbol. At TP4 that
    tile runs half-filled, which costs occupancy rather than bytes.
    """
    # 4 passes = 2 for QK and 2 for AV, the same count csa_core_attn uses;
    # stage 1 runs both, so it must not be charged the 2 of a single pass.
    f, y, fe, _be = _core_attn(cfg, b, kind, 4.0, part="fused")
    M, n_h, c = attn_M(b), cfg["num_attention_heads"] / b.tp, cfg["head_dim"]
    n = _kv_splits(cfg, b, kind)
    # _core_attn charged a single bf16 output row; stage 1 writes fp32 partials.
    y = y - DTYPE_BYTES["bf16"] * M * n_h * c
    y += DTYPE_BYTES["fp32"] * n * M * n_h * (c + 1)  # partials + lse
    return (
        f,
        y,
        Spec(
            "fp8",
            "fp8",
            "hbm",
            f"split-KV stage 1 over {_n(n)} splits: QK + softmax + AV, fp32 partials out",
            fe,
            summ(
                "Bytes",
                [
                    (
                        "KV + Q + scores",
                        _n(y - DTYPE_BYTES["fp32"] * n * M * n_h * (c + 1)) + " B",
                    ),
                    (
                        "fp32 partials + lse",
                        f"4 x {_n(n)} x {_n(M)} x {_n(n_h)} x ({_n(c)}+1)",
                    ),
                ],
                y,
            ),
        ),
    )


def attn_merge(cfg, b, kind):
    """Stage 2: the FlashAttention LSE merge across the num_kv_splits axis.

    `_fwd_kernel_stage2_asm` reads stage 1's fp32 partials and their LSEs and
    writes one bf16 row per (token, head). **It never touches the KV cache** --
    modelling it as the AV pass of core attention had it re-read every visible KV
    entry, 590 KB per firing on a CSA layer, for a kernel whose whole input is
    `splits * n_h * head_dim` floats.

    Renamed from `attn_reduce`: "reduce" reads as the AV reduction, which is what
    the wrong model called it.
    """
    M, n_h, c = attn_M(b), cfg["num_attention_heads"] / b.tp, cfg["head_dim"]
    n = _kv_splits(cfg, b, kind)
    flops = 3.0 * n * M * n_h * c  # rescale by exp(lse_i - lse_max) and accumulate
    byts = (
        DTYPE_BYTES["fp32"] * n * M * n_h * (c + 1) + DTYPE_BYTES["bf16"] * M * n_h * c
    )
    return (
        flops,
        byts,
        Spec(
            "fp32",
            "fp32",
            "hbm",
            f"split-KV stage 2: LSE merge over {_n(n)} splits; reads no KV",
            summ(
                "FLOPs",
                [
                    (
                        "rescale + accumulate",
                        f"3 x {_n(n)} x {_n(M)} x {_n(n_h)} x {_n(c)}",
                    )
                ],
                flops,
            ),
            summ(
                "Bytes",
                [
                    (
                        "fp32 partials + lse in",
                        f"4 x {_n(n)} x {_n(M)} x {_n(n_h)} x ({_n(c)}+1)",
                    ),
                    ("bf16 merged out", f"2 x {_n(M)} x {_n(n_h)} x {_n(c)}"),
                ],
                byts,
            ),
        ),
    )


def swa_write(cfg, b, kind):
    """_swa_write: a SCATTER into the sliding-window ring, not attention.

    Settled from the source (`state_writes.py` swa_write_2buff_prepacked;
    cited by symbol -- line numbers drift):
    "Native 2buff fp8 SWA ring write: scatter of the LAST
    min(tok_n_b, write_per_batch) tokens of every seq into the two SWA ring pools
    (fp8 NoPE + bf16 RoPE) ... This is a pure dtype-agnostic scatter ... NO torch
    quantization happens here."

    Only the last `window` tokens of a sequence can be in the window, so a long
    prefill writes `window` rows, not M. The kernel is called once per plane; this
    models the dominant NoPE plane (512 fp8), which makes the smaller RoPE firing
    (64 bf16, 4x less) a slight over-estimate.
    """
    w = cfg.get("sliding_window", 128)
    rows = min(attn_M(b), float(w))
    width = 512.0  # V4_DIM_QK_PACKED, fp8
    byts = DTYPE_BYTES["fp8"] * 2 * rows * width
    return (
        0.0,
        byts,
        Spec(
            "fp8",
            "fp8",
            "hbm",
            "pure scatter into the ring; no arithmetic",
            "FLOPs = 0  (a scatter, not an arithmetic kernel)",
            prod(
                "Bytes",
                [
                    ("2 (read+write)", 2),
                    ("1 (fp8)", 1),
                    (f"min(M, window {w})", rows),
                    ("V4_DIM_QK_PACKED", width),
                ],
                byts,
            ),
        ),
    )


def compressor_pool(cfg, b, kind):
    """The Main compressor's pool, at head_dim, into the mixed rope/nope cache."""
    return _pool(cfg, b, kind)


def compressor_pool_index(cfg, b, kind):
    """The Indexer's pool, at index_head_dim, into its own per-row fp8 cache.

    `quant=True` in `fused_compress_attn`: "per-row amax -> ue8m0 scale -> fp8 cast
    -> preshuffled write into the FP8 kv_cache, plus fp32 scale into cache_scale".
    Flat fp8 with one fp32 scale per row -- not the Main path's mixed entry, so it
    does NOT share `kv_entry_bytes`.
    """
    c = cfg["index_head_dim"]
    return _pool(cfg, b, "csa", width=c, entry_bytes=index_entry_bytes(cfg, b.dtypes))


def _compressor_epilogue(cfg, entries, width=None, entry_bytes=None):
    """RMSNorm + RoPE + cache scatter on one compressed entry.

    CSA fuses this into `fused_compress_attn`; HCA runs it as the separate
    `hca_norm_rope_scatter`. Same three operations either way, so they share one
    algebra -- pricing them apart had the fused copy at 4*c + 6*rope FLOPs and the
    split copy at a flat 8*c, a 1.7x disagreement about one epilogue, with the
    split copy spreading RoPE across all `head_dim` lanes when it only touches
    `qk_rope_head_dim` of them.

    The scatter lands in the cache `_core_attn` reads, so it is `kv_entry_bytes`.
    """
    c = width if width else cfg["head_dim"]
    rope = cfg["qk_rope_head_dim"]
    # The Indexer scatters per-row fp8 plus one fp32 scale; the Main path writes
    # the mixed rope-bf16 / nope-fp8 entry `_core_attn` reads.
    entry = entry_bytes if entry_bytes else kv_entry_bytes(cfg)
    flops = 4.0 * entries * c + 6.0 * entries * rope
    byts = entries * entry
    terms_f = [
        ("RMSNorm on the entry", f"4 x {_n(entries)} x {_n(c)}"),
        ("RoPE on the rope lanes only", f"6 x {_n(entries)} x {_n(rope)}"),
    ]
    terms_b = [("scattered entry", f"{_n(entries)} x {_n(entry)}")]
    return flops, byts, terms_f, terms_b


def _pool(cfg, b, kind, width=None, entry_bytes=None):
    """fused_compress_attn / hca_compress_forward: pool `ratio` rows into one.

    The kernel's name contains "attn" and it is not attention. Settled from the
    source (`fused_compress.py` fused_compress_attn; cited by symbol): "Batched fused
    per-source-position pool + RMSNorm + RoPE + cache scatter". No queries are
    involved -- there is no head count and no context length in its signature,
    only `kv_in [num_q_tokens, dim_full]`, a learned `ape [ratio, dim_full]`
    pooling weight, and a compressed output at `position // ratio`.

    It is a per-token weighted add, not a per-query scan over the selected KV.

    **One program per boundary, not per token**, and each program walks a window of
    `K = 2*ratio if overlap else ratio` rows: Phase 1 reads them from the
    `kv_state`/`score_state` ring buffer, Phase 2 reads the tail from the ragged
    input, and together they cover `k_static in [0, K)`. The dominant read is the state
    ring, and it is walked `K/ratio` times per token rather than once -- 2x with
    overlap, which is why CSA and HCA do not share a count here.

    `overlap` is read from the kernel name, which carries `_OVL_` when it is on --
    observed beats derived, as with the all-reduce path. The `ape [ratio, dim_full]`
    and `rms_weight [head_dim]` tables are deliberately not counted: `window_len =
    K - min(j_in_seq+1, K)` (fused_compress.py _fused_compress_attn_kernel), so at decode Phase 2 covers
    exactly the boundary token and reads one ape row. At prefill `window_len` falls
    to 0 and Phase 2 reads up to K of them -- that exclusion is decode-specific.

    The scatter writes the SAME cache that `_core_attn` reads, so it is priced with
    the same `kv_entry_bytes` rather than a flat fp8 row. ATOM picks the layout in
    `deepseek_v4_attn.py` (`module.quant_mode = "group_fp8"` set under `self._kv_fp8`;
    cited by symbol, line drifts -- ~1810 at time of writing) under an fp8 KV cache,
    and group_fp8 is the `main_2buff_fp8` path that writes e8m0 nope-fp8 into
    `kv_cache` and bf16 rope into `kv_cache_rope`. Writer and reader disagreeing
    about one cache is a contradiction no measurement would have surfaced here.
    """
    ratio = _HCA_RATIO if kind == "hca" else _CSA_RATIO
    name = b.kernel.lower()
    overlap = "ovl" in name if name else kind != "hca"
    window = 2.0 if overlap else 1.0  # K / ratio
    # CSA fuses the epilogue: `fused_compress_attn` is pool + norm + rope +
    # scatter in one kernel. HCA splits it -- `hca_compress_forward` pools and
    # nothing else, and `hca_norm_rope_scatter` (priced by `compressor_epilogue`)
    # does the rest. Charging the epilogue to both billed HCA for it twice.
    fused = not ("compress_forward" in name if name else kind == "hca")
    M = attn_M(b)
    c = width if width else cfg["head_dim"]
    entries = M / ratio
    # pool: one weighted add per row of the K-wide window.
    flops = 2.0 * window * M * c
    byts = DTYPE_BYTES["fp32"] * 2 * window * M * c  # kv + score per row
    epi_f, epi_b, epi_tf, epi_tb = _compressor_epilogue(cfg, entries, c, entry_bytes)
    if fused:
        flops += epi_f
        byts += epi_b
    else:
        # HCA's pool kernel has to land the pooled entry somewhere for
        # `compressor_epilogue` to read back. Dropping the epilogue without adding
        # this write left a reader with no writer.
        epi_tf, epi_tb = [], [
            ("pooled entry handed to the epilogue", f"2 x {_n(entries)} x {_n(c)}")
        ]
        byts += DTYPE_BYTES["bf16"] * entries * c
    return (
        flops,
        byts,
        Spec(
            "bf16",
            "fp32",
            "hbm",
            f"pool {ratio}->1 over a K={_n(window * ratio)} window"
            f"{' (overlapping)' if overlap else ''} + norm + rope + scatter; no queries",
            summ(
                "FLOPs",
                [
                    (
                        "pool over the K-wide window",
                        f"2 x {_n(window)} x {_n(M)} x {_n(c)}",
                    ),
                ]
                + epi_tf,
                flops,
            ),
            summ(
                "Bytes",
                [
                    (
                        "kv + score fp32 over the window",
                        f"4 x 2 x {_n(window)} x {_n(M)} x {_n(c)}",
                    ),
                ]
                + epi_tb,
                byts,
            ),
        ),
    )


def csa_indexer(cfg, b, kind):
    """Lightning Indexer: ReLU-weighted scores over every compressed block, FP4.

    index_n_heads is FULL, not divided by tp: the indexer Q projection is a
    ReplicatedLinear because every rank needs all 64 heads to compute the per-token
    top-k locally without a cross-rank all-reduce. (The main attention heads ARE
    tp-sharded -- proven by the q_b weight load not exceeding HBM peak.)
    """
    M = attn_M(b)
    n_i, d_i = cfg["index_n_heads"], cfg["index_head_dim"]
    blocks = float(b.kv_seq_len) / _CSA_RATIO
    flops = 2.0 * M * n_i * d_i * blocks
    # The scores it produces are what the top-k then selects, so they are written
    # out; omitting them understated prefill's indexer by 2.8x.
    scores = DTYPE_BYTES["fp32"] * M * blocks
    idx = b.dtypes.index
    # The compressed rows it scores carry a per-row scale, because that is what
    # `compressor_pool_index` wrote. Reading them as a bare `d_i` row is the same
    # writer/reader split that had the Main compressor scattering flat fp8 into a
    # cache `_core_attn` reads as rope-bf16 + nope-fp8.
    byts = (
        blocks * index_entry_bytes(cfg, b.dtypes)  # compressed index KV (MQA)
        + DTYPE_BYTES[idx] * M * n_i * d_i  # Q, all 64 heads
        + scores
    )
    # fp8, NOT fp4. The V4 report describes the Lightning Indexer as FP4 (p.13) and
    # I took the paper's word for it; the kernel is `_gluon_fp8_mqa_logits_kernel` /
    # `_gluon_deepgemm_fp8_paged_mqa_logits`, imported from
    # `aiter.ops.triton.fp8_mqa_logits`. The trace is the authority for what this
    # BUILD does. Same error, same direction, as the 2026-06-02 mla_kv_a / o_proj_b
    # fp4->fp8 correction -- reading a dtype off the model description instead of
    # the kernel signature. It halves the ceiling: 10066 -> 5033.
    # Storage and ceiling are different answers here and both are now derived:
    # `--index_cache_dtype fp4` sets the cache, and the hardware rule says fp4
    # computes at the fp8 peak -- which is what the kernel name
    # (`_gluon_fp8_mqa_logits`) independently says. Asserting either one by hand is
    # how this operator carried an fp4 ceiling for a session.
    return (
        flops,
        byts,
        Spec(
            compute_dtype(idx),
            idx,
            "hbm",
            "scores every compressed block; heads NOT tp-sharded",
            prod(
                "FLOPs",
                [
                    ("2", 2),
                    ("M", M),
                    ("index_n_heads", n_i),
                    ("index_head_dim", d_i),
                    ("blocks kv/4", blocks),
                ],
                flops,
            ),
            summ(
                "Bytes",
                [
                    ("index keys fp4", f"0.5 x {_n(blocks)} x {_n(d_i)}"),
                    ("queries fp4", f"0.5 x {_n(M)} x {_n(n_i)} x {_n(d_i)}"),
                    ("scores out fp32", f"4 x {_n(M)} x {_n(blocks)}"),
                ],
                byts,
            ),
        ),
    )


def indexer_topk(cfg, b, kind):
    """radix top-k over the indexer scores: scan scores TWICE, write k.

    aiter's `radix_topk_one_block_kernel<float,int,12,1024>` is a 3-pass radix
    select, but the HBM cost is not 3x the score array. Verified against the HIP
    (`csrc/kernels/topk_per_row_kernels.cu`):
      - pass 0 builds the histogram only -- its lambda does `atomicAdd(hist)` and
        writes no survivors;
      - pass 1 must therefore RE-READ the full `in` to scatter the winners, and
        `previous_len (=len) > buf_len (=len/32)` forces `in_buf = in` again;
      - 12-bit passes = 4096 buckets, so after pass 0 the candidate set collapses
        to one boundary bucket (~len/4096, far below buf_len) -- passes >=2 read
        only the tiny scratch buffer, negligible.
    So the residency-independent floor is 2 full score scans + the k indices out
    (both full scans are data-independent; scratch round-trips are not). This is
    still a pure classifier (eff ~ 2e-4): the ~9.6us/firing is single-block
    occupancy (1 decode row -> 1 CU of 256) plus per-pass histogram/sync latency,
    neither of which is HBM traffic or FLOPs -- correctly left to regime=latency.
    """
    M = attn_M(b)
    blocks = float(b.kv_seq_len) / _CSA_RATIO
    # 2 full score scans (histogram pass + filter pass) + k output indices
    byts = DTYPE_BYTES["fp32"] * M * (2.0 * blocks + cfg["index_topk"])
    return (
        0.0,
        byts,
        Spec(
            "bf16",
            "fp32",
            "hbm",
            "selection only, no matmul; scores scanned twice (histogram+filter)",
            "FLOPs = 0  (a selection, not an arithmetic kernel)",
            prod(
                "Bytes",
                [
                    ("4 (fp32)", 4),
                    ("M", M),
                    ("2 x blocks + topk", 2.0 * blocks + cfg["index_topk"]),
                ],
                byts,
            ),
        ),
    )


def _ring_write(cfg, b, width, ratio, overlap, label):
    """update_compressor_states: an in-place write into the Compressor's ring buffer.

    Settled from the source (`state_writes.py` update_compressor_states;
    cited by symbol): "In-place update of
    Compressor's per-request kv_state/score_state ring buffer ... the kernel writes
    unconditionally, no in-kernel mask ... kernel fuses ape addition". It compresses
    nothing: the pooling belongs to `fused_compress_attn` and is paid there.

    What it does cost, and the reason the total is not obvious by inspection --
    two of these scale with M in opposite directions:

      width   `kv`/`score` are `coff_d = (1+overlap) * width` wide, the two halves
              of `wkv_gate`'s output (`deepseek_v4.py:1295`). The trace confirms
              it: wkv_gate measures N = 2*coff_d, 2048 for CSA and 1024 for HCA.
      dtype   reads are bf16 (wkv_gate's output), the states are fp32.
      rows    the host pre-filters the plan to each sequence's last `K_pool =
              (1+overlap)*ratio` positions, so a long prefill writes K_pool rows,
              not M.
      FLOPs   one ape add per element, not a windowed weighted combine.

    `ape [ratio, coff_d]` is not counted, matching `compressor_pool`: one row per
    token out of a table small enough to stay resident.
    """
    coff = (1.0 + overlap) * width
    k_pool = (1.0 + overlap) * ratio
    rows = min(attn_M(b), k_pool)
    flops = rows * coff  # score += ape
    byts = (
        rows
        * coff
        * (
            2 * DTYPE_BYTES["bf16"]  # kv + score in
            + 2 * DTYPE_BYTES["fp32"]  # kv_state + score_state out
        )
    )
    return (
        flops,
        byts,
        Spec(
            "fp32",
            "fp32",
            "hbm",
            f"ring-buffer write of the last K_pool={_n(k_pool)} positions"
            f" ({label} x {_n(1 + overlap)}); no compression happens here",
            prod("FLOPs", [("ape add", 1), ("rows", rows), ("coff_d", coff)], flops),
            summ(
                "Bytes",
                [
                    ("kv + score in bf16", f"2 x 2 x {_n(rows)} x {_n(coff)}"),
                    (
                        "kv_state + score_state out fp32",
                        f"2 x 4 x {_n(rows)} x {_n(coff)}",
                    ),
                ],
                byts,
            ),
        ),
    )


def kv_compress(cfg, b, kind):
    """The Main compressor's ring write, at head_dim."""
    ratio = _HCA_RATIO if kind == "hca" else _CSA_RATIO
    return _ring_write(cfg, b, cfg["head_dim"], ratio, kind != "hca", "head_dim")


def kv_compress_index(cfg, b, kind):
    """The Indexer's own ring write: same kernel, at index_head_dim, always 4:1."""
    return _ring_write(
        cfg, b, cfg["index_head_dim"], _CSA_RATIO, True, "index_head_dim"
    )


def compressor_epilogue(cfg, b, kind):
    """hca_norm_rope_scatter: RMSNorm + RoPE + cache scatter, fused into one kernel.

    HCA splits the compressor in two; this is the second half. CSA runs the same
    three operations inside `fused_compress_attn`, so both call
    `_compressor_epilogue`: the name says RoPE, but it norms and scatters too, and
    it shares that algebra with the copy fused into `fused_compress_attn`.

    It runs on M/ratio compressed entries, not on M.

    The bf16 read is the pooled entry that `compressor_pool` handed over; the write
    goes into the same cache `_core_attn` reads.
    """
    ratio = _HCA_RATIO if kind == "hca" else _CSA_RATIO
    entries = attn_M(b) / ratio
    c = cfg["head_dim"]
    flops, byts, terms_f, terms_b = _compressor_epilogue(cfg, entries)
    byts += DTYPE_BYTES["bf16"] * entries * c  # pooled entry read back
    return (
        flops,
        byts,
        Spec(
            "bf16",
            "bf16",
            "hbm",
            f"HCA's split-out epilogue over M/{ratio} entries: norm + rope + scatter",
            summ("FLOPs", terms_f, flops),
            summ(
                "Bytes",
                [("pooled entry read back bf16", f"2 x {_n(entries)} x {_n(c)}")]
                + terms_b,
                byts,
            ),
        ),
    )


def rope(cfg, b, kind):
    """attn.inverse_rope: in-place inverse rotation of the attention output.

    The width was the one thing the capture could not settle -- the call sits
    inside the opaque `v4_attention_with_output`, so no `Input Dims` record names
    the tensor. The source does: `DeepseekV4Attention._wo_a_path` hands
    `RotaryEmbedding.inverse` `o.view(num_tokens, self.n_local_heads,
    self.head_dim)` with `self.rope_head_dim`, and `inverse_rope_inplace`
    (`v4_kernels/inverse_rope.py`) slices `x[..., -rope_dim:]` itself. So the
    touched width is `n_local_heads x rope_head_dim` = `n_h/tp x qk_rope_head_dim`
    -- exactly what is charged here. `rope_head_dim` is `qk_rope_head_dim`
    (`deepseek_v4.py` `V4Args`, default 64); the nope lanes are not read.

    In-place, hence one read and one write of the rope slice and no separate
    output buffer. The cos/sin cache loads are deliberately not counted: the
    kernel's grid is `(H, cdiv(S, BLOCK_S))`, so every head re-loads the same
    `rd/2` cos and sin entries for the same position -- 128 B of unique data per
    token, re-read 32 times. Charging the duplicates at HBM bandwidth would
    double this row's floor with bytes that cannot miss cache. The row is
    latency-bound by three orders of magnitude either way.
    """
    M = attn_M(b)
    width = cfg["num_attention_heads"] / b.tp * cfg["qk_rope_head_dim"]
    flops = 6.0 * M * width
    byts = DTYPE_BYTES["bf16"] * 2 * M * width
    return (
        flops,
        byts,
        Spec(
            "bf16",
            "bf16",
            "hbm",
            "sin/cos rotate over the rope dims only",
            prod("FLOPs", [("6", 6), ("M", M), ("n_h/tp x rope_dim", width)], flops),
            prod(
                "Bytes",
                [
                    ("2 (read+write)", 2),
                    ("2 (bf16)", 2),
                    ("M", M),
                    ("n_h/tp x rope_dim", width),
                ],
                byts,
            ),
        ),
    )


def norm_quant(cfg, b, kind):
    """RMSNorm + fp8 quant on the residual stream, at hidden_size."""
    return _norm_quant(cfg, b)


def q_norm_quant(cfg, b, kind):
    """The same kernel on the Q LoRA, which is q_lora_rank wide, not hidden_size.

    The capture trace settles it: `attn.q_norm` runs
    `aiter::rmsnorm_quant [[1, 1536], [1, 1536], [1, 12], [1536]]` -- 1536 is
    q_lora_rank, and the 12 is 1536/128 per-block scales. `attn.wqkv_a`'s norm runs
    the same kernel at hidden_size: two markers, one kernel name, two widths, so
    they cannot share a formula.
    """
    return _norm_quant(cfg, b, cfg["q_lora_rank"])


def indexer_rope(cfg, b, kind):
    """RoPE + Hadamard rotate + quant on the Indexer's Q, at its own head shape.

    `aiter::rope_hadamard_rotate_activation_quant [[1, 64, 128], [1, 64, 128], ...]`
    -- index_n_heads x index_head_dim, and index_n_heads is NOT tp-sharded (the
    Indexer Q projection is replicated so each rank can pick top-k locally). The
    main-attention `rope` formula is `num_attention_heads/tp * qk_rope_head_dim` =
    2048 wide; this is 8192, so the two cannot share one width.

    Three operations, three different widths, settled from the kernel
    (`dsv4_rotate_quant.cu rope_hadamard_rotate_activation_quant_kernel`):

    * RoPE runs on the TRAILING `rope_dim` lanes only -- `rope_start = dim -
      rope_dim` guards it -- and the capture names `rope_dim`: cos/sin arrive as
      `[.., 1, 1, 32]`, and that 32 is `rope_dim/2`, so 64 of the 128 lanes.
    * The Hadamard is a butterfly over the full `index_head_dim`, two nested
      `static_for` stages (within a thread, then across threads by shuffle), so
      log2(d_i) add/sub per element rather than a constant. Charging RoPE's
      6 per element for all three operations was both too wide and too cheap.
    * The 1/sqrt(dim) normalisation and the quant's abs-max pass are one and two
      more per element.

    All of it reaches the AI column only: the row is latency-bound by three
    orders (measured/floor ~ 4000x at c1).
    """
    M = attn_M(b)
    d_i = cfg["index_head_dim"]
    width = cfg["index_n_heads"] * d_i
    rope_dim = cfg["qk_rope_head_dim"]
    rope_w = cfg["index_n_heads"] * rope_dim
    stages = math.log2(d_i)
    flops = (
        6.0 * M * rope_w  # rotate, trailing rope lanes only
        + stages * M * width  # Hadamard butterfly, log2(d_i) stages
        + M * width  # 1/sqrt(dim) normalisation
        + 2.0 * M * width  # quant: abs-max, then scale
    )
    byts = DTYPE_BYTES["bf16"] * M * width + DTYPE_BYTES["fp8"] * M * width
    return (
        flops,
        byts,
        Spec(
            "bf16",
            "bf16",
            "hbm",
            "indexer rope + hadamard + quant; index heads are NOT tp-sharded",
            summ(
                "FLOPs",
                [
                    ("rope on n_i x rope_dim", f"6 x {_n(M)} x {_n(rope_w)}"),
                    (
                        f"hadamard, log2({_n(d_i)}) stages",
                        f"{stages:.0f} x {_n(M)} x {_n(width)}",
                    ),
                    ("1/sqrt(dim) + quant", f"3 x {_n(M)} x {_n(width)}"),
                ],
                flops,
            ),
            summ(
                "Bytes",
                [
                    ("read bf16", f"2 x {_n(M)} x {_n(width)}"),
                    ("write fp8", f"1 x {_n(M)} x {_n(width)}"),
                ],
                byts,
            ),
        ),
    )


def qk_norm_rope(cfg, b, kind):
    """Fused Q RMSNorm + KV RMSNorm + GPT-J RoPE on the tail rope lanes.

    One kernel, three widths, and they are not interchangeable. RoPE does not run
    on every lane, and the kernel's own template arguments say so:
    `fuse_qk_norm_rope_finegrained_kernel<..., 64, false, false, 512, 1>` -- RD=64
    against D=512, and HCA's split-out copy names the same pair in its kernel name
    (`hca_norm_rope_scatter_D512_RD64_...`). So RMSNorm runs on all `head_dim`
    lanes and RoPE on `qk_rope_head_dim` of them, exactly as the epilogue prices
    it. A flat `8 * c` over both would overstate the FLOPs 1.68x.

    Three widths, from the call site (`deepseek_v4.py` `_attn_pre` ->
    `qk_norm_rope_maybe_quant`):

    * Q: `[M, n_local_heads, head_dim]`, weightless per-head RMSNorm.
    * KV: `[M, head_dim]` -- ONE row. V4 is MQA (`num_key_value_heads=1`), so the
      weighted KV norm is `n_h/tp` times narrower than the Q norm, and it is a
      separate term rather than part of the Q side.
    * RoPE: the trailing `rd` lanes of both, so `(n_h/tp + n_kv) * rd`, not
      `n_h/tp * head_dim`.

    The write side follows the KV-cache layout, because the kernel emits what the
    attention halves consume with no requant. At `--kv-cache-dtype fp8` the 2buff
    path is on and a row is `V4_DIM_QK_PACKED` (512) fp8 + `V4_DIM_ROPE` (64) bf16
    = 640 B, not `head_dim` bf16 = 1024 B. The packed 512 is named, not derived --
    it is NoPE fp8 (448)
    plus 14 duplicated e8m0 scale bytes plus padding (`v4_quant.py`
    `V4_DIM_QK_PACKED`), and deriving `head_dim - rope_head_dim` = 448 instead is
    the exact mistake that once tripped `assert_size_stride` in ATOM itself.

    At decode this launch also fuses the SWA window scatter (`swa_dest_rows` /
    `swa_*_buff`), which is why no separate `_swa_write_kernel` appears per layer
    in the decode trace. It writes one K row per token into the ring, in the same
    layout the output uses, so it is counted here rather than left out: a second
    operator sharing one launch is still work this launch performs. It is small --
    one row against the `n_h/tp + 1` this kernel already writes.
    """
    M = attn_M(b)
    c = cfg["head_dim"]
    rd = cfg["qk_rope_head_dim"]
    h = cfg["num_attention_heads"] / b.tp
    n_kv = cfg.get("num_key_value_heads", 1)
    norm_w = (h + n_kv) * c  # Q heads + the single KV row
    rope_w = (h + n_kv) * rd  # rope touches only the tail lanes of each
    flops = 4.0 * M * norm_w + 6.0 * M * rope_w
    read = DTYPE_BYTES["bf16"] * M * norm_w
    packed = b.dtypes.kv == "fp8"
    # 512 fp8 + 64 bf16 per row -- `v4_quant.V4_DIM_QK_PACKED` / `V4_DIM_ROPE`.
    row_out = (
        (512.0 + 64.0 * DTYPE_BYTES["bf16"]) if packed else c * DTYPE_BYTES["bf16"]
    )
    write = M * (h + n_kv) * row_out
    # Decode fuses the SWA ring scatter into this launch: one K row per token, in
    # the same packed layout. Prefill passes swa_*=None and scatters separately,
    # because a chunked prefix must see the PRIOR chunk's window.
    swa = M * row_out if b.seq_len == 1 else 0.0
    byts = read + write + swa
    return (
        flops,
        byts,
        Spec(
            "bf16",
            "bf16",
            "hbm",
            "norm on all lanes, rope on the rope lanes; "
            + ("fp8 2buff out" if packed else "bf16 out"),
            summ(
                "FLOPs",
                [
                    ("RMSNorm on every lane", f"4 x {_n(M)} x {_n(norm_w)}"),
                    ("RoPE on the rope lanes only", f"6 x {_n(M)} x {_n(rope_w)}"),
                ],
                flops,
            ),
            summ(
                "Bytes",
                [
                    ("read bf16 q + kv row", f"2 x {_n(M)} x {_n(norm_w)}"),
                    (
                        "write "
                        + ("512B fp8 packed + 64 bf16 rope" if packed else "bf16"),
                        f"{_n(M)} x {_n(h + n_kv)} x {_n(row_out)}",
                    ),
                ]
                + (
                    [("SWA ring scatter, one K row", f"{_n(M)} x {_n(row_out)}")]
                    if swa
                    else []
                ),
                byts,
            ),
        ),
    )


# --------------------------------------------------------------------------
# norm / quant / elementwise
# --------------------------------------------------------------------------


def residual_add(cfg, b, kind):
    """A local elementwise add in the ATTENTION region, NOT a collective.

    `triton_poi_fused_add_all_reduce__2` has "all_reduce" in its name because it is
    the residual add fused *after* one, but it moves nothing across the fabric.
    Pricing it as a full-message all-reduce made it 27x its measured time -- a floor
    above the measurement, which is the loudest possible sign a formula is wrong.

    Shapes confirmed from the capture: `[[1, 7168], [1, 7168], []]`, two reads and
    one write of `[M, hidden]` bf16.
    """
    return _residual_add(attn_M(b), cfg)


def residual_add_moe(cfg, b, kind):
    """The same add, fused after the MoE-side all-reduce, on the GATHERED tokens.

    Split from `residual_add` for the same reason `comm_allreduce_moe` is split
    from `comm_allreduce_attn`: it sits after the DP gather, so it runs on
    `batch x dp` tokens while the attention-side copy runs on `batch`. The two
    coincide at dp-attn off and differ by a factor of dp the moment it is on, and
    the catalog runs DPA from concurrency 64 to 2048. Leaving one formula on both
    markers would also have made this row disagree with the all-reduce it is fused
    to -- one tensor, two rows, one token count.
    """
    return _residual_add(moe_B(b), cfg)


def _residual_add(M, cfg):
    H = cfg["hidden_size"]
    flops = 1.0 * M * H
    byts = DTYPE_BYTES["bf16"] * 3 * M * H
    return (
        flops,
        byts,
        Spec(
            "bf16",
            "bf16",
            "hbm",
            "local add, no fabric traffic",
            prod("FLOPs", [("1", 1), ("M", M), ("hidden", H)], flops),
            prod(
                "Bytes",
                [("3 (2 read + 1 write)", 3), ("2 (bf16)", 2), ("M", M), ("hidden", H)],
                byts,
            ),
        ),
    )


def _norm_quant(cfg, b, width=None):
    """add_rmsnorm_quant: RMSNorm fused with the fp8 cast, so the write is fp8."""
    M, H = attn_M(b), width if width else cfg["hidden_size"]
    flops = 6.0 * M * H
    byts = (
        DTYPE_BYTES["bf16"] * M * H
        + DTYPE_BYTES["fp8"] * M * H
        + DTYPE_BYTES["fp32"] * M * H / 128
        + DTYPE_BYTES["bf16"] * H
    )
    return (
        flops,
        byts,
        Spec(
            "bf16",
            "bf16",
            "hbm",
            "norm + fp8 cast fused: the write is fp8, not bf16",
            prod("FLOPs", [("6", 6), ("M", M), ("hidden", H)], flops),
            summ(
                "Bytes",
                [
                    ("read bf16", f"2 x {_n(M)} x {_n(H)}"),
                    ("write fp8", f"1 x {_n(M)} x {_n(H)}"),
                    ("scales fp32/128", f"4 x {_n(M)} x {_n(H)}/128"),
                    ("weights", f"2 x {_n(H)}"),
                ],
                byts,
            ),
        ),
    )


def _quant(M, width, note):
    """Dynamic per-group fp8 quantization: read bf16, write fp8 + fp32 scales."""
    flops = 2.0 * M * width
    byts = (
        DTYPE_BYTES["bf16"] * M * width
        + DTYPE_BYTES["fp8"] * M * width
        + DTYPE_BYTES["fp32"] * M * width / 128
    )
    return (
        flops,
        byts,
        Spec(
            "bf16",
            "bf16",
            "hbm",
            note,
            prod("FLOPs", [("2", 2), ("M", M), ("width", width)], flops),
            summ(
                "Bytes",
                [
                    ("read bf16", f"2 x {_n(M)} x {_n(width)}"),
                    ("write fp8", f"1 x {_n(M)} x {_n(width)}"),
                    ("scales fp32/128", f"4 x {_n(M)} x {_n(width)}/128"),
                ],
                byts,
            ),
        ),
    )


def quant_hidden(cfg, b, kind):
    return _quant(attn_M(b), cfg["hidden_size"], "width = hidden_size")


def quant_heads(cfg, b, kind):
    """Quantizing the attention output before wo_b: width is the o-LoRA rank."""
    return _quant(
        attn_M(b),
        cfg["o_groups"] * cfg["o_lora_rank"] / b.tp,
        "width = o_groups x o_lora_rank / tp",
    )


def quant_moe_intermediate(cfg, b, kind):
    """The quant BEFORE w2: its input is the shared expert's own intermediate."""
    return _quant(
        moe_B(b),
        cfg["moe_intermediate_size"] * cfg["n_shared_experts"] / b.tp,
        "width = shared-expert I_local",
    )


def quant_moe_hidden(cfg, b, kind):
    """The quant BEFORE gate_up: its input is the hidden state, not the intermediate.

    Both shared-expert markers contain the same quant kernel, and sharing one
    formula between them charged this one the intermediate's 768 instead of the
    hidden state's 7168 -- 9.3x low. The capture trace names the width outright:

        layers.30.ffn.shared_experts.gate_up_proj
          aiter::dynamic_per_token_scaled_quant  [[1, 7168], [56, 128], [1, 56]]
          aiter::gemm_a8w8_blockscale_bpreshuffle [[1, 7168], [1536, 7168], ...]

    -- the quant feeds the GEMM's K, which is the full hidden size, while w2's
    quant feeds a K of `moe_intermediate_size * n_shared / tp`. Same kernel, same
    marker family, different widths. Sixth instance of one formula being shared by
    markers that do not share a width.

    `moe_B`, not `attn_M`: this is the MoE region, where the token count follows a
    different rule even though the two coincide at dp-attn off.
    """
    return _quant(moe_B(b), cfg["hidden_size"], "width = hidden_size (MoE region)")


# --------------------------------------------------------------------------
# MoE
# --------------------------------------------------------------------------


def active_experts(cfg: dict, b: Bench) -> float:
    """Distinct experts receiving at least one token, assuming UNIFORM routing.

    Real routing (noaux_tc, learned bias) CONCENTRATES into fewer experts, so at
    moderate M this over-estimates the weight load -- and an over-estimated floor
    is not a floor, which is why a skewed run can show efficiency above 1 on these
    two rows. That concentration is a learned runtime property, not derivable from
    the config, so it is left in the gap on purpose rather than fitted; a fitted
    constant would silently rot when the checkpoint changes.

    Where imbalance does NOT reach: at decode c1 there is one token, its k experts
    are distinct by construction, and every one of them is padded to a full tile.
    Both terms are exact there, whatever the routing does. Imbalance starts to
    matter once an expert can hold more than one tile's worth of tokens, i.e. from
    M x topk / E > 32, which is concurrency ~2048 at this config.
    """
    E = cfg["n_routed_experts"]
    k = cfg["num_experts_per_tok"]
    # One TOKEN draws k DISTINCT experts, so the miss probability per token is
    # (1 - k/E), not (1 - 1/E) repeated k times. The two agree at large M and
    # differ exactly where this run lives: at M=1 the correct form gives 6.000
    # -- top-k cannot pick the same expert twice -- against 5.953 for the
    # independent-draw version.
    return E * (1.0 - (1.0 - k / E) ** moe_B(b))


MOE_TILE_M = 32.0
"""Rows the MoE sorter pads each expert's token list up to.

Measured, not assumed: the capture trace's `moe_sorting_opus_fwd` takes a
`sorted_ids [12288]` buffer and `12288 = n_routed_experts 384 x 32`, and the
expert GEMM names its own tile (`mfma_moe1_..._t32x128x256`).

NOT corroborated by prefill, and the discrepancy is left open rather than
explained away: prefill uses different tiles (`t128x256x256` / `t64x128x256`)
and its launch grids -- the only ones in the trace, since decode is a graph
replay -- give 723 M-tiles x 128 = 92,544 rows against 43,422 real
assignments. Padding each of 384 experts to 128 rows reaches at most 49,152,
so a factor of ~1.9 in the prefill grid is unaccounted for. Whatever it is, it
does not touch the decode numbers, which rest on the 12288 buffer and the t32
tile.
"""


def moe_padded_rows(cfg: dict, b: Bench) -> float:
    """Rows the expert pipeline actually moves, padding included.

    One token routed to `topk` experts is `M x topk` rows of real work, but the
    sorter pads EACH expert's list up to the GEMM's tile, so at decode c1 six
    tokens become six tiles of 32 -- 192 rows, of which 186 are zeros that are
    still written, still read by the GEMM, and still multiplied. That is a 32x
    amplification on every per-row term, and it vanishes as M grows.

    `max` rather than an exact per-expert ceiling: the exact count needs each
    expert's token count, which is the routing distribution and is not derivable
    from the config. At small M the tile term dominates and is exact; at large M
    the real rows dominate and the padding is at most 31 rows per active expert.
    """
    rows = moe_B(b) * cfg["num_experts_per_tok"]
    return max(rows, active_experts(cfg, b) * MOE_TILE_M)


def _moe_ffn(cfg, b, which):
    """mfma_moe1 (gate_up) / mfma_moe2 (down).

    Expert weights are STORED fp4 (MXFP4) but COMPUTED fp8: per the V4 report,
    FP4xFP8 runs at the FP8 peak -- there is no fp4 compute speedup on this
    hardware, fp4 is dequantized to fp8 losslessly. So the compute roof is fp8
    while the bytes are fp4. Only the GEMM is modelled; the fused elementwise work
    (SiLU-mul, token sort, fp8 quant) is second-order against the active-expert
    weight load, which dominates decode.
    """
    H = cfg["hidden_size"]
    I_local = cfg["moe_intermediate_size"] / b.tp
    tokens = moe_B(b) * cfg["num_experts_per_tok"]
    # The GEMM does not see the real rows, it sees the padded ones: at decode c1
    # that is 192 against 6. Weights still dominate, so this moves the floor ~5%,
    # but the activation terms were 32x low.
    rows = moe_padded_rows(cfg, b)
    active = active_experts(cfg, b)
    w_dtype = b.dtypes.expert_w
    if which == "gate_up":
        K, N, wshape = H, 2 * I_local, ("hidden", "N=2*I_local")
    else:
        K, N, wshape = I_local, H, ("K=I_local", "hidden")
    flops = 2.0 * tokens * K * N
    wb = w_bytes(active * K * N, w_dtype)
    # afp8_wfp4: the activation the kernel reads is fp8, the output it writes bf16.
    ab = (
        DTYPE_BYTES[b.dtypes.act if w_dtype == "bf16" else "fp8"] * rows * K
        + DTYPE_BYTES["bf16"] * rows * N
    )
    return (
        flops,
        wb + ab,
        prod(
            "FLOPs",
            [("2", 2), ("B x topk", tokens), (wshape[0], K), (wshape[1], N)],
            flops,
        ),
        summ(
            "Bytes",
            [
                (
                    "active experts x W fp4",
                    f"{_n(active)} x {_n(K)} x {_n(N)} x 0.5 x 1.0625",
                ),
                ("act in fp8, padded rows", f"1 x {_n(rows)} x {_n(K)}"),
                ("act out bf16, padded rows", f"2 x {_n(rows)} x {_n(N)}"),
            ],
            wb + ab,
        ),
    )


def _moe_spec(b, fe, be, active=None, of=None) -> Spec:
    """Storage from the run's expert_dtype, ceiling from the hardware rule.

    Hard-coding fp4 here was wrong for half the catalog: GLM-5.2, MiniMax-M3 and
    Qwen3.5 each ship an FP8 and an MXFP4 build of the SAME architecture, so the
    precision is a property of the run, not of the operator.
    """
    w = b.dtypes.expert_w
    return Spec(
        compute_dtype(w),
        w,
        "hbm",
        f"experts stored {w}, computed {compute_dtype(w)}; "
        f"the active-expert weight load dominates"
        + (
            f" | active={active:,.1f} of {of:,.0f} experts, UNIFORM routing assumed"
            " -- real routing concentrates, which makes this an over-estimate and"
            " the floor with it"
            if active is not None
            else ""
        ),
        fe,
        be,
    )


def moe_gate_up(cfg, b, kind):
    f, y, fe, be = _moe_ffn(cfg, b, "gate_up")
    return f, y, _moe_spec(b, fe, be, active_experts(cfg, b), cfg["n_routed_experts"])


def moe_down(cfg, b, kind):
    f, y, fe, be = _moe_ffn(cfg, b, "down")
    return f, y, _moe_spec(b, fe, be, active_experts(cfg, b), cfg["n_routed_experts"])


def moe_router(cfg, b, kind):
    """ffn.gate: the gating GEMM [M,H] x [H,E], on the gathered tokens."""
    M, H, E = moe_B(b), cfg["hidden_size"], cfg["n_routed_experts"]
    flops = 2.0 * M * H * E
    byts = DTYPE_BYTES["bf16"] * (M * H + H * E + M * E)
    return (
        flops,
        byts,
        Spec(
            "bf16",
            "bf16",
            "hbm",
            "gating GEMM over all experts",
            prod("FLOPs", [("2", 2), ("B", M), ("hidden", H), ("n_experts", E)], flops),
            summ(
                "Bytes",
                [
                    ("tokens", f"2 x {_n(M)} x {_n(H)}"),
                    ("gate weights", f"2 x {_n(H)} x {_n(E)}"),
                    ("logits", f"2 x {_n(M)} x {_n(E)}"),
                ],
                byts,
            ),
        ),
    )


def moe_topk(cfg, b, kind):
    """topk_softplus / hash_topk: score selection over the router logits.

    Reads what `moe_router` wrote, which is bf16 -- this charged fp32 for the E
    term and so double-counted the dominant read. Two adjacent rows have to agree
    about one tensor; the GEMM pair next door (quant writes fp8, gemm reads fp8)
    is the same seam done right. From the source
    (`topk_gating_kernels.cu topk_softplus_kernel_opt`):

        float score = compute_score<SCORE_FUNC>(static_cast<float>(input_ptr[e]));

    -- `DTYPE_I` is the gating output's own dtype and fp32 is the register
    accumulator, not the load width. Outputs are `float* topk_weights` and
    `int* topk_ids`, k of each.

    Not counted: an optional `[E]` correction bias, read once per block when the
    router supplies one. It would add one more bf16 E-read; whether V4-Pro passes
    one is not visible in the trace.

    The cost is not in these bytes and not in the FLOPs. `token_idx = blockIdx.x`
    means ONE block per token, so at decode M=1 this kernel occupies a single
    workgroup of 256 CUs and runs a register-only sort network (EPT = E/warp = 6)
    plus a warp k-way merge. 6.35 us per firing against a 0.0002 us floor --
    30,000x -- is the same reading as the indexer's radix top-k: a selection
    kernel at M=1 is occupancy-bound, and no bandwidth ceiling explains it.
    """
    M, E = moe_B(b), cfg["n_routed_experts"]
    k = cfg["num_experts_per_tok"]
    flops = 4.0 * M * E
    byts = (
        DTYPE_BYTES["bf16"] * M * E  # the logits, as moe_router wrote them
        + DTYPE_BYTES["fp32"] * M * k  # topk_weights
        + I32_BYTES * M * k  # topk_ids
    )
    return (
        flops,
        byts,
        Spec(
            "bf16",
            "fp32",
            "hbm",
            "softplus + top-k over E logits",
            prod("FLOPs", [("4", 4), ("B", M), ("n_experts", E)], flops),
            summ(
                "Bytes",
                [
                    ("read bf16 logits", f"2 x {_n(M)} x {_n(E)}"),
                    ("write fp32 weights", f"4 x {_n(M)} x {_n(k)}"),
                    ("write int32 ids", f"4 x {_n(M)} x {_n(k)}"),
                ],
                byts,
            ),
        ),
    )


def moe_dispatch(cfg, b, kind):
    """MoE sorting: permute the INDICES, and zero the fused-MoE output buffer.

    Settled from the source. `aiter/fused_moe.py`'s `moe_sorting` (cited by
    symbol, not line: aiter line numbers drift across commits) passes only
    `topk_ids [M, topk]` (i32) and `topk_weights [M, topk]` (fp32) in, and gets
    sorted ids / weights / expert-ids out -- the token data is never touched. The
    one large term is a buffer clear, which CK's `moe_sorting_kernel.hpp` states
    three times: "we fused the setzero of output of fused-moe buffer", "must be
    cleard before use", "we fuse this clearing inside sorting kernel", implemented
    as `moe_buf_set_zero_kernel`.

    So it is WRITE-ONLY over [M, model_dim]: a "sort" that moves indices, not the
    tokens they point at. The sort latency itself is not modelled and stays in the
    gap, where it reads as latency rather than bandwidth.
    """
    M, H = moe_B(b), cfg["hidden_size"]
    k = cfg["num_experts_per_tok"]
    clear = DTYPE_BYTES["bf16"] * M * H  # write-only zero fill
    idx = DTYPE_BYTES["fp32"] * 2 * (2 * M * k)  # ids + weights, in and out
    return (
        0.0,
        clear + idx,
        Spec(
            "bf16",
            "bf16",
            "hbm",
            "index permute + fused zero-fill of the output buffer",
            "FLOPs = 0  (a permutation and a memset)",
            summ(
                "Bytes",
                [
                    ("moe_buf zero-fill, WRITE ONLY", f"2 x {_n(M)} x {_n(H)}"),
                    ("topk ids+weights in/out", f"4 x 2 x 2 x {_n(M)} x {_n(k)}"),
                ],
                clear + idx,
            ),
        ),
    )


def moe_quant_sort(cfg, b, kind):
    """fused_mx_quant_moe_sort: quantise the tokens INTO expert order.

    Not the same operator as `moe_dispatch`, which shares its marker and whose
    kernel name shares the substring "moe_sort". `opus_moe_sorting_entry` permutes
    indices and clears the output buffer; this one moves the token data, so the two
    cannot share a formula.

    The source states the one thing that decides the byte count -- aiter's
    `fused_dynamic_mx_quant_moe_sort` docstring on the small-M path this decode
    takes: "single fused HIP kernel ... which does quant + sort + swizzle in one
    pass. Saves a kernel launch but **re-quantises each input row up to topk
    times**." The kernel agrees: it indexes `input + offset_base * input_stride`
    with `offset_base = token_idx * topk + topk_id`. So it reads `M x topk` rows,
    not `M`.

    Output dtype is the MoE GEMM's activation dtype, which the expert kernel's own
    name states (`mfma_moe1_silu_mul_afp8_wfp4`): fp8 activations against fp4
    weights. The e8m0 scale is `(pad32(rows), H/group_size)` bytes -- at decode the
    row padding dominates it, 32 padded rows of scale for 6 real ones.
    """
    M, H = moe_B(b), cfg["hidden_size"]
    k = cfg["num_experts_per_tok"]
    group = 32.0  # MX block
    rows = M * k  # real tokens, each read once per expert that selected it
    padded = moe_padded_rows(cfg, b)  # what it writes: every expert's tile
    read = DTYPE_BYTES["bf16"] * rows * H
    write = DTYPE_BYTES[b.dtypes.act if b.dtypes.act != "bf16" else "fp8"] * padded * H
    scale = padded * (H / group)  # one e8m0 byte per group
    flops = 2.0 * rows * H  # abs-max reduce + scale multiply
    byts = read + write + scale
    return (
        flops,
        byts,
        Spec(
            "bf16",
            "bf16",
            "hbm",
            f"quant + permute into expert order; each row re-read topk times"
            f" | {padded:,.0f} rows written for {rows:,.0f} real ones"
            f" ({padded / rows:.0f}x), the sorter's per-expert tile padding",
            prod("FLOPs", [("2", 2), ("M x topk", rows), ("hidden", H)], flops),
            summ(
                "Bytes",
                [
                    ("read bf16, topk times", f"2 x {_n(rows)} x {_n(H)}"),
                    ("write fp8, padded rows", f"1 x {_n(padded)} x {_n(H)}"),
                    ("e8m0 scale, padded rows", f"{_n(padded)} x {_n(H)}/32"),
                ],
                byts,
            ),
        ),
    )


def shared_expert_gate_up(cfg, b, kind):
    M, H = moe_B(b), cfg["hidden_size"]
    I_local = cfg["moe_intermediate_size"] * cfg["n_shared_experts"] / b.tp
    w = b.dtypes.linear_w
    f, y, fe, be = gemm(M, H, 2 * I_local, w)
    return (
        f,
        y,
        Spec(compute_dtype(w), w, "hbm", "fused gate+up, N = 2 x I_local", fe, be),
    )


def shared_expert_down(cfg, b, kind):
    M, H = moe_B(b), cfg["hidden_size"]
    I_local = cfg["moe_intermediate_size"] * cfg["n_shared_experts"] / b.tp
    w = b.dtypes.linear_w
    f, y, fe, be = gemm(M, I_local, H, w)
    return f, y, Spec(compute_dtype(w), w, "hbm", "down projection", fe, be)


# --------------------------------------------------------------------------
# output projection
# --------------------------------------------------------------------------


def o_proj_a(cfg, b, kind):
    """wo_a: a GROUPED LoRA down-projection, `bsgd,grd->bsgr`, batched over groups.

    The capture trace gives the real shapes at TP4: input `[M, 4, 4096]`, weight
    `[4, 1024, 4096]`, output `[M, 4, 1024]`. So `o_groups` is tp-sharded (16 -> 4)
    and each group maps `n_heads*head_dim/o_groups` down to `o_lora_rank`. The
    group count checks out twice over: `4 * 4096 = 16384` is exactly the local
    32 heads x head_dim that attention produced, and `4 * 1024 = 4096` is exactly
    `wo_b`'s measured K.

    Writing it as one `gemm(M, 4096, 4096)` gave the right FLOPs and the right
    weight bytes -- the group count cancels out of both -- but read one 4096-wide
    activation instead of four, and showed an algebra with no groups in it.
    """
    M = attn_M(b)
    groups = cfg["o_groups"] / b.tp
    k_g = cfg["num_attention_heads"] * cfg["head_dim"] / cfg["o_groups"]
    n_g = cfg["o_lora_rank"]
    flops = 2.0 * M * groups * k_g * n_g
    byts = DTYPE_BYTES["bf16"] * (
        groups * k_g * n_g  # grouped weights
        + M * groups * k_g  # activations in, one per group
        + M * groups * n_g  # activations out
    )
    return (
        flops,
        byts,
        Spec(
            "bf16",
            "bf16",
            "hbm",
            f"grouped LoRA: {_n(groups)} groups of {_n(k_g)}x{_n(n_g)} (o_groups is tp-sharded)",
            prod(
                "FLOPs",
                [("2", 2), ("M", M), ("groups", groups), ("k_g", k_g), ("n_g", n_g)],
                flops,
            ),
            summ(
                "Bytes",
                [
                    (
                        "grouped weights bf16",
                        f"2 x {_n(groups)} x {_n(k_g)} x {_n(n_g)}",
                    ),
                    ("act in bf16", f"2 x {_n(M)} x {_n(groups)} x {_n(k_g)}"),
                    ("act out bf16", f"2 x {_n(M)} x {_n(groups)} x {_n(n_g)}"),
                ],
                byts,
            ),
        ),
    )


def mhc_pre_gemm(cfg, b, kind):
    M = attn_M(b)
    hc = cfg["hc_mult"]
    hc_dim = hc * cfg["hidden_size"]
    mix_hc = (2 + hc) * hc
    flops = 2.0 * M * hc_dim * mix_hc
    byts = DTYPE_BYTES["bf16"] * M * hc_dim + DTYPE_BYTES["fp32"] * hc_dim * mix_hc
    return (
        flops,
        byts,
        Spec(
            "fp32",
            "bf16",
            "hbm",
            "hc-fn linear + sqrsum; fp32 MFMA, and fp32 buys no Matrix Core uplift",
            prod(
                "FLOPs",
                [("2", 2), ("M", M), ("hc x dim", hc_dim), ("(2+hc) x hc", mix_hc)],
                flops,
            ),
            summ(
                "Bytes",
                [
                    ("widened residual bf16", f"2 x {_n(M)} x {_n(hc_dim)}"),
                    ("hc_fn weights fp32", f"4 x {_n(hc_dim)} x {_n(mix_hc)}"),
                ],
                byts,
            ),
        ),
    )


def mhc_fused_post_pre(cfg, b, kind):
    """hc_post of one sub-layer fused with the *GEMM half* of the next hc_pre.

    Settled against the aiter kernel signature rather than the python-level
    decomposition (`csrc/include/mhc.h mhc_fused_post_pre_gemm_sqrsum`):

        out:  gemm_out_mul (split_k, m, mix_hc) | gemm_out_sqrsum (split_k, m)
              next_residual (m, hc, dim)
        in:   layer_input (m, dim) | residual_in (m, hc, dim)
              post_layer_mix (m, hc) | comb_res_mix (m, hc, hc) | fn (mix_hc, hc*dim)

    Two consequences for the byte and FLOP counts. It does NOT emit `y (m, dim)`:
    the weighted reduce, Sinkhorn and rmsnorm all live in the *next* kernel
    (`mhc_pre_big_fuse_rmsnorm`, priced by `mhc_pre_fuse`), so no `y` write is
    charged here. And the post half is not just the gate: `hc_post` is
    `post * x + sum(comb * residual)` (`deepseek_v4.py hc_post`), whose second
    term is a batched `[M,hc,hc] x [M,hc,dim]` matmul -- `hc` times larger than
    the gate beside it.

    `fn` stays fp32 in memory. `is_fn_pack_bf16` splits each fp32 into hi/lo bf16
    *in registers* to reach the bf16 MFMA ("selects the bf16 (fn hi/lo split)
    matrix compute over the fp32 one" -- mhc_kernels.cu:24); the layout is
    untouched either way. ATOM never passes the flag, so this GEMM runs the fp32
    MFMA path -- see the note below, which the ceiling does not yet reflect.
    """
    M = attn_M(b)
    dim, hc = cfg["hidden_size"], cfg["hc_mult"]
    hc_dim = hc * dim
    mix_hc = (2 + hc) * hc
    # aiter picks split_k per (arch, num_cu, m) at runtime; it scales only the
    # small fp32 spill below, not any of the terms that set the floor.
    split_k = 1
    flops = (
        1.0 * M * hc * dim  # post: gate the sub-layer output into hc streams
        + 2.0 * M * hc * hc * dim  # post: comb mixes the hc residual streams
        + 2.0 * M * hc_dim * mix_hc  # pre: the hc-fn linear
        + 2.0 * M * hc_dim  # pre: sqrsum (square + accumulate) over the residual
    )
    byts = (
        DTYPE_BYTES["bf16"] * M * dim  # layer_input in
        + DTYPE_BYTES["bf16"] * M * hc_dim  # residual_in
        + DTYPE_BYTES["bf16"] * M * hc_dim  # next_residual out
        + DTYPE_BYTES["fp32"] * hc_dim * mix_hc  # fn weights -- fp32 in memory
        + DTYPE_BYTES["fp32"] * M * hc * (1 + hc)  # post_layer_mix + comb_res_mix
        + DTYPE_BYTES["fp32"] * split_k * M * (mix_hc + 1)  # gemm_out_mul + sqrsum
    )
    return (
        flops,
        byts,
        Spec(
            "fp32",
            "bf16",
            "hbm",
            "hc_post fused with the next hc_pre's GEMM; the reduce is a separate kernel"
            " | CONFIG: this GEMM runs the fp32 MFMA path (v_mfma_f32_16x16x4_f32)"
            " because ATOM never passes aiter's is_fn_pack_bf16, which IS compiled in"
            " on gfx950. The Matrix Core gives no uplift at fp32 -- 157.3 TF, the same"
            " as the vector units -- so the unused bf16 path is a 16x higher ceiling."
            " The bound does not move (this operator is memory-bound either way), but"
            " the headroom does: an A/B run is what would settle whether it helps.",
            summ(
                "FLOPs",
                [
                    ("post gate", f"{_n(M)} x {_n(hc)} x {_n(dim)}"),
                    ("post comb mix", f"2 x {_n(M)} x {_n(hc)}^2 x {_n(dim)}"),
                    ("pre hc-fn linear", f"2 x {_n(M)} x {_n(hc_dim)} x {_n(mix_hc)}"),
                    ("pre sqrsum", f"2 x {_n(M)} x {_n(hc_dim)}"),
                ],
                flops,
            ),
            summ(
                "Bytes",
                [
                    ("residual in+out bf16", f"2 x 2 x {_n(M)} x {_n(hc_dim)}"),
                    ("layer_input bf16", f"2 x {_n(M)} x {_n(dim)}"),
                    ("fn weights fp32", f"4 x {_n(hc_dim)} x {_n(mix_hc)}"),
                    ("post + comb fp32", f"4 x {_n(M)} x {_n(hc)} x {_n(1 + hc)}"),
                    ("gemm_out fp32", f"4 x {_n(M)} x {_n(mix_hc + 1)}"),
                ],
                byts,
            ),
        ),
    )


def mhc_pre_fuse(cfg, b, kind):
    """The second half of hc_pre: Sinkhorn, weighted reduce, and the output rmsnorm.

    `mhc_pre_big_fuse_rmsnorm` (mhc.h) consumes the GEMM half's spill and emits
    `out (m, dim)` plus the `post_mix`/`comb_mix` the *next* hc_post will need.
    Its `norm_weight (dim)` argument is why a kernel whose name says rmsnorm is
    priced as a reduce as well: it does both, and both are counted.

    The rms divisor rides on `gemm_out_sqrsum`: the hc-fn linear is homogeneous,
    so aiter normalises the mix_hc-wide GEMM result instead of the hc*dim-wide
    input. That is why the GEMM kernel only accumulates squares and no separate
    normalise pass appears on either side.
    """
    M, dim, hc = attn_M(b), cfg["hidden_size"], cfg["hc_mult"]
    mix_hc = (2 + hc) * hc
    iters = int(cfg.get("hc_sinkhorn_iters", 20))
    flops = (
        2.0 * M * hc * dim  # weighted reduce [M,hc,dim] -> [M,dim]
        + 4.0 * M * dim  # rmsnorm on the reduced output
        + 4.0 * iters * M * hc * hc  # Sinkhorn: row- then column-normalise
    )
    byts = (
        DTYPE_BYTES["bf16"] * M * hc * dim  # residual in
        + DTYPE_BYTES["bf16"] * M * dim  # out
        + DTYPE_BYTES["bf16"] * dim  # norm_weight
        + DTYPE_BYTES["fp32"] * M * (mix_hc + 1)  # gemm_out_mul + sqrsum in
        + DTYPE_BYTES["fp32"] * M * hc * (1 + hc)  # post_mix + comb_mix out
    )
    return (
        flops,
        byts,
        Spec(
            "bf16",
            "bf16",
            "hbm",
            "Sinkhorn + weighted reduce [M,hc,dim] -> [M,dim] + rmsnorm",
            summ(
                "FLOPs",
                [
                    ("weighted reduce", f"2 x {_n(M)} x {_n(hc)} x {_n(dim)}"),
                    ("rmsnorm", f"4 x {_n(M)} x {_n(dim)}"),
                    ("Sinkhorn", f"4 x {_n(iters)} x {_n(M)} x {_n(hc)}^2"),
                ],
                flops,
            ),
            summ(
                "Bytes",
                [
                    ("read [M,hc,dim]", f"2 x {_n(M)} x {_n(hc)} x {_n(dim)}"),
                    ("write [M,dim]", f"2 x {_n(M)} x {_n(dim)}"),
                    ("gemm_out fp32", f"4 x {_n(M)} x {_n(mix_hc + 1)}"),
                    ("post + comb fp32", f"4 x {_n(M)} x {_n(hc)} x {_n(1 + hc)}"),
                ],
                byts,
            ),
        ),
    )


def mhc_post(cfg, b, kind):
    """hc_post standalone: `post * x + sum(comb * residual)`.

    `mhc_post(out, x, residual, post_layer_mix, comb_res_mix)` (mhc.h) takes the
    pre-layer residual as an input: it is both read and written, and the comb term
    that reads it is a batched `[M,hc,hc] x [M,hc,dim]` matmul, `hc` times the gate
    beside it.
    """
    M, dim, hc = attn_M(b), cfg["hidden_size"], cfg["hc_mult"]
    flops = (
        1.0 * M * hc * dim  # gate the sub-layer output into hc streams
        + 2.0 * M * hc * hc * dim  # comb mixes the hc residual streams
    )
    byts = DTYPE_BYTES["bf16"] * (
        M * dim + M * hc * dim + M * hc * dim  # x in  # residual in  # out
    ) + DTYPE_BYTES["fp32"] * M * hc * (
        1 + hc
    )  # post_layer_mix + comb_res_mix
    return (
        flops,
        byts,
        Spec(
            "bf16",
            "bf16",
            "hbm",
            "gate [M,dim] into [M,hc,dim] and mix the hc residual streams",
            summ(
                "FLOPs",
                [
                    ("gate", f"{_n(M)} x {_n(hc)} x {_n(dim)}"),
                    ("comb mix", f"2 x {_n(M)} x {_n(hc)}^2 x {_n(dim)}"),
                ],
                flops,
            ),
            summ(
                "Bytes",
                [
                    ("read [M,dim]", f"2 x {_n(M)} x {_n(dim)}"),
                    ("residual in+out", f"2 x 2 x {_n(M)} x {_n(hc)} x {_n(dim)}"),
                    ("post + comb fp32", f"4 x {_n(M)} x {_n(hc)} x {_n(1 + hc)}"),
                ],
                byts,
            ),
        ),
    )


def _layout_copy(cfg, b, n_wide):
    """An Inductor-generated pointwise copy that lands inside an ATOM marker.

    `triton_poi_fused_as_strided_clone{_copy}` comes from torch.compile, not from
    aiter: neither the fused nor the un-fused mHC path in `aiter/ops/mhc.py` calls
    `.contiguous()`, and the name appears nowhere in aiter's sources. It fires
    once per layer under each of four markers -- and the mHC one splits 60/1
    between the fused and un-fused markers, the same layer-0 split that
    `enable_fused_hc` produces.

    The shapes are read from the capture trace's `Input Dims`, not inferred:
    `[[M, dim], [dim], [M, dim]]` under the mHC markers and `[[M, dim], [dim]]`
    under the FFN ones, with an `[M, dim]` output. They are on the INDUCTOR
    kernel's own cpu_op in the capture trace; the same cpu_op in the run trace
    carries `Input Dims: None`, so this is one of the shapes only the capture can
    give.

    WHY torch.compile needs the copy is now settled from the capture, at least
    for the FFN pair: the op that immediately follows it inside the marker is the
    opaque aiter custom op that consumes its output --

        layers.30.ffn.gate
          triton_poi_fused_as_strided_clone_1  [[1, 7168], [7168], []]
          aiter::gemm_a16w16                   [[1, 7168], [384, 7168], ...]

    -- so the copy materialises a strided `[M, dim]` view into a contiguous
    buffer for a custom op that cannot take strides, with the `[dim]` vector
    fused into the same pointwise kernel. The stride comes from the mHC side,
    which carries its state as `[M, hc_mult, dim]`; a per-lane `[M, dim]` out of
    that is a view, not a tensor.

    That makes the cost structural rather than incidental: 4 copies per layer at
    decode c1 = 15.9 ms of 312.7 ms (5.1%), against a byte floor of 21.9 us
    (0.14% of their own measured time). 91% of it is the per-kernel launch floor,
    so what is being paid for is four kernel LAUNCHES per layer, not the 57 KB.
    Making the producer hand out a contiguous buffer removes the launches; making
    the bytes cheaper would recover nothing.

    FLOPs are zero: this moves data and computes nothing. At decode it is three
    orders of magnitude under the per-kernel launch floor, so pricing it does not
    change any bound; it moves the time out of `gap`, where an unpriced row is
    indistinguishable from an unmodelled one.
    """
    M, dim = attn_M(b), cfg["hidden_size"]
    byts = DTYPE_BYTES["bf16"] * ((n_wide + 1) * M * dim + dim)
    return (
        0.0,
        byts,
        Spec(
            "bf16",
            "bf16",
            "hbm",
            f"layout copy: {n_wide} x [M,dim] + [dim] in, [M,dim] out"
            " (shapes from the capture trace's Input Dims; torch.compile emits it,"
            " aiter does not)",
            "FLOPs = 0 (pure data movement)",
            summ(
                "Bytes",
                [
                    (
                        f"{n_wide + 1} x [M,dim] bf16",
                        f"2 x {n_wide + 1} x {_n(M)} x {_n(dim)}",
                    ),
                    ("[dim] bf16", f"2 x {_n(dim)}"),
                ],
                byts,
            ),
        ),
    )


def layout_copy_wide(cfg, b, kind):
    """`..._as_strided_clone_copy__0`: two [M,dim] inputs plus a [dim] vector."""
    return _layout_copy(cfg, b, 2)


def layout_copy_narrow(cfg, b, kind):
    """`..._as_strided_clone_1`: one [M,dim] input plus a [dim] vector."""
    return _layout_copy(cfg, b, 1)


def scale_indexer_weights(cfg, b, kind):
    """`weights * q_scale.squeeze(-1) * weights_scale` in one Triton launch.

    Settled from ATOM's own kernel, not inferred
    (`model_ops/v4_kernels/indexer_weights.py`): `weights [T, H]` and
    `q_scale [T, H, 1]` in, `out [T, H]` fp32 out, where H is `index_n_heads`.
    That H is the same 64 the trace reports as `attn.indexer.weights_proj`'s N,
    which is the cross-check that the head count is the right one.

    The output is fp32 even though `weights` arrives bf16 -- `torch.empty_like(
    weights, dtype=torch.float32)` -- so the write costs twice the read.
    """
    M, H = attn_M(b), float(cfg["index_n_heads"])
    flops = 2.0 * M * H  # scale by q_scale, then by the scalar
    byts = (
        DTYPE_BYTES["bf16"] * M * H  # weights in
        + DTYPE_BYTES["fp32"] * M * H  # q_scale in
        + DTYPE_BYTES["fp32"] * M * H  # out
    )
    return (
        flops,
        byts,
        Spec(
            "fp32",
            "bf16",
            "hbm",
            "scale the indexer weights by the per-head q scale; fp32 out from bf16 in",
            summ("FLOPs", [("2 scales per element", f"2 x {_n(M)} x {_n(H)}")], flops),
            summ(
                "Bytes",
                [
                    ("weights bf16", f"2 x {_n(M)} x {_n(H)}"),
                    ("q_scale + out fp32", f"2 x 4 x {_n(M)} x {_n(H)}"),
                ],
                byts,
            ),
        ),
    )


def csa_translate_pack(cfg, b, kind):
    """Translate the indexer's seq-local top-k rows into packed physical KV slots.

    From `model_ops/v4_kernels/csa_translate_pack.py`: reads `topk_local
    [T, index_topk]` int32 but only the `[0, valid_k)` prefix of each row, looks
    each entry up in `block_tables`, and writes the result into `kv_indices_csa`.
    `valid_k` is the Indexer's per-row visibility, `min(kv/ratio, index_topk)` --
    the same quantity `kv_entries` already computes for the CSA core attention,
    so the two cannot drift apart.

    FLOPs are zero: this is integer index arithmetic, the same convention used
    for the other gather/scatter operators here. The block_tables lookups are
    counted as one int32 each; they are scattered, but the table is small enough
    to sit in cache, so counting them at HBM cost is the pessimistic side.
    """
    M = attn_M(b)
    k = kv_entries(cfg, b, "csa")
    byts = I32_BYTES * (
        M * k  # topk_local, valid prefix only
        + M * k  # block_tables lookup, one per entry
        + M * k  # kv_indices_csa out
        + 3 * M  # indptr, batch_id, skip, per token
    )
    return (
        0.0,
        byts,
        Spec(
            "fp32",
            "fp32",
            "hbm",
            f"index translation over the top-k prefix (k={_n(k)} = min(kv/4, index_topk))",
            "FLOPs = 0 (integer index arithmetic)",
            summ(
                "Bytes",
                [
                    ("topk + lookup + out int32", f"4 x 3 x {_n(M)} x {_n(k)}"),
                    ("per-token metadata int32", f"4 x 3 x {_n(M)}"),
                ],
                byts,
            ),
        ),
    )


# --------------------------------------------------------------------------
# collectives -- interconnect-bound, NOT HBM
#
# Roofline = per-rank bytes / interconnect peak, flops ~ 0. The measured/roofline
# gap here captures latency, protocol overhead and stragglers, none of which are
# modelled -- that gap IS the comm signal.
#
# ATOM reaches three implementations (AITER 1-stage, AITER 2-stage, RCCL), and
# `allreduce_path` decides which from the kernel name or, failing that, AITER's
# size thresholds. Only 1-stage differs in bytes: it moves the whole message per
# link, where any bandwidth-optimal all-reduce moves 2*message/N. 2-stage and
# RCCL therefore share one floor -- not an approximation, but the consequence of
# both using every link. Bytes are counted in one direction against the
# per-direction xGMI peak, because a collective sends and receives at once.
#
# OPEN: two collectives of identical message size measure 118 ms and 59 ms in the
# same prefill, so they are not the same operation -- one is very likely a
# reduce-scatter moving message/N. Until that is read out of the ATOM source both
# are priced as a full-message all-reduce, and the reduce-scatter one trips the
# floor-above-measurement flag, which is the correct outcome for a model known to
# be wrong.
# --------------------------------------------------------------------------


# AITER dispatches an all-reduce three ways by message size, and the thresholds
# are hardcoded in its source. Getting this wrong is not a small error: the three
# paths differ by a factor of N in per-link bytes.
#
#   csrc/include/custom_all_reduce.cuh  CustomAllreduce::allreduce()
#   (cited by symbol; aiter line numbers drift across commits)
#       world_size == 2                                        -> 1-stage
#       full_nvlink and (ws<=4 and bytes<160KB)
#                     or (ws<=8 and bytes<80KB)                 -> 1-stage
#       otherwise                                               -> 2-stage
#   dist/device_communicators/custom_all_reduce.py  CustomAllreduce.should_custom_ar()
#       decode path: inp_size <= 8192*8192 = 64 MiB (cited by symbol; aiter
#       line numbers drift across commits)
#       above that, the custom kernel is skipped and NCCL runs instead
#
# Neither is a ring. `cross_device_reduce_2stage_naive` indexes peers directly --
# `ptrs[i] = _dp->ptrs[(rank + i) % ngpus]` -- so every rank talks to every other
# rank over its own link, in two rounds, rather than forwarding around a circle in
# 2(N-1) hops. On this part each GPU has 7 xGMI links and a TP group of 4 or 8
# sits inside one node, so full connectivity holds and a ring would waste it.
# 64 MiB, and it is a BUFFER CAPACITY, not a performance heuristic. The custom
# path works by having each rank read its peers' memory directly, which needs an
# IPC handle registered up front -- so the input is first copied into a
# pre-registered staging buffer of exactly this size. A message that does not fit
# cannot be staged, so `should_custom_ar` returns False and NCCL runs instead.
# Neither call site overrides it and there is no environment variable for it
# (QuickAllReduce has AITER_QUICK_REDUCE_MAX_SIZE_BYTES_MB; this does not), but it
# is a constructor default rather than a hardware constant -- so a deployment that
# raised it would move the boundary, and this stays configurable rather than
# hardcoded.
CUSTOM_AR_MAX_BYTES = 8192 * 1024 * 8


# The kernel name states the path outright, so a measured run never has to infer
# it: `aiter::cross_device_reduce_1stage` and `ncclDevKernel_Generic_1` both say
# what they are. Reading what ran beats deriving it from a threshold -- the
# threshold is a constructor default this tool cannot see, and today's traces
# already disagree with it in the direction that matters (prefill's 99 MB message
# exceeds the cap and does run NCCL, which the name confirms independently).
_PATH_TOKENS = (("nccl", "nccl"), ("2stage", "2stage"), ("1stage", "1stage"))


def allreduce_path(
    bytes_: float, world: int, kernel: str = "", max_bytes: float | None = None
) -> str:
    """Which all-reduce path this collective takes: observed if possible, else derived.

    The derivation mirrors AITER's own dispatch, thresholds included:
        custom_all_reduce.py should_custom_ar()  inp_size <= 8192*8192 (64 MiB) else NCCL
        custom_all_reduce.cuh CustomAllreduce::allreduce()  world==2, or <160 KB at
                                      TP<=4, or <80 KB at TP<=8 -> 1-stage; else 2-stage
                                      (cited by symbol; aiter line numbers drift)

    Stated, not modelled: those size thresholds sit behind a `full_nvlink_` gate.
    On a fully connected xGMI node it is true and the thresholds decide, which is
    this platform. On a partial mesh neither branch is taken and the call leaves
    the new-kernel path entirely -- so a DERIVED path is only trustworthy on a
    full mesh. A path read from the kernel name is unaffected, which is one more
    reason to prefer the observed one.

    It is only needed when predicting a concurrency nobody ran.
    """
    text = kernel.lower()
    for token, path in _PATH_TOKENS:
        if token in text:
            return path
    cap = CUSTOM_AR_MAX_BYTES if max_bytes is None else max_bytes
    if bytes_ > cap:
        return "nccl"
    if world == 2:
        return "1stage"
    if (world <= 4 and bytes_ < 160 * 1024) or (world <= 8 and bytes_ < 80 * 1024):
        return "1stage"
    return "2stage"


def _allreduce(cfg, b, tokens: float, where: str):
    """Per-link bytes for a TP all-reduce, branched on the path AITER will take.

    **1-stage** (`cross_device_reduce_1stage`; there is also a `..._naive` variant
    -- same bytes, and this run used the non-naive one, which stages the peer
    values through LDS with one warp per GPU instead of reducing them per thread):
    every rank walks the WHOLE message and reduces each element by reading it from
    all `ngpus` peer pointers, so each link carries the full message. It reads its
    own buffer through the same pointer array, but that read is local HBM and
    never touches the fabric, so per LINK it is still one message. Cheap in
    rounds, expensive in bytes -- which is why it is reserved for small messages,
    where the cost is the launch and the barrier rather than the transfer.

    **2-stage**: each rank reduces only `size/ngpus` and then gathers one slice
    from every peer. Two rounds, each moving `(N-1)/N` of the message across the
    `N-1` links a rank owns, so per link it is `2 * message / N` -- N times less
    than 1-stage at N=4.

    **NCCL** above 64 MiB: a different implementation, but the same bound. Any
    all-reduce must send `2*(N-1)/N * message` per rank -- reduce-scatter and
    all-gather each move `(N-1)/N` -- so once that traffic uses all `N-1` links it
    is `2 * message / N` per link whatever the algorithm. A single-channel ring
    would sit at `1.5 * message` because it uses one link; NCCL builds multiple
    channels over a full mesh, so it lands on the same bound as the direct form.
    The number is therefore a bandwidth-optimal floor rather than an assumption
    about which algorithm NCCL chose, and RCCL 2.27 does not name the algorithm in
    its kernel anyway (`ncclDevKernel_Generic_{1,2,4}` dispatch at runtime).

    Bytes are counted in ONE direction. xGMI links are 153.6 GB/s bidirectional,
    which is 76.8 GB/s each way simultaneously, and a rank sends and receives at
    once during a collective -- so one direction's bytes over the one-way rate is
    the right pairing. Counting send+receive against 76.8 would double the floor.
    """
    H = cfg["hidden_size"]
    message = tokens * H * DTYPE_BYTES["bf16"]
    n = max(int(b.tp), 2)
    path = allreduce_path(message, n, b.kernel, b.custom_ar_max_bytes)
    if path == "1stage":
        per_link = message
        how = "every rank reads the whole message from each peer"
        terms = [("whole message per link", f"{_n(tokens)} x {_n(H)} x 2")]
    else:
        per_link = 2.0 * message / n
        how = (
            "reduce-scatter + all-gather"
            if path == "2stage"
            else "NCCL; the bandwidth-optimal bound is the same for any all-reduce "
            "that uses every link"
        )
        terms = [
            (f"message ({where})", f"{_n(tokens)} x {_n(H)} x 2"),
            (f"x 2 stages / {n} ranks", f"2 / {n}"),
        ]
    knob = ""
    if path == "nccl":
        # The path here was decided by a software default, not by the hardware, and
        # a reader cannot see that from a gap. The roofline cannot say whether the
        # other path would be faster -- both share the same bandwidth-optimal floor,
        # so only a measurement can answer it -- but it can say that there is a knob
        # and name it. "This kernel is slow" and "this build never tried the other
        # implementation" are different findings with different fixes.
        cap = (
            CUSTOM_AR_MAX_BYTES
            if b.custom_ar_max_bytes is None
            else b.custom_ar_max_bytes
        )
        knob = (
            f" | CONFIG: fell back to NCCL because the message ({message / 1e6:,.1f} MB) "
            f"exceeds AITER's {cap / 1024 / 1024:,.0f} MiB IPC staging buffer -- a "
            f"CustomAllreduce constructor default, not a hardware limit. No call site "
            f"overrides it and no env var exposes it. Whether the custom path would be "
            f"faster is not something a roofline can answer: both have the same "
            f"bandwidth-optimal floor, so it needs an A/B run."
        )
    return (
        0.0,
        per_link,
        Spec(
            "bf16",
            "bf16",
            "interconnect",
            f"{path} over {n} ranks, {where}: {how}"
            + (" [path read from the kernel name]" if b.kernel else " [path derived]")
            + knob,
            "FLOPs ~ 0  (the reduction adds are negligible against the transfer)",
            summ("Bytes", terms, per_link),
        ),
    )


def comm_allreduce_attn(cfg, b, kind):
    """The attention-side all-reduce (wo_b's RowParallelLinear).

    Inside the attention region, so it reduces the per-rank token count: with
    dp-attn on, attention is DP-partitioned and each rank holds batch/dp.
    """
    return _allreduce(cfg, b, attn_M(b), "attention region")


def comm_allreduce_moe(cfg, b, kind):
    """The MoE-side all-reduce (combine_outputs).

    This one runs AFTER the DP gather, so it reduces the GATHERED tokens of all DP
    ranks -- batch x dp, not batch. Sharing the attention formula made the two
    identical, which is true at dp=1 and wrong by a factor of dp the moment
    --enable-dp-attention is on. The catalog runs DPA from concurrency 64 to 2048,
    so that is not a hypothetical operating point.
    """
    return _allreduce(cfg, b, moe_B(b), "after the DP gather")


# --------------------------------------------------------------------------
# marker path -> formula
#
# Keyed on the ATOM marker path first and a kernel substring second. Marker paths
# survive kernel renames across builds; kernel names do not, so the less they are
# relied on the better -- but a marker names a MODULE, and a module emits several
# kernels, so the second key is what stops an auxiliary kernel from inheriting the
# module's headline formula. That inheritance is what priced an update-states
# kernel as a full core-attention scan at 7.6x its measured time.
#
# Deliberately absent: `triton_poi_fused_as_strided_clone*` layout copies and the
# smallest indexer glue kernels. Their byte counts depend on transient tensor
# layouts the config does not describe, and a guessed formula would be worse than
# an honest blank -- these stay in the gap.
# --------------------------------------------------------------------------

Formula = Callable[[dict, Bench, str], tuple[float, float, Spec]]

# One map per ARCHITECTURE, keyed exactly as ATOM's own
# `support_model_arch_dict` is, so there is no second naming convention to learn.
#
# The formulas above are shared -- a GEMM is a GEMM -- but the marker PATHS are
# not. `@mark_trace` lives in atom/model_ops (linear.py, layernorm.py, moe.py...),
# yet the name it emits comes from the caller's `prefix=`, so `attn.wo_b` is
# DeepSeek-V4's word for an output projection and Llama will call the same
# operation something else. A single flat map would therefore be a pile of one
# model's vocabulary pretending to be universal.
#
# Adding a model is a new map, not new formulas -- and an architecture with no map
# prices nothing rather than matching another model's paths by accident. That
# matters: `attn.compressor` currently means CSA compression, and the day another
# model uses the same word, a flat map would hand it a CSA formula in silence.
# Three catch-all needles did exactly that inside this one model already.
DSV4_MAP: list[tuple[str, str | None, Formula]] = [
    # (marker path suffix, kernel substring or None, formula)
    ("ffn.experts.fused_moe", "mfma_moe1", moe_gate_up),
    ("ffn.experts.fused_moe", "mfma_moe2", moe_down),
    # Before the generic "moe_sort": that needle is a substring of BOTH
    # opus_moe_sorting_entry and fused_mx_quant_moe_sort_kernel, and they are
    # different operators -- one permutes indices, one moves the tokens.
    ("ffn.experts.fused_moe", "mx_quant_moe_sort", moe_quant_sort),
    ("ffn.experts.fused_moe", "moe_sort", moe_dispatch),
    ("ffn.experts.fused_moe", "topk", moe_topk),
    # No catch-all. This marker emits the gating GEMM, a top-k selection and a
    # layout clone; the clone was being priced as a full [B,H]x[H,E] GEMM (5.2 ms
    # of decode) purely because it sat under the same annotation.
    # torch.compile drops a pointwise copy inside four markers. Needled on the
    # exact kernel suffix so the two variants cannot swap: "as_strided_clone_1"
    # and "as_strided_clone_copy__0" share a prefix but not a suffix.
    ("ffn.gate", "as_strided_clone_1", layout_copy_narrow),
    ("ffn.shared_experts.gate_up_proj", "as_strided_clone_1", layout_copy_narrow),
    ("mhc_fused_post_pre", "as_strided_clone_copy", layout_copy_wide),
    ("mhc_post_pre", "as_strided_clone_copy", layout_copy_wide),
    ("ffn.gate", "topk", moe_topk),
    ("ffn.gate", "gemm", moe_router),
    ("ffn.gate", "cijk", moe_router),
    ("ffn.shared_experts.gate_up_proj", "quant", quant_moe_hidden),
    ("ffn.shared_experts.gate_up_proj", "gemm", shared_expert_gate_up),
    ("ffn.shared_experts.w2", "quant", quant_moe_intermediate),
    ("ffn.shared_experts.w2", "gemm", shared_expert_down),
    # Order matters: the fused local add must be caught before the "all_reduce"
    # substring sends it to a collective model.
    ("ffn.combine_outputs", "triton_poi_fused_add", residual_add_moe),
    ("ffn.combine_outputs", "nccl", comm_allreduce_moe),
    ("ffn.combine_outputs", "reduce", comm_allreduce_moe),
    ("ffn.combine_outputs", "all_reduce", comm_allreduce_moe),
    ("ffn.combine_outputs", "mhc_post", mhc_post),
    # Two builds, two names for the same GEMM: decode gets aiter's
    # `_batched_gemm_bf16_kernel`, prefill gets rocBLAS/Tensile's
    # `Cijk_Alik_Bljk_...`, which contains no "gemm" at all. Needle on both rather
    # than on a word that happens to be in one of them.
    ("attn.wo_a", "gemm", o_proj_a),
    ("attn.wo_a", "cijk", o_proj_a),
    ("attn.wo_b", "quant", quant_heads),
    ("attn.wo_b", "triton_poi_fused_add", residual_add),
    ("attn.wo_b", "nccl", comm_allreduce_attn),
    ("attn.wo_b", "reduce", comm_allreduce_attn),
    ("attn.wqkv_a", "rmsnorm", norm_quant),
    ("attn.wqkv_a", "quant", quant_hidden),
    ("attn.sparse_attn_decode", "stage2", attn_merge),
    ("attn.sparse_attn_decode", "mla_", attn_logits),
    # Prefill's paged-attention kernel fuses QK and AV in one launch, like the
    # decode compress kernel, so it takes the 4-pass form. The sliding-window write
    # under the same marker is a different operator and must not inherit it.
    ("attn.sparse_attn_prefill", "swa_write", swa_write),
    ("attn.sparse_attn_prefill", "pa_prefill", csa_core_attn),
    ("attn.csa_translate_pack", "pa_prefill", csa_core_attn),
    ("attn.csa_translate_pack", "csa_translate_pack", csa_translate_pack),
    (
        "attn.indexer.scale_indexer_weights",
        "scale_indexer_weights",
        scale_indexer_weights,
    ),
    (
        "attn.indexer.compressor.update_compressor_states",
        "compressor",
        kv_compress_index,
    ),
    ("attn.indexer.compressor.fused_compress_attn", "compress", compressor_pool_index),
    ("attn.compressor.update_compressor_states", "compressor", kv_compress),
    # One marker, four kernels. Each gets its own model; nothing falls through to a
    # catch-all, because a catch-all here is how an update-states kernel ended up
    # priced as a full core-attention scan (7.6x its measured time).
    ("attn.compressor.fused_compress_attn", "update_compressor_states", kv_compress),
    ("attn.compressor.fused_compress_attn", "norm_rope_scatter", compressor_epilogue),
    ("attn.compressor.fused_compress_attn", "compress_forward", compressor_pool),
    ("attn.compressor.fused_compress_attn", "fused_compress_attn", compressor_pool),
    ("attn.indexer", "topk", indexer_topk),
    ("attn.indexer", "logits", csa_indexer),
    ("attn.indexer", "rope", indexer_rope),
    ("attn.q_norm", "rmsnorm", q_norm_quant),
    # Likewise: prefill puts a paged-attention kernel and a GEMM under the qk-norm
    # marker, and a Tensile GEMM under inverse_rope. Decode does not, which is why
    # a catch-all survived here long after the others were removed -- it only ever
    # misfired on the phase with more kernels per module.
    ("attn.qk_norm_rope_maybe_quant", "qk_norm_rope", qk_norm_rope),
    ("attn.inverse_rope", "inverse_rope", rope),
    ("mhc_fused_post_pre", "rmsnorm", mhc_pre_fuse),
    # The marker says fused; the kernel decides. aiter un-fuses at M >= 1024 on
    # gfx950 (`ops/mhc.py: fused_m_upper_bound`) and issues mhc_post + mhc_pre as
    # separate kernels under the same ATOM record_function. A bare "sqrsum" needle
    # therefore charged the fused formula -- post AND pre -- to the un-fused pre
    # GEMM, while mhc_post_kernel was charged for the post half beside it. The
    # residual was paid for twice. Match the kernel, and only the kernel.
    ("mhc_fused_post_pre", "fused_post_pre", mhc_fused_post_pre),
    ("mhc_fused_post_pre", "pre_gemm_sqrsum", mhc_pre_gemm),
    ("mhc_fused_post_pre", "mhc_post", mhc_post),
    # `Cijk_...` is rocBLAS/Tensile running the same hc-fn GEMM -- that one is a
    # real backend swap. No needle for `a16w16 / mono_tile`: neither appears in
    # aiter's mHC path, so whatever that kernel is, it is not this GEMM. Left
    # unpriced, which keeps it visible; a guess would make it wrong and invisible.
    ("mhc_fused_post_pre", "cijk", mhc_pre_gemm),
    ("mhc_post_pre", "cijk", mhc_pre_gemm),
    ("mhc_post_pre", "rmsnorm", mhc_pre_fuse),
    ("mhc_post_pre", "sqrsum", mhc_pre_gemm),
    ("mhc_post_pre", "mhc_post", mhc_post),
]


MARKER_MAPS: dict[str, list[tuple[str, str | None, Formula]]] = {
    "DeepseekV4ForCausalLM": DSV4_MAP,
}

# Kept so existing callers and checks keep working while only one model is mapped.
MARKER_MAP = DSV4_MAP


def lookup(path: str, kernel: str, arch: str | None = None) -> Formula | None:
    """Find the formula for one marker path + kernel under one architecture.

    An unmapped architecture matches nothing. That is deliberate: pricing zero
    operators is a visible, reportable state, while matching another model's marker
    vocabulary is a silent wrong answer.
    """
    table = MARKER_MAPS.get(arch, DSV4_MAP if arch is None else [])
    kernel_l = kernel.lower()
    for suffix, needle, fn in table:
        if not path.endswith(suffix) and f".{suffix}" not in f".{path}":
            continue
        if needle is None or needle in kernel_l:
            return fn
    return None
