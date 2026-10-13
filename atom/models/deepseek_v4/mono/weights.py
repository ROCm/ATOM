# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""One DeepSeek-V4 layer's TP shard: its shape (``V4Config``), its weights
(``LayerWeights``) and their loading from the HF checkpoint.

Formats are kept as ATOM runs them: FP8 128x128 (shared expert included), MXFP4 routed
experts, BF16 compressor projections and FP32 hyper-connection mixers.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, replace

import torch
from safetensors import safe_open

from atom.models.deepseek_v4.mono.config import (
    COMPRESS_CSA,
    COMPRESS_ROPE_THETA,
    COMPRESS_SWA,
    EPS,
    HC_EPS,
    HC_MULT,
    HC_SINKHORN_ITERS,
    HEAD_DIM,
    HIDDEN,
    INDEX_HEAD_DIM,
    INDEX_HEADS,
    INDEX_TOPK,
    INTER,
    KEY_BLOCK,
    N_EXPERTS,
    O_GROUPS,
    O_LORA,
    Q_LORA,
    ROPE_DIM,
    ROUTE_SCALE,
    SWIGLU_LIMIT,
    TOP_K,
    WINDOW,
)


@dataclass
class V4Config:
    """One rank's shard of a DeepSeek-V4 layer. Defaults are V4-Pro at TP8."""

    heads: int = 16  # local; 128 global / 8 ranks
    hidden: int = HIDDEN
    q_lora: int = Q_LORA
    head_dim: int = HEAD_DIM  # K and V share this; rope lives in its tail
    rope_dim: int = ROPE_DIM
    o_groups: int = O_GROUPS  # local; 16 global / 8 ranks
    o_lora: int = O_LORA
    n_experts: int = N_EXPERTS
    top_k: int = TOP_K
    inter: int = INTER  # local; 3072 global / 8 ranks
    window: int = WINDOW
    route_scale: float = ROUTE_SCALE
    swiglu_limit: float = SWIGLU_LIMIT
    eps: float = EPS
    rope_theta: float = 1.0e4
    compress_ratio: int = COMPRESS_SWA  # 0 SWA, 4 CSA, 128 HCA
    index_heads: int = (
        INDEX_HEADS  # all of them on every rank (the indexer is replicated)
    )
    index_head_dim: int = INDEX_HEAD_DIM
    index_topk: int = INDEX_TOPK
    compress_rope_theta: float = COMPRESS_ROPE_THETA
    max_seq: int = 4096  # sizes the compressed half of the KV cache
    hc_mult: int = HC_MULT  # 1 = plain residual
    hc_sinkhorn_iters: int = HC_SINKHORN_ITERS
    hc_eps: float = HC_EPS
    # Hadamard-rotate indexer q/k as the model does; ATOM's cache is unrotated, so serving it needs False
    indexer_hadamard: bool = True

    @property
    def hc_mix(self) -> int:
        return (2 + self.hc_mult) * self.hc_mult

    def for_layer(self, layer_id: int, ratios=None) -> V4Config:
        """This shard with layer ``layer_id``'s attention variant (``ratios`` defaults to V4-Pro's)."""
        from atom.models.deepseek_v4.mono.config import compress_ratios

        ratios = compress_ratios() if ratios is None else ratios
        if not 0 <= layer_id < len(ratios):
            raise ValueError(f"layer {layer_id} is outside a schedule of {len(ratios)}")
        return replace(self, compress_ratio=ratios[layer_id])

    @property
    def rope_base(self) -> float:
        """Per layer, not per consumer: a compressing layer rotates everything on ``compress_rope_theta``."""
        return self.compress_rope_theta if self.compress_ratio else self.rope_theta

    @property
    def overlap(self) -> bool:
        """CSA (ratio 4) pools 2 * ratio tokens per entry at a stride of ratio."""
        return self.compress_ratio == COMPRESS_CSA

    @property
    def indexed(self) -> bool:
        """CSA selects compressed entries with the lightning indexer; HCA takes them all."""
        return self.compress_ratio == COMPRESS_CSA

    @property
    def n_index(self) -> int:
        """Compressed entries the attention can reach in one step."""
        if not self.compress_ratio:
            return 0
        return (
            min(self.index_topk, self.n_compressed)
            if self.indexed
            else self.n_compressed
        )

    @property
    def c_coff(self) -> int:
        """Compressor channel multiplier: overlapping windows carry previous and current halves."""
        return 2 if self.overlap else 1

    @property
    def c_rows(self) -> int:
        """Rows of compressor state: two windows' worth when overlapping."""
        return self.compress_ratio * self.c_coff

    @property
    def n_compressed(self) -> int:
        """Compressed cache slots, stored after the ``window`` rows of the same cache."""
        return 0 if self.compress_ratio == 0 else self.max_seq // self.compress_ratio

    @property
    def cache_rows(self) -> int:
        return self.window + self.n_compressed

    @property
    def n_keys(self) -> int:
        """Length of the attention's index list (window + compressed), padded to the key tile."""
        rows = self.window + self.n_index
        return (rows + KEY_BLOCK - 1) // KEY_BLOCK * KEY_BLOCK

    @property
    def hc_rows(self) -> int:
        """hc_mix padded to the MFMA row group (24 -> 32 at hc_mult 4)."""
        return (self.hc_mix + 15) // 16 * 16

    @property
    def nope_dim(self) -> int:
        return self.head_dim - self.rope_dim

    @property
    def group_dim(self) -> int:
        """Slice of the concatenated heads that one ``o_a`` group consumes."""
        return self.heads * self.head_dim // self.o_groups

    @property
    def shared_expert(self) -> int:
        """Last in an FP8 bank; an MXFP4 bank keeps it as FP8 ``w_sug`` / ``w_sdn`` beside it."""
        return self.n_experts

    @property
    def softmax_scale(self) -> float:
        return self.head_dim**-0.5

    def validate(self) -> None:
        assert self.compress_ratio == 0 or self.window % self.compress_ratio == 0
        assert self.nope_dim % 64 == 0, "act_quant blocks the nope part by 64"
        assert self.heads % self.o_groups == 0
        assert self.head_dim % 2 == 0 and self.rope_dim % 2 == 0


def fp8_mats(cfg: V4Config):
    """(rows, K, BK) of every FP8 attention matrix in one rank's shard."""
    return {
        # q_a and kv only: the compressors' rows are BF16 (w_qkv_c), as ATOM applies them
        "qkv_a": (cfg.q_lora + cfg.head_dim, cfg.hidden, 128),
        "q_b": (cfg.heads * cfg.head_dim, cfg.q_lora, 128),
        **(
            {"i_q_b": (cfg.index_heads * cfg.index_head_dim, cfg.q_lora, 128)}
            if cfg.indexed
            else {}
        ),
        "o_a": (cfg.o_groups * cfg.o_lora, cfg.group_dim, 128),
        "o_b": (cfg.hidden, cfg.o_groups * cfg.o_lora, 128),
    }


@dataclass
class LayerWeights:
    cfg: V4Config
    t: dict  # name -> tensor


def config_for_layer(path: str, layer: int, tp: int) -> V4Config:
    """This rank's ``V4Config`` for ``layer``, from the checkpoint's own config.json."""
    with open(os.path.join(path, "config.json")) as f:
        c = json.load(f)
    cfg = V4Config(
        heads=c["num_attention_heads"] // tp,
        hidden=c["hidden_size"],
        q_lora=c["q_lora_rank"],
        head_dim=c["head_dim"],
        rope_dim=c["qk_rope_head_dim"],
        o_groups=c["o_groups"] // tp,
        o_lora=c["o_lora_rank"],
        n_experts=c["n_routed_experts"],
        top_k=c["num_experts_per_tok"],
        inter=c["moe_intermediate_size"] // tp,
        window=c["sliding_window"],
        route_scale=c["routed_scaling_factor"],
        swiglu_limit=c["swiglu_limit"],
        eps=c["rms_norm_eps"],
        rope_theta=c["rope_theta"],
        index_heads=c["index_n_heads"],  # replicated, not sharded
        index_head_dim=c["index_head_dim"],
        index_topk=c["index_topk"],
        compress_rope_theta=c["compress_rope_theta"],
        hc_mult=c["hc_mult"],
        hc_sinkhorn_iters=c["hc_sinkhorn_iters"],
        hc_eps=c["hc_eps"],
    )
    return replace(cfg, compress_ratio=c["compress_ratios"][layer])


class Checkpoint:
    """Lazy, sliced reads of a sharded safetensors checkpoint."""

    def __init__(self, path: str):
        self.path = path
        with open(os.path.join(path, "model.safetensors.index.json")) as f:
            self.index = json.load(f)["weight_map"]
        self._files = {}

    def has(self, name: str) -> bool:
        return name in self.index

    def get(self, name: str, rows=None, cols=None) -> torch.Tensor:
        fname = self.index[name]
        if fname not in self._files:
            self._files[fname] = safe_open(os.path.join(self.path, fname), "pt")
        sl = self._files[fname].get_slice(name)
        r = slice(*rows) if rows else slice(None)
        if len(sl.get_shape()) == 1:
            return sl[r]
        return sl[r, slice(*cols) if cols else slice(None)]


def e8m0_float(s: torch.Tensor) -> torch.Tensor:
    """E8M0 scale bytes -> their float32 values, 2**(byte - 127)."""
    return torch.exp2(s.view(torch.uint8).float() - 127.0)


def layer_prefix(path: str, layer: int) -> str:
    """``layers.N.`` for the main stack, ``mtp.M.`` for layer num_hidden_layers + M (ATOM's MTP layer_id)."""
    with open(os.path.join(path, "config.json")) as f:
        n = json.load(f)["num_hidden_layers"]
    return f"layers.{layer}." if layer < n else f"mtp.{layer - n}."


def load_expert(ck: Checkpoint, layer: int, e: int, rank: int, tp: int, device="cuda"):
    """Routed expert ``e``'s MXFP4 [gate; up] and down matrices and E8M0 scales (uint8, row-major)."""
    cfg = config_for_layer(ck.path, layer, tp)
    inter = cfg.inter

    def shard(n):
        return (n * rank, n * (rank + 1))

    x = layer_prefix(ck.path, layer) + f"ffn.experts.{e}."
    rows = shard(inter)
    ug = [ck.get(x + w + ".weight", rows).view(torch.uint8) for w in ("w1", "w3")]
    ugs = [ck.get(x + w + ".scale", rows).view(torch.uint8) for w in ("w1", "w3")]
    dn = ck.get(x + "w2.weight", cols=shard(inter // 2)).view(torch.uint8)
    dns = ck.get(x + "w2.scale", cols=shard(inter // 32)).view(torch.uint8)
    return tuple(
        v.to(device).contiguous() for v in (torch.cat(ug), torch.cat(ugs), dn, dns)
    )


def load_layer(
    ck: Checkpoint,
    layer: int,
    rank: int,
    tp: int,
    device="cuda",
    experts: bool = True,
) -> LayerWeights:
    """``layer``'s weights for TP rank ``rank`` of ``tp``, in ``make_weights``' layout.

    ``experts=False`` skips the routed bank, for callers packing their own.
    """
    cfg = config_for_layer(ck.path, layer, tp)
    cfg.validate()
    p = layer_prefix(ck.path, layer)
    t = {}

    def fp8(name, rows=None, cols=None):
        """An FP8 matrix and its block scales, sliced to this rank's shard."""
        q = ck.get(name + ".weight", rows, cols).view(torch.float8_e4m3fn)
        sr = (rows[0] // 128, rows[1] // 128) if rows else None
        sc = (cols[0] // 128, cols[1] // 128) if cols else None
        return q.to(device), e8m0_float(ck.get(name + ".scale", sr, sc)).to(device)

    def shard(n):
        return (n * rank, n * (rank + 1))

    t["g_in"] = ck.get(p + "attn_norm.weight").to(device)
    t["g_q"] = ck.get(p + "attn.q_norm.weight").to(device)
    t["g_kv"] = ck.get(p + "attn.kv_norm.weight").to(device)
    t["g_post"] = ck.get(p + "ffn_norm.weight").to(device)
    t["attn_sink"] = ck.get(p + "attn.attn_sink", shard(cfg.heads)).float().to(device)

    # qkv_a, in qkv_a_split's order: wq_a | wkv | compressor wkv | wgate | indexer's pair
    parts = [fp8(p + "attn.wq_a"), fp8(p + "attn.wkv")]
    bf16_rows = []
    if cfg.compress_ratio:
        bf16_rows += [
            p + "attn.compressor.wkv.weight",
            p + "attn.compressor.wgate.weight",
        ]
        if cfg.indexed:
            bf16_rows += [
                p + "attn.indexer.compressor.wkv.weight",
                p + "attn.indexer.compressor.wgate.weight",
            ]
    t["w_qkv_a"] = torch.cat([q for q, _ in parts])
    t["s_qkv_a"] = torch.cat([s for _, s in parts])
    if bf16_rows:
        t["w_qkv_c"] = torch.cat(
            [ck.get(name).to(device).to(torch.bfloat16) for name in bf16_rows]
        )

    t["w_q_b"], t["s_q_b"] = fp8(p + "attn.wq_b", rows=shard(cfg.heads * cfg.head_dim))
    t["w_o_a"], t["s_o_a"] = fp8(p + "attn.wo_a", rows=shard(cfg.o_groups * cfg.o_lora))
    t["w_o_b"], t["s_o_b"] = fp8(p + "attn.wo_b", cols=shard(cfg.o_groups * cfg.o_lora))

    if cfg.compress_ratio:
        t["ape"] = ck.get(p + "attn.compressor.ape").float().to(device)
        t["g_ckv"] = ck.get(p + "attn.compressor.norm.weight").to(device)
        if cfg.indexed:
            t["w_i_q_b"], t["s_i_q_b"] = fp8(p + "attn.indexer.wq_b")
            t["i_ape"] = ck.get(p + "attn.indexer.compressor.ape").float().to(device)
            t["g_ickv"] = ck.get(p + "attn.indexer.compressor.norm.weight").to(device)
            t["i_w"] = ck.get(p + "attn.indexer.weights_proj.weight").to(device)

    if cfg.hc_mult > 1:
        for side in ("attn", "ffn"):
            fn = torch.zeros(
                cfg.hc_rows,
                cfg.hc_mult * cfg.hidden,
                dtype=torch.float32,
                device=device,
            )
            fn[: cfg.hc_mix] = ck.get(p + f"hc_{side}_fn").float().to(device)
            t[f"hc_{side}_fn"] = fn  # fp32: packing splits it into bf16 hi / lo
            t[f"hc_{side}_base"] = ck.get(p + f"hc_{side}_base").float().to(device)
            t[f"hc_{side}_scale"] = ck.get(p + f"hc_{side}_scale").float().to(device)

    t["w_r"] = ck.get(p + "ffn.gate.weight").to(device)
    bias = p + "ffn.gate.bias"
    # explicit dtypes: ATOM sets torch's default dtype to bf16
    t["bias"] = (
        ck.get(bias).float().to(device)
        if ck.has(bias)
        else torch.zeros(cfg.n_experts, dtype=torch.float32, device=device)
    )
    if ck.has(p + "ffn.gate.tid2eid"):
        t["tid2eid"] = ck.get(p + "ffn.gate.tid2eid").to(torch.int32).to(device)

    inter = cfg.inter
    rows = shard(inter)
    if experts:
        bank = [
            load_expert(ck, layer, e, rank, tp, device) for e in range(cfg.n_experts)
        ]
        t["w_ug"], t["s_ug"], t["w_dn"], t["s_dn"] = (
            torch.stack(v) for v in zip(*bank)
        )
    sh = p + "ffn.shared_experts."
    (gq, gs), (uq, us) = fp8(sh + "w1", rows=rows), fp8(sh + "w3", rows=rows)
    t["w_sug"], t["s_sug"] = torch.cat([gq, uq]), torch.cat([gs, us])
    # contiguous: a column slice, read as dense row-major
    t["w_sdn"], t["s_sdn"] = (v.contiguous() for v in fp8(sh + "w2", cols=rows))

    for name, (r, k, _bk) in fp8_mats(cfg).items():
        assert t[f"w_{name}"].shape == (
            r,
            k,
        ), f"{name}: {tuple(t[f'w_{name}'].shape)} != {(r, k)}"
    return LayerWeights(cfg, t)
