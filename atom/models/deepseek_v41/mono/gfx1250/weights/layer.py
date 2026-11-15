# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""The step kernel's per-layer weights (``buffers``: WEIGHT entries of a layer-table row, ``kernels/step.py`` LKEYS),
from a loaded V4.1 ``Block`` (target ``layers.i`` or draft ``mtp.j``), in the layout the kernel reads:

- attention (TP4 rank shard): ``wqkv`` [1792][5120] FP8 = ``wqkv_a`` (q_lora 1280 rows, then kv 512), ``wqb`` [8192][1280]
  = ``wq_b``, ``woa`` [2048][4096] = ``wo_a`` (2 groups x o_rank 1024), ``wob`` [5120][2048] = ``wo_b``, each in checkpoint
  row order (K contiguous) with its 32 x 32 block E8M0 scales ``*_s`` [rows/32][K/32]; ``qnw`` / ``kvnw`` the q / kv norm
  weights in f32; ``sink`` the 16 heads' attention sinks (f32);
- FFN front: ``fnh`` / ``fnl`` the mHC ``hc_ffn_fn`` (f32 [24][20480]) as bf16 hi / lo halves (fn = hi + lo), zero padded to
  128 x 5120; ``hsc`` / ``hbs`` its scale [3] / base [24] (f32); ``nw`` the ffn_norm weight (bf16); ``gate`` the router
  weight [384][5120] (bf16), ``gbias`` its score-correction bias (f32);
- ``a*``: the attention front that ENDS layer i, i.e. layer i + 1's ``hc_attn_*`` and ``attn_norm`` (the last layer of a
  segment: the model's final collapse + norm, only ``anw`` read: ``model.norm`` / the draft's ``norm``).

Kept apart from ``runner.py`` so a CPU test can build a row from a module stub."""

import torch

from atom.models.deepseek_v41.mono.gfx1250.weights.planes import (
    checkpoint_weight,
)
from atom.mono.runtime.consensus import MonoUnsupported

H = 5120
FN_ROWS = 128  # fnh / fnl: 32 x 4 rows of H (the hc fn's 24 x 4 H, zero padded)
ATTN_SHAPES = {  # name -> (rows, K): the rank shard the kernel reads
    "wqkv": (1792, 5120),
    "wqb": (8192, 1280),
    "woa": (2048, 4096),
    "wob": (5120, 2048),
}
QKV_ROW = ATTN_SHAPES["wqkv"][0]  # q_lora 1280 + kv 512
ATTN_SRC = {"wqkv": "wqkv_a", "wqb": "wq_b", "woa": "wo_a", "wob": "wo_b"}


def hi_lo(fn: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """f32 -> (bf16 hi, bf16 lo), fn ~= hi + lo, flat and zero padded to FN_ROWS x H."""
    f = fn.detach().float().reshape(-1)
    if f.numel() > FN_ROWS * H:
        raise MonoUnsupported(f"hc fn {tuple(fn.shape)} > {FN_ROWS} x {H}")
    hi = torch.zeros(FN_ROWS * H, dtype=torch.bfloat16, device=f.device)
    lo = torch.zeros_like(hi)
    hi[: f.numel()] = f.to(torch.bfloat16)
    lo[: f.numel()] = (f - hi[: f.numel()].float()).to(torch.bfloat16)
    return hi, lo


def _fp8_rows(linear, name):
    rows, k = ATTN_SHAPES[name]
    w = checkpoint_weight(linear)
    s = linear.weight_scale.data.view(torch.uint8)
    if tuple(w.shape) != (rows, k) or tuple(s.shape) != (rows // 32, k // 32):
        raise MonoUnsupported(
            f"{name}: weight {tuple(w.shape)} scale {tuple(s.shape)}, expected ({rows}, {k}) / ({rows // 32}, {k // 32})"
        )
    return w.contiguous(), s.contiguous()


def attention(block) -> dict[str, torch.Tensor]:
    attn = block.attn
    out = {}
    for name, src in ATTN_SRC.items():
        out[name], out[name + "_s"] = _fp8_rows(getattr(attn, src), name)
    # qnw padded to the whole qkv row (1792): attention' C epilogue loads qnw[col] for every column, the kv ones too,
    # before it selects (a page past 1280 floats may be unmapped -- the 11-07 memory fault)
    qnw = attn.q_norm.weight.data.float()
    out["qnw"] = torch.zeros(QKV_ROW, dtype=torch.float32, device=qnw.device)
    out["qnw"][: qnw.numel()].copy_(qnw)
    out["kvnw"] = attn.kv_norm.weight.data.float().contiguous()
    out["sink"] = attn.attn_sink.data.float().contiguous()
    return out


def ffn_front(block) -> dict[str, torch.Tensor]:
    fnh, fnl = hi_lo(block.hc_ffn_fn)
    gate = block.ffn.gate
    return {
        "fnh": fnh,
        "fnl": fnl,
        "hsc": block.hc_ffn_scale.data.float().contiguous(),
        "hbs": block.hc_ffn_base.data.float().contiguous(),
        "nw": block.ffn_norm.weight.data.to(torch.bfloat16).contiguous(),
        "gate": gate.weight.data.to(torch.bfloat16).contiguous(),
        "gbias": gate.e_score_correction_bias.data.float().contiguous(),
    }


def attention_front(next_block=None, final_norm=None) -> dict[str, torch.Tensor]:
    """The a* of the layer before ``next_block``; with ``final_norm`` (the segment's last layer): the final collapse, of
    which the kernel reads only the norm weight -- the mHC parts are given as zeros of the right size.
    """
    if next_block is not None:
        afnh, afnl = hi_lo(next_block.hc_attn_fn)
        return {
            "afnh": afnh,
            "afnl": afnl,
            "ahsc": next_block.hc_attn_scale.data.float().contiguous(),
            "ahbs": next_block.hc_attn_base.data.float().contiguous(),
            "anw": next_block.attn_norm.weight.data.to(torch.bfloat16).contiguous(),
        }
    dev = final_norm.weight.device
    z = torch.zeros(FN_ROWS * H, dtype=torch.bfloat16, device=dev)
    return {
        "afnh": z,
        "afnl": z,
        "ahsc": torch.zeros(3, dtype=torch.float32, device=dev),
        "ahbs": torch.zeros(24, dtype=torch.float32, device=dev),
        "anw": final_norm.weight.data.to(torch.bfloat16).contiguous(),
    }


def layer_row(block, next_block=None, final_norm=None) -> dict[str, torch.Tensor]:
    """A layer-table row's weight tensors (the planes, rope tables, KV / indexer / compressor pointers are bound apart)."""
    return {
        **attention(block),
        **ffn_front(block),
        **attention_front(next_block, final_norm),
    }
