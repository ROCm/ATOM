# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""What the Kimi-K3 DSpark drafter's MLA layers run on one gfx1250 GPU, per
row count: the kernel the production dispatch picks for fused_qkv_a_proj
(FlyDSL mxfp8_128, its split-K), for q_b_proj (fp32 activation scales: the
gluon blockscale GEMM, its block / split-K config, and the order its split-K
reduce sums the partials in) and the instructions the gluon GEMM's epilogue
multiplies its WMMA results with; the rotary embedding's cos / sin cache; the
DCP decode's head padding.

    python tools/kimi_k3_mono_mla_draft_probe.py
    env: ROWS (7,14,21,28,32,35,42,49,56,63,64), HEADS (64,96)
"""

import glob
import json
import os
import re

import torch
from aiter import dtypes
from aiter.ops.shuffle import shuffle_weight

ROWS = [
    int(r) for r in os.environ.get("ROWS", "7,14,21,28,32,35,42,49,56,63,64").split(",")
]
HEADS = [int(h) for h in os.environ.get("HEADS", "64,96").split(",")]
dev = torch.device("cuda", torch.cuda.current_device())


def qkv_a_configs():
    from aiter.ops.gemm_op_a8w8 import get_mxscale_bpreshuffle_config

    return {
        m: (get_mxscale_bpreshuffle_config(m, 2176, 7168, "128x128") or {}).get(
            "kernelName"
        )
        for m in ROWS
    }


def q_b_config(m, heads):
    from aiter.ops.triton._triton_kernels.gemm.basic.gemm_a8w8_blockscale import (
        _get_config,
    )
    from aiter.ops.triton.utils.gemm_config_utils import compute_splitk_params

    cfg, _ = _get_config(m, heads * 192, 1536, True, backend="gluon")
    compute_splitk_params(cfg, 1536)
    return {
        k: cfg.get(k)
        for k in (
            "BLOCK_SIZE_M",
            "BLOCK_SIZE_N",
            "BLOCK_SIZE_K",
            "NUM_KSPLIT",
            "SPLITK_BLOCK_SIZE",
            "kernel_type",
            "NUM_BUFFERS",
        )
    }


def q_b_operands(m, heads):
    n, k = heads * 192, 1536
    x = (torch.randn(m, k, device=dev) * 2).to(dtypes.fp8)
    # contiguous [m, k / 128] whose bytes are [k / 128][m], as
    # fused_qk_rmsnorm_group_quant(transpose_scale=True) writes it
    xs = (torch.rand(k // 128, m, device=dev) + 0.5).contiguous().view(m, k // 128)
    w = (torch.randn(n, k, device=dev) * 2).to(dtypes.fp8)
    codes = torch.randint(118, 124, (n // 128, k // 128), device=dev, dtype=torch.uint8)
    ws = codes.view(dtypes.fp8_e8m0)
    return x, xs, shuffle_weight(w, layout=(16, 16)).contiguous(), ws


def reduce_order(m, heads):
    """Production's reduced q against the split partials summed in a few
    orders (each add rounded to f32, as torch does elementwise)."""
    from aiter import gemm_a8w8_blockscale_bpreshuffle
    from aiter.ops.triton.gemm.basic.gemm_a8w8_blockscale import (
        gemm_a8w8_blockscale_preshuffle,
    )

    x, xs, w, ws = q_b_operands(m, heads)
    y = gemm_a8w8_blockscale_bpreshuffle(x, w, xs, ws, torch.bfloat16)
    n, k = w.shape
    parts = gemm_a8w8_blockscale_preshuffle(
        x,
        w.reshape(n // 16, k * 16),
        xs,
        ws.to(torch.float32),
        torch.bfloat16,
        skip_reduce=True,
        is_x_scale_tranposed=True,
    )
    if parts.dim() != 3:
        return {"splits": 1, "equal": torch.equal(parts.view(-1), y.view(-1))}
    p = [parts[i] for i in range(parts.size(0))]
    pad = 1 << (len(p) - 1).bit_length()
    z = torch.zeros_like(p[0])
    q = p + [z] * (pad - len(p))

    def seq(vals):
        acc = vals[0]
        for v in vals[1:]:
            acc = acc + v
        return acc

    def tree(vals):
        while len(vals) > 1:
            vals = [vals[2 * i] + vals[2 * i + 1] for i in range(len(vals) // 2)]
        return vals[0]

    def halves(vals):
        h = len(vals) // 2
        while h:
            vals = [vals[i] + vals[i + h] for i in range(h)]
            h //= 2
        return vals[0]

    cands = {
        "seq": seq(p),
        "seq_rev": seq(p[::-1]),
        "tree_pad": tree(q),
        "halves_pad": halves(q),
        "seq_pad": seq(q),
    }
    return {
        "splits": len(p),
        "match": {
            name: (c.to(torch.bfloat16) == y).double().mean().item()
            for name, c in cands.items()
        },
    }


def gluon_isa():
    roots = [
        os.environ.get("TRITON_CACHE_DIR", ""),
        os.path.expanduser("~/.triton/cache"),
    ]
    found = []
    for root in roots:
        if root and os.path.isdir(root):
            found += glob.glob(
                os.path.join(root, "**", "*blockscale_preshuffle*bandwidth*.amdgcn"),
                recursive=True,
            )
    out = []
    for path in found:
        with open(path) as f:
            text = f.read()
        ops = re.findall(r"^\s+(v_[a-z0-9_]+)", text, re.MULTILINE)
        counts = {}
        for op in ops:
            if any(t in op for t in ("wmma", "fma", "mul_f32", "add_f32")):
                counts[op] = counts.get(op, 0) + 1
        out.append({"path": path.split("/")[-2][:8], "counts": counts})
    return out


def rope_cache():
    from aiter.rotary_embedding import get_rope

    default = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    rope = get_rope(
        64,
        rotary_dim=64,
        max_position=1048576,
        base=50000.0,
        is_neox_style=False,
        rope_scaling={
            "rope_type": "deepseek_yarn",
            "factor": 32.0,
            "original_max_position_embeddings": 32768,
            "beta_fast": 32,
            "beta_slow": 1,
            "mscale": 1.0,
            "mscale_all_dim": 1.0,
        },
    )
    torch.set_default_dtype(default)
    return {
        "class": type(rope).__name__,
        "cos": [str(rope.cos_cache.dtype), list(rope.cos_cache.shape)],
        "is_neox": rope.is_neox_style,
    }


def dcp_widths():
    from atom.distributed.dcp_utils import dcp_persistent_supported
    from atom.model_ops.attention_mla import mla_dcp_kernel_num_heads

    return {
        heads: {
            "persistent_supported": dcp_persistent_supported(heads, "fp8", 8),
            "kernel_heads_persistent": mla_dcp_kernel_num_heads(
                heads // 8, 8, 16, kv_cache_dtype="fp8", persistent=True
            ),
        }
        for heads in HEADS
    }


print(json.dumps({"qkv_a": qkv_a_configs()}), flush=True)
for heads in HEADS:
    for m in ROWS:
        print(
            json.dumps(
                {
                    "heads": heads,
                    "rows": m,
                    "q_b": q_b_config(m, heads),
                    "reduce": reduce_order(m, heads),
                }
            ),
            flush=True,
        )
torch.cuda.synchronize()
print(json.dumps({"gluon_isa": gluon_isa()}), flush=True)
print(json.dumps({"rope": rope_cache()}), flush=True)
try:
    print(json.dumps({"dcp": dcp_widths()}), flush=True)
except Exception as why:  # noqa: BLE001
    print(json.dumps({"dcp_error": repr(why)}), flush=True)
