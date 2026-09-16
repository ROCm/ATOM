# SPDX-License-Identifier: MIT
"""Does V4's fused compressor reproduce CSA2's latent at ratio 1 and 2?

P2 replaces V4.1's per-request compressor loop with V4's plan-driven
`fused_compress_attn`. That only works if the kernel's pooling IS CSA2's:
`kv_acc / w_acc` over the K window is a per-dimension softmax-weighted mean,
and `(values * scores.softmax(ratio_axis)).sum(ratio_axis)` is the same
expression -- but "the same expression" is a reading of two sources, not a
measurement, and the two do not round identically (the kernel normalizes in
fp32 throughout; the eager path casts the pooled value to BF16 first).

So: same projections into both, compare the scattered latent. Two ratios,
because `overlap=False, K=RATIO` makes ratio 1 a degenerate window the kernel
has never been run at -- a single element whose softmax weight is 1 regardless
of its score, which is exactly CSA2's ungated ratio-1 path.

Single GPU, no model weights, no TP group. Run under the pinned AITER:

    python -m tests.attentions.deepseek_v41.validate_compressor_kernel \\
        --output /app/logs_claude/<run>/result.json
"""

import argparse
import json
import sys

import numpy as np
import torch
from atom.model_ops.deepseek_v41.compressor import Compressor
from atom.model_ops.deepseek_v41.rotary import RotaryEmbedding

from atom.model_ops import layernorm
from atom.model_ops.v4_kernels import fused_compress_attn, make_compress_plans
from atom.utils import CpuGpuBuffer

DEV = "cuda"
HIDDEN, HEAD_DIM, ROPE_DIM = 1024, 512, 64
EPS = 1e-6


def declare_single_rank():
    """What `tests/models/deepseek_v41/conftest.py::single_rank` declares.

    `Compressor` is two `nn.Linear` and an `RMSNorm`; only the norm asks for a
    TP width. Standing up a real process group to answer `1` would make this
    probe a distributed test for no reason.
    """
    layernorm.get_tensor_model_parallel_world_size = lambda: 1


def plan_for(tokens, ratio, block_size):
    """One request of `tokens` tokens from position 0, eager (tight) slices."""
    buffers = {
        ratio: {
            name: CpuGpuBuffer(
                tokens + 1, 4, dtype=torch.int32, device=DEV, pin_memory=True
            )
            for name in ("compress", "write")
        }
    }

    return make_compress_plans(
        np.asarray([tokens], dtype=np.int32),
        np.asarray([tokens], dtype=np.int32),
        ((ratio, False),),
        plan_buffers=buffers,
        extra_write=0,  # One committed prefill; nothing rejects it.
    )[ratio]


def eager_latent(compressor, rope, hidden, ratio):
    """CSA2's own path: project, pool, norm, rotate.

    Returns the rotated latent (what the main cache holds) and the unrotated
    one (what the index key projects from) -- the kernel has to reproduce both
    or only half of CSA2 moves onto it.
    """
    values, scores = compressor.project(hidden)
    latent, _ = compressor.pool(values, scores, 0, None, dtype=torch.bfloat16)
    begin = torch.arange(latent.shape[1], device=DEV) * ratio
    # `rope` rotates in place -- which is why `_update_global` derives the
    # index key before calling it. Keep a copy or this compares the rotated
    # latent against itself and only the 64 rope dims disagree.
    unrotated = latent.clone()
    return rope(latent, begin), unrotated


def kernel_latent(compressor, rope, hidden, ratio, tokens, block_size, spec):
    """V4's path: the same projections through the plan-driven kernel."""
    values, scores = compressor.project(hidden)
    if scores is None:
        # Ratio 1 has no gate. A constant score leaves the single-element
        # softmax at 1, which is what the ungated path computes.
        scores = torch.zeros_like(values)
    flat_kv = values[0].contiguous().float()
    flat_score = scores[0].contiguous().float()

    state_size = ratio + spec
    kv_state = torch.zeros(1, state_size, HEAD_DIM, dtype=torch.float32, device=DEV)
    score_state = torch.zeros_like(kv_state)
    rows = block_size // ratio
    compressed = -(-tokens // ratio)
    blocks = -(-compressed // rows) + 1
    kv_cache = torch.zeros(blocks, rows, HEAD_DIM, dtype=torch.bfloat16, device=DEV)
    block_tables = torch.arange(blocks, dtype=torch.int32, device=DEV).unsqueeze(0)
    slots = torch.zeros(1, dtype=torch.int32, device=DEV)
    plan = plan_for(tokens, ratio, block_size)
    latent_out = torch.zeros(
        plan.compress_plan_gpu.shape[0], HEAD_DIM, dtype=torch.bfloat16, device=DEV
    )

    fused_compress_attn(
        kv_in=flat_kv,
        score_in=flat_score,
        kv_state=kv_state,
        score_state=score_state,
        plan=plan,
        state_slot_mapping=slots,
        # CSA2 has no absolute position encoding on the gate.
        ape=torch.zeros(ratio, HEAD_DIM, dtype=torch.float32, device=DEV),
        rms_weight=compressor.norm.weight.float().contiguous(),
        rms_eps=EPS,
        cos_cache=rope.cos_cache,
        sin_cache=rope.sin_cache,
        kv_cache=kv_cache,
        block_tables=block_tables,
        k_per_block=rows,
        overlap=False,
        ratio=ratio,
        head_dim=HEAD_DIM,
        rope_head_dim=ROPE_DIM,
        quant_mode="none",
        latent_out=latent_out,
        prefix="validate.fused_compress_attn",
    )
    count = tokens // ratio
    ids = torch.arange(count, device=DEV)
    # Plan rows are emitted in compression-boundary order for this one request,
    # so the first `count` of them are its compressed rows in order.
    return kv_cache[ids // rows, ids % rows].unsqueeze(0), latent_out[:count][None]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--tokens", type=int, nargs="+", default=[8, 64, 130])
    parser.add_argument("--ratios", type=int, nargs="+", default=[1, 2])
    parser.add_argument("--block-size", type=int, default=256)
    parser.add_argument("--speculative-tokens", type=int, default=5)
    parser.add_argument("--seed", type=int, default=21)
    args = parser.parse_args()

    torch.set_default_dtype(torch.bfloat16)
    declare_single_rank()
    records = []
    for ratio in args.ratios:
        torch.manual_seed(args.seed)
        with torch.device(DEV):
            compressor = Compressor(HIDDEN, HEAD_DIM, ratio, EPS)
            rope = RotaryEmbedding(ROPE_DIM, 4096, base=10000)
        for parameter in compressor.parameters():
            parameter.data.normal_(0, 0.05)
        compressor.process_weights_after_loading()
        for tokens in args.tokens:
            row = {"ratio": ratio, "tokens": tokens}
            try:
                hidden = torch.randn(1, tokens, HIDDEN, device=DEV) * 0.5
                with torch.inference_mode():
                    rotated, unrotated = eager_latent(compressor, rope, hidden, ratio)
                    scattered, echoed = kernel_latent(
                        compressor,
                        rope,
                        hidden,
                        ratio,
                        tokens,
                        args.block_size,
                        args.speculative_tokens,
                    )
                row["rows"] = min(rotated.shape[1], scattered.shape[1])
                for name, expected, actual in (
                    ("main", rotated, scattered),
                    ("index_key", unrotated, echoed),
                ):
                    width = min(expected.shape[1], actual.shape[1])
                    left = expected[:, :width].float()
                    right = actual[:, :width].float()
                    row[f"{name}_max_abs_error"] = float((left - right).abs().max())
                    row[f"{name}_relative_l2"] = float(
                        (left - right).norm() / left.norm()
                    )
                    # On the FP4 grid an aggregate norm hides the shape of the
                    # disagreement: a 1e-3 difference before the snap either
                    # vanishes or costs a whole level, so what matters is how
                    # many elements land on a different level, not the average.
                    row[f"{name}_exact_fraction"] = float(
                        (left == right).float().mean()
                    )
                    row[f"{name}_cosine"] = float(
                        torch.nn.functional.cosine_similarity(
                            left.flatten(), right.flatten(), dim=0
                        )
                    )
                row["ok"] = (
                    max(row["main_relative_l2"], row["index_key_relative_l2"]) < 1e-2
                )
            except Exception as error:  # noqa: BLE001 - a refusal is a result here
                row["error"] = f"{type(error).__name__}: {error}"
                row["ok"] = False
            records.append(row)
            print("COMPRESS", json.dumps(row), flush=True)

    passed = all(row["ok"] for row in records)
    with open(args.output, "w") as handle:
        json.dump({"records": records, "passed": passed}, handle, indent=2)
    print("COMPRESSOR_KERNEL_PASSED" if passed else "COMPRESSOR_KERNEL_FAILED")
    return 0 if passed else 2


sys.exit(main())
