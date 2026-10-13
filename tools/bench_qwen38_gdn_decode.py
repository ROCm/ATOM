# SPDX-License-Identifier: MIT
"""Qwen3.8 GDN recurrent core: current path versus stock AITER FlyDSL.

Uses checkpoint geometry and synthetic activations/state on one gfx950 TP rank.
The convolution, projections, output norm, HC, MoE and TP reductions are not
included. The read/write state slots and query starts are prepared outside
both timed graphs, as they are during serving.
"""

import argparse
import json
import statistics

import torch
import triton.testing

from atom.model_ops.attention_gdn import fused_gdn_gating
from atom.model_ops.fla_ops import gdn_flydsl
from atom.model_ops.fla_ops.fused_recurrent import (
    fused_recurrent_gated_delta_rule,
)


def _time(fn, repeats: int) -> float:
    fn()
    torch.cuda.synchronize()
    return round(
        statistics.median(
            triton.testing.do_bench_cudagraph(fn, rep=50) * 1000 for _ in range(repeats)
        ),
        3,
    )


def run(rows: int, tp: int, state_dtype: torch.dtype, repeats: int) -> dict:
    if rows < 1 or tp not in (2, 4):
        raise ValueError("rows must be positive and TP must be 2 or 4")
    torch.manual_seed(812 + rows + tp)
    device = "cuda"
    key_heads, value_heads = 16 // tp, 48 // tp
    q = torch.randn(1, rows, key_heads, 128, device=device, dtype=torch.bfloat16)
    k = torch.randn_like(q)
    v = torch.randn(1, rows, value_heads, 128, device=device, dtype=torch.bfloat16)
    ba = torch.randn(rows, 2 * value_heads, device=device, dtype=torch.bfloat16)
    b, a = ba.split(value_heads, -1)
    A_log = torch.full((value_heads,), -2.0, device=device, dtype=torch.float32)
    dt_bias = torch.randn(value_heads, device=device, dtype=torch.bfloat16)
    source = (
        torch.randn(
            2 * rows,
            value_heads,
            128,
            128,
            device=device,
            dtype=state_dtype,
        )
        * 0.1
    )
    state_vk = source.transpose(-1, -2).contiguous().transpose(-1, -2)
    reads = torch.arange(rows, device=device, dtype=torch.int32)
    writes = reads + rows
    query_starts = torch.arange(rows + 1, device=device, dtype=torch.int32)
    if not gdn_flydsl.decode_supported(
        q, k, v, a, b, state_vk, A_log, dt_bias, reads, writes
    ):
        raise RuntimeError("stock AITER FlyDSL does not support this GDN geometry")

    def native(state):
        g, beta = fused_gdn_gating(A_log, a, b, dt_bias)
        return fused_recurrent_gated_delta_rule(
            q,
            k,
            v,
            g,
            beta,
            initial_state=state,
            inplace_final_state=True,
            cu_seqlens=query_starts,
            ssm_state_indices=writes,
            ssm_state_indices_in=reads,
            use_qk_l2norm_in_kernel=True,
        )[0]

    def flydsl(state):
        return gdn_flydsl.decode(q, k, v, a, b, state, A_log, dt_bias, reads, writes)[0]

    reference_state = source.clone()
    expected = native(reference_state)
    actual = flydsl(state_vk)
    output_error = float(
        torch.linalg.norm((actual - expected).float())
        / torch.linalg.norm(expected.float())
    )
    state_error = float(
        torch.linalg.norm((state_vk[rows:] - reference_state[rows:]).float())
        / torch.linalg.norm(reference_state[rows:].float())
    )
    if output_error >= 0.015 or state_error >= 0.015:
        raise AssertionError(
            f"GDN mismatch: output NRMSE {output_error}, state NRMSE {state_error}"
        )
    native_us = _time(lambda: native(reference_state), repeats)
    flydsl_us = _time(lambda: flydsl(state_vk), repeats)
    return {
        "rows": rows,
        "tp": tp,
        "state_dtype": str(state_dtype),
        "native_us": native_us,
        "flydsl_us": flydsl_us,
        "speedup": round(native_us / flydsl_us, 3),
        "output_nrmse": round(output_error, 6),
        "state_nrmse": round(state_error, 6),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, nargs="+", default=[1, 2, 8, 48])
    parser.add_argument("--tp", type=int, nargs="+", default=[2, 4])
    parser.add_argument("--fp32-state", action="store_true")
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    dtype = torch.float32 if args.fp32_state else torch.bfloat16
    print(
        json.dumps(
            {
                "gpu": torch.cuda.get_device_name(),
                "cases": [
                    run(rows, tp, dtype, args.repeats)
                    for tp in args.tp
                    for rows in args.rows
                ],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
