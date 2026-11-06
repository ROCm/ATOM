"""Stage-1 FP8xFP4 G1 with the minimal async-copy wait (BF16 output, tile_m 64)."""

from functools import cache

import aiter.ops.flydsl.moe_kernels as mk
from aiter.ops.flydsl.moe_common import GateMode

from .flydsl_minimal import compile_mixed_moe_gemm1_common


@cache
def compile_minimal(**kwargs):
    assert kwargs["a_dtype"] == "fp8" and kwargs["b_dtype"] == "fp4"
    assert kwargs["out_dtype"] == "bf16" and kwargs["tile_m"] == 64
    return compile_mixed_moe_gemm1_common(
        **dict(kwargs, gate_mode=GateMode(kwargs["gate_mode"]))
    )


def minimal_wait(**kwargs):
    return mk._flydsl_moe_stage1_impl(**kwargs, _compile_kernel=compile_minimal)
