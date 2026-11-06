# ruff: noqa: I001
# Keep measured import ordering/namespace; AST parity is verified by prepare.py.
"""Default-off corrective G1 builder: tested scope starts at M4096 FP8 sorted output."""

from contextlib import contextmanager
from functools import cache
import aiter.ops.flydsl.moe_kernels as mk
from aiter.ops.flydsl.moe_common import GateMode
from minimal_common import compile_mixed_moe_gemm1_common


@cache
def compile_fp8_wait0(**kwargs):
    assert kwargs["a_dtype"] == "fp8" and kwargs["b_dtype"] == "fp4"
    assert kwargs["out_dtype"] == "fp8" and kwargs["v2_output_layout"] is True
    assert (kwargs["tile_m"], kwargs["tile_n"], kwargs["tile_k"]) == (64, 128, 256)
    assert kwargs["model_dim"] == 5120 and kwargs["inter_dim"] == 640
    return compile_mixed_moe_gemm1_common(
        **dict(kwargs, gate_mode=GateMode(kwargs["gate_mode"]))
    )


def corrected(**kwargs):
    assert tuple(kwargs["a"].shape) == (4096, 5120)
    assert kwargs["out_dtype"] == "fp8" and kwargs.get("v2_output_layout") is True
    return mk._flydsl_moe_stage1_impl(**kwargs, _compile_kernel=compile_fp8_wait0)


@contextmanager
def selected(enabled):
    original = mk.flydsl_moe_stage1

    def guarded(**kwargs):
        if (
            tuple(kwargs["a"].shape) == (4096, 5120)
            and kwargs["out_dtype"] == "fp8"
            and kwargs.get("v2_output_layout") is True
        ):
            return corrected(**kwargs)
        return original(**kwargs)

    if enabled:
        mk.flydsl_moe_stage1 = guarded
    try:
        yield
    finally:
        mk.flydsl_moe_stage1 = original
