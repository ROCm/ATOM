# ruff: noqa: I001
# Keep measured import ordering/namespace; AST parity is verified by prepare.py.
"""Native G1, fully drained async waits, synchronous-copy comparison."""

from contextlib import contextmanager
from functools import cache
import aiter.ops.flydsl.moe_kernels as mk
from aiter.ops.flydsl.moe_common import GateMode
from wait_common import compile_mixed_moe_gemm1_common


@cache
def compile_wait(**kwargs):
    assert kwargs["a_dtype"] == "fp8" and kwargs["b_dtype"] == "fp4"
    assert kwargs["out_dtype"] == "bf16" and kwargs["tile_m"] == 64
    kwargs = dict(kwargs, gate_mode=GateMode(kwargs["gate_mode"]))
    return compile_mixed_moe_gemm1_common(**kwargs)


def full_wait(**kwargs):
    return mk._flydsl_moe_stage1_impl(**kwargs, _compile_kernel=compile_wait)


def synchronous(**kwargs):
    return mk._flydsl_moe_stage1_impl(**dict(kwargs, use_async_copy=False))


@contextmanager
def selected(name):
    before = mk.flydsl_moe_stage1
    if name == "full_wait":
        mk.flydsl_moe_stage1 = full_wait
    elif name == "native_fresh":
        mk.flydsl_moe_stage1 = native_fresh
    elif name == "minimal_wait":
        mk.flydsl_moe_stage1 = minimal_wait
    elif name == "sync_copy":
        mk.flydsl_moe_stage1 = synchronous
    else:
        assert name == "native_async"
    try:
        yield
    finally:
        mk.flydsl_moe_stage1 = before


@cache
def compile_minimal(**kwargs):
    from minimal_common import compile_mixed_moe_gemm1_common as compile_common

    assert kwargs["a_dtype"] == "fp8" and kwargs["b_dtype"] == "fp4"
    assert kwargs["out_dtype"] == "bf16" and kwargs["tile_m"] == 64
    return compile_common(**dict(kwargs, gate_mode=GateMode(kwargs["gate_mode"])))


def minimal_wait(**kwargs):
    return mk._flydsl_moe_stage1_impl(**kwargs, _compile_kernel=compile_minimal)


@cache
def compile_fresh(**kwargs):
    from fresh_common import compile_mixed_moe_gemm1_common as compile_common

    assert kwargs["a_dtype"] == "fp8" and kwargs["b_dtype"] == "fp4"
    return compile_common(**dict(kwargs, gate_mode=GateMode(kwargs["gate_mode"])))


def native_fresh(**kwargs):
    return mk._flydsl_moe_stage1_impl(**kwargs, _compile_kernel=compile_fresh)
