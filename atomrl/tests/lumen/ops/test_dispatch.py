# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

###############################################################################
#
# See LICENSE for license information.
###############################################################################

"""Tests for atomrl.lumen.ops.dispatch — backend dispatcher and fallback chain.

Covers:
  - Backend enum values
  - FALLBACK_ORDER ordering
  - build_fallback_chain: filters None, preserves order
  - try_backends: returns first success
  - try_backends: falls through on RuntimeError
  - try_backends: raises when all backends fail
  - Probe functions: lru_cache returns consistent bool
"""

import logging

import pytest
from atomrl.lumen.ops.dispatch import (
    _BACKEND_WARMUP_CALLS,
    FALLBACK_ORDER,
    Backend,
    _backend_cache,
    build_fallback_chain,
    try_backends,
)


def test_dispatch_does_not_expose_aiter_probe_api():
    import atomrl.lumen.ops.dispatch as dispatch_module

    assert not any(name.startswith("_probe_") for name in vars(dispatch_module))


@pytest.fixture(autouse=True)
def _cold_backend_cache():
    """Every test starts from a cold dispatcher.

    The warmup cache is module-global and keyed by ``op_name``, so tests
    sharing a name would otherwise inherit each other's streaks and locks.
    """
    _backend_cache.clear()
    yield
    _backend_cache.clear()


# ===================================================================
# Backend enum
# ===================================================================


def test_backend_values():
    assert Backend.ASM.value == "asm"
    assert Backend.HIPBLAS.value == "hipblas"
    assert Backend.TRITON.value == "triton"


def test_fallback_order():
    assert FALLBACK_ORDER == [Backend.ASM, Backend.TRITON]


# ===================================================================
# build_fallback_chain
# ===================================================================


def test_build_chain_filters_none():
    """None values (unavailable backends) are skipped."""
    candidates = {
        Backend.ASM: None,
        Backend.HIPBLAS: lambda: "hipblas_result",
        Backend.TRITON: lambda: "triton_result",
    }
    chain = build_fallback_chain(
        candidates, order=[Backend.ASM, Backend.HIPBLAS, Backend.TRITON]
    )
    assert len(chain) == 2
    assert chain[0][0] == Backend.HIPBLAS
    assert chain[1][0] == Backend.TRITON


def test_build_chain_preserves_order():
    """Chain follows FALLBACK_ORDER."""

    def fn_a():
        return "a"

    def fn_b():
        return "b"

    def fn_c():
        return "c"

    candidates = {
        Backend.TRITON: fn_c,
        Backend.ASM: fn_a,
        Backend.HIPBLAS: fn_b,
    }
    chain = build_fallback_chain(
        candidates, order=[Backend.ASM, Backend.HIPBLAS, Backend.TRITON]
    )
    assert [b for b, _ in chain] == [Backend.ASM, Backend.HIPBLAS, Backend.TRITON]


def test_build_chain_empty():
    """All-None candidates produce an empty chain."""
    candidates = {Backend.ASM: None, Backend.HIPBLAS: None, Backend.TRITON: None}
    chain = build_fallback_chain(candidates)
    assert chain == []


def test_build_chain_custom_order():
    """Custom order is respected."""

    def fn():
        return "ok"

    candidates = {Backend.ASM: fn, Backend.HIPBLAS: fn, Backend.TRITON: fn}
    chain = build_fallback_chain(candidates, order=[Backend.TRITON, Backend.ASM])
    assert [b for b, _ in chain] == [Backend.TRITON, Backend.ASM]


# ===================================================================
# try_backends
# ===================================================================


def test_try_backends_first_success():
    """Returns result from first successful backend."""
    chain = [
        (Backend.ASM, lambda: "asm_ok"),
        (Backend.HIPBLAS, lambda: "hipblas_ok"),
    ]
    result = try_backends(chain, op_name="test")
    assert result == "asm_ok"


def test_try_backends_fallthrough():
    """Falls through on RuntimeError to next backend."""

    def fail_asm():
        raise RuntimeError("ASM not supported")

    chain = [
        (Backend.ASM, fail_asm),
        (Backend.HIPBLAS, lambda: "hipblas_ok"),
    ]
    result = try_backends(chain, op_name="test")
    assert result == "hipblas_ok"


def test_try_backends_all_fail():
    """Raises RuntimeError when all backends exhausted."""

    def fail():
        raise RuntimeError("nope")

    chain = [
        (Backend.ASM, fail),
        (Backend.HIPBLAS, fail),
        (Backend.TRITON, fail),
    ]
    with pytest.raises(RuntimeError, match="all AITER backends exhausted"):
        try_backends(chain, op_name="test")


def test_try_backends_not_implemented():
    """NotImplementedError triggers fallback."""

    def not_impl():
        raise NotImplementedError("missing")

    chain = [
        (Backend.ASM, not_impl),
        (Backend.TRITON, lambda: "triton_ok"),
    ]
    result = try_backends(chain, op_name="test")
    assert result == "triton_ok"


def test_try_backends_type_error():
    """TypeError triggers fallback."""

    def bad_types():
        raise TypeError("wrong args")

    chain = [
        (Backend.ASM, bad_types),
        (Backend.HIPBLAS, lambda: "hipblas_ok"),
    ]
    result = try_backends(chain, op_name="test")
    assert result == "hipblas_ok"


def test_try_backends_value_error():
    """ValueError triggers fallback."""

    def bad_value():
        raise ValueError("bad config")

    chain = [
        (Backend.ASM, bad_value),
        (Backend.TRITON, lambda: "ok"),
    ]
    result = try_backends(chain, op_name="test")
    assert result == "ok"


def test_try_backends_passes_args():
    """Arguments are forwarded to backend callables."""

    def backend_fn(a, b, c=None):
        return a + b + (c or 0)

    chain = [(Backend.ASM, backend_fn)]
    result = try_backends(chain, 1, 2, c=3, op_name="test")
    assert result == 6


# ===================================================================
# try_backends — what the warmup cache remembers
# ===================================================================


def test_try_backends_cache_is_keyed_by_label_not_position():
    """A verdict earned by one chain must not be applied to a different one.

    Ops that rebuild their chain per call -- ``gemm_mxfp4`` reorders by the
    autotuned winner and drops entries the operands make illegal -- give the
    same position a different meaning from call to call. Caching the position
    let a shape that had to reach the last-resort entry hand that slot to every
    later shape, which is how a quantized run ends up running the dequant→BF16
    fallback throughout with nothing but a debug line to say so.
    """
    op = "test_label_cache"

    def fail():
        raise RuntimeError("these operands are not supported")

    lock_chain = [
        (Backend.ASM, fail, "asm"),
        (Backend.TRITON, lambda: "fallback", "dequant_bf16"),
    ]
    for _ in range(_BACKEND_WARMUP_CALLS):
        assert try_backends(lock_chain, op_name=op) == "fallback"
    assert _backend_cache[op] == "dequant_bf16"

    # Same op, operands that make every kernel legal: nothing here is the
    # backend that got cached, and position 1 is now a real kernel.
    other_chain = [
        (Backend.ASM, lambda: "asm_result", "asm"),
        (Backend.TRITON, lambda: "shuffled_result", "shuffled"),
    ]
    assert try_backends(other_chain, op_name=op) == "asm_result"


def test_try_backends_streak_does_not_carry_across_backends():
    """Consecutive wins are counted per backend, not per op.

    Reading the running count before comparing it to the previous winner let a
    newly chosen backend inherit the old one's streak and lock on its first
    success, skipping warmup entirely.
    """
    op = "test_streak"

    state = {"fail_first": False}

    def first():
        if state["fail_first"]:
            raise RuntimeError("stopped working")
        return "first"

    chain = [
        (Backend.ASM, first, "first"),
        (Backend.TRITON, lambda: "second", "second"),
    ]

    for _ in range(_BACKEND_WARMUP_CALLS - 1):
        assert try_backends(chain, op_name=op) == "first"
    assert _backend_cache[op + ":hits"] == _BACKEND_WARMUP_CALLS - 1

    state["fail_first"] = True
    assert try_backends(chain, op_name=op) == "second"
    assert (
        _backend_cache[op + ":hits"] == 1
    ), "second backend inherited the first one's streak"
    assert op not in _backend_cache, "locked without serving its own warmup"


def test_try_backends_warns_once_when_it_locks_onto_a_slow_label(caplog):
    """Locking onto a degraded path is a warning, not a debug line."""
    op = "test_slow_lock"

    def fail():
        raise RuntimeError("unsupported")

    chain = [
        (Backend.ASM, fail, "asm"),
        (Backend.TRITON, lambda: "slow", "dequant_bf16"),
    ]
    with caplog.at_level(logging.WARNING, logger="atomrl.lumen.ops.dispatch"):
        for _ in range(_BACKEND_WARMUP_CALLS + 2):
            try_backends(chain, op_name=op, slow_labels=("dequant_bf16",))

    locked = [
        r for r in caplog.records if "locked to the dequant_bf16 fallback" in r.message
    ]
    assert len(locked) == 1, f"expected exactly one lock warning, got {len(locked)}"


def test_try_backends_label_defaults_to_the_backend_name():
    """Chains that don't pass a label keep working, keyed by ``Backend``."""
    op = "test_default_label"

    chain = [(Backend.HIPBLAS, lambda: "hipblas_ok")]
    for _ in range(_BACKEND_WARMUP_CALLS):
        assert try_backends(chain, op_name=op) == "hipblas_ok"
    assert _backend_cache[op] == Backend.HIPBLAS.value


def test_try_backends_cached_backend_degrades_instead_of_raising():
    """A locked backend that starts failing must fall back, not propagate.

    The chain's whole purpose is that a kernel rejecting these operands at
    runtime degrades. Taking the cached shortcut without a guard voided that
    the moment warmup finished.
    """
    op = "test_cached_degrades"

    state = {"broken": False}

    def primary():
        if state["broken"]:
            raise RuntimeError("operands no longer supported")
        return "primary"

    chain = [
        (Backend.ASM, primary, "primary"),
        (Backend.TRITON, lambda: "secondary", "secondary"),
    ]
    for _ in range(_BACKEND_WARMUP_CALLS):
        assert try_backends(chain, op_name=op) == "primary"
    assert _backend_cache[op] == "primary"

    state["broken"] = True
    assert try_backends(chain, op_name=op) == "secondary"
