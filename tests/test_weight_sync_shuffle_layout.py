# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""A weight's layout must not depend on how it arrived.

The initial load decides whether a quantized 2D GEMM weight is held
preshuffled; an online weight update has to reach the same answer, or the
kernel reads the new weight through the old permutation and generation
collapses. The two used to decide separately, and disagreed wherever the rule
was not simply the env var.

`weight_is_stored_preshuffled` is now the single answer. These tests pin its
truth table, and then drive the real loader and the real updater over that
table and assert they shuffle in exactly the same cases -- which is the
property, not the table.
"""

from types import SimpleNamespace

import pytest
import torch
from torch import nn

if not torch.cuda.is_available():
    pytest.skip(
        "atom.model_ops.linear imports aiter, which resolves the chip "
        "architecture through rocminfo",
        allow_module_level=True,
    )

from aiter import QuantType, dtypes

from atom.model_ops import linear as linear_mod
from atom.model_ops.linear import (
    LinearBase,
    weight_is_stored_preshuffled,
)
from atom.rollout.weight_updater import WeightUpdaterMixin

PRESHUFFLE_ENV = "ATOM_FP8_BLOCKSCALE_WEIGHT_PRESHUFFLE"

# (quant_type, params_dtype, needs_preshuffled_weight)
CASES = [
    (QuantType.per_1x128, dtypes.fp8, False),
    (QuantType.per_1x128, dtypes.fp8, True),
    (QuantType.per_1x32, dtypes.fp8, False),
    (QuantType.per_1x32, dtypes.fp4x2, False),
    (QuantType.per_Token, dtypes.fp8, False),
    (QuantType.per_Token, torch.bfloat16, False),
    (QuantType.per_Tensor, dtypes.fp8, False),
    (QuantType.No, torch.bfloat16, False),
]


# ── the truth table ───────────────────────────────────────────────────────


@pytest.mark.parametrize("env_value,expected", [("1", True), ("0", False)])
def test_blockscale_follows_the_preshuffle_env_var(monkeypatch, env_value, expected):
    monkeypatch.setenv(PRESHUFFLE_ENV, env_value)
    assert weight_is_stored_preshuffled(QuantType.per_1x128, dtypes.fp8) is expected


def test_a_module_can_override_the_env_var_off(monkeypatch):
    """DeepSeek's fused qkv_a_proj calls the preshuffle blockscale GEMM directly.

    Its weight is shuffled at load even with the global path off, so a sync
    that only reads the env var leaves that one layer wrong -- and only that
    one, on a configuration (`=0`) the CI recipes use for four models.
    """
    monkeypatch.setenv(PRESHUFFLE_ENV, "0")
    assert (
        weight_is_stored_preshuffled(
            QuantType.per_1x128, dtypes.fp8, needs_preshuffled_weight=True
        )
        is True
    )


def test_the_override_cannot_turn_a_shuffle_off(monkeypatch):
    monkeypatch.setenv(PRESHUFFLE_ENV, "1")
    assert (
        weight_is_stored_preshuffled(
            QuantType.per_1x128, dtypes.fp8, needs_preshuffled_weight=False
        )
        is True
    )


def test_per_token_follows_which_gemm_will_read_it(monkeypatch):
    """The triton a8w8 per_Token GEMM wants (N, K) unshuffled.

    The sync used to shuffle every per_Token fp8 weight, so turning the triton
    GEMM on made every synced weight disagree with the loaded one.
    """
    monkeypatch.setenv("ATOM_USE_TRITON_GEMM", "1")
    expected = linear_mod.gemm_a8w8_triton is None
    assert weight_is_stored_preshuffled(QuantType.per_Token, dtypes.fp8) is expected

    monkeypatch.setenv("ATOM_USE_TRITON_GEMM", "0")
    assert weight_is_stored_preshuffled(QuantType.per_Token, dtypes.fp8) is True


def test_per_token_that_is_not_fp8_is_not_shuffled():
    assert weight_is_stored_preshuffled(QuantType.per_Token, torch.bfloat16) is False


def test_unquantized_and_per_tensor_are_not_shuffled():
    assert weight_is_stored_preshuffled(QuantType.No, torch.bfloat16) is False
    assert weight_is_stored_preshuffled(QuantType.per_Tensor, dtypes.fp8) is False


# ── the property: load and sync agree ─────────────────────────────────────


def _linear_double(quant_type, params_dtype, needs_preshuffled_weight, *, dim=2):
    """The attributes `LinearBase.process_weights_after_loading` reads.

    A double rather than a real Linear so the case matrix does not need a
    quant config and a device allocation per row; the method under test is the
    real one.
    """
    shape = (32, 64) if dim == 2 else (2, 32, 64)
    return SimpleNamespace(
        weight=nn.Parameter(torch.zeros(shape, dtype=torch.uint8), requires_grad=False),
        weight_scale=nn.Parameter(torch.ones(32, 1), requires_grad=False),
        quant_config=None,
        quant_type=quant_type,
        params_dtype=params_dtype,
        # Not bfloat16, so the fp4 online-quantize branch above is not taken.
        source_quant_dtype=torch.float8_e4m3fn,
        need_normalize_e4m3fn_to_e4m3fnuz=False,
        output_partition_sizes=[32],
        needs_preshuffled_weight=needs_preshuffled_weight,
        _maybe_pad_a8w8_preshuffle_output=lambda: False,
        prefix="test",
    )


def _updater_double():
    class _Updater(WeightUpdaterMixin):
        label = "test"
        device = torch.device("cpu")
        rank = 0
        world_size = 1

    return _Updater()


def _loader_shuffles(monkeypatch, case, *, dim=2):
    calls = []
    monkeypatch.setattr(linear_mod, "shuffle_weights", lambda *a, **k: calls.append(a))
    # per_1x32 shuffles its scale through aiter at the end of the method; that
    # is not what is under test and it wants a real e8m0 tensor.
    monkeypatch.setattr(
        linear_mod.fp4_utils, "e8m0_shuffle", lambda scale: scale, raising=False
    )
    LinearBase.process_weights_after_loading(_linear_double(*case, dim=dim))
    return bool(calls)


def _sync_shuffles(monkeypatch, case, *, dim=2):
    import atom.model_ops.utils as utils_mod

    calls = []
    monkeypatch.setattr(utils_mod, "shuffle_weights", lambda *a, **k: calls.append(a))
    quant_type, params_dtype, needs_preshuffled_weight = case
    shape = (32, 64) if dim == 2 else (2, 32, 64)
    param = nn.Parameter(torch.zeros(shape, dtype=torch.uint8), requires_grad=False)
    module = SimpleNamespace(
        quant_type=quant_type,
        params_dtype=params_dtype,
        needs_preshuffled_weight=needs_preshuffled_weight,
        weight_scale=None,
        need_normalize_e4m3fn_to_e4m3fnuz=False,
    )
    _updater_double()._post_process_fp8_weight(module, param)
    return bool(calls)


@pytest.mark.parametrize("case", CASES, ids=lambda c: f"{c[0].name}-{c[1]}-{c[2]}")
@pytest.mark.parametrize("preshuffle", ["0", "1"])
@pytest.mark.parametrize("triton_gemm", ["0", "1"])
def test_load_and_sync_shuffle_in_the_same_cases(
    monkeypatch, case, preshuffle, triton_gemm
):
    monkeypatch.setenv(PRESHUFFLE_ENV, preshuffle)
    monkeypatch.setenv("ATOM_USE_TRITON_GEMM", triton_gemm)

    assert _loader_shuffles(monkeypatch, case) == _sync_shuffles(monkeypatch, case)


@pytest.mark.parametrize("case", CASES, ids=lambda c: f"{c[0].name}-{c[1]}-{c[2]}")
def test_neither_side_shuffles_a_3d_weight(monkeypatch, case):
    """Qwen3-Next's GDN conv1d expands its weight to 3D and stays row-major.

    The loader has always checked; the sync briefly allowed rank 3 in order to
    reach the fused MoE buffers, which it cannot reach anyway -- FusedMoE has
    no `weight_scale`, so `_is_fp8_param` is false for it and this function is
    never called with one. What rank 3 did reach was conv1d.
    """
    monkeypatch.setenv(PRESHUFFLE_ENV, "1")
    assert _loader_shuffles(monkeypatch, case, dim=3) is False
    assert _sync_shuffles(monkeypatch, case, dim=3) is False
