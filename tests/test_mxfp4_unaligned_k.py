# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""An MXFP4 weight whose K the preshuffled layout cannot hold must not use it.

The preshuffled MXFP4 layout pads K//32 scale groups to a multiple of 8, i.e.
K to a multiple of 256, and the kernels cannot mask the partial group. The
triton preshuffle GEMM refuses such a K; the default ASM ``gemm_a4w4`` accepted
it and returned inf/NaN for M >= 64. Kimi-K3's dense MLP ``down_proj`` at TP8 is
K = 33792 / 8 = 4224 (132 groups, padded to 136): every prefill pushed NaN
through layer 0 and generation collapsed, while decode-sized M looked fine.

Such a weight is now kept unshuffled and read by the non-preshuffled
``gemm_afp4wfp4``. The decision lives in ``weight_is_stored_preshuffled`` so the
initial load and an online weight sync still agree on it.
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
    FP4_PRESHUFFLE_K_ALIGN,
    LinearBase,
    gemm_a4w4_quant,
    logical_k,
    weight_is_stored_preshuffled,
)
from atom.rollout.weight_updater import WeightUpdaterMixin

KIMI_K3_DENSE_DOWN_K = 33792 // 8  # 4224
assert KIMI_K3_DENSE_DOWN_K % FP4_PRESHUFFLE_K_ALIGN != 0


@pytest.fixture(autouse=True)
def _default_backends(monkeypatch):
    # The default path, where the bug was: neither triton override on.
    monkeypatch.setattr(linear_mod, "use_fp4_non_shuffle_triton_gemm", lambda: False)
    monkeypatch.setattr(linear_mod, "use_triton_gemm", lambda: False)


# ── the decision ──────────────────────────────────────────────────────────


def test_unaligned_mxfp4_k_is_not_preshuffled():
    assert not weight_is_stored_preshuffled(
        QuantType.per_1x32, dtypes.fp4x2, k=KIMI_K3_DENSE_DOWN_K
    )


@pytest.mark.parametrize("k", [256, 4096, 4352, 7168])
def test_aligned_mxfp4_k_is_still_preshuffled(k):
    assert weight_is_stored_preshuffled(QuantType.per_1x32, dtypes.fp4x2, k=k)


def test_without_k_the_answer_is_unchanged():
    """Callers that do not pass K keep exactly the old behaviour."""
    assert weight_is_stored_preshuffled(QuantType.per_1x32, dtypes.fp4x2)


def test_the_k_rule_is_mxfp4_only():
    assert weight_is_stored_preshuffled(
        QuantType.per_1x32, dtypes.fp8, k=KIMI_K3_DENSE_DOWN_K
    )


def test_logical_k_unpacks_fp4x2():
    w = torch.empty(8, KIMI_K3_DENSE_DOWN_K // 2, dtype=torch.uint8)
    assert logical_k(w, dtypes.fp4x2) == KIMI_K3_DENSE_DOWN_K
    assert logical_k(w, dtypes.fp8) == KIMI_K3_DENSE_DOWN_K // 2


# ── load and sync agree, and the layer is flagged ─────────────────────────


def _mxfp4_linear_double(k):
    """The attributes `LinearBase.process_weights_after_loading` reads."""
    return SimpleNamespace(
        weight=nn.Parameter(
            torch.zeros(32, k // 2, dtype=torch.uint8), requires_grad=False
        ),
        weight_scale=nn.Parameter(
            torch.zeros(32, k // 32, dtype=torch.uint8), requires_grad=False
        ),
        quant_config=None,
        quant_type=QuantType.per_1x32,
        params_dtype=dtypes.fp4x2,
        source_quant_dtype=None,
        need_normalize_e4m3fn_to_e4m3fnuz=False,
        output_partition_sizes=[32],
        needs_preshuffled_weight=False,
        native_a8_group_rows=None,
        _maybe_pad_a8w8_preshuffle_output=lambda: False,
        prefix="test",
    )


def _load(monkeypatch, k):
    weight_calls, scale_calls = [], []
    monkeypatch.setattr(
        linear_mod, "shuffle_weights", lambda *a, **kw: weight_calls.append(a)
    )
    monkeypatch.setattr(
        linear_mod.fp4_utils,
        "e8m0_shuffle",
        lambda s: scale_calls.append(s) or s,
        raising=False,
    )
    layer = _mxfp4_linear_double(k)
    LinearBase.process_weights_after_loading(layer)
    return layer, bool(weight_calls), bool(scale_calls)


def _sync_shuffles(monkeypatch, k):
    import atom.model_ops.utils as utils_mod

    calls = []
    monkeypatch.setattr(utils_mod, "shuffle_weights", lambda *a, **kw: calls.append(a))
    param = nn.Parameter(
        torch.zeros(32, k // 2, dtype=torch.uint8), requires_grad=False
    )
    module = SimpleNamespace(
        quant_type=QuantType.per_1x32,
        params_dtype=dtypes.fp4x2,
        needs_preshuffled_weight=False,
        weight_scale=None,
        need_normalize_e4m3fn_to_e4m3fnuz=False,
    )

    class _Updater(WeightUpdaterMixin):
        label = "test"
        device = torch.device("cpu")
        rank = 0
        world_size = 1

    _Updater()._post_process_fp8_weight(module, param)
    return bool(calls)


@pytest.mark.parametrize("k", [KIMI_K3_DENSE_DOWN_K, 4096])
def test_load_and_sync_agree(monkeypatch, k):
    _, load_shuffled, _ = _load(monkeypatch, k)
    assert load_shuffled == _sync_shuffles(monkeypatch, k)


def test_unaligned_layer_keeps_weight_and_scale_raw(monkeypatch):
    layer, weight_shuffled, scale_shuffled = _load(monkeypatch, KIMI_K3_DENSE_DOWN_K)
    assert layer._fp4_unshuffled
    assert not weight_shuffled
    assert not scale_shuffled


def test_aligned_layer_is_untouched(monkeypatch):
    layer, weight_shuffled, scale_shuffled = _load(monkeypatch, 4096)
    assert not layer._fp4_unshuffled
    assert weight_shuffled
    assert scale_shuffled


# ── the numbers ───────────────────────────────────────────────────────────

_LUT = torch.tensor([0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6])


def _dequant(wq, ws):
    p = wq.view(torch.uint8).cpu()
    v = torch.stack([_LUT[(p & 0xF).long()], _LUT[(p >> 4).long()]], -1)
    v = v.reshape(p.shape[0], -1)
    s = torch.pow(2.0, ws.view(torch.uint8).cpu().float() - 127)
    return v * s.repeat_interleave(32, 1)[:, : v.shape[1]]


@pytest.mark.parametrize("k", [KIMI_K3_DENSE_DOWN_K, 4096])
@pytest.mark.parametrize("m", [1, 64, 300])
def test_gemm_matches_reference(k, m):
    """Through the same load decision and forward a real layer takes.

    The reference is the SAME quantized weight dequantized, so the remaining
    gap is the fp4 activation quantization (~11% relative, cosine ~0.993).
    The bug was not a small error: it was inf/NaN.
    """
    torch.manual_seed(0)
    n = 512
    quant = linear_mod.get_hip_quant(QuantType.per_1x32)
    w = torch.randn(n, k, dtype=torch.bfloat16, device="cuda") * 0.02
    wq, ws = quant(w, quant_dtype=dtypes.fp4x2, shuffle=False)
    ref_w = _dequant(wq, ws).cuda()

    unshuffled = not weight_is_stored_preshuffled(QuantType.per_1x32, dtypes.fp4x2, k=k)
    weight = nn.Parameter(wq.clone(), requires_grad=False)
    if not unshuffled:
        linear_mod.shuffle_weights(weight)
        ws = linear_mod.fp4_utils.e8m0_shuffle(ws.clone())

    x = torch.randn(m, k, dtype=torch.bfloat16, device="cuda")
    y = gemm_a4w4_quant(
        x, None, weight.data, torch.bfloat16, ws, dtypes.fp4x2, None, n, unshuffled
    ).float()
    ref = x.float() @ ref_w.T

    assert torch.isfinite(y).all()
    cos = torch.nn.functional.cosine_similarity(y.flatten(), ref.flatten(), 0)
    assert cos > 0.98, f"k={k} m={m} cosine={cos.item():.4f}"
