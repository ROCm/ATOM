# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""The preshuffled MXFP4 GEMMs cannot handle a K that is not a multiple of 256.

None of the ASM BpreShuffle f4gemm kernels mask a K tail: the host rounds K up
to 256 and the last step reads the padded scale columns and the bytes past each
row, so the result depends on uninitialized memory and fails at an M that varies
from run to run. Kimi-K3's dense MLP down_proj at TP8 (K = 33792 / 8 = 4224) hit
this and every generation was garbage.

Pinned here, at the kernel and load-check level:
  * the bug, made deterministic (a positive control that goes red if the
    kernel ever masks the tail);
  * zero-padding K to 256 -- what an MLP pair now does inside the parallel
    linears -- is exact on the fast path;
  * a preshuffled MXFP4 weight still left unaligned is refused at load, loudly.
How the pair pads and loads is covered by tests/test_mxfp4_k_pad_pair.py.
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
    MXFP4_PRESHUFFLE_K_ALIGN,
    LinearBase,
    gemm_a4w4_quant,
)

KIMI_K3_DENSE_DOWN_K = 33792 // 8  # 4224
PADDED_K = 4352


@pytest.fixture(autouse=True)
def _default_backends(mxfp4_default_backends):
    """The default ASM path, where the bug was."""


# ── refused at load ───────────────────────────────────────────────────────


def _mxfp4_linear_double(k):
    """The attributes `LinearBase.process_weights_after_loading` reads."""
    return SimpleNamespace(
        weight=nn.Parameter(
            torch.zeros(32, k // 2, dtype=torch.uint8), requires_grad=False
        ),
        weight_scale=nn.Parameter(
            torch.full((32, k // 32), 127, dtype=torch.uint8), requires_grad=False
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
        prefix="layers.0.mlp.down_proj",
    )


def _load(monkeypatch, k, double=None):
    monkeypatch.setattr(linear_mod, "shuffle_weights", lambda *a, **kw: None)
    monkeypatch.setattr(
        linear_mod.fp4_utils, "e8m0_shuffle", lambda s: s, raising=False
    )
    LinearBase.process_weights_after_loading(double or _mxfp4_linear_double(k))


def test_unaligned_mxfp4_k_is_refused_at_load(monkeypatch):
    with pytest.raises(RuntimeError, match=r"layers\.0\.mlp\.down_proj.*K=4224"):
        _load(monkeypatch, KIMI_K3_DENSE_DOWN_K)


@pytest.mark.parametrize("k", [256, 4096, PADDED_K, 7168])
def test_aligned_mxfp4_k_loads(monkeypatch, k):
    _load(monkeypatch, k)


def test_non_shuffle_path_is_not_refused(monkeypatch):
    """gemm_afp4wfp4 reads the raw layout and has no K-tail problem."""
    monkeypatch.setattr(linear_mod, "use_fp4_non_shuffle_triton_gemm", lambda: True)
    _load(monkeypatch, KIMI_K3_DENSE_DOWN_K)


def test_the_32x4_scale_layout_is_not_refused(monkeypatch):
    """gfx1250's 32x4-scale kernels take K % 128, so K=4224 is fine there."""
    import aiter.ops.shuffle

    monkeypatch.setattr(linear_mod, "use_mxfp4_n32k4_scale", lambda w: True)
    monkeypatch.setattr(aiter.ops.shuffle, "shuffle_mxfp8fp4_scale", lambda s: s)
    double = _mxfp4_linear_double(KIMI_K3_DENSE_DOWN_K)
    _load(monkeypatch, KIMI_K3_DENSE_DOWN_K, double)
    assert double.fp4_n32k4_scale

    # An owner that reads the legacy e8m0 layout itself is still refused.
    double = _mxfp4_linear_double(KIMI_K3_DENSE_DOWN_K)
    double.needs_e8m0_shuffled_scale = True
    with pytest.raises(RuntimeError, match=r"K=4224"):
        _load(monkeypatch, KIMI_K3_DENSE_DOWN_K, double)


def test_the_refusal_is_mxfp4_only(monkeypatch):
    """Other schemes do not walk K in 256-wide fp4 steps."""
    double = _mxfp4_linear_double(KIMI_K3_DENSE_DOWN_K)
    double.quant_type = QuantType.No
    double.params_dtype = torch.bfloat16
    double.weight_scale = None
    LinearBase.process_weights_after_loading(double)


# ── zero-padding K is exact on the fast path ──────────────────────────────

_LUT = torch.tensor([0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6])


def _dequant(wq, ws):
    p = wq.view(torch.uint8).cpu()
    v = torch.stack([_LUT[(p & 0xF).long()], _LUT[(p >> 4).long()]], -1)
    v = v.reshape(p.shape[0], -1)
    s = torch.pow(2.0, ws.view(torch.uint8).cpu().float() - 127)
    return v * s.repeat_interleave(32, 1)[:, : v.shape[1]]


# M values the review saw fail at K=4224 (16, 33, 64) plus decode and prefill.
@pytest.mark.parametrize("m", [1, 16, 33, 64, 300])
def test_zero_padded_k_matches_the_unpadded_reference(m):
    """Real K=4224 data, zero-padded to 4352, through the default ASM path.

    Weights are prepared exactly as LinearBase does for a preshuffled MXFP4
    weight (shuffle_weights + e8m0_shuffle) and called as forward calls
    gemm_a4w4_quant. The reference is the UNPADDED quantized weight, so equality
    up to the fp4 activation error (~0.99 cosine) shows the zero tail adds
    nothing. N is Kimi-K3's hidden size, the real down_proj output.
    """
    torch.manual_seed(0)
    n, k = 7168, KIMI_K3_DENSE_DOWN_K
    quant = linear_mod.get_hip_quant(QuantType.per_1x32)
    w = torch.randn(n, k, dtype=torch.bfloat16, device="cuda") * 0.02
    x = torch.randn(m, k, dtype=torch.bfloat16, device="cuda")

    wq, ws = quant(w, quant_dtype=dtypes.fp4x2, shuffle=False)
    ref = x.float() @ _dequant(wq, ws).cuda().T

    tail = PADDED_K - k
    wq_p = torch.cat(
        [
            wq.view(torch.uint8),
            torch.zeros(n, tail // 2, dtype=torch.uint8, device="cuda"),
        ],
        dim=1,
    )
    ws_p = torch.cat(
        [
            ws.view(torch.uint8),
            torch.full((n, tail // 32), 127, dtype=torch.uint8, device="cuda"),
        ],
        dim=1,
    )
    x_p = torch.nn.functional.pad(x, (0, tail))

    weight = nn.Parameter(wq_p.view(dtypes.fp4x2), requires_grad=False)
    linear_mod.shuffle_weights(weight)
    ws_sh = linear_mod.fp4_utils.e8m0_shuffle(ws_p.view(ws.dtype))
    y = gemm_a4w4_quant(
        x_p, None, weight.data, torch.bfloat16, ws_sh, dtypes.fp4x2, None, n
    ).float()

    assert torch.isfinite(y).all(), f"m={m}"
    cos = torch.nn.functional.cosine_similarity(y.flatten(), ref.flatten(), 0)
    assert cos > 0.98, f"m={m} cosine={cos.item():.4f}"


def test_align_constant_matches_the_kernel_step():
    assert MXFP4_PRESHUFFLE_K_ALIGN == 256


def test_unpadded_k_reads_the_scale_tail_positive_control():
    """The bug itself, made deterministic: without padding the fix is needed.

    At K=4224 the ASM kernel's last 256-wide step reads 4 scale groups past the
    real 132 (e8m0_shuffle pads to 136). Normally those hold whatever memory
    did; here they are NaN (e8m0 0xFF), so the unpadded GEMM must come out
    non-finite, while the zero-padded weight of the same data stays correct. If
    the kernel ever learns to mask the tail, this goes red and the padding can
    be revisited.
    """
    torch.manual_seed(0)
    n, k, m = 7168, KIMI_K3_DENSE_DOWN_K, 64
    quant = linear_mod.get_hip_quant(QuantType.per_1x32)
    w = torch.randn(n, k, dtype=torch.bfloat16, device="cuda") * 0.02
    x = torch.randn(m, k, dtype=torch.bfloat16, device="cuda")
    wq, ws = quant(w, quant_dtype=dtypes.fp4x2, shuffle=False)

    tail = (PADDED_K - k) // 32
    poisoned = torch.cat(
        [
            ws.view(torch.uint8),
            torch.full((n, tail), 0xFF, dtype=torch.uint8, device="cuda"),
        ],
        dim=1,
    )
    weight = nn.Parameter(wq.clone(), requires_grad=False)
    linear_mod.shuffle_weights(weight)
    ws_sh = linear_mod.fp4_utils.e8m0_shuffle(poisoned.view(ws.dtype))
    y = gemm_a4w4_quant(
        x, None, weight.data, torch.bfloat16, ws_sh, dtypes.fp4x2, None, n
    ).float()

    assert not torch.isfinite(y).all(), (
        "the unpadded K=4224 GEMM no longer reads the scale tail -- the kernel "
        "may now mask it"
    )
