# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Loading a Kimi-K3 checkpoint whose attention is statically quantized.

`amd/Kimi-K3-Quark-MXFP4-AttnFP8` ships its self_attn projections as
per-output-channel FP8 (ptpc). Three things assumed the target's static quant
config was empty, which held only while attention was BF16 on disk and
quantized online:

  * the KDA in-proj fusion asserted QuantType.No on in_proj / b_proj / f_a_proj;
  * b_proj was built with quant_config=None, so its FP8 rows would load
    value-cast with their scales dropped;
  * a standalone draft (bf16) inherited the target's MXFP4 global spec.

These drive the real methods with stand-ins, as the neighbouring tests do.
"""

from types import SimpleNamespace

import pytest
import torch
from torch import nn

if not torch.cuda.is_available():
    pytest.skip(
        "atom.models.kimi_k3 imports aiter, which resolves the chip "
        "architecture through rocminfo",
        allow_module_level=True,
    )

from aiter import QuantType

from atom.models.kimi_k3.model import KimiKDAAttention
from atom.spec_decode.drafter import Drafter

LP, NLH, HD, K = 8, 2, 4, 16  # local_proj_size, local heads, head_dim, hidden
CONV_K = 4


def _proj(rows, quant_type, seed):
    g = torch.Generator().manual_seed(seed)
    w = torch.randn(rows, K, generator=g)
    scale = None
    if quant_type == QuantType.per_Token:
        w = w.to(torch.float8_e4m3fn)
        scale = nn.Parameter(
            torch.rand(rows, 1, generator=g) + 0.5, requires_grad=False
        )
    return SimpleNamespace(
        weight=nn.Parameter(w, requires_grad=False),
        weight_scale=scale,
        quant_type=quant_type,
    )


def _kda_double(in_type, b_type=None, f_a_type=None):
    b_type = in_type if b_type is None else b_type
    f_a_type = in_type if f_a_type is None else f_a_type
    conv = lambda: SimpleNamespace(weight=torch.randn(LP, 1, CONV_K))
    return SimpleNamespace(
        in_proj=_proj(4 * LP, in_type, 0),
        b_proj=_proj(NLH, b_type, 1),
        f_a_proj=_proj(HD, f_a_type, 2),
        q_conv1d=conv(),
        k_conv1d=conv(),
        v_conv1d=conv(),
        local_proj_size=LP,
        conv_kernel_size=CONV_K,
    )


# ── the fusion ────────────────────────────────────────────────────────────


def test_fp8_fusion_concatenates_weights_and_scales_in_order():
    """A per-output-channel scale is indexed by the axis being concatenated, so
    each row keeps its own scale: the fused scale is the concatenation."""
    attn = _kda_double(QuantType.per_Token)
    parts = [attn.in_proj, attn.b_proj, attn.f_a_proj]
    want_w = torch.cat([p.weight.data.view(torch.uint8) for p in parts])
    want_s = torch.cat([p.weight_scale.data for p in parts])

    KimiKDAAttention.process_weights_after_loading(attn)

    assert attn.in_proj.weight.shape == (4 * LP + NLH + HD, K)
    assert torch.equal(attn.in_proj.weight.data.view(torch.uint8), want_w)
    assert torch.equal(attn.in_proj.weight_scale.data, want_s)
    # the tails are released; the modules stay as empty shells
    assert attn.b_proj.weight.numel() == 0
    assert attn.f_a_proj.weight.numel() == 0


def test_fp8_fusion_equals_quantizing_the_fused_weight():
    """The order online quant uses (fuse bf16, then quantize the grown weight)
    and the order a static checkpoint arrives in (quantize, then fuse) agree,
    because a per-row scale does not depend on the rows beside it."""
    fp8, fmax = torch.float8_e4m3fn, torch.finfo(torch.float8_e4m3fn).max

    def ptpc(w):
        s = w.abs().amax(dim=1, keepdim=True) / fmax
        return (w / s).clamp(-fmax, fmax).to(fp8), s

    g = torch.Generator().manual_seed(7)
    rows = [torch.randn(n, K, generator=g) for n in (4 * LP, NLH, HD)]
    fused_then_q = ptpc(torch.cat(rows))

    attn = _kda_double(QuantType.per_Token)
    for proj, w in zip((attn.in_proj, attn.b_proj, attn.f_a_proj), rows, strict=True):
        q, s = ptpc(w)
        proj.weight = nn.Parameter(q, requires_grad=False)
        proj.weight_scale = nn.Parameter(s, requires_grad=False)
    KimiKDAAttention.process_weights_after_loading(attn)

    assert torch.equal(
        attn.in_proj.weight.data.view(torch.uint8),
        fused_then_q[0].view(torch.uint8),
    )
    assert torch.equal(attn.in_proj.weight_scale.data, fused_then_q[1])


def test_bf16_fusion_is_unchanged():
    attn = _kda_double(QuantType.No)
    parts = [attn.in_proj, attn.b_proj, attn.f_a_proj]
    want = torch.cat([p.weight.data for p in parts])
    KimiKDAAttention.process_weights_after_loading(attn)
    assert torch.equal(attn.in_proj.weight.data, want)
    assert attn.in_proj.weight_scale is None


def test_a_mixed_scheme_is_refused():
    """What a b_proj built without quant_config produced: an unquantized tail
    beside an FP8 in_proj. Concatenating them would mix dtypes."""
    attn = _kda_double(QuantType.per_Token, b_type=QuantType.No)
    with pytest.raises(AssertionError, match="one scheme"):
        KimiKDAAttention.process_weights_after_loading(attn)


def test_an_unsupported_scheme_is_refused():
    attn = _kda_double(QuantType.per_1x128)
    with pytest.raises(AssertionError, match="per_Token"):
        KimiKDAAttention.process_weights_after_loading(attn)


def test_a_missing_per_token_scale_is_reported_by_the_guard():
    """The guard, not an AttributeError on `.data`, reports a missing scale."""
    attn = _kda_double(QuantType.per_Token)
    attn.b_proj.weight_scale = None
    with pytest.raises(AssertionError, match="one row per output channel"):
        KimiKDAAttention.process_weights_after_loading(attn)


def test_glm_refuses_a_quantized_in_proj_before_its_gate_fold():
    """GLM-5.3's KDA subclass folds a bf16 gate into in_proj by a plain cast;
    with the parent now accepting per_Token, the subclass must refuse it
    rather than write unscaled FP8 rows."""
    from atom.models.glm5_next import Glm5NextKDAAttention

    attn = _kda_double(QuantType.per_Token)
    with pytest.raises(NotImplementedError, match="statically quantized in_proj"):
        Glm5NextKDAAttention.process_weights_after_loading(attn)


def test_b_proj_is_deferred_with_the_other_fused_projections():
    """Under online quant b_proj must stay bf16 until the fuse, like in_proj and
    f_a_proj, or it reaches the fuse as a per_Token tail beside a bf16 in_proj."""
    attn = _kda_double(QuantType.No)
    deferred = KimiKDAAttention.get_streaming_deferred_modules(attn)
    assert deferred == (attn.in_proj, attn.b_proj, attn.f_a_proj)


# ── a standalone draft keeps its own static quantization ──────────────────


def _drafter_double(online_raw):
    return SimpleNamespace(
        speculative_config=SimpleNamespace(
            draft_model_hf_config=SimpleNamespace(
                torch_dtype=torch.bfloat16, quantization_config=None
            )
        ),
        config=SimpleNamespace(
            quant_config=SimpleNamespace(online_quant_config_raw=online_raw)
        ),
    )


def test_standalone_draft_does_not_inherit_the_target_static_scheme():
    """A bf16 draft beside an MXFP4 target used to be declared fp4 and fail in
    copy_. Its static spec now comes from its own (unquantized) config."""
    qc = Drafter.standalone_draft_quant_config(_drafter_double(None))
    assert qc.global_spec.quant_type == QuantType.No
    assert not qc.online_quant


def test_standalone_draft_still_gets_the_online_override():
    """Online quant is a property of the run, and the draft was always subject
    to it; only the checkpoint-derived half is re-sourced."""
    online = {"global_quant_config": "ptpc_fp8"}
    qc = Drafter.standalone_draft_quant_config(_drafter_double(online))
    assert qc.online_quant
    assert qc.online_quant_config_raw == online


class _ForeignQuantType:
    """A QuantType member as a differently identified aiter import yields it in
    a plugin process: same value, but unequal (and unhashable-equal) to this
    import's members -- and to each other."""

    def __init__(self, member):
        self.value = member.value
        self.name = member.name

    def __eq__(self, other):
        return self is other

    def __hash__(self):
        return id(self)


def test_fp8_fusion_accepts_quant_types_from_another_aiter_import():
    attn = _kda_double(QuantType.per_Token)
    for proj in (attn.in_proj, attn.b_proj, attn.f_a_proj):
        proj.quant_type = _ForeignQuantType(QuantType.per_Token)  # one each
    want_s = torch.cat(
        [p.weight_scale.data for p in (attn.in_proj, attn.b_proj, attn.f_a_proj)]
    )
    KimiKDAAttention.process_weights_after_loading(attn)
    assert torch.equal(attn.in_proj.weight_scale.data, want_s)
