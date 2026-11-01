# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Loading a Kimi-K3 checkpoint whose attention is statically quantized.

`amd/Kimi-K3-Quark-MXFP4-AttnFP8` ships its self_attn projections (b_proj
included) as per-output-channel FP8 (per_Token): the KDA in-proj fusion
concatenates their rows and per-row scales, and a standalone bf16 draft beside
that MXFP4 target takes its static quant spec from its own checkpoint.

These drive the real methods with stand-ins, as the neighbouring tests do.
"""

from types import SimpleNamespace

import pytest
import torch
from torch import nn

pytest.importorskip("aiter")  # atom.models.kimi_k3 imports aiter at module scope

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
    # the tails are released, scales included; the modules stay as empty shells
    for tail in (attn.b_proj, attn.f_a_proj):
        assert tail.weight.numel() == 0 and tail.weight_scale.numel() == 0


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
    with pytest.raises(NotImplementedError, match="one scheme"):
        KimiKDAAttention.process_weights_after_loading(attn)


def test_an_unsupported_scheme_is_refused():
    attn = _kda_double(QuantType.per_1x128)
    with pytest.raises(NotImplementedError, match="per_Token"):
        KimiKDAAttention.process_weights_after_loading(attn)


def test_a_missing_per_token_scale_is_reported_by_the_guard():
    """The guard, not an AttributeError on `.data`, reports a missing scale."""
    attn = _kda_double(QuantType.per_Token)
    attn.b_proj.weight_scale = None
    with pytest.raises(ValueError, match="one row per output channel"):
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


DRAFT_PATH = "/ckpt/draft"


def _drafter_double(online):
    """What Drafter._standalone_draft_config reads from a proposer."""
    target_qc = SimpleNamespace(online_quant_config_raw={"stale": "target copy"})
    return SimpleNamespace(
        speculative_config=SimpleNamespace(
            draft_model_hf_config=SimpleNamespace(
                torch_dtype=torch.bfloat16, quantization_config=None
            ),
            model=DRAFT_PATH,
        ),
        config=SimpleNamespace(
            hf_config=SimpleNamespace(name="target"),
            quant_config=target_qc,
            online_quant_config=online,
            compilation_config=SimpleNamespace(level=3),
        ),
    )


def test_standalone_draft_does_not_inherit_the_target_static_scheme():
    """A bf16 draft beside an MXFP4 target is built unquantized, from its own
    config, with compilation off and the target config untouched."""
    drafter = _drafter_double(None)
    cfg = Drafter._standalone_draft_config(drafter)
    assert cfg.hf_config is drafter.speculative_config.draft_model_hf_config
    assert cfg.quant_config.global_spec.quant_type == QuantType.No
    assert not cfg.quant_config.online_quant
    assert cfg.compilation_config.level != 3
    assert drafter.config.compilation_config.level == 3


def test_standalone_draft_gets_the_runs_online_override():
    """The online override is the run's (Config.online_quant_config), not a
    copy kept on the target's quant config."""
    online = {"global_quant_config": "ptpc_fp8"}
    cfg = Drafter._standalone_draft_config(_drafter_double(online))
    assert cfg.quant_config.online_quant
    assert cfg.quant_config.online_quant_config_raw == online


def test_standalone_draft_quant_config_probes_the_draft_checkpoint(monkeypatch):
    """model= is the draft's path, so layers it stores in bf16 can fall back."""
    import atom.config as config_mod

    seen = []
    monkeypatch.setattr(
        config_mod,
        "unquantized_layer_keys",
        lambda model: seen.append(model) or frozenset(),
    )
    Drafter._standalone_draft_config(_drafter_double(None))
    assert seen == [DRAFT_PATH]


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
