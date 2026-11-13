# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Loading a Kimi-K3 checkpoint whose attention is statically quantized.

`amd/Kimi-K3-Quark-MXFP4-AttnFP8` ships its self_attn projections (b_proj
included) as per-output-channel FP8 (per_Token). b_proj loads as in_proj's
fifth shard, and the KDA in-proj fusion appends f_a_proj's rows and per-row
scales (`append_linear_rows`, tested in tests/model_ops).

These drive the real methods with stand-ins, as the neighbouring tests do.
"""

from types import SimpleNamespace

import pytest
import torch
from torch import nn

pytest.importorskip("aiter")  # atom.models.kimi_k3 imports aiter at module scope

from aiter import QuantType

from atom.models.kimi_k3.model import KimiKDAAttention, _kda_packed_modules_mapping

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
        bias=None,
        quant_type=quant_type,
        output_size=rows,
        output_partition_sizes=[rows],
    )


def _kda_double(quant_type):
    conv = lambda: SimpleNamespace(weight=torch.randn(LP, 1, CONV_K))
    return SimpleNamespace(
        in_proj=_proj(4 * LP + NLH, quant_type, 0),  # [q | k | v | g | b]
        f_a_proj=_proj(HD, quant_type, 2),
        q_conv1d=conv(),
        k_conv1d=conv(),
        v_conv1d=conv(),
        local_proj_size=LP,
        conv_kernel_size=CONV_K,
    )


@pytest.mark.parametrize("quant_type", [QuantType.No, QuantType.per_Token])
def test_the_fusion_appends_f_a_to_the_loaded_in_proj(quant_type):
    """in_proj already holds [q | k | v | g | b]; the fusion appends f_a, rows
    and (FP8) per-row scales, giving the forward's [... | b | f_a] layout."""
    attn = _kda_double(quant_type)
    parts = [attn.in_proj, attn.f_a_proj]
    want_w = torch.cat([p.weight.data.view(torch.uint8) for p in parts])

    KimiKDAAttention.process_weights_after_loading(attn)

    assert attn.in_proj.weight.shape == (4 * LP + NLH + HD, K)
    assert attn.in_proj.output_size == 4 * LP + NLH + HD
    assert torch.equal(attn.in_proj.weight.data.view(torch.uint8), want_w)
    if quant_type == QuantType.per_Token:
        want_s = torch.cat([p.weight_scale.data for p in parts])
        assert torch.equal(attn.in_proj.weight_scale.data, want_s)
    assert attn.f_a_proj.weight.numel() == 0


def test_b_proj_loads_as_the_fifth_in_proj_shard():
    """b_proj is TP-split like q/k/v/g, so the loader writes it (and its
    per_Token scale) straight into in_proj; nothing is appended for it."""
    mapping = _kda_packed_modules_mapping([1])
    prefix = ".layers.1.self_attn."
    assert mapping[f"{prefix}b_proj"] == (f"{prefix}in_proj", 4)
    assert KimiKDAAttention.get_streaming_deferred_modules(
        SimpleNamespace(in_proj="in", f_a_proj="f_a")
    ) == ("in", "f_a")


def test_glm_maps_b_proj_into_in_proj():
    from atom.models.glm5_next import _KDA_PACKED_MODULES_MAPPING

    assert _KDA_PACKED_MODULES_MAPPING[".b_proj"] == (".in_proj", 4)


def test_glm_refuses_a_checkpoint_without_b_proj(monkeypatch):
    """Once b is a shard of in_proj, the loader's parameter-level report can't
    see it missing; GLM's per-shard coverage check must."""
    import atom.models.glm5_next as glm
    from atom.models.glm5_next import Glm5NextKDAAttention

    monkeypatch.setattr(
        glm, "get_current_atom_config", lambda: SimpleNamespace(load_dummy=False)
    )
    attn = _kda_double(QuantType.No)
    attn._loaded_input_shards = {0, 1, 2}
    with pytest.raises(RuntimeError, match=r"missing checkpoint shards \[4\]"):
        Glm5NextKDAAttention.process_weights_after_loading(attn)


def test_glm_refuses_a_quantized_in_proj_before_its_gate_fold():
    """GLM-5.3's KDA subclass folds a bf16 gate into in_proj by a plain cast;
    with the parent accepting per_Token, the subclass must refuse it rather
    than write unscaled FP8 rows."""
    from atom.models.glm5_next import Glm5NextKDAAttention

    attn = _kda_double(QuantType.per_Token)
    with pytest.raises(NotImplementedError, match="statically quantized in_proj"):
        Glm5NextKDAAttention.process_weights_after_loading(attn)
