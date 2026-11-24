# SPDX-License-Identifier: MIT
"""GLM-5.2 mono reads the attention storage production loaded.

A PTPC deployment keeps its preshuffled FP8 projections; a BF16 checkpoint
(``amd/GLM-5.2-MXFP4`` as shipped) hands the kernel its row-major BF16
projections in place -- a packed copy would pin another ~100 MB a layer -- and
anything else is refused before a kernel is built. CPU-only.
"""

from types import SimpleNamespace

import pytest
import torch

from tests.aiter_stub import stubbed_aiter

with stubbed_aiter():
    from atom.models.glm52.mono.kernels.geometry_config import (
        AttentionWeight,
        glm5_tp_config,
    )
    from atom.models.glm52.mono.weights import (
        attention_format,
        layer_weights,
        projection_weights,
    )
    from atom.mono.runtime.consensus import MonoUnsupported

CFG = glm5_tp_config(4)
FP8 = torch.float8_e4m3fn
SHAPES = {
    "fused_qkv_a_proj": (CFG.qkv_a_rows, CFG.hidden),
    "q_b_proj": (CFG.local_heads * (CFG.nope_dim + CFG.pe_dim), CFG.q_lora),
    "o_proj": (CFG.hidden, CFG.local_heads * CFG.v_dim),
}
# what a BF16 layer with its own indexer folds after q_a / kv_a
INDEX_ROWS = [128, 32]


def _linear(kind, rows, cols):
    if kind == "No":
        return SimpleNamespace(
            quant_type=SimpleNamespace(name="No"),
            params_dtype=torch.bfloat16,
            input_size=cols,
            weight=torch.empty(rows, cols, dtype=torch.bfloat16),
        )
    weight = torch.empty(rows, cols, dtype=FP8)
    weight.is_shuffled = True
    return SimpleNamespace(
        quant_type=SimpleNamespace(name=kind),
        params_dtype=FP8,
        weight=weight,
        weight_scale=torch.ones(rows, 1),
    )


def _layer(kinds=("No", "No", "No"), *, folded_indexer=True):
    split = [CFG.q_lora, CFG.kv_lora, CFG.pe_dim]
    if folded_indexer:
        split += INDEX_ROWS
    rows = {name: shape[0] for name, shape in SHAPES.items()}
    rows["fused_qkv_a_proj"] = sum(split)
    attn = SimpleNamespace(
        qkv_a_split=split,
        mla_attn=SimpleNamespace(W_K_scale=torch.ones(1), W_V_scale=torch.ones(1)),
        **{
            name: _linear(kind, rows[name], SHAPES[name][1])
            for name, kind in zip(SHAPES, kinds)
        },
    )
    return SimpleNamespace(self_attn=attn)


@pytest.mark.parametrize(
    "kind, attention",
    [("No", AttentionWeight.BF16_ABSORB_FP8), ("per_Token", AttentionWeight.FP8_PTPC)],
)
def test_projections_are_read_in_place_in_the_format_production_loaded(kind, attention):
    layer = _layer((kind,) * 3, folded_indexer=kind == "No")
    attn = layer.self_attn
    assert attention_format(layer) is attention
    tensors = projection_weights(attn, CFG, attention)
    for key, name in zip(("qkv_a", "q_b", "o"), SHAPES):
        assert tensors[f"w_{key}"].data_ptr() == getattr(attn, name).weight.data_ptr()
        if kind == "per_Token":
            assert (
                tensors[f"s_{key}"].data_ptr()
                == getattr(attn, name).weight_scale.data_ptr()
            )
    if kind == "No":
        # q_a / kv_a lead the folded indexer's rows
        assert tensors["w_qkv_a"].shape == SHAPES["fused_qkv_a_proj"]


def test_mixed_attention_storage_is_refused():
    with pytest.raises(MonoUnsupported, match="mix formats"):
        attention_format(_layer(kinds=("per_Token", "No", "No")))


def test_block_fp8_attention_storage_is_refused():
    with pytest.raises(MonoUnsupported, match="not PTPC FP8 or BF16"):
        attention_format(_layer(kinds=("per_1x32",) * 3))


def test_bf16_absorbed_attention_storage_is_refused():
    layer = _layer()
    layer.self_attn.mla_attn.W_K_scale = None
    layer.self_attn.mla_attn.W_V_scale = None
    with pytest.raises(MonoUnsupported, match="ATOM_MLA_ABSORB_BF16"):
        attention_format(layer)


def test_a_layer_stored_unlike_the_deployment_is_refused():
    with pytest.raises(MonoUnsupported, match="attention is bf16_absorb_fp8"):
        layer_weights(_layer(), 0, 4, attention=AttentionWeight.FP8_PTPC)


def test_a_preshuffled_bf16_projection_is_refused():
    attn = _layer().self_attn
    attn.o_proj.weight.is_shuffled = True
    with pytest.raises(
        MonoUnsupported, match="o_proj BF16 weight must not be preshuffled"
    ):
        projection_weights(attn, CFG, AttentionWeight.BF16_ABSORB_FP8)


def test_a_fused_qkv_a_not_led_by_q_a_and_kv_a_is_refused():
    attn = _layer().self_attn
    attn.qkv_a_split = [CFG.kv_lora, CFG.q_lora, CFG.pe_dim]
    with pytest.raises(MonoUnsupported, match="fused_qkv_a split"):
        projection_weights(attn, CFG, AttentionWeight.BF16_ABSORB_FP8)
