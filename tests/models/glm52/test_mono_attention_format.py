# SPDX-License-Identifier: MIT
"""GLM-5.2 mono reads the attention storage production loaded.

A PTPC deployment keeps its preshuffled FP8 projections; a BF16 checkpoint
(``amd/GLM-5.2-MXFP4`` as shipped) hands the kernel its row-major BF16
projections in place -- a packed copy would pin another ~100 MB a layer -- and
anything else is refused before a kernel is built. CPU-only.
"""

from types import SimpleNamespace

import pytest

pytest.importorskip("aiter", reason="needs the AITER GPU kernel library")
import torch

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


def _layer(kinds=("No", "No", "No"), *, folded_indexer=True, absorb_fp8=True):
    split = [CFG.q_lora, CFG.kv_lora, CFG.pe_dim]
    if folded_indexer:
        split += INDEX_ROWS
    rows = {name: shape[0] for name, shape in SHAPES.items()}
    rows["fused_qkv_a_proj"] = sum(split)
    scale = torch.ones(1) if absorb_fp8 else None
    attn = SimpleNamespace(
        qkv_a_split=split,
        mla_attn=SimpleNamespace(W_K_scale=scale, W_V_scale=scale),
        **{
            name: _linear(kind, rows[name], SHAPES[name][1])
            for name, kind in zip(SHAPES, kinds)
        },
    )
    return SimpleNamespace(self_attn=attn)


def test_bf16_checkpoint_maps_production_storage_in_place():
    layer = _layer()
    attn = layer.self_attn
    assert attention_format(layer) is AttentionWeight.BF16_ABSORB_FP8
    tensors = projection_weights(attn, CFG, AttentionWeight.BF16_ABSORB_FP8)
    assert set(tensors) == {"w_qkv_a", "w_q_b", "w_o"}
    for key, name in zip(("qkv_a", "q_b", "o"), SHAPES):
        weight = getattr(attn, name).weight
        assert tensors[f"w_{key}"].data_ptr() == weight.data_ptr()
        assert tensors[f"w_{key}"].shape == weight.shape
    # the folded indexer rows trail q_a / kv_a; the kernel reads the first
    # `qkv_a_rows` of them
    assert tensors["w_qkv_a"].shape[0] == CFG.qkv_a_rows + sum(INDEX_ROWS)


def test_ptpc_checkpoint_keeps_its_mapping():
    layer = _layer(("per_Token",) * 3, folded_indexer=False)
    attn = layer.self_attn
    assert attention_format(layer) is AttentionWeight.FP8_PTPC
    tensors = projection_weights(attn, CFG, AttentionWeight.FP8_PTPC)
    for key, name in zip(("qkv_a", "q_b", "o"), SHAPES):
        linear = getattr(attn, name)
        assert tensors[f"w_{key}"].data_ptr() == linear.weight.data_ptr()
        assert tensors[f"s_{key}"].shape == (linear.weight.shape[0],)


@pytest.mark.parametrize(
    "storage, why",
    [
        ({"kinds": ("per_Token", "No", "No")}, "mix formats"),
        ({"kinds": ("per_1x32",) * 3}, "not PTPC FP8 or BF16"),
        ({"absorb_fp8": False}, "ATOM_MLA_ABSORB_BF16"),
    ],
)
def test_other_attention_storage_is_refused(storage, why):
    with pytest.raises(MonoUnsupported, match=why):
        attention_format(_layer(**storage))


def test_a_layer_stored_unlike_the_deployment_is_refused():
    with pytest.raises(MonoUnsupported, match="attention is bf16_absorb_fp8"):
        layer_weights(_layer(), 0, 4, attention=AttentionWeight.FP8_PTPC)


def test_bf16_storage_the_kernel_cannot_read_in_place_is_refused():
    layer = _layer()
    attn = layer.self_attn
    attn.o_proj.weight.is_shuffled = True
    with pytest.raises(MonoUnsupported, match="o_proj BF16 weight must be row-major"):
        projection_weights(attn, CFG, AttentionWeight.BF16_ABSORB_FP8)
    attn.o_proj.weight.is_shuffled = False
    attn.qkv_a_split = [CFG.kv_lora, CFG.q_lora, CFG.pe_dim]
    with pytest.raises(MonoUnsupported, match="fused_qkv_a split"):
        projection_weights(attn, CFG, AttentionWeight.BF16_ABSORB_FP8)
