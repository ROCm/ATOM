"""Qwen3.5 GDN input projection: when `in_proj_qkvz` + `in_proj_ba` merge into one layer.

The layer construction and the checkpoint mapping must make the same decision --
if they disagree, `in_proj_b` / `in_proj_a` are loaded into a module that was
never built, or the merged weight keeps two uninitialised shards. Both ask
`gdn_in_proj_is_merged`, so these tests pin that one function and the mapping
built from it.
"""

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("aiter")

from atom.models.qwen3_5 import (  # noqa: E402
    _QWEN3_5_PACKED_MODULES_MAPPING,
    _apply_merged_in_proj_mapping,
    gdn_in_proj_is_merged,
)
from atom.quant_spec import LayerQuantConfig, QuantType  # noqa: E402


class _Cfg:
    """Stand-in for the parts of `Config` that the decision reads."""

    def __init__(
        self, quant_config, layer_types=("full_attention", "linear_attention")
    ):
        self.hf_config = type("HF", (), {"layer_types": list(layer_types)})()
        self.quant_config = quant_config


class _Quant:
    """Stand-in for `QuantizationConfig` with a global spec and per-layer specs."""

    def __init__(self, global_spec, per_layer=None):
        self.global_quant_config = global_spec
        self._per_layer = per_layer or {}

    def get_layer_quant_config(self, name, **kwargs):
        for suffix, spec in self._per_layer.items():
            if name.endswith(suffix):
                return spec
        return self.global_quant_config


def _spec(quant_type, dtype=torch.float8_e4m3fn, block=None):
    return LayerQuantConfig(
        quant_type=quant_type, quant_dtype=dtype, weight_block_size=block
    )


BF16 = LayerQuantConfig(quant_type=QuantType.No, quant_dtype=torch.bfloat16)
PER_TOKEN = _spec(QuantType.per_Token)
PER_1X128 = _spec(QuantType.per_1x128, block=(128, 128))


def test_bf16_checkpoint_merges_and_ignores_the_env(monkeypatch):
    monkeypatch.setenv("ATOM_QWEN35_GDN_MERGE_QKVZBA", "0")
    assert gdn_in_proj_is_merged(_Cfg(_Quant(BF16)))


def test_per_token_fp8_merges():
    assert gdn_in_proj_is_merged(_Cfg(_Quant(PER_TOKEN)))


def test_block_scale_never_merges():
    # in_proj_ba is 32 columns per rank at TP4, not a whole number of 128-column
    # scale blocks, so concatenating it onto in_proj_qkvz would misalign them.
    assert not gdn_in_proj_is_merged(_Cfg(_Quant(PER_1X128)))


def test_env_off_keeps_the_two_fp8_layers(monkeypatch):
    monkeypatch.setenv("ATOM_QWEN35_GDN_MERGE_QKVZBA", "0")
    assert not gdn_in_proj_is_merged(_Cfg(_Quant(PER_TOKEN)))


def test_mismatched_specs_do_not_merge():
    quant = _Quant(PER_TOKEN, per_layer={"in_proj_ba": PER_1X128})
    assert not gdn_in_proj_is_merged(_Cfg(quant))


def test_model_without_a_gdn_layer_does_not_merge():
    cfg = _Cfg(_Quant(PER_TOKEN), layer_types=("full_attention",))
    assert not gdn_in_proj_is_merged(cfg)


def test_mapping_follows_the_decision():
    merged = _apply_merged_in_proj_mapping(
        dict(_QWEN3_5_PACKED_MODULES_MAPPING), _Cfg(_Quant(PER_TOKEN))
    )
    assert merged["in_proj_qkv"] == ("in_proj_qkvzba", (0, 1, 2))
    assert merged["in_proj_z"] == ("in_proj_qkvzba", 3)
    assert merged["in_proj_b"] == ("in_proj_qkvzba", 4)
    assert merged["in_proj_a"] == ("in_proj_qkvzba", 5)

    split = _apply_merged_in_proj_mapping(
        dict(_QWEN3_5_PACKED_MODULES_MAPPING), _Cfg(_Quant(PER_1X128))
    )
    assert split["in_proj_qkv"] == ("in_proj_qkvz", (0, 1, 2))
    assert split["in_proj_b"] == ("in_proj_ba", 0)
    assert split["in_proj_a"] == ("in_proj_ba", 1)


def test_mapping_does_not_mutate_the_module_level_dict():
    _apply_merged_in_proj_mapping(
        dict(_QWEN3_5_PACKED_MODULES_MAPPING), _Cfg(_Quant(PER_TOKEN))
    )
    assert _QWEN3_5_PACKED_MODULES_MAPPING["in_proj_b"] == ("in_proj_ba", 0)
