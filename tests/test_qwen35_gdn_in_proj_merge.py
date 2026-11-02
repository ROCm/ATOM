"""Qwen3.5 GDN input projection: when `in_proj_qkvz` + `in_proj_ba` merge into one layer.

The layer construction and the checkpoint mapping must make the same decision --
if they disagree, `in_proj_b` / `in_proj_a` are loaded into a module that was
never built, or the merged weight keeps two uninitialised shards. Both ask
`gdn_in_proj_is_merged`, so these tests pin that one function and the mapping
built from it.
"""

import enum
import sys
from types import ModuleType

import pytest
import torch

from atom import quant_spec
from atom.config import QuantizationConfig
from atom.models.qwen3_5_gdn_in_proj import (
    _QWEN3_5_PACKED_MODULES_MAPPING,
    _apply_merged_in_proj_mapping,
    gdn_in_proj_is_merged,
)
from atom.quant_spec import LayerQuantConfig, QuantType


class _FakeQuantType(enum.IntEnum):
    No = 0
    per_Token = 1
    per_Tensor = 2
    per_1x32 = 3
    per_1x128 = 4


@pytest.fixture(autouse=True)
def _quant_type_without_aiter(monkeypatch):
    """`atom.quant_spec.QuantType` resolves `aiter.QuantType` on first use; the
    CPU CI has no AITER, so stand in for that one enum there (and forget it
    afterwards). With AITER installed the real enum is used."""
    try:
        quant_spec.QuantType.No  # noqa: B018 - resolves the lazy AITER name
    except Exception:  # noqa: BLE001 - no AITER, or AITER without a GPU
        fake = ModuleType("aiter")
        fake.QuantType = _FakeQuantType
        monkeypatch.setitem(sys.modules, "aiter", fake)
        monkeypatch.setattr(quant_spec.QuantType, "_value", None)


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


def bf16():
    return LayerQuantConfig(quant_type=QuantType.No, quant_dtype=torch.bfloat16)


def per_token():
    return _spec(QuantType.per_Token)


def per_1x128():
    return _spec(QuantType.per_1x128, block=(128, 128))


def test_bf16_checkpoint_merges_and_ignores_the_env(monkeypatch):
    monkeypatch.setenv("ATOM_QWEN35_GDN_MERGE_QKVZBA", "0")
    assert gdn_in_proj_is_merged(_Cfg(_Quant(bf16())))


def test_per_token_fp8_merges():
    assert gdn_in_proj_is_merged(_Cfg(_Quant(per_token())))


def test_block_scale_never_merges():
    # in_proj_ba is 32 columns per rank at TP4, not a whole number of 128-column
    # scale blocks, so concatenating it onto in_proj_qkvz would misalign them.
    assert not gdn_in_proj_is_merged(_Cfg(_Quant(per_1x128())))


def test_env_off_keeps_the_two_fp8_layers(monkeypatch):
    monkeypatch.setenv("ATOM_QWEN35_GDN_MERGE_QKVZBA", "0")
    assert not gdn_in_proj_is_merged(_Cfg(_Quant(per_token())))


def test_mismatched_specs_do_not_merge():
    quant = _Quant(per_token(), per_layer={"in_proj_ba": per_1x128()})
    assert not gdn_in_proj_is_merged(_Cfg(quant))


def test_model_without_a_gdn_layer_does_not_merge():
    cfg = _Cfg(_Quant(per_token()), layer_types=("full_attention",))
    assert not gdn_in_proj_is_merged(cfg)


def test_mapping_follows_the_decision():
    merged = _apply_merged_in_proj_mapping(
        dict(_QWEN3_5_PACKED_MODULES_MAPPING), _Cfg(_Quant(per_token()))
    )
    assert merged["in_proj_qkv"] == ("in_proj_qkvzba", (0, 1, 2))
    assert merged["in_proj_z"] == ("in_proj_qkvzba", 3)
    assert merged["in_proj_b"] == ("in_proj_qkvzba", 4)
    assert merged["in_proj_a"] == ("in_proj_qkvzba", 5)

    split = _apply_merged_in_proj_mapping(
        dict(_QWEN3_5_PACKED_MODULES_MAPPING), _Cfg(_Quant(per_1x128()))
    )
    assert split["in_proj_qkv"] == ("in_proj_qkvz", (0, 1, 2))
    assert split["in_proj_b"] == ("in_proj_ba", 0)
    assert split["in_proj_a"] == ("in_proj_ba", 1)


def test_mapping_does_not_mutate_the_module_level_dict():
    _apply_merged_in_proj_mapping(
        dict(_QWEN3_5_PACKED_MODULES_MAPPING), _Cfg(_Quant(per_token()))
    )
    assert _QWEN3_5_PACKED_MODULES_MAPPING["in_proj_b"] == ("in_proj_ba", 0)


# The checks below use the real `QuantizationConfig` lookup (exact names,
# fnmatch patterns, excludes), remapped through the model's split mapping
# before construction, as `ModelRunner` does.

GDN_LAYERS = ("linear_attention", "linear_attention", "linear_attention")


def _real_quant(global_spec, per_layer=(), exclude=()):
    quant = QuantizationConfig()
    quant.global_spec = global_spec
    quant.layer_pattern_specs = list(per_layer)
    quant.exclude_layers = list(exclude)
    hf_config = type("HF", (), {"model_type": "qwen3_5_moe"})()
    quant.remap_layer_name(
        hf_config, packed_modules_mapping=dict(_QWEN3_5_PACKED_MODULES_MAPPING)
    )
    return quant


def test_real_lookup_per_token_merges():
    assert gdn_in_proj_is_merged(_Cfg(_real_quant(per_token()), GDN_LAYERS))


def test_a_later_layer_that_does_not_qualify_keeps_all_split():
    # Layer 2's in_proj_b / in_proj_a are stored in bf16(); layer 1 alone would
    # say "merge", but the checkpoint mapping is model-wide.
    exclude = [
        "model.layers.2.linear_attn.in_proj_b",
        "model.layers.2.linear_attn.in_proj_a",
    ]
    quant = _real_quant(per_token(), exclude=exclude)
    assert (
        quant.get_layer_quant_config("model.layers.0.linear_attn.in_proj_ba")
        == per_token()
    )
    assert not gdn_in_proj_is_merged(_Cfg(quant, GDN_LAYERS))


def test_per_token_overrides_on_a_block_scale_default_do_not_merge():
    # Per-projection per-token overrides agree for in_proj_qkvz and in_proj_ba,
    # but the merged layer's own name falls through to the block-scale default,
    # so a merged layer would allocate the wrong scale layout.
    per_layer = [
        (f"model.layers.{i}.linear_attn.{proj}", per_token())
        for i in range(3)
        for proj in ("in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a")
    ]
    quant = _real_quant(per_1x128(), per_layer=per_layer)
    prefix = "model.layers.0.linear_attn"
    assert quant.get_layer_quant_config(f"{prefix}.in_proj_qkvz") == per_token()
    assert quant.get_layer_quant_config(f"{prefix}.in_proj_ba") == per_token()
    assert quant.get_layer_quant_config(f"{prefix}.in_proj_qkvzba") == per_1x128()
    assert not gdn_in_proj_is_merged(_Cfg(quant, GDN_LAYERS))


def test_pattern_specs_covering_the_merged_name_merge():
    quant = _real_quant(per_1x128(), per_layer=[("*linear_attn*", per_token())])
    assert gdn_in_proj_is_merged(_Cfg(quant, GDN_LAYERS))
