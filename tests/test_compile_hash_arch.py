# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""The compile cache key must change with the HF architecture, not just its shape."""

import logging
from types import SimpleNamespace

from transformers import PretrainedConfig

from tests.test_config_compile_hash import _hash as _config_hash


def _hash(hf_config):
    # The shared Config stub, so a field added to Config.compute_hash reaches
    # this file too instead of raising AttributeError here.
    return _config_hash(hf_config=hf_config)


def _hf(path="/models/a", **kw):
    fields = {
        "architectures": ["DeepseekV4ForCausalLM"],
        "num_hidden_layers": 61,
        "hidden_size": 7168,
        "compress_ratios": [128, 4, 128],
    }
    cfg = PretrainedConfig(**(fields | kw))
    cfg._name_or_path = path
    return cfg


def _self_ref(cfg):
    # get_hf_config's multimodal text path sets text_config to the config itself.
    cfg.text_config = cfg
    return cfg


def _text_config(**fields):
    # A separate text_config, as a multimodal checkpoint's config carries it.
    return SimpleNamespace(
        architectures=["SomeVLM"], text_config=PretrainedConfig(**fields)
    )


def test_layer_count_keys_the_compile_cache():
    # The reported case: a 61-layer config started over the cache a 4-layer
    # slice of it had just written.
    assert _hash(_hf(num_hidden_layers=4)) != _hash(_hf(num_hidden_layers=61))
    assert _hash(_self_ref(_hf(num_hidden_layers=4))) != _hash(
        _self_ref(_hf(num_hidden_layers=61))
    )


def test_widths_patterns_and_constants_key_the_compile_cache():
    assert _hash(_hf(hidden_size=4096)) != _hash(_hf(hidden_size=7168))
    assert _hash(_hf(compress_ratios=[128, 4])) != _hash(_hf())
    # The same length with a different per-layer layout is the real case: the
    # list decides each layer's attention type, so its order changes the
    # graph even though the length does not.
    assert _hash(_hf(compress_ratios=[128, 4, 128])) != _hash(
        _hf(compress_ratios=[4, 128, 128])
    )
    assert _hash(_hf(compress_ratios=[128, 4, 128])) == _hash(
        _hf(compress_ratios=[128, 4, 128])
    )
    # Same shapes, but the graph bakes the value in as a constant.
    assert _hash(_hf(routed_scaling_factor=1.5)) != _hash(
        _hf(routed_scaling_factor=2.5)
    )


def test_index_cache_fields_key_the_compile_cache():
    # The four index-cache fields key the cache from hf_config and from a
    # separate text_config, where a multimodal checkpoint carries them.
    variants = {
        "use_index_cache": (False, True),
        "index_topk_freq": (1, 4),
        "index_topk_pattern": (["S", "D"], ["D", "S"]),
        "index_skip_topk_offset": (1, 2),
    }
    for name, (left, right) in variants.items():
        assert _hash(_hf(**{name: left})) != _hash(_hf(**{name: right})), name
        assert _hash(_text_config(**{name: left})) != _hash(
            _text_config(**{name: right})
        ), name


def _yarn(factor=32.0, mscale_all_dim=0):
    # transformers 5 validates YaRN and needs original_max_position_embeddings
    # inside the dict; keep it consistent with max_position_embeddings so the
    # config builds quietly.
    return _hf(
        max_position_embeddings=8192,
        rope_parameters={
            "rope_type": "yarn",
            "factor": float(factor),
            "mscale_all_dim": mscale_all_dim,
            "original_max_position_embeddings": int(8192 / factor),
        },
    )


def test_rope_dict_keys_the_compile_cache():
    # The mscale becomes a graph constant (see _hf_dict_entries); a key that
    # missed the dict ran the other checkpoint's attention scale to
    # completion, with no error.
    assert _hash(_yarn(factor=32.0)) != _hash(_yarn(factor=16.0))
    assert _hash(_yarn(mscale_all_dim=0)) != _hash(_yarn(mscale_all_dim=1))
    # Same values, same key: keying a dict must not make the factor unstable.
    assert _hash(_yarn()) == _hash(_yarn())
    # The value ATOM writes onto hf_config in __post_init__ is a plain dict on
    # a plain config too.
    a = SimpleNamespace(rope_parameters={"factor": 32.0, "mscale_all_dim": 0})
    b = SimpleNamespace(rope_parameters={"factor": 32.0, "mscale_all_dim": 1})
    assert _hash(a) != _hash(b)
    # Keying a dict must be order-independent: the same entries inserted in a
    # different order are the same config and must not force a recompile.
    c = SimpleNamespace(rope_parameters={"mscale_all_dim": 0, "factor": 32.0})
    assert _hash(a) == _hash(c)


def test_quantization_config_keys_the_compile_cache():
    # The stub leaves quant_config unset, so this keys through the one-level
    # dict rule; in production quant_config.compute_hash() covers it as well.
    base = {"quant_method": "fp8", "fmt": "e4m3"}
    assert _hash(_hf(quantization_config=base)) != _hash(
        _hf(quantization_config={"quant_method": "fp8", "fmt": "e5m2"})
    )
    assert _hash(_hf(quantization_config=base)) != _hash(
        _hf(quantization_config={"quant_method": "mxfp4"})
    )
    assert _hash(_hf(quantization_config=base)) == _hash(
        _hf(quantization_config=dict(base))
    )


def test_plain_dict_text_config_keys_the_compile_cache():
    # PretrainedConfig.from_dict leaves a nested ``text_config`` mapping as a
    # dict; it must be read like a config object, not silently skipped.
    def mm(**fields):
        return SimpleNamespace(architectures=["SomeVLM"], text_config=fields)

    assert _hash(mm(num_hidden_layers=4)) != _hash(mm(num_hidden_layers=61))
    assert _hash(mm(rope_parameters={"factor": 32.0})) != _hash(
        mm(rope_parameters={"factor": 16.0})
    )
    assert _hash(mm(num_hidden_layers=61)) == _hash(mm(num_hidden_layers=61))


def test_architecture_factor_is_logged_at_debug(caplog):
    # hf_config gains attributes in several places (rope_parameters, dspark_*,
    # an expert_dtype probed from disk). Now that they move the key, the debug
    # log is the only way to see why a cache stopped hitting.
    with caplog.at_level(logging.DEBUG, logger="atom"):
        _hash(_hf())
    assert any(
        "HF architecture factor" in record.getMessage() for record in caplog.records
    )


def test_multimodal_text_config_keys_the_compile_cache():
    def mm(layers):
        return SimpleNamespace(
            architectures=["SomeVLM"], text_config=_hf(num_hidden_layers=layers)
        )

    assert _hash(mm(4)) != _hash(mm(61))

    # A wrapper field the text config does not carry still keys it.
    def wrapper(layers):
        return SimpleNamespace(text_config=SimpleNamespace(), num_hidden_layers=layers)

    assert _hash(wrapper(4)) != _hash(wrapper(61))


def test_same_model_hashes_identically():
    # Guard against keying on something unstable, which would never cache:
    # the checkpoint path, a sub-config object or a dict of unstable reprs.
    a = _hf(path="/mnt/x/model", quantization_config={"quant_method": "fp8"})
    b = _hf(path="/data/model", quantization_config={"quant_method": "fp8"})
    _self_ref(a)
    _self_ref(b)
    a.some_module = object()
    b.some_module = object()
    assert _hash(a) == _hash(b)
    assert _hash(_hf()) == _hash(_hf())
