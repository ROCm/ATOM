# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import logging
from types import SimpleNamespace

from transformers import PretrainedConfig

from atom.config import Config


def _hash(hf_config):
    # Only the fields compute_hash reads; quant/compilation/parallel are falsy
    # so their sub-hashes are skipped.
    cfg = SimpleNamespace(
        quant_config=None,
        compilation_config=None,
        parallel_config=None,
        tensor_parallel_size=1,
        max_model_len=8192,
        max_num_seqs=512,
        prefill_context_parallel_size=1,
        dcp_config=None,
        enable_dp_attention=False,
        index_cache_dtype="fp8",
        enable_decoder_swa_bounded_replay=True,
        hf_config=hf_config,
        speculative_config=None,
    )
    return Config.compute_hash(cfg)


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


def test_layer_count_keys_the_compile_cache():
    # A 61-layer config started over the cache a 4-layer slice of it had just
    # written unpacked the wrong number of graph inputs and crashed.
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


def _yarn(factor=32.0, mscale_all_dim=0):
    # transformers 5 validates YaRN and derives the implicit factor from
    # max_position_embeddings / original_max_position_embeddings; keep the two
    # consistent so the config builds quietly. It also normalizes the
    # ``rope_scaling`` keyword into the ``rope_parameters`` dict ATOM reads.
    return _hf(
        max_position_embeddings=8192,
        rope_scaling={
            "type": "yarn",
            "factor": float(factor),
            "mscale_all_dim": mscale_all_dim,
            "original_max_position_embeddings": int(8192 / factor),
        },
    )


def test_rope_dict_keys_the_compile_cache():
    # deepseek_v2 folds the mscale derived from rope_parameters into
    # ``self.scaling``, a Python float the graph bakes in as a constant. A key
    # that missed the dict would run the other checkpoint's attention scale to
    # completion -- silent wrong numerics, not the crash the layer-count case
    # gives.
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


def test_unexpected_text_config_warns(caplog):
    # The helper must not fall back to an empty half without saying so.
    hf_config = SimpleNamespace(architectures=["SomeVLM"], text_config=object())
    with caplog.at_level(logging.WARNING, logger="atom"):
        _hash(hf_config)
    assert any(
        "text_config" in record.getMessage() and "object" in record.getMessage()
        for record in caplog.records
    )


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
