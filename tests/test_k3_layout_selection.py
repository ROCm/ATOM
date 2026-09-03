# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Layout selection for the Kimi-K3 offload family (pure python, no GPU)."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from atom.kv_transfer.offload import config as offcfg


def _config(hf_config, **kv_transfer_config):
    return SimpleNamespace(
        model="org/model",
        model_tag="org/model",
        kv_cache_dtype="fp8",
        index_cache_dtype="auto",
        kv_cache_block_size=64,
        tensor_parallel_size=1,
        decode_context_parallel_size=1,
        speculative_config=None,
        kv_transfer_config=dict(kv_transfer_config),
        hf_config=hf_config,
    )


def _plain_hf(**overrides):
    fields = {
        "model_type": "deepseek_v3",
        "num_hidden_layers": 8,
        "num_attention_heads": 16,
        "num_key_value_heads": 16,
        "hidden_size": 1024,
        "head_dim": 64,
    }
    fields.update(overrides)
    return SimpleNamespace(**fields)


def test_kimi_linear_selects_kimi_k3():
    config = _config(_plain_hf(model_type="kimi_linear"))

    assert offcfg.select_offload_layout(config) == "kimi_k3"


def test_compress_ratios_still_selects_hybrid():
    config = _config(_plain_hf(compress_ratios=[4, 128, 0]))

    assert offcfg.select_offload_layout(config) == "hybrid"


def test_plain_config_still_selects_dense():
    assert offcfg.select_offload_layout(_config(_plain_hf())) == "dense"


def test_explicit_override_wins_over_sniffing():
    # A dense-looking config forced onto the K3 layout, and a K3 config forced
    # back to dense: the override is consulted before any hf sniffing.
    forced_k3 = _config(_plain_hf(), offload_layout="kimi_k3")
    forced_dense = _config(_plain_hf(model_type="kimi_linear"), offload_layout="dense")

    assert offcfg.select_offload_layout(forced_k3) == "kimi_k3"
    assert offcfg.select_offload_layout(forced_dense) == "dense"


def test_unknown_override_raises_value_error():
    config = _config(_plain_hf(), offload_layout="kimi_k4")

    with pytest.raises(ValueError, match="unknown offload_layout"):
        offcfg.select_offload_layout(config)


def test_nested_text_config_selects_kimi_k3():
    # Plugin-mode shape: the outer config's model_type is the wrapper's, and
    # only the text sub-config names kimi_linear.
    hf_config = _plain_hf(
        model_type="kimi_k3_wrapper",
        text_config=_plain_hf(model_type="kimi_linear"),
    )

    assert offcfg.select_offload_layout(_config(hf_config)) == "kimi_k3"


def test_text_config_helper_failure_falls_back_to_bare_attribute(monkeypatch):
    import atom.utils

    def _boom(_config):
        raise RuntimeError("no transformers here")

    monkeypatch.setattr(atom.utils, "get_hf_text_config", _boom)

    config = _config(_plain_hf(model_type="kimi_linear"))
    assert offcfg.select_offload_layout(config) == "kimi_k3"


def test_page_namespace_puts_kimi_k3_in_the_dense_namespace():
    lmcache_cfg = SimpleNamespace(chunk_size=256)
    k3 = offcfg.build_page_namespace(
        _config(_plain_hf(model_type="kimi_linear")), lmcache_cfg, 1
    )
    dense = offcfg.build_page_namespace(
        _config(_plain_hf(model_type="kimi_linear"), offload_layout="dense"),
        lmcache_cfg,
        1,
    )
    dsv4 = offcfg.build_page_namespace(
        _config(_plain_hf(model_type="kimi_linear"), offload_layout="hybrid"),
        lmcache_cfg,
        1,
    )

    # Same hf geometry + same page_mode => byte-identical namespace as dense.
    assert k3 == dense
    assert k3 != dsv4
