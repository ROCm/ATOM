# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""GLM-5.3-Flash on the vllm-atom backend: MLA width, geometry, and guards.

CPU-only. The config fixture mirrors the real checkpoint's shape rather than
reading the 328 GB model, so this runs anywhere.
"""

import pytest

pytest.importorskip("vllm")

from types import SimpleNamespace

import torch
from vllm.transformers_utils.configs.glm5_next import Glm5NextConfig

from atom.plugin.vllm.glm5_next_mla_head_size import patch_vllm_mla_head_size
from atom.plugin.vllm.kimi_k3_kda_autotune_patch import (
    apply_vllm_kimi_k3_kda_autotune_patch,
)


@pytest.fixture(autouse=True, scope="module")
def _kimi_k3_kda_autotune_patch():
    """The plugin model reuses Kimi-K3's KDA adapter, which imports vLLM's
    Kimi-K3 kernels. A server installs this Triton fix in register_model before
    that import; these tests import the model directly, so install it here too
    -- and put `triton.autotune` back afterwards, since the patch's own tests
    assert that they are the ones installing it."""
    import triton

    original = triton.autotune
    apply_vllm_kimi_k3_kda_autotune_patch()
    yield
    triton.autotune = original


def _text_config_dict(index_kpool: int = 4) -> dict:
    # 45 layers, KDA everywhere except the 11 "full" MLA layers, as shipped.
    full = [3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43]
    return {
        "model_type": "glm5_next_text",
        "num_hidden_layers": 45,
        "hidden_size": 4096,
        "num_attention_heads": 64,
        "kv_lora_rank": 512,
        "qk_nope_head_dim": 256,
        "qk_rope_head_dim": 0,
        "v_head_dim": 256,
        "index_topk": 2048,
        "index_kpool": index_kpool,
        "index_head_dim": 128,
        "first_k_dense_replace": 1,
        "n_routed_experts": 288,
        "num_experts_per_tok": 8,
        "layer_types": [
            "deepseek_sparse_attention" if i in full else "linear_attention"
            for i in range(45)
        ],
        "linear_attn_config": {
            "num_heads": 64,
            "head_dim": 128,
            "short_conv_kernel_size": 4,
            "kda_layers": [i for i in range(45) if i not in full],
            "full_attn_layers": full,
        },
    }


def _config(index_kpool: int = 4) -> Glm5NextConfig:
    return Glm5NextConfig(text_config=_text_config_dict(index_kpool))


def _vllm_config(
    cfg,
    *,
    pcp=1,
    dcp=1,
    spec=None,
    block_size=1024,
    prefix_caching=False,
    kv_transfer=None,
) -> SimpleNamespace:
    # _check_unsupported / the shape helpers only read attributes.
    return SimpleNamespace(
        model_config=SimpleNamespace(hf_config=cfg, dtype=torch.bfloat16),
        parallel_config=SimpleNamespace(
            tensor_parallel_size=8,
            prefill_context_parallel_size=pcp,
            decode_context_parallel_size=dcp,
        ),
        cache_config=SimpleNamespace(
            mamba_cache_dtype="auto",
            block_size=block_size,
            enable_prefix_caching=prefix_caching,
        ),
        speculative_config=spec,
        kv_transfer_config=kv_transfer,
    )


def test_mla_head_size_is_the_nope_padded_entry():
    """vLLM's own answer is kv_lora_rank + qk_rope_head_dim = 512, because the
    checkpoint keeps the rope dim at 0; ATOM's MLA entry carries the 64-wide
    zero rope pad, so the cache must be sized at 576 or the first reshape fails.
    """
    # Patch first: it imports vllm.config, which the convertor module needs
    # loaded before it can be imported (the two import each other).
    patch_vllm_mla_head_size()
    from vllm.transformers_utils.model_arch_config_convertor import (
        ModelArchConfigConvertorBase,
    )

    cfg = _config()
    convertor = ModelArchConfigConvertorBase(cfg, cfg.get_text_config())
    assert convertor.is_deepseek_mla()
    assert convertor.get_head_size() == 576


def test_mla_head_size_patch_leaves_other_models_alone():
    from transformers import PretrainedConfig

    patch_vllm_mla_head_size()
    from vllm.transformers_utils.model_arch_config_convertor import (
        ModelArchConfigConvertorBase,
    )

    other = PretrainedConfig(
        model_type="glm_moe_dsa", kv_lora_rank=512, qk_rope_head_dim=64
    )
    convertor = ModelArchConfigConvertorBase(other, other)
    assert convertor.get_head_size() == 576  # vLLM's sum, not the patch


def test_text_config_fields_survive_verbatim():
    # ATOM reads these straight off vLLM's config object; a field vLLM's class
    # renamed or defaulted would silently disagree with the checkpoint.
    t = _config().get_text_config()
    assert t.num_hidden_layers == 45
    assert t.index_topk == 2048
    assert t.index_kpool == 4
    assert isinstance(t.linear_attn_config, dict)


def test_state_shape_matches_glms_kda_geometry():
    """The recurrent state vLLM allocates has to match GLM's KDA dims.

    Wrong dims here do not fail loudly -- vLLM would allocate a state of the
    wrong width and the KDA kernels would read it as if it were right.
    """
    from atom.plugin.vllm.models.glm5_next import _get_glm5_state_shape

    tp = 8
    heads_per_rank = 64 // tp
    conv_shape, state_shape = _get_glm5_state_shape(_vllm_config(_config()))

    # conv: (kernel - 1, q|k|v packed across the rank's heads)
    assert conv_shape == (4 - 1, 3 * heads_per_rank * 128)
    # recurrent state: one (head_k x head_v) matrix per head on this rank
    assert state_shape == (heads_per_rank, 128, 128)


def test_state_shape_scales_with_tensor_parallel():
    from atom.plugin.vllm.models.glm5_next import _get_glm5_state_shape

    vc = _vllm_config(_config())
    vc.parallel_config.tensor_parallel_size = 4
    conv_shape, state_shape = _get_glm5_state_shape(vc)
    assert state_shape == (64 // 4, 128, 128)
    assert conv_shape == (3, 3 * (64 // 4) * 128)


def test_kda_layer_count_matches_the_shipped_split():
    from atom.models.glm5_next import _normalize_glm5_next_config

    t = _config().get_text_config()
    _normalize_glm5_next_config(t)
    assert len(t.glm5_kda_layers) == 34
    assert len(t.glm5_full_attn_layers) == 11
    # 0-based, unlike Kimi's 1-based list -- layer 0 really is KDA.
    assert 0 in t.glm5_kda_layers


@pytest.mark.parametrize(
    "kwargs, expected",
    [
        ({"pcp": 2}, "PCP"),
        ({"dcp": 2}, "DCP"),
        ({"spec": object()}, "speculative decoding"),
    ],
)
def test_unsupported_features_are_refused(kwargs, expected):
    """Plugin mode never runs Config.__post_init__, so these guards would
    otherwise be silently absent and the run would produce wrong output."""
    from atom.plugin.vllm.models.glm5_next import _check_unsupported

    with pytest.raises(ValueError, match=expected):
        _check_unsupported(_vllm_config(_config(index_kpool=1), **kwargs))


def test_kpool_runs_when_the_pool_size_divides_the_block_size(monkeypatch):
    """The pooled cache holds one row per `index_kpool` tokens, so the pool size
    has to divide the block size vLLM settled on. 1024 (TP4) and 512 (TP8) both
    do, which is the only reason the pooled path can be allowed at all here."""
    from atom.plugin.vllm.models import glm5_next as mod

    monkeypatch.setattr(mod, "pooled_path_enabled", lambda _k: True)
    mod._check_unsupported(_vllm_config(_config(index_kpool=4), block_size=1024))
    mod._check_unsupported(_vllm_config(_config(index_kpool=4), block_size=512))


def test_kpool_is_refused_when_the_pool_does_not_divide_the_block(monkeypatch):
    """Refuse loudly rather than mis-address the pooled cache.

    1022 is deliberately not a multiple of the pool size; the block sizes vLLM
    actually picks for this model are, but nothing in vLLM promises that, and a
    mismatch would put pooled rows in the wrong block instead of raising.
    """
    from atom.plugin.vllm.models import glm5_next as mod

    monkeypatch.setattr(mod, "pooled_path_enabled", lambda _k: True)
    with pytest.raises(ValueError, match="divide the KV block size"):
        mod._check_unsupported(_vllm_config(_config(index_kpool=4), block_size=1022))


def test_prefix_caching_and_kv_transfer_are_refused():
    """Both move a request off the KDA state slot the pooled tail buffer is
    keyed on, and neither migration knows the buffer exists."""
    from atom.plugin.vllm.models import glm5_next as mod

    with pytest.raises(ValueError, match="prefix caching"):
        mod._check_unsupported(
            _vllm_config(_config(index_kpool=1), prefix_caching=True)
        )
    with pytest.raises(ValueError, match="KV transfer"):
        mod._check_unsupported(
            _vllm_config(_config(index_kpool=1), kv_transfer=SimpleNamespace())
        )


def test_kpool_disabled_is_allowed():
    from atom.plugin.vllm.models import glm5_next as mod

    # index_kpool == 1 makes pooled_path_enabled False regardless of the env.
    mod._check_unsupported(_vllm_config(_config(index_kpool=1)))


def test_tail_buffer_is_not_allocated_from_the_forward():
    """The op must not create the tail buffer itself: the first forward that
    carries metadata is a CUDA graph capture, and a buffer born there has its
    zero-fill replayed every step."""
    from atom.plugin.vllm.attention import glm5_kpool_indexer as op

    with pytest.raises(RuntimeError, match="never allocated"):
        op._tail_cache("model.layers.999.self_attn.indexer.k_cache")


def test_binding_the_kda_state_sizes_the_tail_buffers(monkeypatch):
    """One tail buffer per pooled indexer, sized off the bound KDA state slots,
    allocated when the state is bound -- before warmup and graph capture."""
    from atom.plugin.vllm.attention import glm5_kpool_indexer as op
    from atom.plugin.vllm.models import glm5_next as mod

    monkeypatch.setattr(op, "_TAIL_CACHES", {})
    layer = mod.Glm5NextKDAAttentionVllm.__new__(mod.Glm5NextKDAAttentionVllm)
    object.__setattr__(
        layer,
        "pooled_tail_indexers",
        (("a.indexer.k_cache", 4, 128), ("b.indexer.k_cache", 4, 128)),
    )

    def bind_state(self, kv_cache):
        object.__setattr__(self, "kv_cache", (torch.empty(33, 3, 8), kv_cache))

    # Stand in for vLLM's MambaBase unpacking; the override's own work is what
    # is under test.
    monkeypatch.setattr(mod.KimiKDAAttentionVllm, "bind_kv_cache", bind_state)
    layer.bind_kv_cache(torch.empty(0))

    assert set(op._TAIL_CACHES) == {"a.indexer.k_cache", "b.indexer.k_cache"}
    for buf in op._TAIL_CACHES.values():
        assert buf.shape == (33, 2, 4, 128)
        assert buf.dtype == torch.bfloat16
        assert not buf.any()


def test_plugin_classes_replace_native_ones_only_during_construction():
    """The plugin subclasses the native KDA layer and indexer and swaps them in
    while the model is built; importing it must leave the native module alone,
    so a native ATOM engine in the same process keeps its own classes."""
    from atom.models import glm5_next as native
    from atom.plugin.vllm.models import glm5_next as mod

    assert issubclass(mod.Glm5NextIndexerVllm, native.Glm5NextIndexer)
    assert issubclass(mod.Glm5NextKDAAttentionVllm, native.Glm5NextKDAAttention)
    assert native.Glm5NextIndexer is not mod.Glm5NextIndexerVllm
    assert native.Glm5NextKDAAttention is not mod.Glm5NextKDAAttentionVllm
