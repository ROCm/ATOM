"""VllmConfig -> ATOM offload config projection.

The offload path reads these fields off ATOM's Config; in plugin mode vLLM owns
them. Getting the KV dtype wrong is not a crash but a sizing error -- the codec
derives every byte segment from it -- so the mapping fails closed on anything
it does not recognise.
"""

from __future__ import annotations

from types import SimpleNamespace as NS

import pytest

from atom.plugin.vllm.kv_transfer.offload_config import build_offload_config


def _vllm_config(cache_dtype="fp8", model_dtype="bfloat16", block_size=128, **kw):
    return NS(
        cache_config=NS(block_size=block_size, cache_dtype=cache_dtype),
        model_config=NS(
            hf_config=NS(num_hidden_layers=60, model_type="minimax_m3"),
            dtype=model_dtype,
            model="/models/MiniMax-M3-MXFP4",
        ),
        parallel_config=NS(
            pipeline_parallel_size=kw.get("pp", 1),
            decode_context_parallel_size=kw.get("dcp", 1),
        ),
        kv_transfer_config=NS(
            kv_role=kw.get("role", "kv_both"),
            kv_connector_extra_config=kw.get("extra", {}),
        ),
    )


def test_projects_the_fields_offload_reads():
    cfg = build_offload_config(_vllm_config())

    assert cfg.kv_cache_block_size == 128
    assert cfg.kv_cache_dtype == "fp8"
    assert cfg.hf_config.num_hidden_layers == 60
    assert cfg.kv_transfer_config["kv_role"] == "kv_both"


def test_auto_kv_dtype_resolves_to_the_model_dtype():
    cfg = build_offload_config(_vllm_config(cache_dtype="auto", model_dtype="bfloat16"))
    assert cfg.kv_cache_dtype == "bf16"

    cfg = build_offload_config(
        _vllm_config(cache_dtype="auto", model_dtype="torch.float16")
    )
    assert cfg.kv_cache_dtype == "fp16"


@pytest.mark.parametrize("raw", ["fp8_e4m3", "fp8_e5m2", "fp8_inc"])
def test_fp8_spellings_all_map_to_fp8(raw):
    assert build_offload_config(_vllm_config(cache_dtype=raw)).kv_cache_dtype == "fp8"


def test_unknown_kv_dtype_fails_closed():
    # Guessing a width here would mis-size every byte segment silently.
    with pytest.raises(ValueError, match="unsupported KV cache dtype"):
        build_offload_config(_vllm_config(cache_dtype="int4_magic"))


def test_missing_block_size_is_rejected():
    with pytest.raises(ValueError, match="block_size"):
        build_offload_config(_vllm_config(block_size=None))


def test_missing_hf_config_is_rejected():
    bad = _vllm_config()
    bad.model_config.hf_config = None
    with pytest.raises(ValueError, match="hf_config"):
        build_offload_config(bad)


def test_extra_connector_config_is_carried_through():
    cfg = build_offload_config(_vllm_config(extra={"lmcache.foo": 7}))
    assert cfg.kv_transfer_config["lmcache.foo"] == 7
