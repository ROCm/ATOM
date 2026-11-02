# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

from types import SimpleNamespace

from atom.config import Config


def _hash(spec=None, kv_cache_dtype="fp8"):
    # Only the fields compute_hash reads; quant/compilation/parallel are falsy
    # so their sub-hashes are skipped.
    cfg = SimpleNamespace(
        quant_config=None,
        compilation_config=None,
        parallel_config=None,
        tensor_parallel_size=8,
        prefill_context_parallel_size=1,
        dcp_config=None,
        enable_dp_attention=True,
        index_cache_dtype="fp4",
        kv_cache_dtype=kv_cache_dtype,
        enable_decoder_swa_bounded_replay=True,
        hf_config=SimpleNamespace(),
        max_model_len=4096,
        max_num_seqs=256,
        speculative_config=spec,
    )
    return Config.compute_hash(cfg)


def _spec(method="dspark", k=3, **kw):
    return SimpleNamespace(method=method, num_speculative_tokens=k, **kw)


def test_draft_width_keys_the_compile_cache():
    assert _hash(_spec(k=3)) != _hash(_spec(k=6))


def test_drafter_method_keys_the_compile_cache():
    assert _hash(_spec(method="dspark")) != _hash(_spec(method="mtp"))


def test_spec_on_and_off_differ():
    assert _hash(None) != _hash(_spec())


def test_acceptance_length_does_not_key_the_compile_cache():
    assert _hash(_spec(synthetic_acceptance_length=3.01)) == _hash(
        _spec(synthetic_acceptance_length=3.77)
    )


def test_kv_cache_dtype_keys_the_compile_cache():
    """The DSpark drafter branches on the KV dtype while tracing: a bf16 run
    over an fp8 run's cache loaded the fp8 window's graph and died in warmup."""
    assert _hash(_spec(), kv_cache_dtype="bf16") != _hash(_spec(), kv_cache_dtype="fp8")
