# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

from types import SimpleNamespace

from atom.config import Config


def _hash(spec=None):
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
        enable_decoder_swa_bounded_replay=True,
        hf_config=SimpleNamespace(),
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
