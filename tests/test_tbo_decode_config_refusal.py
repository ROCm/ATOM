# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""What `Config` refuses for TBO: a linear-attention hybrid, whose state
metadata a micro-batch does not get; under PCP, decode TBO and speculation; and
for decode TBO speculation on a sparse MLA model or under DCP, whose verify
runs per-token metadata a decode micro-batch does not build."""

from types import SimpleNamespace

import pytest

from atom.config import Config

SPARSE_MLA = SimpleNamespace(
    model_type="glm_moe_dsa", kv_lora_rank=512, index_topk=2048
)
DENSE_MLA = SimpleNamespace(model_type="deepseek_v3", kv_lora_rank=512)
SPARSE_V4 = SimpleNamespace(
    model_type="deepseek_v3", architectures=["DeepseekV4ForCausalLM"], index_topk=512
)
MHA = SimpleNamespace(model_type="gpt_oss")


def _check(hf_config, *, tbo_decode=True, spec=True, dcp=1, pcp=1):
    Config._check_tbo_support(
        SimpleNamespace(
            enable_tbo=True,
            enable_tbo_decode=tbo_decode,
            speculative_config=SimpleNamespace(method="mtp") if spec else None,
            hf_config=hf_config,
            decode_context_parallel_size=dcp,
            prefill_context_parallel_size=pcp,
        )
    )


def test_pcp_refuses_decode_tbo():
    with pytest.raises(ValueError, match="decode TBO"):
        _check(MHA, spec=False, pcp=8)


def test_pcp_refuses_speculation_under_tbo():
    with pytest.raises(ValueError, match="speculative"):
        _check(MHA, tbo_decode=False, pcp=8)


def test_pcp_allows_prefill_tbo():
    _check(MHA, tbo_decode=False, spec=False, pcp=8)


def test_speculation_on_a_sparse_mla_model_refuses_decode_tbo():
    with pytest.raises(ValueError, match="sparse MLA"):
        _check(SPARSE_MLA)


@pytest.mark.parametrize(
    "hf_config",
    [
        SimpleNamespace(model_type="qwen3_next"),
        SimpleNamespace(model_type="qwen4_exp_text"),
        SimpleNamespace(model_type="kimi_linear", kv_lora_rank=512),
    ],
)
@pytest.mark.parametrize("tbo_decode", [False, True])
def test_a_linear_attention_hybrid_refuses_tbo(hf_config, tbo_decode):
    with pytest.raises(ValueError, match="linear-attention"):
        _check(hf_config, tbo_decode=tbo_decode, spec=False)


def test_speculation_under_dcp_refuses_decode_tbo():
    with pytest.raises(ValueError, match="DCP"):
        _check(DENSE_MLA, dcp=8)


@pytest.mark.parametrize(
    "hf_config, tbo_decode, spec",
    [
        (SPARSE_MLA, False, True),
        (SPARSE_MLA, True, False),
        (DENSE_MLA, True, True),
        (SPARSE_V4, True, True),
        (MHA, True, True),
    ],
)
def test_everything_else_is_allowed(hf_config, tbo_decode, spec):
    _check(hf_config, tbo_decode=tbo_decode, spec=spec)


def test_a_run_tbo_is_turned_off_for_is_not_refused(monkeypatch):
    """TBO is resolved before decode TBO is checked: a run whose TBO is turned
    off (here the DP-attention + EP collective fallback) starts as it would
    without it, rather than being refused for what it no longer runs."""
    import atom.config as config_module

    monkeypatch.setattr(config_module.envs, "ATOM_DISABLE_MORI_EP", True)
    config = SimpleNamespace(
        enable_tbo=True,
        enable_tbo_decode=True,
        enable_dp_attention=True,
        enable_expert_parallel=True,
        moe_all2all_backend="auto",
        prefill_context_parallel_size=1,
        speculative_config=SimpleNamespace(method="mtp"),
        hf_config=SPARSE_MLA,
        decode_context_parallel_size=1,
    )
    Config._resolve_tbo(config)
    Config._check_tbo_support(config)
    assert not config.enable_tbo and not config.enable_tbo_decode


@pytest.mark.parametrize(
    "max_num_seqs, max_num_batched_tokens, spec_tokens, expected",
    [
        (3, 8192, 0, 2),  # never past the user's bound: down
        (512, 1024, 2, 340),  # 1024 // 3 = 341 requests' tokens: down to 340
        (255, 8192, 3, 254),
        (256, 8192, 3, 256),
    ],
)
def test_decode_tbo_bounds_a_decode_at_an_even_number_of_requests(
    max_num_seqs, max_num_batched_tokens, spec_tokens, expected
):
    config = SimpleNamespace(
        enable_tbo_decode=True,
        max_num_seqs=max_num_seqs,
        max_num_batched_tokens=max_num_batched_tokens,
        speculative_config=(
            SimpleNamespace(num_speculative_tokens=spec_tokens) if spec_tokens else None
        ),
    )
    Config._even_decode_requests(config)
    assert config.max_num_seqs == expected


@pytest.mark.parametrize("max_num_seqs, max_num_batched_tokens", [(1, 8192), (64, 7)])
def test_decode_tbo_refuses_a_bound_of_fewer_than_two_requests(
    max_num_seqs, max_num_batched_tokens
):
    config = SimpleNamespace(
        enable_tbo_decode=True,
        max_num_seqs=max_num_seqs,
        max_num_batched_tokens=max_num_batched_tokens,
        speculative_config=SimpleNamespace(num_speculative_tokens=3),
    )
    with pytest.raises(ValueError, match="two halves"):
        Config._even_decode_requests(config)
