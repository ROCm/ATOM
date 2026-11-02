# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""What `Config` refuses under PP: P/D consumers, and speculation beyond
Kimi-K3 with DSpark."""

from types import SimpleNamespace

import pytest

from atom.config import Config


def _cfg(pp=2, arch="KimiK3ForConditionalGeneration", method="dspark", **kw):
    spec = SimpleNamespace(method=method, use_dspark=lambda: method == "dspark")
    return SimpleNamespace(
        pipeline_parallel_size=pp,
        hf_config=SimpleNamespace(architectures=[arch]),
        speculative_config=kw.pop("spec", spec),
        dspark=SimpleNamespace(confidence_schedule=kw.pop("confidence", False)),
        kv_transfer_config=kw.pop("kv", None),
    )


def _check(cfg):
    Config._check_pp_support(cfg)


def test_kimi_k3_dspark_under_pp_is_allowed():
    _check(_cfg())


def test_pp1_is_never_refused():
    _check(_cfg(pp=1, arch="Qwen3NextForCausalLM", method="mtp"))


def test_pp_without_speculation_is_allowed_for_any_model():
    _check(_cfg(arch="Qwen3NextForCausalLM", spec=None))


@pytest.mark.parametrize(
    "arch, method",
    [
        ("KimiK3ForConditionalGeneration", "eagle3"),
        ("Qwen3NextForCausalLM", "mtp"),
        ("MiniMaxM3ForCausalLM", "dspark"),
    ],
)
def test_unwired_spec_pairs_under_pp_are_refused(arch, method):
    with pytest.raises(ValueError, match="only Kimi-K3 with DSpark"):
        _check(_cfg(arch=arch, method=method))


def test_confidence_schedule_under_pp_is_refused():
    with pytest.raises(ValueError, match="confidence_schedule"):
        _check(_cfg(confidence=True))


def test_kv_consumer_under_pp_is_refused():
    kv = {"kv_connector": "mooncake", "kv_role": "kv_consumer"}
    with pytest.raises(ValueError, match="kv_consumer"):
        _check(_cfg(spec=None, kv=kv))
