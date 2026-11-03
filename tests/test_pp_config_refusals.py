# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""What `Config` refuses under PP: TBO, P/D consumers, and speculation an
architecture does not declare in `registry.pp_speculative_methods`."""

from types import SimpleNamespace
from unittest import mock

import pytest

from atom.config import Config
from atom.models.registry import pp_speculative_methods


def _cfg(pp=2, arch="KimiK3ForConditionalGeneration", method="dspark", **kw):
    spec = SimpleNamespace(method=method, use_dspark=lambda: method == "dspark")
    return SimpleNamespace(
        pipeline_parallel_size=pp,
        hf_config=SimpleNamespace(architectures=[arch]),
        speculative_config=kw.pop("spec", spec),
        dspark=SimpleNamespace(confidence_schedule=kw.pop("confidence", False)),
        kv_transfer_config=kw.pop("kv", None),
        enable_tbo=kw.pop("tbo", False),
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
    with pytest.raises(ValueError, match="pp_speculative_methods declares"):
        _check(_cfg(arch=arch, method=method))


def test_a_declared_architecture_is_allowed_without_a_config_edit():
    with mock.patch.dict(pp_speculative_methods, {"Qwen3NextForCausalLM": ("mtp",)}):
        _check(_cfg(arch="Qwen3NextForCausalLM", method="mtp"))


def test_confidence_schedule_under_pp_is_refused():
    with pytest.raises(ValueError, match="confidence_schedule"):
        _check(_cfg(confidence=True))


def test_tbo_under_pp_is_refused():
    with pytest.raises(ValueError, match="TBO"):
        _check(_cfg(spec=None, tbo=True))


def test_kv_consumer_under_pp_is_refused():
    kv = {"kv_connector": "mooncake", "kv_role": "kv_consumer"}
    with pytest.raises(ValueError, match="kv_consumer"):
        _check(_cfg(spec=None, kv=kv))


def test_kv_producer_under_pp_drafts_with_any_method():
    # P/D prefill with `--method mtp` on PP (GLM-5.2's Atomesh cells): the
    # producer never speculates, so the Kimi-K3 + DSpark gate does not apply.
    kv = {"kv_connector": "mooncake", "kv_role": "kv_producer"}
    _check(_cfg(arch="Glm5ForCausalLM", method="mtp", kv=kv))
