# SPDX-License-Identifier: MIT

from types import SimpleNamespace

import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("requires the AITER GPU runtime", allow_module_level=True)

from atom.spec_decode import deepseek_v4_proposer as v4_proposer_module
from atom.spec_decode import factory
from atom.spec_decode.deepseek_v4_proposer import DeepseekV4Proposer
from atom.spec_decode.eagle_proposer import EagleProposer


class _CapturedGraph:
    def is_captured(self, batch_size):
        return batch_size == 1


def _selection_fixture(monkeypatch, *, is_prefill):
    proposer = object.__new__(DeepseekV4Proposer)
    proposer.step0 = _CapturedGraph()
    proposer.mtp_k = 3
    proposer.dtype = torch.bfloat16
    context = SimpleNamespace(is_prefill=is_prefill, running_bs=1)
    forward_context = SimpleNamespace(
        context=context,
        attn_metadata=SimpleNamespace(max_seqlen_q=4),
    )
    monkeypatch.setattr(
        v4_proposer_module, "get_forward_context", lambda: forward_context
    )
    return proposer


def test_v4_step0_graph_is_decode_only(monkeypatch):
    proposer = _selection_fixture(monkeypatch, is_prefill=True)
    inputs = (
        1,
        torch.zeros(4, dtype=torch.int32),
        torch.zeros(4, dtype=torch.int64),
        torch.zeros(4, 2, 8, dtype=torch.bfloat16),
        torch.tensor([3], dtype=torch.int32),
    )

    assert proposer._select_step0_graph(*inputs) is None

    v4_proposer_module.get_forward_context().context.is_prefill = False
    assert proposer._select_step0_graph(*inputs) is proposer.step0


def test_v4_owns_the_full_width_step0_graph(monkeypatch):
    monkeypatch.setattr(
        EagleProposer, "_declare_draft_graphs", lambda self: ("common-mid-step",)
    )
    proposer = object.__new__(DeepseekV4Proposer)
    proposer.speculative_config = SimpleNamespace(
        draft_model_hf_config=SimpleNamespace(hc_mult=2, hidden_size=8)
    )
    proposer.config = SimpleNamespace(
        parallel_config=SimpleNamespace(
            data_parallel_size=1, decode_context_parallel_size=1
        ),
        prefill_context_parallel_size=1,
        enable_tbo=False,
        enable_expert_parallel=False,
    )
    proposer.runner = SimpleNamespace(use_mrope=False)
    proposer.mtp_k = 3
    proposer.dtype = torch.bfloat16

    graphs = proposer._declare_draft_graphs()

    assert graphs == (proposer.step0, "common-mid-step")
    assert proposer.step0.tokens_per_seq == 4


def test_factory_routes_only_v4_serial_mtp_to_v4_proposer(monkeypatch):
    built = []

    def make(name):
        return lambda config, device, runner: built.append(name) or name

    monkeypatch.setattr(factory, "DeepseekV4Proposer", make("v4"))
    monkeypatch.setattr(factory, "EagleProposer", make("eagle"))
    monkeypatch.setattr(factory, "DSparkProposer", make("dspark"))

    spec = SimpleNamespace(
        use_dspark=lambda: False,
        draft_model_hf_config=SimpleNamespace(architectures=["DeepseekV4MTPModel"]),
    )
    config = SimpleNamespace(speculative_config=spec)
    assert factory.build_drafter(config, "device", "runner") == "v4"

    spec.draft_model_hf_config.architectures = ["DeepSeekMTPModel"]
    assert factory.build_drafter(config, "device", "runner") == "eagle"

    spec.use_dspark = lambda: True
    assert factory.build_drafter(config, "device", "runner") == "dspark"
    assert built == ["v4", "eagle", "dspark"]
