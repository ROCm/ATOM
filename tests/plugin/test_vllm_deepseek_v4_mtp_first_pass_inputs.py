# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""DeepSeek-V4 MTP first-pass inputs on vLLM 0.31's draft proposer.

The plugin replaces ``SpecDecodeBaseProposer.set_inputs_first_pass`` for the
V4 MTP draft: the target token ids go in unshifted, the sampled token lands at
each request's last slot, positions and hidden states are copied as-is. vLLM
0.31 removed XD-RoPE from speculative decoding, so the proposer has no
``uses_xdrope_dim`` / ``draft_uses_xdrope_dim``; the fast path must not read
them (it raised AttributeError on the first MTP request and killed the engine).
"""

from types import SimpleNamespace as NS

import pytest
import torch

llm_base_proposer = pytest.importorskip("vllm.v1.spec_decode.llm_base_proposer")
SpecDecodeBaseProposer = llm_base_proposer.SpecDecodeBaseProposer

from atom.plugin.vllm import spec_decode_patch

HIDDEN = 4
CAPACITY = 16


def _proposer(is_v4_mtp=True):
    # Only what the fast path and vLLM's own _set_positions read. No
    # uses_xdrope_dim / draft_uses_xdrope_dim: vLLM 0.31 does not set them.
    proposer = SpecDecodeBaseProposer.__new__(SpecDecodeBaseProposer)
    proposer.model = NS(_is_deepseek_v4_mtp=is_v4_mtp)
    proposer.needs_extra_input_slots = False
    proposer.uses_mrope = False
    proposer.vllm_config = NS(model_config=NS(uses_mrope=False))
    proposer.input_ids = torch.zeros(CAPACITY, dtype=torch.int64)
    proposer.positions = torch.zeros(CAPACITY, dtype=torch.int64)
    proposer.hidden_states = torch.zeros(CAPACITY, HIDDEN)
    return proposer


@pytest.fixture
def patched(monkeypatch):
    # Registers the original for restore; the patch then wraps whatever is
    # installed at call time.
    monkeypatch.setattr(
        SpecDecodeBaseProposer,
        "set_inputs_first_pass",
        SpecDecodeBaseProposer.set_inputs_first_pass,
    )
    spec_decode_patch._patch_vllm_deepseek_v4_mtp_first_pass_inputs()
    return monkeypatch


def _batch():
    # Three requests of 1, 2 and 3 tokens.
    cad = NS(query_start_loc=torch.tensor([0, 1, 3, 6]))
    target_token_ids = torch.tensor([11, 21, 22, 31, 32, 33])
    target_positions = torch.tensor([5, 7, 8, 2, 3, 4])
    target_hidden = torch.arange(6 * HIDDEN, dtype=torch.float32).view(6, HIDDEN)
    next_token_ids = torch.tensor([12, 23, 34])
    return cad, target_token_ids, target_positions, target_hidden, next_token_ids


def test_v4_mtp_fast_path_without_xdrope_attributes(patched):
    proposer = _proposer()
    cad, ids, pos, hidden, nxt = _batch()

    num_tokens, sample_idx, out_cad = proposer.set_inputs_first_pass(
        ids, nxt, pos, hidden, None, cad, None
    )

    assert num_tokens == 6
    assert out_cad is cad
    assert sample_idx.tolist() == [0, 2, 5]
    # Unshifted target ids, the sampled token at each request's last slot.
    assert proposer.input_ids[:6].tolist() == [12, 21, 23, 31, 32, 34]
    assert proposer.positions[:6].tolist() == pos.tolist()
    assert torch.equal(proposer.hidden_states[:6], hidden)


def test_v4_mtp_fast_path_keeps_given_sample_indices(patched):
    proposer = _proposer()
    cad, ids, pos, hidden, nxt = _batch()
    given = torch.tensor([0, 1, 4])

    _, sample_idx, _ = proposer.set_inputs_first_pass(
        ids, nxt, pos, hidden, given, cad, None
    )

    assert sample_idx is given
    assert proposer.input_ids[:6].tolist() == [12, 23, 22, 31, 34, 33]


def test_other_drafts_take_vllm_path(monkeypatch):
    calls = []

    def fake_original(self, *args):
        calls.append(args)
        return "vllm"

    monkeypatch.setattr(SpecDecodeBaseProposer, "set_inputs_first_pass", fake_original)
    spec_decode_patch._patch_vllm_deepseek_v4_mtp_first_pass_inputs()
    cad, ids, pos, hidden, nxt = _batch()

    assert (
        _proposer(is_v4_mtp=False).set_inputs_first_pass(
            ids, nxt, pos, hidden, None, cad, None
        )
        == "vllm"
    )
    assert len(calls) == 1
