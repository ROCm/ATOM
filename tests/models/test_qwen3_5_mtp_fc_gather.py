# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""The Qwen3.5 MTP fc all-gather takes aiter's custom collective.

`GroupCoordinator.all_gather` defaults to `use_custom=False`, which is RCCL:
~14 us per draft step at decode plus an idle gap after it. The LM head's
gather already asks for the custom one through `ATOM_USE_CUSTOM_ALL_GATHER`;
these pin that the MTP fc gather asks the same way, along the last dim, and
that the knob still turns it off. CPU-only: the TP group is a stand-in.
"""

import pytest
import torch

# `qwen3_5_mtp` imports AITER at module scope.
pytest.importorskip("aiter")

from atom.models import qwen3_5_mtp

HIDDEN = 64
TP = 4


class _RecordingGroup:
    """Concatenates `world_size` copies along `dim` and records the call."""

    world_size = TP

    def __init__(self):
        self.calls = []

    def all_gather(self, x, use_custom=False, dim=-1):
        self.calls.append(
            {"shape": tuple(x.shape), "use_custom": use_custom, "dim": dim}
        )
        return torch.cat([x] * self.world_size, dim=dim)


def _predictor():
    """A predictor whose submodules are shape-correct identity stand-ins."""
    pred = qwen3_5_mtp.Qwen3_5MultiTokenPredictor.__new__(
        qwen3_5_mtp.Qwen3_5MultiTokenPredictor
    )
    torch.nn.Module.__init__(pred)
    pred.num_mtp_layers = 1
    pred.embed_input_ids = lambda ids: torch.zeros(ids.shape[0], HIDDEN)
    pred.pre_fc_norm_embedding = lambda x: x
    pred.pre_fc_norm_hidden = lambda x: x
    # ColumnParallelLinear(2 * hidden -> hidden) leaves hidden / tp per rank.
    pred.fc = lambda x: x[:, : HIDDEN // TP].contiguous()
    pred.layers = [lambda positions, hidden_states, residual: (hidden_states, None)]
    pred.norm = lambda x, residual: (x, None)
    return pred


def _run(monkeypatch, n_tokens=4):
    group = _RecordingGroup()
    monkeypatch.setattr(qwen3_5_mtp, "get_tp_group", lambda: group)
    out = _predictor()(
        input_ids=torch.zeros(n_tokens, dtype=torch.int64),
        positions=torch.zeros(n_tokens, dtype=torch.int64),
        hidden_states=torch.zeros(n_tokens, HIDDEN),
    )
    return group, out


@pytest.mark.parametrize("n_tokens", [1, 4, 32])
def test_fc_gather_asks_for_the_custom_collective(monkeypatch, n_tokens):
    monkeypatch.delenv("ATOM_USE_CUSTOM_ALL_GATHER", raising=False)
    group, out = _run(monkeypatch, n_tokens)

    assert group.calls == [
        {"shape": (n_tokens, HIDDEN // TP), "use_custom": True, "dim": -1}
    ]
    # The gather restores the full hidden size the decoder layer expects.
    assert out.shape == (n_tokens, HIDDEN)


def test_fc_gather_follows_the_knob_off(monkeypatch):
    monkeypatch.setenv("ATOM_USE_CUSTOM_ALL_GATHER", "0")
    group, _ = _run(monkeypatch)

    assert group.calls[0]["use_custom"] is False


def test_qwen35_fc_shard_qualifies_for_the_custom_last_dim_gather():
    """aiter's `GroupCoordinator.all_gather` takes the custom kernel for a
    last-dim gather only when a row is a multiple of 16 bytes, else it falls
    back to RCCL silently. Qwen3.5-397B: hidden 4096, TP 2/4/8, bf16 fc out."""
    hidden = 4096
    for tp in (2, 4, 8):
        row_bytes = (hidden // tp) * torch.tensor(
            [], dtype=torch.bfloat16
        ).element_size()
        assert row_bytes % 16 == 0, (tp, row_bytes)


def test_env_knob_defaults_on(monkeypatch):
    """The default the change relies on, shared with the LM head's gather."""
    from atom.utils import envs

    monkeypatch.delenv("ATOM_USE_CUSTOM_ALL_GATHER", raising=False)
    assert envs.ATOM_USE_CUSTOM_ALL_GATHER is True
