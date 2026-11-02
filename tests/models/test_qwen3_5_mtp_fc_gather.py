# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""The Qwen3.5 and Qwen3-Next MTP fc all-gathers take aiter's custom collective.

`GroupCoordinator.all_gather` defaults to `use_custom=False`, which is RCCL:
~14 us per draft step at decode plus an idle gap after it. The LM head's
gather already asks for the custom one through `ATOM_USE_CUSTOM_ALL_GATHER`;
these pin that both MTP fc gathers ask the same way, along the last dim, and
that the knob still turns it off. CPU-only: the TP group is a stand-in.
"""

import pytest
import torch

# Both MTP modules import AITER at module scope.
pytest.importorskip("aiter")

from atom.models import qwen3_5_mtp, qwen3_next_mtp

# (module, predictor class): both forwards share the fc -> all_gather shape.
PREDICTORS = [
    pytest.param(qwen3_5_mtp, "Qwen3_5MultiTokenPredictor", id="qwen3_5"),
    pytest.param(qwen3_next_mtp, "Qwen3NextMultiTokenPredictor", id="qwen3_next"),
]

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


def _predictor(module, cls_name):
    """A predictor whose submodules are shape-correct identity stand-ins."""
    cls = getattr(module, cls_name)
    pred = cls.__new__(cls)
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


def _run(monkeypatch, module, cls_name, n_tokens=4):
    group = _RecordingGroup()
    monkeypatch.setattr(module, "get_tp_group", lambda: group)
    out = _predictor(module, cls_name)(
        input_ids=torch.zeros(n_tokens, dtype=torch.int64),
        positions=torch.zeros(n_tokens, dtype=torch.int64),
        hidden_states=torch.zeros(n_tokens, HIDDEN),
    )
    return group, out


@pytest.mark.parametrize("module, cls_name", PREDICTORS)
@pytest.mark.parametrize("n_tokens", [1, 4, 32])
def test_fc_gather_asks_for_the_custom_collective(
    monkeypatch, module, cls_name, n_tokens
):
    monkeypatch.delenv("ATOM_USE_CUSTOM_ALL_GATHER", raising=False)
    group, out = _run(monkeypatch, module, cls_name, n_tokens)

    assert group.calls == [
        {"shape": (n_tokens, HIDDEN // TP), "use_custom": True, "dim": -1}
    ]
    # The gather restores the full hidden size the decoder layer expects.
    assert out.shape == (n_tokens, HIDDEN)


@pytest.mark.parametrize("module, cls_name", PREDICTORS)
def test_fc_gather_follows_the_knob_off(monkeypatch, module, cls_name):
    monkeypatch.setenv("ATOM_USE_CUSTOM_ALL_GATHER", "0")
    group, _ = _run(monkeypatch, module, cls_name)

    assert group.calls[0]["use_custom"] is False


@pytest.mark.parametrize("module, cls_name", PREDICTORS)
def test_prefill_sized_fc_gather_stays_on_rccl(monkeypatch, module, cls_name):
    """aiter's custom all-gather has no size fallback, and step 0 of the draft
    runs over a whole prefill chunk. Above the per-rank bound the gather asks
    for RCCL; at the bound it still asks for custom."""
    from atom.models import utils as model_utils

    monkeypatch.delenv("ATOM_USE_CUSTOM_ALL_GATHER", raising=False)
    row_bytes = (HIDDEN // TP) * 4  # fp32 stand-in rows
    # Shrink the bound so that 8 rows a rank fit exactly and 9 do not.
    monkeypatch.setattr(
        model_utils, "MTP_FC_CUSTOM_AG_BYTES_PER_RANK", 8 * row_bytes // TP
    )
    group, _ = _run(monkeypatch, module, cls_name, n_tokens=8)
    assert group.calls[0]["use_custom"] is True
    group, _ = _run(monkeypatch, module, cls_name, n_tokens=9)
    assert group.calls[0]["use_custom"] is False


def test_qwen35_prefill_chunk_is_over_the_bound_and_decode_is_under():
    """The shipped bound on the shapes it is meant to separate: Qwen3.5-397B
    (hidden 4096) bf16 fc output at TP4, a 16384-token prefill chunk vs a
    64-request x 6-row decode step."""
    from atom.models.utils import mtp_fc_gather_use_custom

    cols = 4096 // 4
    prefill = torch.empty(16384, cols, dtype=torch.bfloat16, device="meta")
    decode = torch.empty(64 * 6, cols, dtype=torch.bfloat16, device="meta")
    assert mtp_fc_gather_use_custom(prefill, 4) is False
    assert mtp_fc_gather_use_custom(decode, 4) is True


def test_env_knob_defaults_on(monkeypatch):
    """The default the change relies on, shared with the LM head's gather."""
    from atom.utils import envs

    monkeypatch.delenv("ATOM_USE_CUSTOM_ALL_GATHER", raising=False)
    assert envs.ATOM_USE_CUSTOM_ALL_GATHER is True
