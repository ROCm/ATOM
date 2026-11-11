# SPDX-License-Identifier: MIT
"""A DP step on which this rank scheduled no request.

The rank still forwards its running rows for its peers' collectives, the LM
head included; `postprocess` then cuts the logits to its zero scheduled rows
and samples nothing, handing back only the previous step's deferred output.

Skipped where `model_engine.model_runner` cannot import (it needs aiter at
module load); nothing here touches a GPU.
"""

import types

import numpy as np
import pytest
import torch

from atom.model_ops.attentions.token_layout.batch_ids import build_query_layout

# See tests/test_postprocess_width.py for why not `importorskip("aiter")`.
mod = pytest.importorskip(
    "atom.model_engine.model_runner",
    reason="model_runner imports aiter at module load",
    exc_type=ImportError,
)


@pytest.mark.parametrize("running_bs,running_tokens", [(1, 1), (4, 2), (8, 32)])
def test_query_layout_without_a_request_is_all_padding(running_bs, running_tokens):
    cu = np.full(running_bs + 1, 7, dtype=np.int32)
    batch_ids = np.full(running_tokens, 7, dtype=np.int32)

    build_query_layout(
        np.zeros(0, dtype=np.int32),
        running_bs,
        running_tokens,
        cu_seqlens_q=cu,
        batch_ids=batch_ids,
    )

    # Every request is zero long and every token belongs to none.
    assert cu.tolist() == [0] * (running_bs + 1)
    assert batch_ids.tolist() == [-1] * running_tokens


def test_postprocess_samples_nothing_without_a_request(monkeypatch):
    ctx = types.SimpleNamespace(scheduled_bs=0, is_prefill=False)
    fc = types.SimpleNamespace(context=ctx, spec_decode_metadata=None)
    monkeypatch.setattr(mod, "get_forward_context", lambda: fc)
    monkeypatch.setattr(
        mod, "get_tp_group", lambda: types.SimpleNamespace(world_size=1)
    )
    monkeypatch.setattr(mod, "pcp_is_enabled", lambda: False)
    monkeypatch.setattr(mod, "sp_is_enabled", lambda: False)
    runner = object.__new__(mod.ModelRunner)
    runner.device = torch.device("cpu")
    runner.sampler = lambda logits, *args, **kwargs: logits.argmax(-1)
    runner.forward_done_event = types.SimpleNamespace(record=lambda: None)
    sampled = {}
    runner.tokenID_processor = types.SimpleNamespace(
        is_deferred_out=False,
        prev_batch=None,
        zero_spec_counts=torch.zeros(8, dtype=torch.int32),
        prepare_sampled_ids=lambda b, ids, *a, **k: (
            sampled.update(ids=ids) or ({}, {})
        ),
    )
    batch = types.SimpleNamespace(
        total_seqs_num=0,
        total_tokens_num=0,
        req_ids=[],
        return_logprobs=[],
        any_logprobs=False,
        required_channel=None,
        num_spec_step=0,
    )

    out = runner.postprocess(
        batch=batch,
        # The running rows the LM head emitted; none of them is scheduled.
        logits=torch.zeros(2, 8),
        temperatures=torch.ones(0),
        top_ks=None,
        top_ps=None,
        all_greedy=True,
        hidden_states=torch.zeros(0, 8),
    )

    assert sampled["ids"].numel() == 0
    assert out.num_rejected.tolist() == []
