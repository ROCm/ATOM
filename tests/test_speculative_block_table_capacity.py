# SPDX-License-Identifier: MIT
"""Speculative lookahead must fit even at the public context limit."""

from types import SimpleNamespace

import pytest
import torch

from atom.model_ops.attentions import backends
from atom.utils import CpuGpuBuffer
from atom.utils.block_tables import block_table_state


@pytest.mark.parametrize("limit", [16384, 16385])
@pytest.mark.parametrize("drafts", [0, 3, 65])
@pytest.mark.parametrize("ratio", [1, 4])
def test_pending_pages_fit_at_context_limit(monkeypatch, limit, drafts, ratio):
    monkeypatch.setattr(backends, "get_dcp_world_size", lambda: 1)
    monkeypatch.setattr(backends, "get_dcp_rank", lambda: 0)
    monkeypatch.setattr(backends, "get_tp_group", lambda: SimpleNamespace(world_size=1))
    monkeypatch.setattr(
        backends,
        "CpuGpuBuffer",
        lambda *args, **kwargs: CpuGpuBuffer(*args, **kwargs, pin_memory=False),
    )
    config = SimpleNamespace(
        max_model_len=limit,
        speculative_config=(
            SimpleNamespace(num_speculative_tokens=drafts) if drafts else None
        ),
        hf_config=SimpleNamespace(num_attention_heads=8),
    )
    runner = SimpleNamespace(
        config=config,
        block_size=64,
        device="cpu",
        max_bs=2,
        max_num_batched_tokens=256,
        forward_vars={},
    )
    builder = SimpleNamespace(block_size=64 // ratio)
    backends.CommonAttentionBuilder.__init__(builder, runner)
    buf = runner.forward_vars["block_tables"]
    # Scheduler rows include pending tokens; exercise publication, not just
    # the width calculation, including multiple extra pages and split pages.
    pending = 2 * drafts + 1 if drafts else 0
    pages = (limit + pending + 63) // 64
    row = list(range(pages))
    state = block_table_state(buf).prepare([row], pad_to=2)
    state.publish(2)
    assert buf.gpu[0, :pages].tolist() == row
    assert not torch.any(buf.gpu[1])
    assert builder.max_num_blocks_per_seq == pages * ratio
    assert config.max_model_len == limit
    with pytest.raises(ValueError, match="exceeds the destination columns"):
        state.prepare([list(range(pages + 1))])
