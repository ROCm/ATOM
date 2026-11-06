# SPDX-License-Identifier: MIT
"""Speculative lookahead must fit even at the public context limit."""

from types import SimpleNamespace

import pytest
import torch

backends = pytest.importorskip(
    "atom.model_ops.attentions.backends",
    reason="the common builder imports optional GPU dependencies",
    exc_type=ImportError,
)

from conftest import MockConfig

from atom.model_engine.block_manager import BlockManager, draft_kv_lookahead
from atom.model_engine.sequence import Sequence
from atom.utils import CpuGpuBuffer
from atom.utils.block_tables import block_table_state


@pytest.fixture
def make_builder(monkeypatch):
    monkeypatch.setattr(backends, "get_dcp_world_size", lambda: 1)
    monkeypatch.setattr(backends, "get_dcp_rank", lambda: 0)
    monkeypatch.setattr(backends, "get_tp_group", lambda: SimpleNamespace(world_size=1))
    monkeypatch.setattr(
        backends,
        "CpuGpuBuffer",
        lambda *args, **kwargs: CpuGpuBuffer(*args, **kwargs, pin_memory=False),
    )

    def make(limit, drafts, ratio, block=64):
        config = SimpleNamespace(
            max_model_len=limit,
            speculative_config=(
                SimpleNamespace(num_speculative_tokens=drafts) if drafts else None
            ),
            hf_config=SimpleNamespace(num_attention_heads=8),
        )
        runner = SimpleNamespace(
            config=config,
            block_size=block,
            device="cpu",
            max_bs=2,
            max_num_batched_tokens=256,
            forward_vars={},
        )
        builder = SimpleNamespace(block_size=block // ratio)
        backends.CommonAttentionBuilder.__init__(builder, runner)
        return builder, runner

    return make


@pytest.mark.parametrize("limit", [16384, 16385])
@pytest.mark.parametrize("drafts", [0, 3, 65])
@pytest.mark.parametrize("ratio", [1, 4])
def test_pending_pages_fit_at_context_limit(make_builder, limit, drafts, ratio):
    builder, runner = make_builder(limit, drafts, ratio)
    buf = runner.forward_vars["block_tables"]
    # Scheduler rows include pending tokens; exercise publication, not just
    # the width calculation, including multiple extra pages and split pages.
    pending = 2 * drafts + 1 if drafts else 0
    pages = (limit + pending + draft_kv_lookahead(drafts) + 63) // 64
    row = list(range(pages))
    state = block_table_state(buf).prepare([row], pad_to=2)
    state.publish(2)
    assert buf.gpu[0, :pages].tolist() == row
    assert not torch.any(buf.gpu[1])
    assert builder.max_num_blocks_per_seq == pages * ratio
    assert runner.config.max_model_len == limit
    with pytest.raises(ValueError, match="exceeds the destination columns"):
        state.prepare([list(range(pages + 1))])


@pytest.mark.parametrize("block", [16, 64, 128])
@pytest.mark.parametrize("ratio", [1, 4])
def test_allocator_lookahead_row_fits_near_limit(make_builder, block, ratio):
    drafts = 3
    pending = 2 * drafts + 1
    # Without allocator lookahead, the reserved capacity ends exactly at a page.
    limit = 257 * block - pending
    _, runner = make_builder(limit, drafts, ratio, block)
    manager = BlockManager(
        MockConfig(
            kv_cache_block_size=block,
            num_kvcache_blocks=300,
            max_model_len=limit,
        )
    )
    seq = Sequence(list(range(limit - 1)), block)
    manager.allocate(seq)
    for _ in range(pending):
        seq.append_token(1)
    manager.may_append(seq, pending, lookahead=draft_kv_lookahead(drafts))
    assert len(seq.block_table) == 258
    buf = runner.forward_vars["block_tables"]
    state = block_table_state(buf).prepare([seq.block_table], pad_to=2)
    state.publish(2)
    assert buf.gpu[0, :258].tolist() == list(seq.block_table)
    assert not torch.any(buf.gpu[1])
    assert runner.config.max_model_len == limit
