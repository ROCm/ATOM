# SPDX-License-Identifier: MIT
"""Mixed TBO uses local request rows, DP token padding, then parent draft state."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

pytest.importorskip("aiter")

from atom.model_ops.attentions.deepseek_v4_attn import (
    DeepseekV4AttentionMetadataBuilder,
    _mixed_carrier,
)
from atom.model_ops.moe import pad_for_all_gather
from atom.utils.forward_context import (
    AttnState,
    Context,
    DPMetadata,
    ForwardContext,
    _forward_context_local,
    get_forward_context,
)
from atom.utils.tbo.ubatch_splitting import split_mixed_token_midpoint
from atom.utils.tbo.ubatch_wrapper import UBatchWrapper

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires GPU streams for the TBO workers"
)


@pytest.mark.parametrize("rank", [0, 1])
@pytest.mark.parametrize("query_width", [1, 4])
def test_unequal_dp_rows_keep_token_padding_and_restore_full_draft_state(
    monkeypatch, rank, query_width
):
    layouts = [([32], [query_width]), ([24, 24], [query_width] * 3)]
    splits = [
        split_mixed_token_midpoint(len(p) + len(d), len(p), sum(p), np.asarray(p + d))
        for p, d in layouts
    ]
    # The packed DP exchange supplies TOKEN counts, not a per-ubatch request max.
    per_ubatch = tuple(
        tuple(
            parts[i].token_slice.stop - parts[i].token_slice.start for parts in splits
        )
        for i in range(2)
    )
    maxima = tuple(max(rows) for rows in per_ubatch)
    prefill, decode = layouts[rank]
    device = torch.device("cuda", torch.cuda.current_device())

    def metadata(lengths, slots, state):
        return SimpleNamespace(
            cu_seqlens_q=torch.tensor(
                [0, *np.cumsum(lengths)], device=device, dtype=torch.int32
            ),
            state_slot_out=torch.tensor(slots, device=device, dtype=torch.int32),
            max_seqlen_q=max(lengths),
            max_seqlen_k=128,
            has_cached=True,
            total_kv=128,
            state=state,
            swa_dest_rows=None,
        )

    pslots = list(range(10, 10 + len(prefill)))
    dslots = list(range(20, 20 + len(decode)))
    parent = _mixed_carrier(
        metadata(prefill, pslots, AttnState.PREFILL_PREFIX),
        metadata(decode, dslots, AttnState.DECODE),
        len(prefill),
        len(decode),
        sum(prefill),
        sum(decode),
        8,
    )
    published = torch.full((8,), -99, device=device, dtype=torch.int32)
    builder = DeepseekV4AttentionMetadataBuilder.__new__(
        DeepseekV4AttentionMetadataBuilder
    )
    builder.model_runner = SimpleNamespace(
        forward_vars={"v4_meta_state_slot_out": SimpleNamespace(gpu=published)}
    )

    def rebuild_prefill(source, part, running_bs, ubatch_idx):
        # Only the prefill kernel-plan construction is stubbed. Exercise the
        # real mixed rebuild, carrier, wrapper workers and final state publish.
        rs, ts = part.request_slice, part.token_slice
        cu = np.asarray([0, *np.cumsum(prefill)])
        lens = np.minimum(cu[rs.start + 1 : rs.stop + 1], ts.stop) - np.maximum(
            cu[rs.start : rs.stop], ts.start
        )
        return metadata(lens, pslots[rs], AttnState.PREFILL_PREFIX)

    monkeypatch.setattr(builder, "_rebuild_prefill_segment", rebuild_prefill)
    tokens = sum(prefill + decode)
    positions = torch.arange(tokens, device=device)
    ctx = ForwardContext(
        attn_metadata=parent,
        context=Context(
            positions=positions,
            is_prefill=True,
            scheduled_bs=len(prefill + decode),
            running_bs=8,
            scheduled_tokens=tokens,
            running_tokens=tokens,
            running_tokens_are_unified=False,
            running_tokens_across_dp=tuple(sum(p + d) for p, d in layouts),
        ),
        dp_metadata=DPMetadata(np.asarray([sum(p + d) for p, d in layouts])),
        ubatch_slices=splits[rank],
        ub_max_tokens_across_dp=maxima,
        ub_tokens_across_dp=per_ubatch,
    )
    observed = []

    class Model(torch.nn.Module):
        def forward(self, ids, positions):
            child = get_forward_context()
            i = int(child.context.ubatch_token_offset > 0)
            part = splits[rank][i]
            local_rows = part.request_slice.stop - part.request_slice.start
            assert child.context.running_bs == local_rows
            assert child.attn_metadata.state_slot_out.numel() == local_rows
            assert child.spec_decode_metadata is None
            padded, local_tokens = pad_for_all_gather(ids.float()[:, None], fill=0)
            assert padded.shape == (maxima[i], 1)
            assert child.dp_metadata.max_tokens_across_dp == maxima[i]
            observed.append((i, local_rows, padded.shape[0]))
            return padded[:local_tokens]

    wrapper = UBatchWrapper(Model(), builder, dp_gather_scatter=True)
    monkeypatch.setattr(_forward_context_local, "ctx", ctx, raising=False)
    for sentinel in (-99, -123):
        published.fill_(sentinel)
        observed.clear()
        result = wrapper(positions, positions)
        torch.testing.assert_close(result[:, 0], positions.float())
        assert len(observed) == 2
        assert get_forward_context() is ctx
        assert torch.all(published == sentinel)  # no draft publication in workers
        builder.publish_draft_state(get_forward_context().attn_metadata)
        assert published.tolist() == pslots + dslots + [0] * (8 - len(pslots + dslots))
