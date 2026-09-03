# SPDX-License-Identifier: MIT
"""MTP verify under DCP: the sparse indexer's per-query-token local KV lengths.

At qlen=1 an indexer row is a request, so this rank's local KV length is a
per-request number and `dcp_local_context_lens` is all the exchange needs. MTP
verify breaks that: a row is a query token, and draft position j of a request
only sees the global prefix ``ctx - next_n + 1 + j``. How many of those tokens
live on this rank does NOT grow by one per j -- under interleave S it grows by S
once every W positions -- so the whole verify window needs its own length per
(request, draft position) pair.

Getting that distinction wrong is silent: the paged-MQA-logits kernel would
simply score a slightly short prefix and drop the most recent owned tokens from
the local candidate set, which costs accuracy without ever raising. Hence a
brute-force reference rather than an algebraic restatement of the implementation.
"""

import pytest
import torch

try:
    from atom.model_ops.dcp_ops import dcp_local_context_lens
except ImportError as _e:  # triton/aiter absent on a CPU-only runner
    pytest.skip(f"requires full atom import env: {_e}", allow_module_level=True)

if not torch.cuda.is_available():
    pytest.skip("Triton kernels need a GPU", allow_module_level=True)

DEV = "cuda"


class _FakeMeta:
    """Metadata builders that do not publish the buffers hit the fallback."""

    def __init__(self, ctx, max_seqlen_q=1):
        self.context_lens = ctx
        self.max_seqlen_q = max_seqlen_q


def _owned_before(prefix_len, rank, world, interleave):
    return sum(
        1 for p in range(max(0, prefix_len)) if (p // interleave) % world == rank
    )


def test_fused_dcp_topk_merge_availability_is_cached_bool():
    from atom.model_ops.dcp_ops import fused_dcp_topk_merge_available

    first = fused_dcp_topk_merge_available()
    assert isinstance(first, bool)
    assert fused_dcp_topk_merge_available() is first


@pytest.mark.parametrize("rank, world", [(0, 4), (2, 4), (3, 4), (1, 2)])
@pytest.mark.parametrize("interleave", [1, 16])
@pytest.mark.parametrize("next_n", [1, 2, 4])
def test_local_lens_are_per_query_token(rank, world, interleave, next_n):
    # Short contexts are in here on purpose: a cudagraph-padded batch and a
    # just-started request both produce contexts below the verify window, where
    # the visible prefix must clamp at 0 rather than go negative.
    ctx = torch.tensor(
        [1, 2, 3, 4, 17, 64, 65, 999, 4096, 100000, 100001],
        dtype=torch.int32,
        device=DEV,
    )
    bs = int(ctx.shape[0])
    ref = torch.tensor(
        [
            [
                _owned_before(int(c) - next_n + 1 + j, rank, world, interleave)
                for j in range(next_n)
            ]
            for c in ctx
        ],
        dtype=torch.int32,
        device=DEV,
    )

    per_token = dcp_local_context_lens(
        _FakeMeta(ctx), rank, world, interleave, bs, next_n
    )

    assert per_token.dtype == torch.int32
    # Row-major (request-major): indexer row r is request r // next_n at draft
    # position r % next_n, matching how q/weights/logits are laid out.
    torch.testing.assert_close(per_token, ref.reshape(-1), rtol=0, atol=0)


@pytest.mark.parametrize("world", [2, 4])
@pytest.mark.parametrize("interleave", [1, 16])
def test_last_draft_position_matches_the_plain_per_request_length(world, interleave):
    """next_n-1 sees the full context, so it must equal the qlen=1 answer.

    This is the anchor that catches an off-by-one in the visible prefix: the
    last draft position of an MTP step covers exactly the context a plain decode
    step would have.
    """
    ctx = torch.randint(1, 100000, (23,), dtype=torch.int32, device=DEV)
    bs, next_n = int(ctx.shape[0]), 4
    for rank in range(world):
        plain = dcp_local_context_lens(_FakeMeta(ctx), rank, world, interleave, bs)
        mtp = dcp_local_context_lens(
            _FakeMeta(ctx), rank, world, interleave, bs, next_n
        ).view(bs, next_n)
        torch.testing.assert_close(mtp[:, next_n - 1], plain, rtol=0, atol=0)


@pytest.mark.parametrize("world", [2, 4])
@pytest.mark.parametrize("next_n", [1, 4])
def test_unified_published_buffer_is_used_as_is(world, next_n):
    """One flat buffer publishes both plain and MTP request-major layouts."""
    bs = 5
    meta = _FakeMeta(
        torch.full((bs,), 1000, dtype=torch.int32, device=DEV),
        max_seqlen_q=next_n,
    )
    # Deliberately not the right answer -- identity is the assertion.
    meta.dcp_local_context_lens = torch.arange(
        bs * next_n, dtype=torch.int32, device=DEV
    )
    local_lens = dcp_local_context_lens(meta, 0, world, 1, bs, next_n)
    assert local_lens.data_ptr() == meta.dcp_local_context_lens.data_ptr()


def test_wrong_width_published_buffer_falls_back():
    """A verify [B*N] buffer must never be consumed by qlen=1 draft decode."""
    bs, next_n, world = 5, 4, 4
    ctx = torch.full((bs,), 1000, dtype=torch.int32, device=DEV)
    meta = _FakeMeta(ctx, max_seqlen_q=next_n)
    meta.dcp_local_context_lens = torch.full(
        (bs * next_n,), -1, dtype=torch.int32, device=DEV
    )
    got = dcp_local_context_lens(meta, 0, world, 1, bs)
    expected = dcp_local_context_lens(_FakeMeta(ctx), 0, world, 1, bs)
    torch.testing.assert_close(got, expected, rtol=0, atol=0)
