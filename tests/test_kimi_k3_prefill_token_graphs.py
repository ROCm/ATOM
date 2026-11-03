# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Kimi-K3 prefill on token graphs: the padded-step contract its regions keep.

A K3 prefill padded to a recorded bucket has metadata for its real rows only,
marked on the step (`AttentionMetaData.prefill_real_rows`). Its KDA and MLA
regions cut their bucket-high inputs back to those rows -- an FP8 group scale
written group-major is rebuilt the way an unpadded producer lays it out -- and
hand their output back at bucket height, the shape the next recorded segment
reads. What these pin, on the CPU (`atom/utils/token_graph_rows.py`):

  * the cut and the hand-back, and that both are the identity off such a step;
  * the group-major scale rebuild;
  * that only a marked step is treated as padded;
  * the bucket a step pads to at DP = 1, where the group agreement is the
    rank's own choice;
  * the pad-without-graphs switch, which runs the padded step eagerly.
"""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from atom.utils import envs, forward_context, region_graph
from atom.utils.forward_context import AttentionMetaData, ForwardMode
from atom.utils.token_graph_rows import padded_prefill_rows, real_rows, to_rows

BUCKET, REAL = 512, 300


def _group_major(rows: int, groups: int) -> torch.Tensor:
    """A (rows, groups) scale laid out as `transpose_scale` writes it: row r of
    group g at g * rows + r."""
    return (
        torch.arange(groups * rows, dtype=torch.float32)
        .reshape(groups, rows)
        .view(rows, groups)
    )


# ── cut to the real rows ─────────────────────────────────────────────────────


def test_only_bucket_high_tensors_are_cut():
    q = torch.randn(BUCKET, 8)
    assert torch.equal(real_rows(q, REAL, BUCKET), q[:REAL])
    other = torch.randn(7, 8)
    assert real_rows(other, REAL, BUCKET) is other
    assert real_rows(None, REAL, BUCKET) is None
    scalar = torch.tensor(3.0)
    assert real_rows(scalar, REAL, BUCKET) is scalar


def test_a_group_major_scale_is_rebuilt_at_the_real_height():
    """What an unpadded producer of `REAL` rows would have written."""
    got = real_rows(_group_major(BUCKET, 4), REAL, BUCKET, groups_major=True)
    want = torch.stack(
        [torch.arange(REAL, dtype=torch.float32) + g * BUCKET for g in range(4)]
    ).view(REAL, 4)
    assert got.shape == (REAL, 4) and got.is_contiguous()
    assert torch.equal(got.reshape(4, REAL), want.reshape(4, REAL))


def test_a_per_token_scale_is_a_plain_cut():
    """One group: the two layouts coincide."""
    scale = torch.randn(BUCKET, 1)
    assert torch.equal(real_rows(scale, REAL, BUCKET, groups_major=True), scale[:REAL])


# ── hand back at bucket height ───────────────────────────────────────────────


def test_the_output_goes_back_at_bucket_height_with_zero_pad_rows():
    out = torch.randn(REAL, 16, dtype=torch.bfloat16)
    back = to_rows(out, BUCKET)
    assert back.shape == (BUCKET, 16) and back.dtype == out.dtype
    assert torch.equal(back[:REAL], out)
    assert torch.count_nonzero(back[REAL:]) == 0


def test_an_output_already_at_bucket_height_is_returned_as_is():
    out = torch.randn(BUCKET, 16)
    assert to_rows(out, BUCKET) is out


def test_more_rows_than_the_step_is_an_error():
    with pytest.raises(ValueError, match="rows"):
        to_rows(torch.randn(BUCKET + 1, 4), BUCKET)


def test_a_region_runs_the_real_rows_and_returns_the_buckets():
    """The region's shape: cut every token-major input, attend over the real
    rows only, hand the result back padded."""
    q, k = torch.randn(BUCKET, 4, 8), torch.randn(BUCKET, 8)
    q_scale = _group_major(BUCKET, 2)
    seen = {}

    def attention(q, k, q_scale):
        seen.update(q=q.shape[0], k=k.shape[0], scale=tuple(q_scale.shape))
        return q.sum(-1)

    q2, k2 = (real_rows(t, REAL, BUCKET) for t in (q, k))
    s2 = real_rows(q_scale, REAL, BUCKET, groups_major=True)
    out = to_rows(attention(q2, k2, s2), BUCKET)
    assert seen == {"q": REAL, "k": REAL, "scale": (REAL, 2)}
    assert out.shape[0] == BUCKET
    assert torch.equal(out[:REAL], q[:REAL].sum(-1))


# ── which step is padded ─────────────────────────────────────────────────────


@pytest.fixture
def step(monkeypatch):
    """Point the forward context at a step with `metadata`."""

    def set_step(metadata):
        ctx = SimpleNamespace(attn_metadata=metadata)
        monkeypatch.setattr(forward_context, "get_forward_context", lambda: ctx)

    return set_step


def test_an_unmarked_step_is_not_padded(step):
    """Every builder but a token-graph K3 prefill leaves the field unset."""
    assert AttentionMetaData().prefill_real_rows is None
    step(AttentionMetaData())
    assert padded_prefill_rows() is None
    step(None)
    assert padded_prefill_rows() is None


def test_a_marked_step_reports_its_real_rows(step):
    metadata = AttentionMetaData()
    metadata.prefill_real_rows = REAL
    step(metadata)
    assert padded_prefill_rows() == REAL


# ── the bucket at DP = 1 ─────────────────────────────────────────────────────


@pytest.fixture
def recorded(monkeypatch):
    sizes = (256, 512, 1024, 2048, 3072)
    monkeypatch.setattr(
        region_graph.TokenGraph,
        "captured_sizes",
        staticmethod(lambda phase=None: sizes if phase == "prefill" else ()),
    )
    return sizes


def _pad(tokens, graphable=True, prefill=True):
    sync = SimpleNamespace(
        scheduled_tokens_per_dp_rank=np.array([tokens]),
        prefill_across_dp=[prefill],
    )
    running = np.zeros(1, dtype=np.int64)
    return ForwardMode._pad_to_token_graph(graphable, sync, running), int(running[0])


@pytest.mark.parametrize(
    "tokens, bucket",
    [(1, 256), (300, 512), (512, 512), (513, 1024), (3072, 3072), (3073, 0)],
)
def test_one_rank_pads_to_its_own_smallest_bucket(recorded, tokens, bucket):
    """DP = 1: the group's largest row count is the rank's own, so the group
    agreement picks what the rank would; above the largest bucket, no graph."""
    got, running = _pad(tokens)
    assert got == bucket
    assert running == (bucket or tokens)


def test_no_padding_off_the_graphs_or_off_prefill(recorded):
    assert _pad(300, graphable=False) == (0, 300)
    assert _pad(300, prefill=False) == (0, 300)


# ── pad without graphs ───────────────────────────────────────────────────────


class _Recording:
    def __init__(self):
        self.replays = 0

    def has(self, key, phase=None):
        return True

    def replay(self, args, kwargs, phase=None):
        self.replays += 1
        return "replayed"


@pytest.mark.parametrize("eager_pad", [False, True])
def test_the_eager_pad_switch_runs_the_padded_step_without_the_graph(
    monkeypatch, eager_pad
):
    monkeypatch.setenv("ATOM_TOKEN_GRAPH_EAGER_PAD", "1" if eager_pad else "0")
    ctx = SimpleNamespace(tr_graph="replay")
    monkeypatch.setattr(forward_context, "get_forward_context", lambda: ctx)
    owner = SimpleNamespace(_token_graph=_Recording())
    calls = []

    def forward(x):
        calls.append(x.shape[0])
        return "eager"

    got = region_graph._run_token_graph(owner, forward, (torch.zeros(BUCKET, 2),), {})
    if eager_pad:
        assert got == "eager" and calls == [BUCKET]
        assert owner._token_graph.replays == 0
    else:
        assert got == "replayed" and calls == []


def test_the_eager_pad_switch_is_off_by_default(monkeypatch):
    monkeypatch.delenv("ATOM_TOKEN_GRAPH_EAGER_PAD", raising=False)
    assert envs.ATOM_TOKEN_GRAPH_EAGER_PAD is False
