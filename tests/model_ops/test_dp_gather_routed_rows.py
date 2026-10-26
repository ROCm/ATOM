# SPDX-License-Identifier: MIT
"""A DP gather moves a rank's routed rows -- hidden states, top-k weights,
top-k ids -- in one `all_gatherv`, eager or uniform, not one gather per
tensor. CPU only: the group is a stand-in for two ranks, rank 1 holding
`peer_rows` rows of zeros."""

from types import SimpleNamespace

import torch
from import_guard import skip_if_dependency_missing

try:
    # aiter/triton absent under bare non-GPU pytest
    from atom.model_ops import moe
except ImportError as exc:
    skip_if_dependency_missing(exc, "requires full atom import env")


class _TwoRankGroup:
    world_size = 2

    def __init__(self, peer_rows):
        self.peer_rows = peer_rows
        self.calls = []
        self.device_communicator = SimpleNamespace(all_gatherv=self._all_gatherv)

    def _all_gatherv(self, inputs, dim, sizes):
        self.calls.append(([t.dtype for t in inputs], sizes))
        return [
            torch.cat([t, t.new_zeros(self.peer_rows, *t.shape[1:])]) for t in inputs
        ]


def _rows(n):
    return (
        torch.randn(n, 8, dtype=torch.bfloat16),
        torch.rand(n, 6),
        torch.randint(-1, 384, (n, 6), dtype=torch.int32),
    )


def _check(rows, sent, peer_rows):
    for got, s in zip(rows, sent):
        assert got.dtype == s.dtype
        assert torch.equal(got[: s.shape[0]], s)
        assert got.shape[0] == s.shape[0] + peer_rows


def _set_heights(monkeypatch, scheduled, running):
    monkeypatch.setattr(
        moe,
        "get_forward_context",
        lambda: SimpleNamespace(
            context=SimpleNamespace(
                running_tokens=running, scheduled_tokens=scheduled, is_prefill=True
            )
        ),
    )


def test_uniform_rows_travel_in_one_collective(monkeypatch):
    _set_heights(monkeypatch, scheduled=3, running=3)
    sent = _rows(3)
    group = _TwoRankGroup(peer_rows=3)

    rows, local_tokens, sizes = moe.dp_gather_routed_rows(
        sent, False, SimpleNamespace(), group
    )

    # ids travel as the float of their width, the collectives' contract.
    assert group.calls == [([torch.bfloat16, torch.float32, torch.float32], [3, 3])]
    assert local_tokens == 3 and sizes is None
    _check(rows, sent, 3)


def test_uniform_pad_rows_route_nowhere(monkeypatch):
    """A uniform prefill a bridge pads past its 3 scheduled rows to 5: the pad
    rows are routed already, so they carry ids -1 and weights 0."""
    _set_heights(monkeypatch, scheduled=3, running=5)
    sent = _rows(3)
    group = _TwoRankGroup(peer_rows=5)

    (_, weights, ids), local_tokens, _ = moe.dp_gather_routed_rows(
        sent, False, SimpleNamespace(), group
    )

    assert group.calls[0][1] == [5, 5] and local_tokens == 3
    assert torch.equal(ids[:3], sent[2]) and torch.equal(weights[:3], sent[1])
    assert (ids[3:5] == -1).all() and (weights[3:5] == 0).all()


def test_eager_rows_travel_in_one_collective_with_the_dp_sizes():
    sent = _rows(3)
    group = _TwoRankGroup(peer_rows=5)
    ctx = SimpleNamespace(
        dp_metadata=SimpleNamespace(get_sizes_across_dp=lambda: [3, 5]),
        context=SimpleNamespace(scheduled_tokens=3),
    )

    rows, local_tokens, sizes = moe.dp_gather_routed_rows(sent, True, ctx, group)

    assert group.calls == [([torch.bfloat16, torch.float32, torch.float32], [3, 5])]
    assert local_tokens == 3 and sizes == [3, 5]
    _check(rows, sent, 5)


def test_gathered_shared_ids_split_the_rows_between_ep_ranks(monkeypatch):
    """LEGACY_AITER under EP: rank r's prefill gives row i the real shared id
    only where i % ep == r. Routed on their own ranks, the rows carry their
    owners' marks through the gather -- every rank would run the same rows'
    shared expert. Re-marked by gathered position, the ranks split the rows."""
    from atom.model_ops import topK
    from atom.model_ops.fused_moe.expert_layout import SharedExpertMode

    ep, top_k, shared, fake, local = 2, 2, 8, 9, 2
    layer = SimpleNamespace(
        expert_layout=SimpleNamespace(mode=SharedExpertMode.LEGACY_AITER),
        top_k=top_k,
    )

    def prefill(rank, rows):
        ids = torch.zeros(rows, top_k + 1, dtype=torch.int32)
        ids[:, top_k] = fake
        ids[rank::ep, top_k] = shared
        return torch.ones(rows, top_k + 1), ids

    # Each rank routes its own `local` rows, then the gather stacks them.
    gathered = torch.cat([prefill(rank, local)[1] for rank in range(ep)])
    runs = {}
    for rank in range(ep):
        ids = gathered.clone()
        monkeypatch.setattr(topK, "aiter_topK_meta_data", prefill(rank, 16))
        topK.reassign_gathered_shared_ids(layer, ids)
        runs[rank] = set((ids[:, top_k] == shared).nonzero().flatten().tolist())

    # Before: both ranks would run rows {0, 3}, and none rows {1, 2}.
    assert set((gathered[:, top_k] == shared).nonzero().flatten().tolist()) == {0, 3}
    assert runs[0] == {0, 2} and runs[1] == {1, 3}


def test_gathered_rows_are_placed_after_the_gather_and_uncounted(monkeypatch):
    """Placement (the EPLB remap, which shifts the shared column) follows the
    gather and the shared-id re-mark, and gathered rows carry no count."""
    calls = []
    method = SimpleNamespace(
        place_experts=lambda layer, w, ids, gathered: (
            calls.append(("place", gathered)) or (w, ids)
        ),
        apply=lambda **kw: calls.append(("apply", kw["scheduled_tokens_gpu"])) or 0,
    )
    layer = SimpleNamespace(
        moe_parallel_config=SimpleNamespace(pcp_merge=False),
        quant_method=method,
        global_num_experts=4,
        expert_map=None,
        activation=None,
        apply_router_weight_on_input=False,
        prefix="t",
    )
    count = torch.tensor([2], dtype=torch.int32)
    context = SimpleNamespace(
        is_dummy_run=False,
        pcp_split=False,
        running_tokens=3,
        scheduled_tokens_gpu=count,
    )
    monkeypatch.setattr(
        moe, "get_forward_context", lambda: SimpleNamespace(context=context)
    )
    monkeypatch.setattr(
        moe, "pass_scheduled_tokens_gpu", lambda rows: count if rows == 3 else None
    )
    w, ids = torch.rand(3, 2), torch.zeros(3, 2, dtype=torch.int32)

    moe.FusedMoE._run_experts(layer, torch.zeros(3, 8), w, ids, gathered=True)
    moe.FusedMoE._run_experts(layer, torch.zeros(3, 8), w, ids)

    assert calls == [
        ("place", True),
        ("apply", None),
        ("place", False),
        ("apply", count),
    ]


def test_gathered_rows_pick_one_replica_on_every_rank(monkeypatch):
    """The forced-remote replica hash mixes in the DP rank only for a rank's
    own rows; gathered rows are hashed by the row alone."""
    from atom.model_ops import eplb

    monkeypatch.setattr(eplb, "_MANAGER", SimpleNamespace(replica_spread=(4, 3)))
    assert eplb._replica_spread() == (4, 3)
    assert eplb._replica_spread(gathered=True) == (1, 0)
