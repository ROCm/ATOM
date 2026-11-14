# SPDX-License-Identifier: MIT
"""A DP gather moves a rank's routed rows -- hidden states, top-k weights,
top-k ids -- in one `all_gatherv`, eager or uniform, not one gather per
tensor. CPU only: the group is a stand-in for two ranks, rank 1 holding
`peer_rows` rows of zeros."""

from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch
from aiter_stub import stubbed_aiter

from atom.model_ops.fused_moe.shared_expert_dispatch import _remap_torch
from atom.utils import forward_context as fc

with (
    stubbed_aiter(
        "aiter.fused_moe",
        "aiter.jit.utils.chip_info",
        "aiter.jit.utils.torch_guard",
        "aiter.ops.flydsl.moe_common",
        "aiter.ops.shuffle",
        "aiter.ops.triton.quant",
        "aiter.utility.fp4_utils",
        "triton",
        "triton.language",
        # `topK` imports `triton_topk_softmax`, which takes `libdevice` from here.
        "triton.language.extra",
        "triton.language.extra.hip",
    ),
    # These CPU contracts never call CUDA custom ops. Registering them would
    # infer schemas from the fake AITER enum defaults and retain stub kernels
    # in torch's process-wide registry after stubbed_aiter removes its modules.
    patch("atom.utils.custom_register.direct_register_custom_op"),
):
    from atom.model_ops import eplb, moe, topK
    from atom.model_ops.fused_moe.expert_layout import SharedExpertMode


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
        context=SimpleNamespace(running_tokens=3),
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


@pytest.mark.parametrize("rank", [None, 0, 1], ids=["tp", "ep-0", "ep-1"])
@pytest.mark.parametrize("num_shared", [1, 3])
@pytest.mark.parametrize("rows", [0, 4], ids=["empty", "mixed"])
@pytest.mark.parametrize("compiled", [False, True], ids=["eager", "compiled"])
def test_gather_does_not_restore_shared_experts_on_dummy_or_padded_rows(
    monkeypatch, rank, num_shared, rows, compiled
):
    layer = SimpleNamespace(
        expert_layout=SimpleNamespace(mode=SharedExpertMode.LEGACY_AITER),
        top_k=2,
    )
    # Rank 0's dummy followed by rank 1's two real rows and one padded row.
    # Shared columns were prefilled before routing masked the routed columns.
    routed = torch.tensor([[-1, -1], [0, 1], [1, -1], [-1, -1]], dtype=torch.int32)[
        :rows
    ]
    ids = torch.cat((routed, torch.full((rows, num_shared), 8, dtype=torch.int32)), 1)
    template = torch.zeros_like(ids)
    shared_ids = torch.arange(8, 8 + num_shared, dtype=torch.int32)
    if rank is None:
        # TP computes every real token's shared-expert weight shard.
        template[:, 2:] = shared_ids
    else:
        template[:, 2:] = 8 + num_shared  # fake ID on unassigned EP rows
        template[rank::2, 2:] = shared_ids
    before = template.clone()
    monkeypatch.setattr(topK, "aiter_topK_meta_data", (torch.ones_like(ids), template))

    reassign = topK.reassign_gathered_shared_ids
    if compiled:
        reassign = torch.compile(reassign, backend="eager", fullgraph=True)
    reassign(layer, ids)

    invalid = routed[:, 0] < 0
    assert (ids[invalid] == -1).all()
    assert torch.equal(ids[:, :2], routed)
    assert torch.equal(ids[~invalid, 2:], template[~invalid, 2:])
    assert torch.equal(template, before)
    # A subsequent real replay must still find the complete prefilled table.
    real = torch.zeros_like(ids)
    reassign(layer, real)
    assert torch.equal(real[:, 2:], before[:, 2:])


@pytest.mark.parametrize(
    "mode",
    [
        SharedExpertMode.NONE,
        SharedExpertMode.LOCAL_REPLICA,
        SharedExpertMode.EPLB_ROUTED,
    ],
)
def test_gathered_shared_reassignment_bypasses_other_layouts(monkeypatch, mode):
    layer = SimpleNamespace(expert_layout=SimpleNamespace(mode=mode))
    ids = torch.tensor([[0, 4], [-1, -1]], dtype=torch.int32)
    before = ids.clone()
    copy = Mock(side_effect=AssertionError("non-legacy layout reached shared copying"))
    monkeypatch.setattr(topK, "copy_shared_expert_ids_", copy)
    monkeypatch.setattr(topK, "aiter_topK_meta_data", None)

    topK.reassign_gathered_shared_ids(layer, ids)

    copy.assert_not_called()
    assert torch.equal(ids, before)


@pytest.mark.parametrize("placement", ["local-replica", "eplb"])
@pytest.mark.parametrize("num_shared", [1, 3])
@pytest.mark.parametrize("rows", ["mixed", "all-dummy", "empty"])
@pytest.mark.parametrize("compiled", [False, True], ids=["eager", "compiled"])
def test_shared_placement_preserves_padding(placement, num_shared, rows, compiled):
    """Shared placement agrees across the two torch fallback layouts, including
    a partially masked real row, an idle rank, and multiple shared columns."""
    ids = torch.tensor([[0, 4], [3, -1], [-1, -1]], dtype=torch.int32)
    weights = torch.tensor([[0.7, 0.3], [0.6, 0.0], [0.0, 0.0]])
    if rows == "all-dummy":
        ids.fill_(-1)
        weights.zero_()
    elif rows == "empty":
        ids, weights = ids[:0], weights[:0]

    def append(w, i):
        if placement == "eplb":
            return eplb._append_shared_torch(w, i, num_shared, 8, 0.5)
        return _remap_torch(w, i, 8, 4, 8, num_shared, 0.5)

    if compiled:
        append = torch.compile(append, backend="eager", fullgraph=True)
    out_w, out_ids = append(weights, ids)

    assert out_ids.shape == out_w.shape == (ids.shape[0], 2 + num_shared)
    assert torch.equal(out_w[:, :2], weights)
    if rows == "mixed":
        assert torch.equal(
            out_ids[:2, 2:], torch.arange(8, 8 + num_shared).expand(2, -1)
        )
        assert (out_w[:2, 2:] == 0.5).all()
        assert out_ids[1, 1] == -1  # A masked later slot does not pad a real row.
        assert (out_ids[2] == -1).all() and (out_w[2] == 0).all()
    else:
        assert (out_ids == -1).all() and (out_w == 0).all()


@pytest.mark.parametrize("use_ep", [False, True], ids=["tp", "ep"])
@pytest.mark.parametrize("dummy", [False, True], ids=["real", "dp-dummy"])
def test_gathered_rows_are_placed_after_the_gather_and_uncounted(
    monkeypatch, use_ep, dummy
):
    """Placement (the EPLB remap, which shifts the shared column) follows the
    gather and the shared-id re-mark. Only the pass's whole ungathered rows
    carry its count, under both TP and EP."""
    calls = []
    method = SimpleNamespace(
        place_experts=lambda layer, w, ids, gathered: (
            calls.append(("place", gathered)) or (w, ids)
        ),
        apply=lambda **kw: calls.append(("apply", kw["scheduled_tokens_gpu"])) or 0,
    )
    layer = SimpleNamespace(
        moe_parallel_config=SimpleNamespace(
            execution_plan=SimpleNamespace(needs_pcp_collectives=False)
        ),
        quant_method=method,
        global_num_experts=4,
        expert_map=None,
        activation=None,
        apply_router_weight_on_input=False,
        prefix="t",
        use_ep=use_ep,
    )
    count = torch.tensor([0 if dummy else 2], dtype=torch.int32)
    context = SimpleNamespace(
        is_dummy_run=dummy,
        pcp_split=False,
        running_tokens=3,
        scheduled_tokens_gpu=count,
    )
    forward = SimpleNamespace(context=context)
    monkeypatch.setattr(moe, "get_forward_context", lambda: forward)
    monkeypatch.setattr(fc, "get_forward_context", lambda: forward)
    w, ids = torch.rand(3, 2), torch.zeros(3, 2, dtype=torch.int32)

    moe.FusedMoE._run_experts(layer, torch.zeros(3, 8), w, ids, gathered=True)
    moe.FusedMoE._run_experts(layer, torch.zeros(3, 8), w, ids)
    moe.FusedMoE._run_experts(layer, torch.zeros(1, 8), w[:1], ids[:1])

    assert calls == [
        ("place", True),
        ("apply", None),
        ("place", False),
        ("apply", count),
        ("place", False),
        ("apply", None),
    ]


def test_gathered_rows_pick_one_replica_on_every_rank(monkeypatch):
    """The forced-remote replica hash mixes in the DP rank only for a rank's
    own rows; gathered rows are hashed by the row alone."""
    monkeypatch.setattr(eplb, "_MANAGER", SimpleNamespace(replica_spread=(4, 3)))
    assert eplb._replica_spread() == (4, 3)
    assert eplb._replica_spread(gathered=True) == (1, 0)
