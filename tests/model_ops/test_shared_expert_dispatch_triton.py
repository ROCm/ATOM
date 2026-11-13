"""The Triton remap must match the torch reference bit-exactly.

Covers only the EPLB-off path: with EPLB on the shared expert is an ordinary
routed logical expert and none of this runs.
"""

import pytest
import torch

pytest.importorskip("triton", reason="Triton dispatch tests require triton")

from atom.model_ops.fused_moe.gathered_shared_ids import copy_shared_expert_ids_
from atom.model_ops.fused_moe.shared_expert_dispatch import (
    _remap_torch,
    remap_topk_to_dispatch,
)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="Triton path needs a GPU"
)

NUM_ROUTED, EP_SIZE, NUM_SHARED = 256, 8, 1
ROUTED_PER_RANK = NUM_ROUTED // EP_SIZE  # 32
STRIDE = ROUTED_PER_RANK + NUM_SHARED  # 33


def _shared_base(ep_rank: int) -> int:
    return ep_rank * STRIDE + ROUTED_PER_RANK


@pytest.mark.parametrize("ep_rank", [0, 3, 7])
@pytest.mark.parametrize("num_tokens", [1, 17, 512])
def test_triton_matches_torch_reference(ep_rank, num_tokens):
    torch.manual_seed(1234 + ep_rank + num_tokens)
    topk = 8
    args = (NUM_ROUTED, ROUTED_PER_RANK, _shared_base(ep_rank), NUM_SHARED, 0.4)

    ids = torch.randint(
        0, NUM_ROUTED, (num_tokens, topk), dtype=torch.int32, device="cuda"
    )
    # -1 sentinel: floor division sends it to -2 without the in-range guard.
    ids[torch.rand_like(ids, dtype=torch.float) < 0.1] = -1
    weights = torch.rand((num_tokens, topk), dtype=torch.float32, device="cuda")

    w_ref, ids_ref = _remap_torch(weights.clone(), ids.clone(), *args)
    w_got, ids_got = remap_topk_to_dispatch(weights.clone(), ids.clone(), *args)

    assert ids_got.shape == (num_tokens, topk + NUM_SHARED)
    assert torch.equal(ids_got, ids_ref)
    assert torch.equal(w_got, w_ref)


def test_shared_column_is_this_ranks_constant():
    """The point of the remap: one id per rank, resolving to that rank itself."""
    ids = torch.randint(0, NUM_ROUTED, (32, 8), dtype=torch.int32, device="cuda")
    weights = torch.rand((32, 8), dtype=torch.float32, device="cuda")

    for ep_rank in range(EP_SIZE):
        base = _shared_base(ep_rank)
        _, out_ids = remap_topk_to_dispatch(
            weights.clone(), ids.clone(), NUM_ROUTED, ROUTED_PER_RANK, base, 1, 0.4
        )
        shared_col = out_ids[:, -1]
        assert torch.all(shared_col == base), (ep_rank, shared_col[:4], base)
        assert base // STRIDE == ep_rank


def test_routed_ids_land_on_the_owning_rank():
    """Dispatch ids must resolve to the same rank MoRI would compute."""
    ids = torch.arange(NUM_ROUTED, dtype=torch.int32, device="cuda").reshape(32, 8)
    weights = torch.zeros((32, 8), dtype=torch.float32, device="cuda")

    _, out_ids = remap_topk_to_dispatch(
        weights, ids, NUM_ROUTED, ROUTED_PER_RANK, _shared_base(0), 1, 0.4
    )
    for physical_id, dispatch_id in enumerate(out_ids[:, :8].flatten().tolist()):
        # mori: destPe = destExpert / numExpertPerRank (internode.hpp).
        assert dispatch_id // STRIDE == physical_id // ROUTED_PER_RANK
        assert dispatch_id % STRIDE == physical_id % ROUTED_PER_RANK


def test_empty_batch_matches_torch():
    """ntok==0 takes the torch early-out; shapes must still line up."""
    ids = torch.empty((0, 8), dtype=torch.int32, device="cuda")
    weights = torch.empty((0, 8), dtype=torch.float32, device="cuda")

    w_got, ids_got = remap_topk_to_dispatch(
        weights, ids, NUM_ROUTED, ROUTED_PER_RANK, _shared_base(2), 1, 0.4
    )

    assert ids_got.shape == (0, 9)
    assert w_got.shape == (0, 9)


@pytest.mark.parametrize("rows", [1, 257, 4096])
@pytest.mark.parametrize("num_shared", [1, 3])
@pytest.mark.parametrize("strided", [False, True])
@pytest.mark.parametrize("graph_replay", [False, True], ids=["eager", "graph"])
def test_gathered_shared_ids_match_cpu_through_idle_replay(
    rows, num_shared, strided, graph_replay
):
    """The fused copy preserves routed columns and the read-only template,
    including strided views and a real -> all-idle -> real replay."""
    top_k = 4
    width = top_k + num_shared
    step = 2 if strided else 1
    storage = torch.full(
        (rows + 2, width * step), -77, dtype=torch.int32, device="cuda"
    )
    ids = storage[1:-1, ::step]
    template_storage = torch.full_like(storage, 99)
    template = template_storage[1:-1, top_k * step :: step]
    template[::2] = 8
    before = template_storage.clone()
    routed = (
        torch.arange(rows * top_k, dtype=torch.int32, device="cuda").view(rows, top_k)
        % 8
    )
    routed[::3] = -1
    ids[:, :top_k] = routed

    if graph_replay:
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            copy_shared_expert_ids_(ids, template, top_k)
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            copy_shared_expert_ids_(ids, template, top_k)

    for idle in (False, True, False):
        ids[:, :top_k] = -1 if idle else routed
        expected = ids.cpu()
        copy_shared_expert_ids_(expected, template.cpu(), top_k)
        if graph_replay:
            graph.replay()
        else:
            copy_shared_expert_ids_(ids, template, top_k)
        torch.cuda.synchronize()
        assert torch.equal(ids.cpu(), expected)
        assert torch.equal(template_storage, before)
        assert (storage[[0, -1]] == -77).all()
        if strided:
            assert (storage[:, 1::2] == -77).all()
