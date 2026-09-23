"""The Triton remap must match the torch reference bit-exactly.

Covers only the EPLB-off path: with EPLB on the shared expert is an ordinary
routed logical expert and none of this runs.
"""

import pytest
import torch

pytest.importorskip("triton", reason="Triton dispatch tests require triton")

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
