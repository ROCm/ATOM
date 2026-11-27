# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

"""The slicing reduce-scatter copy-in must be bit-identical to torch._chunk_cat.

Needs one GPU (the reference kernel is CUDA/HIP only) but no model and no
distributed setup. Run: python -m atomrl.tests.test_fsdp_chunk_cat_fallback
"""

import pytest
import torch
from atomrl.engine.training.fsdp_chunk_cat_fallback import reduce_scatter_copy_in

pytestmark = pytest.mark.gpu

WORLD = 8

# One Qwen3-30B-A3B transformer block: the group whose copy-in aborted the queue.
QWEN3_MOE_LAYER = [
    (4096, 2048),
    (512, 2048),
    (512, 2048),
    (2048, 4096),  # q, k, v, o
    (128,),
    (128,),
    (2048,),
    (2048,),  # q_norm, k_norm, 2x layernorm
    (128, 2048),  # mlp.gate
    (128, 1536, 2048),
    (128, 2048, 768),  # fused experts
]


def _padded_numel(shapes, world=WORLD):
    total = 0
    for s in shapes:
        rows = -(-s[0] // world) * world
        total += rows * (torch.Size(s).numel() // s[0])
    return total


def _compare(shapes, dtype=torch.bfloat16, out_dtype=torch.float32, world=WORLD):
    torch.manual_seed(0)
    grads = [torch.randn(s, dtype=dtype, device="cuda") for s in shapes]
    numel = _padded_numel(shapes, world)

    want = torch.zeros(numel, dtype=out_dtype, device="cuda")
    torch._chunk_cat(grads, dim=0, num_chunks=world, out=want.view(world, -1))

    got = torch.zeros(numel, dtype=out_dtype, device="cuda")
    reduce_scatter_copy_in(grads, got, world)

    torch.cuda.synchronize()
    assert torch.equal(
        got, want
    ), f"{shapes}: {int((got != want).sum())} elements differ"


def test_matches_on_the_real_moe_layer():
    _compare(QWEN3_MOE_LAYER)


def test_matches_when_rows_need_padding():
    # dim 0 not divisible by world_size, so chunk_cat zero-pads the tail.
    _compare([(13, 7), (1, 4), (17,), (9, 2, 3)])


def test_matches_when_a_chunk_is_all_padding():
    # rows < world_size: the last chunks get no real rows at all.
    _compare([(3, 5), (1, 1)])


def test_matches_without_dtype_promotion():
    _compare([(64, 32), (33, 8)], dtype=torch.bfloat16, out_dtype=torch.bfloat16)


def test_matches_at_other_world_sizes():
    for world in (1, 2, 4):
        _compare([(64, 32), (33, 8), (7,)], world=world)


def test_matches_on_noncontiguous_grads():
    """Autograd can hand back non-contiguous gradients; copy_ must still agree."""
    torch.manual_seed(1)
    base = torch.randn(64, 128, dtype=torch.bfloat16, device="cuda")
    grads = [base.t()[:32], base[:16]]
    shapes = [tuple(g.shape) for g in grads]
    numel = _padded_numel(shapes)

    want = torch.zeros(numel, dtype=torch.float32, device="cuda")
    torch._chunk_cat(grads, dim=0, num_chunks=WORLD, out=want.view(WORLD, -1))
    got = torch.zeros(numel, dtype=torch.float32, device="cuda")
    reduce_scatter_copy_in(grads, got, WORLD)
    torch.cuda.synchronize()
    assert torch.equal(got, want)


if __name__ == "__main__":
    if not torch.cuda.is_available():
        raise SystemExit("needs one GPU for the torch._chunk_cat reference")
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"  {name} ok")
    print("all FSDP2 chunk_cat fallback tests passed")
