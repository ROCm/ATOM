# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

"""Unit tests for the simulated-TP collective wrappers.

`_patch_group` makes a TP group report a wider `world_size` than it has ranks.
What has to hold is that the collectives keep talking to the ranks that exist
while presenting logical-width results, absent ranks reading as zero.

Single-rank gloo group: the padding and slicing are local, and multi-rank
behaviour belongs to NCCL rather than to this code.
"""

import types

import pytest
import torch
import torch.distributed as dist

from atom.distributed.simulated_tp import _patch_group, reject_simulated_tp

LOGICAL = 4
PHYSICAL = 1


def _unpatched(*args, **kwargs):
    raise AssertionError("collective was not replaced by _patch_group")


@pytest.fixture
def group():
    """A stub TP coordinator backed by a real single-rank gloo group."""
    if not dist.is_available():
        pytest.skip("torch.distributed is unavailable")
    created = not dist.is_initialized()
    if created:
        dist.init_process_group(
            backend="gloo",
            init_method="tcp://127.0.0.1:29591",
            world_size=1,
            rank=0,
        )
    grp = types.SimpleNamespace(
        device_group=dist.group.WORLD,
        rank_in_group=0,
        ranks=[0],
        world_size=PHYSICAL,
        # Raises so the tests below prove the patch replaced it: a lone rank has
        # no communicator to dispatch to.
        all_reduce=_unpatched,
    )
    _patch_group(grp, LOGICAL, PHYSICAL)
    yield grp
    if created:
        dist.destroy_process_group()


def test_world_size_reports_logical(group):
    # The one number every shard-size computation in the tree reads.
    assert group.world_size == LOGICAL


def test_all_reduce_is_identity_on_a_lone_rank(group):
    # PHYSICAL == 1: no peers, and no device communicator was ever built, so
    # the patch must answer locally instead of dispatching.
    x = torch.arange(6, dtype=torch.float32).reshape(2, 3)

    assert group.all_reduce(x) is x


@pytest.mark.parametrize("dim", [0, 1, -1])
def test_all_gather_is_logical_width_with_absent_ranks_zeroed(group, dim):
    x = torch.arange(6, dtype=torch.float32).reshape(2, 3) + 1

    out = group.all_gather(x, dim=dim)

    axis = dim % x.dim()
    expected = list(x.shape)
    expected[axis] *= LOGICAL
    assert list(out.shape) == expected
    present, absent = out.split([x.shape[axis], x.shape[axis] * (LOGICAL - 1)], axis)
    assert torch.equal(present, x)
    assert not absent.any(), "shards of ranks that do not exist must read as 0"


def test_gather_matches_all_gather_on_dst(group):
    x = torch.arange(4, dtype=torch.float32).reshape(2, 2)

    assert torch.equal(group.gather(x, dst=0, dim=0), group.all_gather(x, dim=0))


def test_reduce_scatter_returns_this_ranks_logical_slice(group):
    # LOGICAL shards along dim 0; rank 0 keeps the first.
    x = torch.arange(8, dtype=torch.float32).reshape(LOGICAL * 2, 1)

    out = group.reduce_scatter_tensor(x, dim=0)

    assert torch.equal(out, x[:2])
    assert torch.equal(group.reduce_scatter(x, dim=0), out)


def test_reduce_scatter_rejects_indivisible_input(group):
    with pytest.raises(AssertionError, match="logical TP size"):
        group.reduce_scatter_tensor(torch.zeros(LOGICAL * 2 + 1, 1), dim=0)


def _config(logical: int, physical: int):
    return types.SimpleNamespace(tensor_parallel_size=logical, tp_world_size=physical)


def test_reject_simulated_tp_only_fires_when_simulating():
    reject_simulated_tp(_config(8, 8), "pipeline parallel")  # no-op
    with pytest.raises(ValueError, match="pipeline parallel"):
        reject_simulated_tp(_config(8, 4), "pipeline parallel")
