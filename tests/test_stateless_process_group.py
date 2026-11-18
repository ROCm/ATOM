# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""``stateless_init_torch_distributed_process_group`` building an NCCL group.

Its store keys have to sit where the trainer's end of the group puts them,
under the group's name, or the two ends never meet. The run over real GPUs is
in ``test_rdma_weight_stream_multiprocess.py``.
"""

from datetime import timedelta

import pytest

from atom.utils.distributed import utils


class _Store:
    def __init__(self):
        self.timeouts = []

    def set_timeout(self, timeout):
        self.timeouts.append(timeout)


@pytest.fixture
def built(monkeypatch):
    """What the helper hands torch, with no rendezvous or NCCL behind it."""
    record = {"store": _Store()}

    def rendezvous(url, rank, world_size, timeout):
        record["rendezvous"] = (url, rank, world_size, timeout)
        return iter([(record["store"], rank, world_size)])

    def new_group(size, rank, ranks, backend, store, **kw):
        record["group"] = (size, rank, ranks, str(backend), store, kw)
        return record.setdefault("pg", object()), None

    monkeypatch.setattr(utils, "rendezvous", rendezvous)
    monkeypatch.setattr(utils, "PrefixStore", lambda prefix, store: (prefix, store))
    monkeypatch.setattr(utils, "_new_process_group_helper", new_group)
    yield record
    utils._world.pg_group_ranks.pop(record.get("pg"), None)


def test_an_nccl_group_is_named_registered_and_timed_out_once(built):
    timeout = timedelta(seconds=42)

    pg = utils.stateless_init_torch_distributed_process_group(
        "10.0.0.1", 29500, 2, 3, "nccl", group_name="weights", timeout=timeout
    )

    assert built["rendezvous"] == ("tcp://10.0.0.1:29500", 2, 3, timeout)
    assert built["store"].timeouts == [timeout]
    store = ("weights", built["store"])
    assert built["group"] == (
        3,
        2,
        [],
        "nccl",
        store,
        {"group_name": "weights", "timeout": timeout},
    )
    # broadcast(src=0) maps the group rank through this; the default world
    # knows nothing of the group.
    assert utils._world.pg_group_ranks[pg] == {0: 0, 1: 1, 2: 2}


def test_a_zero_timeout_is_the_one_given(built):
    """``timeout or default`` read timedelta(0) as no timeout given."""
    utils.stateless_init_torch_distributed_process_group(
        "10.0.0.1", 29500, 1, 2, "nccl", group_name="w", timeout=timedelta(0)
    )
    assert built["rendezvous"][3] == timedelta(0)


def test_a_group_whose_shutdown_raised_frees_its_name_once_forgotten():
    """torch drops its record of a group only after the shutdown returns, so a
    group whose shutdown raised kept its name, and none could be built under
    it again."""
    import torch.distributed as dist

    store = dist.HashStore()

    def build():
        pg, _ = utils._new_process_group_helper(
            1,
            0,
            [],
            utils.Backend("gloo"),
            utils.PrefixStore("forget-test", store),
            group_name="forget-test",
            timeout=timedelta(seconds=5),
        )
        return pg

    pg = build()
    with pytest.raises(ValueError, match="already been created"):
        build()

    utils.forget_process_group(pg)
    again = build()

    assert again in utils._world.pg_map and pg not in utils._world.pg_map
    utils.forget_process_group(again)


def test_an_nccl_group_needs_a_name_before_any_rendezvous(built):
    with pytest.raises(ValueError, match="group_name"):
        utils.stateless_init_torch_distributed_process_group(
            "10.0.0.1", 29500, 1, 2, "nccl"
        )
    assert "rendezvous" not in built
