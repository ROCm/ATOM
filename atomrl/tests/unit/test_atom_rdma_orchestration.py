# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

"""ATOM RDMA orchestration: rank layout and the guards around it.

The rank arithmetic is the part most likely to be quietly wrong. Every rank must
call into the rendezvous, and if the base ranks overlap or leave a gap, the
group never forms -- all nine ranks sit until the timeout with nothing in the
logs naming the cause. So it is worth pinning as pure arithmetic, away from Ray.
"""

import pytest
from atomrl.engine.inference.atom_ray_server import ATOMReplicaManager


def _mgr(num_replicas, tp, dp=1):
    """A manager with only the fields the rank layout reads.

    ``__init__`` needs a live worker group and ``create()`` spawns Ray actors,
    so neither can run here.
    """
    mgr = object.__new__(ATOMReplicaManager)
    mgr.num_replicas = num_replicas
    mgr.tensor_parallel_size = tp
    mgr.data_parallel_size = dp
    mgr.rdma_group_name = None
    mgr.servers = [object() for _ in range(num_replicas)]
    return mgr


def _layout(mgr):
    """The (base_rank, world_size) the manager would hand out."""
    per = mgr._ranks_per_replica
    world = 1 + mgr.num_replicas * per
    return [1 + r * per for r in range(mgr.num_replicas)], world


# ── the layout the runbook describes ───────────────────────────────────────


def test_the_qwen3_topology_is_nine_ranks():
    """4 replicas x TP2 + one trainer, which is the documented ATOM shape."""
    bases, world = _layout(_mgr(num_replicas=4, tp=2))
    assert world == 9
    assert bases == [1, 3, 5, 7]


def test_rank_zero_is_left_for_the_trainer():
    for replicas, tp in ((4, 2), (8, 1), (2, 4)):
        bases, _ = _layout(_mgr(replicas, tp))
        assert min(bases) == 1, "rank 0 belongs to the sender"


@pytest.mark.parametrize(
    ("replicas", "tp", "dp"),
    [(4, 2, 1), (8, 1, 1), (2, 4, 1), (2, 2, 2), (1, 8, 1), (3, 2, 2)],
)
def test_every_rank_is_claimed_exactly_once(replicas, tp, dp):
    """No overlap and no gap. Either one hangs the rendezvous."""
    mgr = _mgr(replicas, tp, dp)
    bases, world = _layout(mgr)
    per = mgr._ranks_per_replica

    claimed = [rank for base in bases for rank in range(base, base + per)]
    assert sorted(claimed) == list(
        range(1, world)
    ), f"ranks 1..{world - 1} must be covered once; got {sorted(claimed)}"


def test_dp_inside_a_replica_widens_its_block():
    """A replica running DP is several engines behind one handle, and each of
    their TP ranks joins separately -- so it claims tp*dp ranks, not tp."""
    assert _mgr(2, tp=2, dp=1)._ranks_per_replica == 2
    assert _mgr(2, tp=2, dp=2)._ranks_per_replica == 4

    bases, world = _layout(_mgr(num_replicas=2, tp=2, dp=2))
    assert bases == [1, 5]
    assert world == 9


def test_striding_by_tp_alone_would_overlap_under_dp():
    """Guards the specific bug: using tp as the stride when dp > 1 makes
    replica 1 start inside replica 0's block, and the group never forms."""
    mgr = _mgr(num_replicas=2, tp=2, dp=2)
    wrong_stride = mgr.tensor_parallel_size
    wrong_bases = [1 + r * wrong_stride for r in range(mgr.num_replicas)]
    assert wrong_bases == [1, 3]
    # replica 0 legitimately occupies 1..4, so 3 is already taken
    assert wrong_bases[1] < 1 + mgr._ranks_per_replica


# ── guards ─────────────────────────────────────────────────────────────────


def test_receiving_before_init_is_refused():
    """Otherwise the receivers arm against a group that does not exist and the
    sender broadcasts into nothing."""
    mgr = _mgr(2, tp=2)
    with pytest.raises(RuntimeError, match="not been initialized"):
        mgr.start_receive_weights_rdma(version=1, verify_full_load=True)


def test_teardown_without_a_group_is_a_no_op():
    """Called on paths that may never have built one."""
    mgr = _mgr(2, tp=2)
    mgr.destroy_rdma_weight_group()  # must not raise


def test_the_manager_interface_matches_the_vllm_one():
    """_sync_weights_rdma in the trainer is backend-agnostic: it drives both
    managers through these names. A signature drift here shows up as an
    AttributeError mid-sync."""
    import inspect

    from atomrl.engine.inference.vllm_ray_server import VLLMReplicaManager

    for name in (
        "init_rdma_weight_group",
        "start_receive_weights_rdma",
        "destroy_rdma_weight_group",
    ):
        atom = inspect.signature(getattr(ATOMReplicaManager, name))
        vllm = inspect.signature(getattr(VLLMReplicaManager, name))
        assert atom.parameters.keys() == vllm.parameters.keys(), (
            f"{name}: ATOM takes {list(atom.parameters)}, "
            f"vLLM takes {list(vllm.parameters)}"
        )

    # And the attribute the trainer gates on before syncing at all.
    assert hasattr(_mgr(1, 1), "rdma_group_name")


def test_trainer_side_fp8_is_rejected_not_ignored():
    """ATOM's receive path is BF16-only. Silently dropping the flag would leave
    the trainer quantising and ATOM expecting BF16 -- garbage output, no error.
    """
    import asyncio

    from atomrl.engine.inference.atom_ray_server import ATOMRayServer

    server = object.__new__(ATOMRayServer)
    with pytest.raises(NotImplementedError, match="BF16-only"):
        asyncio.run(server.receive_weights_rdma("g", 1, True, prequantized_fp8=True))


class _Engine:
    """An ATOM engine with the receive as a weight-sync command, or without."""

    def __init__(self, with_receive: bool) -> None:
        self.calls = []
        if with_receive:
            self.receive_weights_rdma = lambda *a, **kw: (
                self.calls.append(("receive_weights_rdma", a, kw)) or [435]
            )

    def collective_rpc(self, method, **kw):
        self.calls.append(("collective_rpc", method, kw))
        return []


def _atom_server(engine):
    from atomrl.engine.inference.atom_ray_server import ATOMRayServer

    server = object.__new__(ATOMRayServer)
    server.engine = engine
    return server


def test_the_stream_is_a_weight_sync_where_atom_has_the_command(monkeypatch):
    """The engine fences serving around it and drops the KV the old weights
    computed, so no prefix-cache reset follows."""
    import asyncio

    monkeypatch.setenv("ATOMRL_RDMA_RECV_TIMEOUT_S", "42")
    engine = _Engine(with_receive=True)

    assert asyncio.run(_atom_server(engine).receive_weights_rdma("g", 3)) == [435]
    assert engine.calls == [("receive_weights_rdma", ("g", 3), {"timeout": 42.0})]


def test_an_atom_without_the_public_receive_api_fails_loudly():
    import asyncio

    engine = _Engine(with_receive=False)

    with pytest.raises(AttributeError, match="receive_weights_rdma"):
        asyncio.run(_atom_server(engine).receive_weights_rdma("g", 3))
    assert engine.calls == []
