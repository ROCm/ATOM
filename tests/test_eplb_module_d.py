# SPDX-License-Identifier: MIT
# Tests for atom/model_ops/eplb.py (Module-D migration planning/execution)

import pytest
from contextlib import contextmanager

torch = pytest.importorskip("torch")

# Keep config import order consistent with existing EPLB tests.
import atom.config  # noqa: F401
import atom.model_ops.eplb as eplb


def test_assign_sender_for_receiver_even_split():
    senders = [0, 1]
    recvers = [2, 3, 4, 5]
    assert eplb._assign_sender_for_receiver(senders, recvers, 2) == 0
    assert eplb._assign_sender_for_receiver(senders, recvers, 3) == 0
    assert eplb._assign_sender_for_receiver(senders, recvers, 4) == 1
    assert eplb._assign_sender_for_receiver(senders, recvers, 5) == 1


def test_assign_sender_for_receiver_with_remainder():
    senders = [0, 1]
    recvers = [2, 3, 4]
    # base=1, rem=1 => sender-0 serves recv[0],recv[2], sender-1 serves recv[1]
    assert eplb._assign_sender_for_receiver(senders, recvers, 2) == 0
    assert eplb._assign_sender_for_receiver(senders, recvers, 3) == 1
    assert eplb._assign_sender_for_receiver(senders, recvers, 4) == 0


def test_plan_single_layer_migration_remote_plus_free_rider():
    # world_size=2, num_local=2, rank-0 wants logical=2 twice, but old rank-0 has no logical=2.
    old_p2l = torch.tensor([0, 1, 2, 3], dtype=torch.int32)
    new_p2l = torch.tensor([2, 2, 2, 3], dtype=torch.int32)

    plan, local_copies, sends, recvs = eplb._plan_single_layer_migration(
        old_p2l_layer=old_p2l,
        new_p2l_layer=new_p2l,
        num_local_physical_experts=2,
        num_gpu_per_node=2,
        rank=0,
        world_size=2,
    )

    assert local_copies == []
    # primary recv into dst-0, dst-1 is free-rider from temp slot 0
    assert plan == [(0, 0), (0, 1)]
    assert len(recvs) == 1
    assert recvs[0].logical_expert_id == 2
    assert recvs[0].peer_rank == 1
    assert recvs[0].local_slot == 0
    assert sends == []


def test_migrate_single_layer_local_copy_only():
    # world_size=1, local-only migration path (no P2P).
    old_p2l = torch.tensor([0, 1], dtype=torch.int32)
    new_p2l = torch.tensor([1, 0], dtype=torch.int32)
    # src slot-1 should be copied into dst slot-0, src slot-0 into dst slot-1.
    w = torch.tensor([[10.0], [20.0]], dtype=torch.float32)
    temp = torch.zeros_like(w)

    plan = eplb._migrate_single_layer(
        routed_experts_weights=[w],
        temp_buffers=[temp],
        old_p2l_layer=old_p2l,
        new_p2l_layer=new_p2l,
        num_local_physical_experts=2,
        num_gpu_per_node=1,
        rank=0,
        world_size=1,
        ep_group=None,
        num_logical_experts=2,
        p2p_batch_chunk_size=32,
    )

    assert plan == [(0, 0), (1, 1)]
    assert temp[0, 0].item() == pytest.approx(20.0)
    assert temp[1, 0].item() == pytest.approx(10.0)


def test_migrate_single_layer_runs_local_copy_on_given_stream(monkeypatch):
    old_p2l = torch.tensor([0, 1], dtype=torch.int32)
    new_p2l = torch.tensor([1, 0], dtype=torch.int32)
    w = torch.tensor([[10.0], [20.0]], dtype=torch.float32)
    temp = torch.zeros_like(w)
    seen = {"stream": None}

    @contextmanager
    def _fake_stream_ctx(s):
        seen["stream"] = s
        yield

    monkeypatch.setattr(torch.cuda, "stream", _fake_stream_ctx)
    stream_obj = object()
    eplb._migrate_single_layer(
        routed_experts_weights=[w],
        temp_buffers=[temp],
        old_p2l_layer=old_p2l,
        new_p2l_layer=new_p2l,
        num_local_physical_experts=2,
        num_gpu_per_node=1,
        rank=0,
        world_size=1,
        ep_group=None,
        num_logical_experts=2,
        p2p_batch_chunk_size=32,
        cuda_stream=stream_obj,
    )
    assert seen["stream"] is stream_obj


def test_effective_p2p_chunk_size_clamp():
    # Chunk must stay < num_logical_experts (ROCm one-shot P2P constraint);
    # oversized requests clamp to num_logical_experts - 1.
    assert eplb._effective_p2p_chunk_size(requested=999, num_logical_experts=8) == 7
    # Within range: kept as-is.
    assert eplb._effective_p2p_chunk_size(requested=4, num_logical_experts=8) == 4
    # Degenerate num_logical_experts <= 1 -> 1.
    assert eplb._effective_p2p_chunk_size(requested=999, num_logical_experts=1) == 1


def test_select_source_rank_prefers_same_node():
    # world_size=4, gpus_per_node=2, nodes: [0,1] and [2,3]
    # recv rank=1 has same-node sender rank=0, so should pick 0 over cross-node 2.
    src = eplb._select_source_rank_for_receiver(
        ranks_to_send=[0, 2],
        ranks_to_recv=[1],
        recv_rank=1,
        num_gpu_per_node=2,
    )
    assert src == 0


def test_migrate_experts_chunk_reads_config_and_returns_plan(monkeypatch):
    class _Meta:
        def __init__(self, p2l):
            self.physical_to_logical_map = p2l
            self.num_logical_experts = int(p2l.max().item()) + 1

    old = _Meta(torch.tensor([[0, 1]], dtype=torch.int32))
    new = _Meta(torch.tensor([[1, 0]], dtype=torch.int32))
    w = [torch.zeros((2, 1), dtype=torch.float32)]
    temp = [torch.zeros((2, 1), dtype=torch.float32)]

    called = {}

    def _fake_migrate_single_layer(**kwargs):
        called["chunk"] = kwargs["p2p_batch_chunk_size"]
        return []

    monkeypatch.setattr(eplb, "_migrate_single_layer", _fake_migrate_single_layer)
    # Avoid touching real config state in this test.
    import atom.config as atom_config

    monkeypatch.setattr(
        atom_config,
        "get_current_atom_config",
        lambda: type(
            "Cfg",
            (),
            {"eplb_config": type("ECfg", (), {"p2p_batch_chunk_size": 17})()},
        )(),
    )

    ret = eplb.migrate_experts_chunk(
        layer_ids=[0],
        old_meta=old,
        new_meta=new,
        expert_weights_of_layer={0: w},
        temp_buffers=temp,
        ep_group=None,
        nnodes=1,
        rank=0,
        p2p_batch_chunk_size=None,
    )
    assert ret == {0: []}
    assert called["chunk"] == 17
