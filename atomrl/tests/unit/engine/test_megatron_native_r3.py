# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

"""CPU-only tests for MegatronNativeEngine vLLM R3 (expert-id) replay."""

from __future__ import annotations

import inspect
import sys
import types

import pytest
import torch
from atomrl.core.protocol import DataProto
from atomrl.engine.training.megatron_native_engine import MegatronNativeEngine
from atomrl.engine.training.megatron_r3_replay import (
    clear_replay,
    install_replay,
    local_router_layer_indices,
    pack_microbatch_routes,
    routes_from_batch,
)


class _ReplayInstance:
    def __init__(self) -> None:
        self.target_indices: list[torch.Tensor] = []
        self.replay_backward_list: list[torch.Tensor] = []

    def set_target_indices(self, indices: torch.Tensor) -> None:
        self.target_indices.append(indices.clone())
        self.replay_backward_list.append(indices.clone())

    def clear_indices(self) -> None:
        self.target_indices.clear()
        self.replay_backward_list.clear()


def _install_router_replay(monkeypatch, instance_count: int):
    instances = [_ReplayInstance() for _ in range(instance_count)]

    class RouterReplayAction:
        REPLAY_FORWARD = "replay_forward"

    class RouterReplay:
        global_router_replay_instances = instances
        action = None
        clear_indices_calls = 0

        @classmethod
        def clear_global_indices(cls):
            cls.clear_indices_calls += 1
            for instance in cls.global_router_replay_instances:
                instance.clear_indices()

        @classmethod
        def clear_global_router_replay_instances(cls):
            cls.global_router_replay_instances.clear()

        @classmethod
        def clear_global_router_replay_action(cls):
            cls.action = None

        @classmethod
        def set_global_router_replay_action(cls, action):
            cls.action = action

        @classmethod
        def set_replay_data(cls, tensors):
            if len(tensors) != len(cls.global_router_replay_instances):
                raise ValueError("layer / instance mismatch")
            for instance, tensor in zip(cls.global_router_replay_instances, tensors):
                instance.set_target_indices(tensor)

    replay_module = types.ModuleType("megatron.core.transformer.moe.router_replay")
    replay_module.RouterReplay = RouterReplay
    replay_module.RouterReplayAction = RouterReplayAction
    monkeypatch.setitem(
        sys.modules,
        "megatron.core.transformer.moe.router_replay",
        replay_module,
    )
    return RouterReplay, instances


def test_native_forwards_require_vllm_routes_not_logit_hooks() -> None:
    update = inspect.getsource(MegatronNativeEngine._pp_update_policy)
    logprob = inspect.getsource(MegatronNativeEngine._pp_compute_log_probs)
    forward = inspect.getsource(MegatronNativeEngine._pp_forward_model)
    assert "routes_from_batch" in update
    assert "routes_from_batch" in logprob
    assert "pack_microbatch_routes" in forward
    assert "install_replay" in forward
    assert "megatron_record_router_logits" not in update + logprob
    assert "megatron_replay_router_logits" not in update + logprob
    assert "self._r3_append = True" in update
    assert "self._r3_append = False" in logprob
    source = inspect.getsource(MegatronNativeEngine.initialize)
    assert "moe_enable_routing_replay" in source
    cleanup = source.index("clear_stale_instances(")
    construction = source.index("model = GPTModel(")
    assert cleanup < construction
    assert "megatron_record_router_logits" not in source
    assert "recompute_granularity" in source
    assert "cannot use activation recomputation" in source


def test_routes_from_batch_requires_vllm_payload() -> None:
    batch = DataProto(tensors={"input_ids": torch.zeros(1, 4, dtype=torch.long)})
    with pytest.raises(RuntimeError, match="rollout_routing"):
        routes_from_batch(batch)


def test_routes_from_batch_rejects_missing_ragged_rows() -> None:
    batch = DataProto(
        tensors={"input_ids": torch.zeros(2, 4, dtype=torch.long)},
        ragged={"rollout_routing": [torch.zeros(3, 2, 2), None]},
    )
    with pytest.raises(RuntimeError, match="missing rows"):
        routes_from_batch(batch)


def test_pack_appends_last_token_and_tp_pad() -> None:
    # One sequence of length 4 -> 3 rollout tokens. Packed length 8 (TP pad).
    routes = torch.arange(3 * 2 * 2, dtype=torch.int16).view(1, 3, 2, 2)
    layers = pack_microbatch_routes(
        routes,
        [(0, 0, 4)],
        packed_len=8,
        num_experts=16,
        layer_indices=[0, 1],
    )
    assert len(layers) == 2
    assert layers[0].shape == (8, 2)
    assert layers[0].dtype == torch.int64
    torch.testing.assert_close(layers[0][:3], routes[0, :, 0, :].to(torch.int64))
    # Last real token + pad use top-k filler ids 0,1.
    assert torch.equal(layers[0][3:], torch.tensor([[0, 1]] * 5))


def test_pack_dummy_microbatch_matches_packed_len() -> None:
    routes = torch.zeros(1, 3, 2, 2, dtype=torch.int16)
    layers = pack_microbatch_routes(
        routes,
        [],
        packed_len=4,
        num_experts=8,
        layer_indices=[0, 1],
        sequence_parallel=True,
        tp_size=2,
        tp_rank=1,
    )
    assert layers[0].shape == (2, 2)
    assert torch.equal(layers[0], torch.tensor([[0, 1], [0, 1]]))


def test_pack_sequence_parallel_and_pipeline_slice() -> None:
    # PP rank owning global layers 2 and 3 of a 4-layer route payload asks for
    # exactly those columns, in RouterReplay-instance order.
    routes = torch.arange(4 * 4 * 2, dtype=torch.int32).view(1, 4, 4, 2)
    layers = pack_microbatch_routes(
        routes,
        [(0, 0, 5)],
        packed_len=8,
        num_experts=32,
        layer_indices=[2, 3],
        sequence_parallel=True,
        tp_size=2,
        tp_rank=0,
    )
    assert len(layers) == 2
    replay = torch.cat(
        [
            routes[0, :, 2, :].to(torch.int64),
            torch.tensor([[0, 1]], dtype=torch.int64),
            torch.tensor([[0, 1]] * 3, dtype=torch.int64),
        ],
        dim=0,
    )
    torch.testing.assert_close(layers[0], replay[:4])


def test_pack_layer_indices_select_owned_layers_in_order() -> None:
    # 3-layer payload; a rank owns global layers 2 then 0 (reversed order still
    # honored, e.g. interleaved virtual pipeline).
    routes = torch.arange(2 * 3 * 2, dtype=torch.int32).view(1, 2, 3, 2)
    layers = pack_microbatch_routes(
        routes,
        [(0, 0, 3)],
        packed_len=3,
        num_experts=64,
        layer_indices=[2, 0],
    )
    assert len(layers) == 2
    torch.testing.assert_close(layers[0][:2], routes[0, :, 2, :].to(torch.int64))
    torch.testing.assert_close(layers[1][:2], routes[0, :, 0, :].to(torch.int64))


def test_pack_rejects_layer_index_out_of_range() -> None:
    routes = torch.zeros(1, 2, 2, 2, dtype=torch.int32)
    with pytest.raises(ValueError, match="local layer index out of range"):
        pack_microbatch_routes(
            routes,
            [(0, 0, 3)],
            packed_len=3,
            num_experts=8,
            layer_indices=[0, 2],
        )


def test_local_router_layer_indices_reads_layer_number(monkeypatch) -> None:
    from torch import nn

    _replay, instances = _install_router_replay(monkeypatch, 2)

    class _Router(nn.Module):
        def __init__(self, replay_instance, layer_number):
            super().__init__()
            self.router_replay = replay_instance
            self.layer_number = layer_number

    class _Model(nn.Module):
        def __init__(self):
            super().__init__()
            # Instances are appended in construction order; layer_number is the
            # global 1-indexed layer (offset already applied by Megatron).
            self.a = _Router(instances[0], 7)
            self.b = _Router(instances[1], 8)

    assert local_router_layer_indices(_Model()) == [6, 7]


def test_pack_cp_zigzag_gathers_local_tokens() -> None:
    # length=8, cp=2, rank 0 owns [0:2] + [6:8].
    routes = torch.arange(7 * 1 * 2, dtype=torch.int32).view(1, 7, 1, 2)
    layers = pack_microbatch_routes(
        routes,
        [(0, 0, 8)],
        packed_len=4,
        num_experts=16,
        layer_indices=[0],
        cp_size=2,
        cp_rank=0,
    )
    full = torch.cat(
        [routes[0], torch.tensor([[[0, 1]]], dtype=torch.int32)],
        dim=0,
    )
    expected = torch.stack([full[0], full[1], full[6], full[7]])[:, 0, :].to(
        torch.int64
    )
    torch.testing.assert_close(layers[0], expected)


def test_pack_rejects_out_of_range_expert_ids() -> None:
    routes = torch.tensor([[[[16, 0]]]], dtype=torch.int32).expand(1, 1, 1, 2).clone()
    with pytest.raises(ValueError, match="outside global range"):
        pack_microbatch_routes(
            routes,
            [(0, 0, 2)],
            packed_len=2,
            num_experts=16,
            layer_indices=[0],
        )


def test_install_overwrite_clears_fifo_append_keeps_it(monkeypatch) -> None:
    replay, instances = _install_router_replay(monkeypatch, 1)
    first = torch.zeros(2, 2, dtype=torch.int64)
    second = torch.ones(2, 2, dtype=torch.int64)

    install_replay([first], append=False)
    install_replay([second], append=True)
    assert replay.action == "replay_forward"
    assert len(instances[0].replay_backward_list) == 2
    assert torch.equal(instances[0].target_indices[-1], second)

    install_replay([first], append=False)
    assert len(instances[0].replay_backward_list) == 1
    assert replay.clear_indices_calls == 2  # two overwrite installs

    clear_replay()
    assert replay.action is None
    assert instances[0].replay_backward_list == []
