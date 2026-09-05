from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("aiter")

from atom.model_ops.fused_moe.rccl_prepare_finalize import RcclPrepareAndFinalize


class _FakePyNccl:
    disabled = False

    def __init__(self, world_size: int):
        self.world_size = world_size
        self.all_gather_calls: list[tuple[torch.dtype, torch.dtype]] = []

    def all_gather(self, output: torch.Tensor, input_: torch.Tensor) -> None:
        self.all_gather_calls.append((output.dtype, input_.dtype))
        output.view(self.world_size, *input_.shape).copy_(
            input_.unsqueeze(0).expand(self.world_size, *input_.shape)
        )

    def group_start(self) -> None:
        pass

    def group_end(self) -> None:
        pass

    def reduce_scatter(self, output: torch.Tensor, input_: torch.Tensor) -> None:
        output.copy_(input_.view(self.world_size, *output.shape).sum(dim=0))


class _FakeDeviceCommunicator:
    def __init__(self, world_size: int):
        self.pynccl_comm = _FakePyNccl(world_size)
        self.peer_payload: torch.Tensor | None = None
        self.all_gatherv_calls: list[tuple[int, list[int]]] = []
        self.reduce_scatterv_calls: list[tuple[int, list[int]]] = []

    def all_gatherv(
        self, input_: torch.Tensor, *, dim: int, sizes: list[int]
    ) -> torch.Tensor:
        self.all_gatherv_calls.append((dim, list(sizes)))
        assert self.peer_payload is not None
        return torch.cat((input_, self.peer_payload), dim=dim)

    def reduce_scatterv(
        self, input_: torch.Tensor, *, dim: int, sizes: list[int]
    ) -> torch.Tensor:
        self.reduce_scatterv_calls.append((dim, list(sizes)))
        assert dim == 0
        # Simulate the sum from two ranks and return rank zero's source slice.
        return input_[: sizes[0]] * 2


def _make_backend(
    world_size: int = 2,
    *,
    num_local_experts: int = 2,
    num_replicated_shared_experts: int = 0,
) -> RcclPrepareAndFinalize:
    device_communicator = _FakeDeviceCommunicator(world_size)
    ep_group = SimpleNamespace(
        device_group=None,
        device_communicator=device_communicator,
        rank_in_group=0,
        world_size=world_size,
    )
    backend = RcclPrepareAndFinalize(
        ep_group,
        num_local_experts=num_local_experts,
        max_tokens_per_rank=8,
        num_replicated_shared_experts=num_replicated_shared_experts,
    )
    backend._use_static_decode_path = lambda: True
    return backend


def test_static_decode_gathers_prerouted_rows_and_reduce_scatters_output():
    backend = _make_backend()
    assert backend.needs_dispatch_output_trim() is False
    hidden = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    topk_ids = torch.tensor([[0, 2], [1, 3]], dtype=torch.int32)
    topk_weights = torch.tensor([[0.6, 0.4], [0.25, 0.75]])

    dispatched, scale, metadata, dispatch_ids, dispatch_weights = backend.prepare(
        hidden,
        topk_weights,
        topk_ids,
        num_experts=4,
        expert_map=None,
        apply_router_weight_on_input=False,
        quant_config=None,
    )

    assert scale is None
    assert metadata.expert_num_tokens is None
    assert torch.equal(dispatched, hidden.repeat(2, 1))
    assert torch.equal(dispatch_ids, topk_ids.repeat(2, 1))
    assert torch.equal(dispatch_weights, topk_weights.repeat(2, 1))
    assert backend._pynccl.all_gather_calls == [(torch.uint8, torch.uint8)]

    output = backend.finalize(
        None,
        dispatched,
        topk_weights,
        topk_ids,
        apply_router_weight_on_input=False,
    )
    assert torch.equal(output, hidden * 2)


def test_variable_gather_uses_scheduler_sizes_and_preserves_source_routing():
    from atom.model_ops.fused_moe.routed_all2all import pack_routed_payload

    backend = _make_backend()
    backend._use_static_decode_path = lambda: False
    backend._variable_gather_sizes = lambda _: [2, 1]

    hidden = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    topk_ids = torch.tensor([[0, 2], [1, 3]], dtype=torch.int32)
    topk_weights = torch.tensor([[0.6, 0.4], [0.25, 0.75]])
    peer_hidden = torch.tensor([[5.0, 6.0]])
    # The peer's locally replicated shared-expert ID deliberately differs from
    # rank zero's IDs. Gathering post-routing metadata must preserve it.
    peer_ids = torch.tensor([[2, 3]], dtype=torch.int32)
    peer_weights = torch.tensor([[0.1, 0.9]])
    peer_payload, _ = pack_routed_payload(peer_hidden, peer_ids, peer_weights)
    backend._device_communicator.peer_payload = peer_payload

    dispatched, scale, metadata, dispatch_ids, dispatch_weights = backend.prepare(
        hidden,
        topk_weights,
        topk_ids,
        num_experts=4,
        expert_map=None,
        apply_router_weight_on_input=False,
        quant_config=None,
    )

    assert scale is None
    assert metadata.expert_num_tokens is None
    assert torch.equal(dispatched, torch.cat((hidden, peer_hidden)))
    assert torch.equal(dispatch_ids, torch.cat((topk_ids, peer_ids)))
    assert torch.equal(dispatch_weights, torch.cat((topk_weights, peer_weights)))
    assert backend._device_communicator.all_gatherv_calls == [(0, [2, 1])]
    assert backend._pynccl.all_gather_calls == []

    output = backend.finalize(
        None,
        dispatched,
        topk_weights,
        topk_ids,
        apply_router_weight_on_input=False,
    )
    assert torch.equal(output, hidden * 2)
    assert backend._device_communicator.reduce_scatterv_calls == [(0, [2, 1])]


def test_variable_gather_rejects_inconsistent_local_token_count(monkeypatch):
    backend = _make_backend()
    backend._use_static_decode_path = lambda: False

    import atom.model_ops.fused_moe.rccl_prepare_finalize as rccl_module

    monkeypatch.setattr(
        rccl_module,
        "get_forward_context",
        lambda: SimpleNamespace(
            context=SimpleNamespace(running_tokens_are_unified=False),
            dp_metadata=SimpleNamespace(get_sizes_across_dp=lambda: [1, 1]),
        ),
    )
    with pytest.raises(ValueError, match="token count disagrees"):
        backend._variable_gather_sizes(torch.empty((2, 4)))


@pytest.mark.parametrize(
    ("is_prefill", "running_tokens_are_unified", "expected"),
    [
        (False, True, True),
        (False, False, False),
        (True, True, False),
    ],
)
def test_static_decode_uses_settled_forward_shape(
    monkeypatch, is_prefill, running_tokens_are_unified, expected
):
    backend = _make_backend()

    import atom.model_ops.fused_moe.rccl_prepare_finalize as rccl_module

    monkeypatch.setattr(
        rccl_module,
        "get_forward_context",
        lambda: SimpleNamespace(
            context=SimpleNamespace(
                is_prefill=is_prefill,
                running_tokens_are_unified=running_tokens_are_unified,
            )
        ),
    )

    assert RcclPrepareAndFinalize._use_static_decode_path(backend) is expected


def test_static_decode_preserves_topk_ids_for_aiter_expert_mask():
    backend = _make_backend()
    hidden = torch.tensor([[1.0, 2.0]])
    topk_ids = torch.tensor([[0, 2]], dtype=torch.int32)
    topk_weights = torch.tensor([[0.6, 0.4]])

    _, _, _, dispatch_ids, dispatch_weights = backend.prepare(
        hidden,
        topk_weights,
        topk_ids,
        num_experts=4,
        expert_map=None,
        apply_router_weight_on_input=False,
        quant_config=None,
    )

    assert torch.equal(dispatch_ids, topk_ids.repeat(2, 1))
    assert torch.equal(dispatch_weights, topk_weights.repeat(2, 1))


def test_gathered_rows_round_robin_replicated_shared_expert_owners():
    backend = _make_backend(
        num_local_experts=3,
        num_replicated_shared_experts=1,
    )
    hidden = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    # Rank zero initially pins its shared column to dispatch slot 2. The fake
    # gather repeats that source payload, after which rows 1 and 3 must move to
    # rank one's replica at slot 5.
    topk_ids = torch.tensor([[0, 3, 2], [1, 4, 2]], dtype=torch.int32)
    topk_weights = torch.tensor([[0.6, 0.4, 1.0], [0.25, 0.75, 1.0]])

    _, _, _, dispatch_ids, _ = backend.prepare(
        hidden,
        topk_weights,
        topk_ids,
        num_experts=6,
        expert_map=None,
        apply_router_weight_on_input=False,
        quant_config=None,
    )

    assert dispatch_ids[:, -1].tolist() == [2, 5, 2, 5]
