# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Qwen3.8-Flash-Next sequence-parallel HC streams, on CPU with simulated TP ranks."""

import threading
from concurrent.futures import ThreadPoolExecutor
from types import MethodType, SimpleNamespace

import pytest
import torch

pytest.importorskip("triton")
pytest.importorskip("aiter.ops.enum", exc_type=ImportError)

from atom.models import qwen4_exp


class _Collectives:
    """In-process TP group: one thread per rank, collectives meet at a barrier."""

    def __init__(self, world: int) -> None:
        self.world = world
        self.barrier = threading.Barrier(world, timeout=30)
        self.slots = [None] * world

    def _exchange(self, rank: int, x: torch.Tensor) -> list[torch.Tensor]:
        self.slots[rank] = x
        self.barrier.wait()
        inputs = list(self.slots)
        self.barrier.wait()
        return inputs

    def group(self, rank: int):
        def all_gather(x, use_custom=False, dim=0):
            assert dim == 0
            return torch.cat(self._exchange(rank, x))

        def reduce_scatter_tensor(x, use_custom=False, dim=0):
            assert dim == 0 and x.shape[0] % self.world == 0
            total = torch.stack(self._exchange(rank, x)).sum(0)
            return total.chunk(self.world)[rank].clone()

        return SimpleNamespace(
            all_gather=all_gather, reduce_scatter_tensor=reduce_scatter_tensor
        )


@pytest.fixture
def ranks(monkeypatch):
    """`ranks(world, fn)` runs `fn(rank)` on `world` simulated TP ranks; outside
    `ranks` the TP world size is 1."""
    local = threading.local()
    monkeypatch.setattr(qwen4_exp, "get_tp_group", lambda: local.group)
    monkeypatch.setattr(qwen4_exp, "get_tensor_model_parallel_rank", lambda: local.rank)
    monkeypatch.setattr(
        qwen4_exp,
        "get_tensor_model_parallel_world_size",
        lambda: getattr(local, "world", 1),
    )
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)

    def run(world, fn):
        collectives = _Collectives(world)

        def on_rank(rank):
            local.rank, local.world = rank, world
            local.group = collectives.group(rank)
            return fn(rank)

        with ThreadPoolExecutor(world) as pool:
            return list(pool.map(on_rank, range(world)))

    return run


@pytest.mark.parametrize("world,tokens", [(2, 8), (4, 8), (4, 10), (4, 1027)])
def test_shard_gather_round_trip_drops_padding(ranks, world, tokens):
    full = torch.randn(tokens, 6)

    def on_rank(rank):
        sp = qwen4_exp._SequenceParallel(tokens, world, rank)
        local = sp.shard(full)
        assert local.shape[0] == -(-tokens // world)
        start = rank * local.shape[0]
        real = max(0, min(local.shape[0], tokens - start))
        assert torch.equal(local[:real], full[start : start + real])
        if real < local.shape[0]:
            # Only the local rows are copied, not the whole padded tensor.
            nbytes = local.untyped_storage().nbytes()
            assert nbytes == local.numel() * local.element_size()
        assert not local[real:].any(), "padded rows must be zero"
        return sp.gather(local)

    for gathered in ranks(world, on_rank):
        assert torch.equal(gathered, full)


@pytest.mark.parametrize("world,tokens", [(2, 8), (4, 10), (4, 1027)])
def test_scatter_sums_partials_into_local_rows(ranks, world, tokens):
    partials = torch.randn(world, tokens, 5, dtype=torch.float64)
    total = partials.sum(0)

    def on_rank(rank):
        sp = qwen4_exp._SequenceParallel(tokens, world, rank)
        summed = sp.scatter(partials[rank])
        start = rank * sp.chunk
        real = max(0, min(sp.chunk, tokens - start))
        torch.testing.assert_close(summed[:real], total[start : start + real])
        assert not summed[real:].any()
        return sp.gather(summed)

    for gathered in ranks(world, on_rank):
        torch.testing.assert_close(gathered, total)


class _Layer:
    """Per-token stand-in for a fused decoder layer; the PLE layer needs every
    token of the request (its causal convolution is a prefix sum here)."""

    def __init__(self, idx: int, ple: bool, calls: list) -> None:
        self.idx, self.ple, self.calls = idx, (object() if ple else None), calls

    def __call__(self, positions, hidden, input_ids, pending, sp=None):
        self.calls.append((self.idx, hidden.shape[0], sp is not None))
        if pending is not None:
            hidden = hidden + pending[0] * pending[1]
        if self.ple is not None:
            assert sp is None and hidden.shape[0] == input_ids.shape[0]
            hidden = hidden + input_ids.double().cumsum(0)[:, None]
        hidden = torch.tanh(hidden * (self.idx + 1.5))
        return hidden, (hidden.sum(1, keepdim=True), hidden[:, :1])


def _model(num_layers, ple_layer, return_hc_state, embed, hc=4):
    calls = []
    layers = [_Layer(i, i == ple_layer, calls) for i in range(num_layers)]
    hidden = embed.shape[1]

    def apply_pending(h, pending):
        return h + pending[0] * pending[1]

    model = SimpleNamespace(
        config=SimpleNamespace(hidden_size=hidden),
        hc_count=hc,
        return_hc_state=return_hc_state,
        get_input_embeddings=lambda ids: embed[ids],
        start_layer=0,
        end_layer=num_layers,
        layers=layers,
        # As `Qwen4ExpModel.__init__`: the layer after the last PLE layer.
        sp_start_layer=ple_layer + 1,
        hyper_connection_mixer=SimpleNamespace(
            apply_pending=apply_pending,
            mix_fused=lambda h, p: (apply_pending(h, p)[:, :hidden] * 0.5, None),
        ),
    )
    model._sequence_parallel = MethodType(
        qwen4_exp.Qwen4ExpModel._sequence_parallel, model
    )
    return model, calls


@pytest.mark.parametrize("return_hc_state", [False, True])
@pytest.mark.parametrize(
    "world,tokens,ple_layer",
    [
        (2, 8, 1),
        (4, 10, 1),
        # PLE in the last layer: nothing is left to shard, SP must stay off.
        (4, 8, 3),
    ],
)
def test_model_forward_matches_unsharded(
    ranks, monkeypatch, world, tokens, ple_layer, return_hc_state
):
    monkeypatch.setenv("ATOM_QWEN4_EXP_SP_MIN_TOKENS", "4")
    num_layers = 4
    torch.manual_seed(0)
    input_ids = torch.randint(0, 64, (tokens,))
    positions = torch.arange(tokens)
    embed = torch.randn(64, 3, dtype=torch.float64)

    reference, _ = _model(num_layers, ple_layer, return_hc_state, embed)
    expected = qwen4_exp.Qwen4ExpModel.forward(reference, input_ids, positions)

    def on_rank(rank):
        model, calls = _model(num_layers, ple_layer, return_hc_state, embed)
        out = qwen4_exp.Qwen4ExpModel.forward(model, input_ids, positions)
        return out, calls

    chunk = -(-tokens // world)
    for out, calls in ranks(world, on_rank):
        torch.testing.assert_close(out, expected)
        for idx, rows, sharded in calls:
            assert sharded == (idx > ple_layer)
            assert rows == (chunk if sharded else tokens)
