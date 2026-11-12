# SPDX-License-Identifier: MIT
"""Kimi-K3 latent MoE on simulated TP8 ranks: above the dual-stream threshold
the routed norm, up-projection and routed + shared add run on each rank's
rows, and the block's pair still sums to the replicated `split_moe_forward`."""

import threading
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest
import torch
from torch import nn

pytest.importorskip("aiter", reason="atom.models.kimi_k3.model imports aiter")

from atom.models import qwen4_exp
from atom.models.kimi_k3 import model as kimi

WORLD, HIDDEN, LATENT, EXPERTS = 8, 16, 8, 4


class _Collectives:
    """In-process TP group: one thread per rank, collectives meet at a barrier."""

    def __init__(self) -> None:
        self.barrier = threading.Barrier(WORLD, timeout=30)
        self.slots = [None] * WORLD

    def total(self, rank: int, x: torch.Tensor) -> torch.Tensor:
        """Rounded once from an exact sum, so the all-reduce and the padded
        reduce-scatter agree on every row."""
        self.slots[rank] = x
        self.barrier.wait()
        total = torch.stack(self.slots).double().sum(0).to(x.dtype)
        self.barrier.wait()
        return total

    def gather(self, rank: int, x: torch.Tensor) -> torch.Tensor:
        self.slots[rank] = x
        self.barrier.wait()
        out = torch.cat(self.slots)
        self.barrier.wait()
        return out


@pytest.fixture
def ranks(monkeypatch):
    """`ranks(fn)` runs `fn(rank)` on WORLD simulated TP ranks."""
    local = threading.local()

    def reduce_scatter_tensor(x, use_custom=False, dim=0):
        assert dim == 0 and x.shape[0] % WORLD == 0
        return local.comm.total(local.rank, x).chunk(WORLD)[local.rank].clone()

    def all_gather(x, use_custom=False, dim=0):
        assert dim == 0
        return local.comm.gather(local.rank, x)

    group = SimpleNamespace(
        reduce_scatter_tensor=reduce_scatter_tensor, all_gather=all_gather
    )
    monkeypatch.setattr(qwen4_exp, "get_tp_group", lambda: group)
    monkeypatch.setattr(kimi, "get_tensor_model_parallel_rank", lambda: local.rank)
    monkeypatch.setattr(
        kimi,
        "tensor_model_parallel_all_reduce",
        lambda x: local.comm.total(local.rank, x),
    )
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)

    def run(fn):
        comm = _Collectives()

        def on_rank(rank):
            local.rank, local.comm = rank, comm
            return fn(rank)

        with ThreadPoolExecutor(WORLD) as pool:
            return list(pool.map(on_rank, range(WORLD)))

    return run


def _weights():
    gen = torch.Generator().manual_seed(3)
    shapes = {
        "gate": (HIDDEN, EXPERTS),
        "down": (HIDDEN, LATENT),
        "up": (LATENT, HIDDEN),
        "shared": (HIDDEN, HIDDEN),
    }
    return {
        name: torch.randn(*shape, generator=gen).to(torch.bfloat16)
        for name, shape in shapes.items()
    }


def _block(w, rank, up_rows):
    """A latent block whose stub experts return this rank's share of the
    sum, as the real ones do before the all-reduce."""
    block = kimi.KimiSparseMoeBlock.__new__(kimi.KimiSparseMoeBlock)
    nn.Module.__init__(block)
    block.tp_size, block.use_latent_moe = WORLD, True
    block.fuse_routed_triton_mxfp4_quant = False
    share = (rank + 1) / (WORLD * (WORLD + 1) / 2)
    block.gate = lambda x: x @ w["gate"]
    block.routed_expert_down_proj = lambda x: x @ w["down"]
    block.experts = lambda x, logits: share * torch.tanh(
        x * logits.sum(-1, keepdim=True)
    )
    block.shared_experts = lambda x: share * torch.tanh(x @ w["shared"])
    # RMS norm with eps, so padded (zero) rows stay zero as in the model.
    block.routed_expert_norm = lambda x: x * torch.rsqrt(
        x.pow(2).mean(-1, keepdim=True) + 1e-6
    )

    def up_proj(x):
        up_rows.append(x.shape[0])
        return x @ w["up"]

    block.routed_expert_up_proj = up_proj
    return block


def _base_and_op_target(ranks, tokens):
    """Per rank: `split_moe_forward`'s pair, the split op's single-stream
    target's pair, and the rows each up-projection saw."""
    w = _weights()
    gen = torch.Generator().manual_seed(tokens)
    x = torch.randn(tokens, HIDDEN, generator=gen).to(torch.bfloat16)

    def on_rank(rank):
        up_rows = []
        block = _block(w, rank, up_rows)
        base = block.split_moe_forward(x)
        return base, block.single_stream_split_moe_forward(x), up_rows

    return ranks(on_rank)


@pytest.mark.parametrize("tokens", [8, 13, 1027])
def test_row_sharded_pair_sums_to_the_base_pair(ranks, monkeypatch, tokens):
    monkeypatch.setenv("ATOM_DUAL_STREAM_MOE_TOKEN_THRESHOLD", "4")
    results = _base_and_op_target(ranks, tokens)
    for (routed, shared), (summed, zeros), up_rows in results:
        # The base up-projects every row; the sharded tail only this rank's.
        assert up_rows == [tokens, -(-tokens // WORLD)]
        assert summed.shape == zeros.shape == (tokens, HIDDEN)
        assert not zeros.any()
        expected = (routed.float() + shared.float()).to(summed.dtype)
        torch.testing.assert_close(summed, expected)


@pytest.mark.parametrize(
    "threshold,capturing",
    [("13", False), ("14", False), ("4", True)],
    ids=["at-threshold", "below-threshold", "graph-capture"],
)
def test_base_pair_kept_bit_for_bit(ranks, monkeypatch, threshold, capturing):
    tokens = 13
    monkeypatch.setenv("ATOM_DUAL_STREAM_MOE_TOKEN_THRESHOLD", threshold)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: capturing)
    results = _base_and_op_target(ranks, tokens)
    for (routed, shared), (routed_out, shared_out), up_rows in results:
        assert up_rows == [tokens, tokens]
        assert torch.equal(routed_out, routed) and torch.equal(shared_out, shared)
