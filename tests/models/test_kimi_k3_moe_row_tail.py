# SPDX-License-Identifier: MIT
"""Kimi-K3 latent MoE on simulated TP8 ranks: above the dual-stream threshold
and the row floor the routed norm, up-projection and routed + shared add run
on each rank's rows, and the block's pair still sums to the replicated
`split_moe_forward`."""

import threading
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest
import torch
from torch import nn

pytest.importorskip("aiter", reason="atom.models.kimi_k3.model imports aiter")

from atom.model_ops import module_dispatch_ops
from atom.models import qwen4_exp
from atom.models.kimi_k3 import model as kimi
from atom.utils.tbo import ubatching

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


class _Stub(nn.Module):
    """Stands in for a sub-module class; the test sets `fn` after `__init__`."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__()

    def forward(self, *args):
        return self.fn(*args)


@pytest.fixture(autouse=True)
def stub_block_parts(monkeypatch):
    """`KimiSparseMoeBlock.__init__` runs for real on CPU: its sub-modules are
    stubs, the TP size is WORLD and its config registers into one context."""
    for name in ("ReplicatedLinear", "FusedMoE", "KimiMLP", "RMSNorm"):
        monkeypatch.setattr(kimi, name, _Stub)
    monkeypatch.setattr(kimi, "get_tensor_model_parallel_world_size", lambda: WORLD)
    config = SimpleNamespace(
        enable_tbo=False,
        compilation_config=SimpleNamespace(static_forward_context={}),
    )
    monkeypatch.setattr(kimi, "get_current_atom_config", lambda: config)
    monkeypatch.setattr(module_dispatch_ops, "get_current_atom_config", lambda: config)
    return config


@pytest.fixture(autouse=True)
def no_row_floor(monkeypatch):
    """Every row count may take the tail; `test_row_floor` sets its own floor."""
    monkeypatch.setenv("ATOM_KIMI_K3_MOE_ROW_SHARD_MIN_TOKENS", "0")


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
    config = SimpleNamespace(
        hidden_size=HIDDEN,
        num_experts=EXPERTS,
        num_experts_per_token=2,
        routed_expert_hidden_size=LATENT,
        moe_intermediate_size=HIDDEN,
        moe_renormalize=True,
        moe_router_activation_func="sigmoid",
        num_shared_experts=1,
        latent_moe_use_norm=True,
        rms_norm_eps=1e-6,
    )
    # The model threads a stream into every block with shared experts.
    block = kimi.KimiSparseMoeBlock(
        config, prefix=f"rank{rank}.mlp", alt_stream=object()
    )
    share = (rank + 1) / (WORLD * (WORLD + 1) / 2)
    block.gate.fn = lambda x: x @ w["gate"]
    block.routed_expert_down_proj.fn = lambda x: x @ w["down"]
    block.experts.fn = lambda x, logits: share * torch.tanh(
        x * logits.sum(-1, keepdim=True)
    )
    block.shared_experts.fn = lambda x: share * torch.tanh(x @ w["shared"])
    # RMS norm with eps, so padded (zero) rows stay zero as in the model; in
    # FP32, as there, since a BF16 CPU mean rounds differently by row count.
    block.routed_expert_norm.fn = lambda x: (
        x.float() * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + 1e-6)
    ).to(x.dtype)

    def up_proj(x):
        up_rows.append(x.shape[0])
        return x @ w["up"]

    block.routed_expert_up_proj.fn = up_proj
    return block


def _base_and_op_target(ranks, tokens, target=None):
    """Per rank: `split_moe_forward`'s pair, `target`'s pair (by default the
    split op's single-stream target) and the rows each up-projection saw."""
    w = _weights()
    gen = torch.Generator().manual_seed(tokens)
    x = torch.randn(tokens, HIDDEN, generator=gen).to(torch.bfloat16)
    target = target or (lambda block, x: block.single_stream_split_moe_forward(x))

    def on_rank(rank):
        up_rows = []
        block = _block(w, rank, up_rows)
        base = block.split_moe_forward(x)
        return base, target(block, x), up_rows

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


@pytest.fixture
def split_op_on_cpu(monkeypatch):
    """The op is registered for CUDA only; call its Python body on CPU tensors."""
    monkeypatch.setattr(
        torch.ops.aiter,
        "maybe_dual_stream_split_forward",
        module_dispatch_ops.maybe_dual_stream_split_forward,
    )


def test_forward_with_dual_stream_off_takes_the_row_sharded_tail(
    ranks, monkeypatch, split_op_on_cpu
):
    """A threshold of 0 turns dual-stream off; the block still goes through
    the split op, whose single-stream target shards the tail."""
    monkeypatch.setenv("ATOM_DUAL_STREAM_MOE_TOKEN_THRESHOLD", "0")
    tokens = 13
    for (routed, shared), (summed, zeros), up_rows in _base_and_op_target(
        ranks, tokens, lambda block, x: block.forward(x)
    ):
        assert up_rows == [tokens, -(-tokens // WORLD)]
        assert not zeros.any()
        expected = (routed.float() + shared.float()).to(summed.dtype)
        torch.testing.assert_close(summed, expected)


def test_forward_with_tbo_enabled_keeps_the_base_pair(
    ranks, monkeypatch, split_op_on_cpu, stub_block_parts
):
    """With TBO enabled the block stays off the split op, so a step that TBO
    does not split keeps the base pair too."""
    monkeypatch.setenv("ATOM_DUAL_STREAM_MOE_TOKEN_THRESHOLD", "0")
    stub_block_parts.enable_tbo = True
    tokens = 13
    for (routed, shared), (routed_out, shared_out), up_rows in _base_and_op_target(
        ranks, tokens, lambda block, x: block.forward(x)
    ):
        assert up_rows == [tokens, tokens]
        assert torch.equal(routed_out, routed) and torch.equal(shared_out, shared)


@pytest.mark.parametrize(
    "floor,sharded",
    [(None, False), ("14", False), ("13", True)],
    ids=["default-floor", "below-floor", "at-floor"],
)
def test_row_floor(ranks, monkeypatch, split_op_on_cpu, floor, sharded):
    """At threshold 0 an eager step below the row floor keeps the base pair."""
    monkeypatch.setenv("ATOM_DUAL_STREAM_MOE_TOKEN_THRESHOLD", "0")
    if floor is None:
        monkeypatch.delenv("ATOM_KIMI_K3_MOE_ROW_SHARD_MIN_TOKENS")
    else:
        monkeypatch.setenv("ATOM_KIMI_K3_MOE_ROW_SHARD_MIN_TOKENS", floor)
    tokens = 13
    for (routed, shared), (first, second), up_rows in _base_and_op_target(
        ranks, tokens, lambda block, x: block.forward(x)
    ):
        assert up_rows == [tokens, -(-tokens // WORLD) if sharded else tokens]
        if not sharded:
            assert torch.equal(first, routed) and torch.equal(second, shared)


@pytest.mark.parametrize(
    "threshold,capturing,tbo_step",
    [
        ("13", False, False),
        ("14", False, False),
        ("4", True, False),
        ("4", False, True),
    ],
    ids=["at-threshold", "below-threshold", "graph-capture", "tbo-step"],
)
def test_base_pair_kept_bit_for_bit(ranks, monkeypatch, threshold, capturing, tbo_step):
    tokens = 13
    monkeypatch.setenv("ATOM_DUAL_STREAM_MOE_TOKEN_THRESHOLD", threshold)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: capturing)
    monkeypatch.setattr(ubatching, "tbo_active", lambda: tbo_step)
    results = _base_and_op_target(ranks, tokens)
    for (routed, shared), (routed_out, shared_out), up_rows in results:
        assert up_rows == [tokens, tokens]
        assert torch.equal(routed_out, routed) and torch.equal(shared_out, shared)
