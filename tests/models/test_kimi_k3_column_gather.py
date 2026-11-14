# SPDX-License-Identifier: MIT
"""Kimi-K3's column tail at decode with its all-gather deferred: every layer
but a stage's last hands back a stand-in for the all-gather of each rank's
columns, which the next layer's attention residual gathers on load through a
peer buffer (`PeerColumnGather`) and a consumer in torch all-gathers first."""

from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("aiter", reason="atom.models.kimi_k3.model imports aiter")

from aiter import dtypes
from aiter.dist import communication_op

from atom.model_ops.kimi_k3 import attention_residual
from atom.models.kimi_k3 import model as kimi
from atom.mono.runtime import peer_memory
from tests.models.test_kimi_k3_moe_column_tail import (  # noqa: F401
    HIDDEN,
    _blocks,
    _weights,
    tail_calls,
)
from tests.models.test_kimi_k3_moe_row_tail import (  # noqa: F401
    WORLD,
    ranks,
    split_op_on_cpu,
)


class _Peers:
    """`SymmetricPeerBuffer` on one device: every rank's buffer is a tensor
    here, and `addresses` points the kernel's pushes at them."""

    def __init__(self, nbytes, rank, npes, group, device="cuda"):
        self.buffers = [
            torch.zeros(nbytes, dtype=torch.uint8, device=device) for _ in range(npes)
        ]
        self.storage = self.buffers[rank]
        self.addresses = torch.tensor([b.data_ptr() for b in self.buffers]).to(device)


def _slots(buffer, gather):
    """``buffer`` as the scratch's [parity, rank, row, column pair] words."""
    return buffer.view(torch.int64).view(2, WORLD, gather.rows, gather.cols // 2)


def _pushed(share, epoch):
    """``share``'s rows as a rank pushes them: column pairs under the epoch."""
    pairs = share.view(torch.int32).to(torch.int64) & 0xFFFFFFFF
    return epoch[:, None].to(torch.int64) << 32 | pairs


def _mix(add, prefix, bank, score_weight, norm, quant):
    """`apply_attn_res` on an MoE layer's column tail: the sum, then a folded
    zero."""
    return attention_residual._apply_attn_res_impl(
        prefix,
        bank,
        score_weight,
        1e-6,
        add,
        norm,
        1e-6,
        attention_residual.folded_zero_like(prefix),
        quant,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="runs the Triton kernel")
@pytest.mark.parametrize("quant", [None, dtypes.fp8], ids=["bf16", "fp8"])
def test_kernel_mix_equals_all_gather_then_mix(monkeypatch, quant):
    """Rank 3 of a TP8 hidden of 7168, the other ranks' pushes posted in its
    scratch by hand: the mix is bitwise the all-gather's, rank 3's share lands
    in every rank's scratch and each gathered row's epoch advances. Rows
    differ in epoch from the second round, so both parities are read."""
    monkeypatch.setattr(peer_memory, "SymmetricPeerBuffer", _Peers)
    rank, hidden = 3, 7168
    gather = attention_residual.PeerColumnGather(hidden, 64, rank, WORLD, None)
    scratch = _slots(gather.scratch, gather)
    gen = torch.Generator(device="cuda").manual_seed(0)

    def randn(*shape):
        return torch.randn(*shape, device="cuda", generator=gen)

    for tokens, slots in ((4, 1), (64, 8), (7, 3), (32, 5), (1, 2)):
        shares = randn(WORLD, tokens, gather.cols).bfloat16()
        prefix = randn(tokens, hidden).bfloat16()
        bank = randn(tokens, slots, hidden).bfloat16()
        score_weight, norm = randn(hidden), (1 + 0.1 * randn(hidden)).bfloat16()
        epoch = gather.epochs[:tokens] + 1
        rows = torch.arange(tokens, device="cuda")
        for src in set(range(WORLD)) - {rank}:
            scratch[epoch % 2, src, rows] = _pushed(shares[src], epoch)

        operands = (prefix, bank, score_weight, norm, quant)
        want = _mix(torch.cat(list(shares), -1), *operands)
        stand_in = gather.defer(shares[rank]).fill_(float("nan"))  # never read
        got = _mix(stand_in, *operands)
        for g, w in zip(got, want, strict=True):
            assert torch.equal(g.view(torch.uint8), w.view(torch.uint8))
        assert torch.equal(gather.epochs[:tokens], epoch)
        for buffer in gather.buffer.buffers:
            pushed = _slots(buffer, gather)[epoch % 2, rank, rows]
            assert torch.equal(pushed, _pushed(shares[rank], epoch))


@pytest.fixture
def column_gather(monkeypatch, tail_calls):  # noqa: F811
    """Peer buffers on CPU, and the all-gather a torch consumer runs."""
    monkeypatch.setattr(
        peer_memory, "SymmetricPeerBuffer", lambda *args: _Peers(*args, device="cpu")
    )
    kimi.get_tp_group().cpu_group = None
    monkeypatch.setattr(
        communication_op,
        "tensor_model_parallel_all_gather",
        kimi.tensor_model_parallel_all_gather,
    )


@pytest.mark.parametrize("tokens", [1, 64])
def test_layer_defers_its_gather(ranks, column_gather, tokens):  # noqa: F811
    """Attached to every MoE layer but the stage's last, the gather makes a
    block hand back an unwritten stand-in that a torch consumer
    (`aux_hidden_state`, a drafter's tap) sees as the all-gather."""
    blocks = _blocks(_weights())
    x = torch.randn(tokens, HIDDEN, generator=torch.Generator().manual_seed(tokens))
    x, prefix = x.bfloat16(), torch.ones(tokens, HIDDEN).bfloat16()

    def on_rank(rank):
        want = blocks[rank](x)
        last = SimpleNamespace(column_tail_rows=64, column_gather=None)
        dense, moe = SimpleNamespace(), SimpleNamespace(block_sparse_moe=blocks[rank])
        kimi._attach_column_gather([dense, moe, SimpleNamespace(block_sparse_moe=last)])
        assert last.column_gather is None
        deferred, zero = blocks[rank](x)
        deferred.fill_(float("nan"))  # an unwritten stand-in, never read
        tapped = kimi.KimiDecoderLayer.aux_hidden_state((prefix, deferred, zero, None))
        return want, deferred, tapped

    for (gathered, _), deferred, tapped in ranks(on_rank):
        assert torch.equal(deferred, gathered)
        assert torch.equal(tapped, prefix + gathered)
