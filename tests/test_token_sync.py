# SPDX-License-Identifier: MIT
"""ATOM_TP_TOKEN_SYNC_ALLGATHER: rank 0's ids via the custom all-gather.

The group is simulated: its `all_gather` returns what aiter's custom gather
returns, the rank-major concatenation of every rank's padded input -- with an
unset (here: garbage) tail, as the real padding buffer has. Every rank must
read rank 0's ids back, exactly, and the transport choice must depend on
nothing a single rank could see differently.
"""

import types

import pytest
import torch

from atom.distributed import token_sync

WORLD = 4


class _CaComm:
    def __init__(self, disabled=False, fits=True):
        self.disabled = disabled
        self.fits = fits
        self.asked = []

    def should_custom_ag(self, inp):
        self.asked.append(inp.numel() * inp.element_size())
        return self.fits


class _Group:
    """One rank's view of a TP group whose ranks hold `per_rank` tokens."""

    def __init__(self, rank, per_rank, ca_comm=None):
        self.world_size = len(per_rank)
        self.rank_in_group = rank
        self.per_rank = per_rank
        self.device_communicator = types.SimpleNamespace(
            ca_comm=ca_comm if ca_comm is not None else _CaComm()
        )
        self.gathers = []
        self.broadcasts = 0

    def broadcast(self, input_, src=0):
        self.broadcasts += 1
        input_.copy_(self.per_rank[src].contiguous())
        return input_

    def all_gather(self, input_, use_custom=False, dim=-1):
        assert use_custom and dim == 0 and input_.dim() == 1
        assert input_.dtype == torch.int32
        assert (input_.numel() * input_.element_size()) % 16 == 0
        self.gathers.append(input_.numel())
        chunks = []
        for r, tokens in enumerate(self.per_rank):
            if r == self.rank_in_group:
                chunks.append(input_)
                continue
            flat = tokens.contiguous().view(-1).view(torch.int32)
            chunk = torch.randint(-(2**31), 2**31 - 1, input_.shape, dtype=torch.int32)
            chunk[: flat.numel()] = flat
            chunks.append(chunk)
        return torch.cat(chunks)


def _per_rank(n, dtype, shape=None):
    gen = torch.Generator().manual_seed(n)
    out = []
    for _ in range(WORLD):
        t = torch.randint(0, 248320, (n,), generator=gen).to(dtype)
        out.append(t.view(shape) if shape else t)
    return out


@pytest.fixture
def enabled(monkeypatch):
    monkeypatch.setenv("ATOM_TP_TOKEN_SYNC_ALLGATHER", "1")
    monkeypatch.delenv("ATOM_USE_CUSTOM_ALL_GATHER", raising=False)


@pytest.mark.parametrize("n", [0, 1, 3, 4, 5, 16, 27, 40])
@pytest.mark.parametrize("dtype", [torch.int32, torch.int64, torch.float32])
def test_every_rank_reads_rank0(enabled, n, dtype):
    per_rank = _per_rank(n, dtype)
    for rank in range(WORLD):
        group = _Group(rank, per_rank)
        mine = per_rank[rank].clone()
        synced = token_sync.broadcast_from_rank0(mine, group)
        assert synced.dtype == dtype and synced.shape == mine.shape
        assert torch.equal(synced, per_rank[0])
        assert group.broadcasts == 0 and len(group.gathers) == 1
        # Unlike the broadcast, the caller's tensor is left alone.
        assert torch.equal(mine, per_rank[rank])


def test_shape_is_kept(enabled):
    per_rank = _per_rank(12, torch.int32, shape=(3, 4))
    synced = token_sync.broadcast_from_rank0(per_rank[2], _Group(2, per_rank))
    assert synced.shape == (3, 4) and torch.equal(synced, per_rank[0])


def test_ids_beyond_fp32_exact_range_survive(enabled):
    """A memcpy, not an fp32 round trip: values above 2**24 come back intact."""
    big = torch.tensor([2**24 + 1, 2**31 - 1, -1, 7], dtype=torch.int32)
    per_rank = [big] + [torch.zeros(4, dtype=torch.int32)] * (WORLD - 1)
    synced = token_sync.broadcast_from_rank0(per_rank[3].clone(), _Group(3, per_rank))
    assert torch.equal(synced, big)


@pytest.mark.parametrize(
    "env, ca_comm",
    [
        ({"ATOM_TP_TOKEN_SYNC_ALLGATHER": "0"}, _CaComm()),
        (
            {"ATOM_TP_TOKEN_SYNC_ALLGATHER": "1", "ATOM_USE_CUSTOM_ALL_GATHER": "0"},
            _CaComm(),
        ),
        ({"ATOM_TP_TOKEN_SYNC_ALLGATHER": "1"}, _CaComm(disabled=True)),
        ({"ATOM_TP_TOKEN_SYNC_ALLGATHER": "1"}, _CaComm(fits=False)),
    ],
)
def test_falls_back_to_the_broadcast(monkeypatch, env, ca_comm):
    monkeypatch.delenv("ATOM_TP_TOKEN_SYNC_ALLGATHER", raising=False)
    monkeypatch.delenv("ATOM_USE_CUSTOM_ALL_GATHER", raising=False)
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    per_rank = _per_rank(5, torch.int32)
    group = _Group(1, per_rank, ca_comm)
    synced = token_sync.broadcast_from_rank0(per_rank[1].clone(), group)
    assert torch.equal(synced, per_rank[0])
    assert group.broadcasts == 1 and group.gathers == []


def test_no_ca_comm_falls_back(enabled):
    per_rank = _per_rank(5, torch.int32)
    group = _Group(1, per_rank)
    group.device_communicator = types.SimpleNamespace(ca_comm=None)
    synced = token_sync.broadcast_from_rank0(per_rank[1].clone(), group)
    assert torch.equal(synced, per_rank[0]) and group.broadcasts == 1


def test_other_dtype_sizes_take_the_broadcast(enabled):
    per_rank = [torch.arange(6, dtype=torch.int16) + r for r in range(WORLD)]
    group = _Group(2, per_rank)
    synced = token_sync.broadcast_from_rank0(per_rank[2].clone(), group)
    assert torch.equal(synced, per_rank[0]) and group.broadcasts == 1


def test_single_rank_is_a_no_op(enabled):
    tokens = torch.arange(3, dtype=torch.int32)
    group = _Group(0, [tokens])
    assert token_sync.broadcast_from_rank0(tokens, group) is tokens


def test_decision_reads_only_padded_bytes(enabled):
    """`should_custom_ag` sees the padded payload, a whole number of 16-byte
    vectors that depends only on the (rank-invariant) element count."""
    ca = _CaComm()
    per_rank = _per_rank(5, torch.int32)
    token_sync.broadcast_from_rank0(per_rank[0].clone(), _Group(0, per_rank, ca))
    assert ca.asked == [32]


def test_flag_defaults_off(monkeypatch):
    from atom.utils import envs

    monkeypatch.delenv("ATOM_TP_TOKEN_SYNC_ALLGATHER", raising=False)
    assert envs.ATOM_TP_TOKEN_SYNC_ALLGATHER is False
