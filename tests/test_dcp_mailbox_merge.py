# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""The ``mailbox`` DCP merge backend (aiter's FlyDSL ``dcp_lse_combine``):
when a merge takes it, when it falls back to all-to-all, and -- across real
ranks -- that it merges what all-to-all merges."""

import os
import tempfile
from types import SimpleNamespace

import pytest
import torch

from atom.config import DCPConfig
from atom.model_ops import dcp_ops


def test_mailbox_is_a_comm_backend():
    assert DCPConfig(comm_backend="mailbox").comm_backend == "mailbox"


def _group(world=4):
    return SimpleNamespace(world_size=world, rank_in_group=0)


@pytest.fixture
def a2a_calls(monkeypatch):
    calls = []

    def fake_a2a(out, lse, group, owned_counts=None, quant_dtype=None):
        calls.append(quant_dtype)
        return "a2a"

    monkeypatch.setattr(dcp_ops, "cp_lse_a2a", fake_a2a)
    return calls


def test_without_a_workspace_the_mailbox_merge_is_all_to_all(a2a_calls):
    group = _group()
    out = torch.zeros(2, 16, 128, dtype=torch.bfloat16)
    lse = torch.zeros(2, 16)
    assert dcp_ops.dcp_lse_merge(out, lse, group, backend="mailbox") == "a2a"
    assert a2a_calls == [None]


@pytest.mark.parametrize(
    "tokens,dtype", [(9, torch.bfloat16), (0, torch.bfloat16), (2, torch.float32)]
)
def test_a_call_the_workspace_cannot_take_is_all_to_all(
    monkeypatch, a2a_calls, tokens, dtype
):
    group = _group()
    key = dcp_ops._mailbox_key(group, 4, 128)
    monkeypatch.setitem(dcp_ops._MAILBOX_WORKSPACES, key, SimpleNamespace(max_tokens=8))
    out = torch.zeros(tokens, 16, 128, dtype=dtype)
    lse = torch.zeros(tokens, 16)
    assert dcp_ops.dcp_lse_merge(out, lse, group, backend="mailbox") == "a2a"


def test_max_tokens_is_the_widest_decode_step():
    spec = SimpleNamespace(num_speculative_tokens=3)
    cfg = SimpleNamespace(max_num_seqs=16, speculative_config=spec)
    assert dcp_ops.dcp_mailbox_max_tokens(cfg) == 64
    cfg.speculative_config = None
    assert dcp_ops.dcp_mailbox_max_tokens(cfg) == 16


@pytest.mark.parametrize("decode,expected", [(True, "mailbox"), (False, "a2a")])
def test_only_decode_takes_the_mailbox(monkeypatch, decode, expected):
    from atom.model_ops.attention_mla import MLAAttention

    seen = []
    monkeypatch.setattr(
        dcp_ops, "dcp_lse_merge", lambda *a, **k: seen.append(a[3]) or "out"
    )
    layer = object.__new__(MLAAttention)
    layer.dcp_group = _group()
    layer.dcp_comm_backend = "mailbox"
    layer._dcp_merge(None, None, decode=decode)
    assert seen == [expected]


# ------------------------------------------------------------ across ranks
def _mailbox_available():
    try:
        from aiter.ops.flydsl.dcp_mailbox import dcp_lse_combine  # noqa: F401
    except ImportError:
        return False
    return True


def _worker(rank, world, init, results):
    import torch.distributed as dist

    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", init_method=init, rank=rank, world_size=world)
    group = SimpleNamespace(
        world_size=world,
        rank_in_group=rank,
        cpu_group=dist.new_group(backend="gloo"),
        device_group=dist.group.WORLD,
    )
    heads, dim, tokens = 4, 128, 8
    assert dcp_ops.prepare_dcp_mailbox(group, heads, dim, max_tokens=16)
    gen = torch.Generator(device="cuda").manual_seed(100 + rank)
    out = torch.randn(
        tokens, world * heads, dim, device="cuda", dtype=torch.bfloat16, generator=gen
    )
    lse = torch.randn(tokens, world * heads, device="cuda", generator=gen) * 3
    # rank 1 holds no key for row 2: its partial there is garbage
    owned = torch.full((tokens,), 5, dtype=torch.int32, device="cuda")
    if rank == 1:
        owned[2] = 0
        out[2] = float("nan")
    for _ in range(3):  # epochs and both mailbox parities
        got = dcp_ops.dcp_lse_merge(out, lse, group, "mailbox", owned_counts=owned)
        ref = dcp_ops.dcp_lse_merge(out, lse, group, "a2a", owned_counts=owned)
        torch.testing.assert_close(got, ref, rtol=1e-2, atol=1e-2)
    assert torch.isfinite(got).all()
    q, scale = dcp_ops.dcp_lse_merge(
        out, lse, group, "mailbox", owned_counts=owned, quant_dtype=torch.float8_e4m3fn
    )
    assert scale is None and q.dtype == torch.bfloat16
    torch.cuda.synchronize()
    dist.barrier()
    results[rank] = True
    dist.destroy_process_group()


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.device_count() < 2,
    reason="needs two GPUs",
)
@pytest.mark.skipif(not _mailbox_available(), reason="aiter has no dcp_mailbox")
def test_the_mailbox_merges_what_all_to_all_merges():
    import torch.multiprocessing as mp

    world = min(torch.cuda.device_count(), int(os.environ.get("DCP_MAILBOX_WORLD", "8")))
    with tempfile.TemporaryDirectory() as tmp:
        results = mp.Manager().dict()
        mp.spawn(
            _worker,
            args=(world, f"file://{tmp}/init", results),
            nprocs=world,
            join=True,
        )
    assert all(results.get(r) for r in range(world))
