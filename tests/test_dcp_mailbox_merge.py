# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""The ``mailbox`` DCP merge backend (aiter's FlyDSL ``dcp_lse_combine``):
when a merge takes it, when it falls back to all-to-all, and -- across real
ranks, in and out of a CUDA graph -- that it merges what all-to-all merges."""

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


@pytest.fixture
def capturing(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)


def test_without_a_workspace_the_mailbox_merge_is_all_to_all(a2a_calls, capturing):
    group = _group()
    out = torch.zeros(2, 16, 128, dtype=torch.bfloat16)
    lse = torch.zeros(2, 16)
    assert dcp_ops.dcp_lse_merge(out, lse, group, backend="mailbox") == "a2a"
    assert a2a_calls == [None]


@pytest.mark.parametrize(
    "tokens,dtype", [(9, torch.bfloat16), (0, torch.bfloat16), (2, torch.float32)]
)
def test_a_call_the_workspace_cannot_take_is_all_to_all(
    monkeypatch, a2a_calls, capturing, tokens, dtype
):
    group = _group()
    key = dcp_ops._mailbox_key(group, 4, 128)
    monkeypatch.setitem(dcp_ops._MAILBOX_WORKSPACES, key, SimpleNamespace(max_tokens=8))
    out = torch.zeros(tokens, 16, 128, dtype=dtype)
    lse = torch.zeros(tokens, 16)
    assert dcp_ops.dcp_lse_merge(out, lse, group, backend="mailbox") == "a2a"


def test_an_eager_step_is_all_to_all(monkeypatch, a2a_calls):
    """Each token count is its own kernel build; outside a capture it would
    compile mid-serving."""
    group = _group()
    key = dcp_ops._mailbox_key(group, 4, 128)
    monkeypatch.setitem(dcp_ops._MAILBOX_WORKSPACES, key, SimpleNamespace(max_tokens=8))
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    out = torch.zeros(2, 16, 128, dtype=torch.bfloat16)
    lse = torch.zeros(2, 16)
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


class _PerTokenFp8Proj:
    """An o_proj that wants a per-token FP8 activation, so the merge is asked
    to fuse the quant."""

    params_dtype = torch.float8_e4m3fn
    input_scale = None

    def __init__(self):
        from atom.model_ops.attention_mla import QuantType

        self.quant_type = QuantType.per_Token

    def __call__(self, x, scale=None):
        return x


def _merge_layer(monkeypatch, seen):
    """An MLA layer under DCP whose attention and projections are stubs, with
    its merge recorded as (backend, quant dtype)."""
    from atom.model_ops.attention_mla import MLAAttention

    def fake_merge(
        o, lse, group, backend, ctx=None, owned_counts=None, quant_dtype=None
    ):
        seen.append((backend, quant_dtype))
        return o if quant_dtype is None else (o, None)

    monkeypatch.setattr(dcp_ops, "dcp_lse_merge", fake_merge)
    layer = object.__new__(MLAAttention)
    layer.dcp_group = _group()
    layer.dcp_comm_backend = "mailbox"
    layer.dcp_world_size = 4
    layer.is_sparse_mla = True
    layer.dcp_owned_counts_buffer = torch.ones(8, dtype=torch.int32)
    layer._cp_triton_ctx = None
    layer.dcp_prefill_merge_bf16_ok = True
    layer.dcp_sparse_prefill_num_heads = 16
    layer.pbm_enabled = True
    layer.W_V_dcp = layer.W_V_dcp_scale = None
    layer.num_heads, layer.v_head_dim = 4, 128
    layer._v_up_proj = lambda o, *a, **k: o
    layer.o_proj = _PerTokenFp8Proj()
    attn = (torch.zeros(2, 16, 128, dtype=torch.bfloat16), torch.zeros(2, 16))
    layer._forward_decode = lambda *a, **k: attn
    layer._forward_prefill_mla = lambda *a, **k: attn
    return layer


def test_decode_hands_its_merge_to_the_mailbox(monkeypatch):
    seen = []
    layer = _merge_layer(monkeypatch, seen)
    layer._dcp_decode(torch.zeros(2, 16, 576), None, None, use_qrep=True)
    assert seen == [("mailbox", torch.float8_e4m3fn)]


def test_prefill_keeps_all_to_all_and_its_fused_quant(monkeypatch):
    seen = []
    layer = _merge_layer(monkeypatch, seen)
    layer._dcp_sparse_prefill(torch.zeros(2, 16, 576), None, None, use_qrep=True)
    assert seen == [("a2a", torch.float8_e4m3fn)]


@pytest.mark.parametrize("prepared,backend", [(True, "mailbox"), (False, "a2a")])
def test_a_layer_without_a_workspace_merges_with_all_to_all(
    monkeypatch, prepared, backend
):
    from atom.model_ops import attention_mla
    from atom.model_ops.attention_mla import MLAAttention

    monkeypatch.setattr(dcp_ops, "prepare_dcp_mailbox", lambda *a: prepared)
    monkeypatch.setattr(dcp_ops, "dcp_mailbox_max_tokens", lambda cfg: 8)
    monkeypatch.setattr(attention_mla, "get_current_atom_config", lambda: None)
    layer = object.__new__(MLAAttention)
    layer.dcp_group = _group()
    layer.dcp_comm_backend = "mailbox"
    layer.num_heads, layer.v_head_dim, layer.pbm_enabled = 4, 128, True
    layer._prepare_dcp_mailbox()
    assert layer.dcp_comm_backend == backend


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
    # captured the way decode is: a mailbox merge first built in the capture,
    # three layers in one graph, replays with fresh inputs and an eager
    # (all-to-all) merge in between, over both mailbox parities
    graph, side = torch.cuda.CUDAGraph(), torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side), torch.cuda.graph(graph, stream=side):
        got = [
            dcp_ops.dcp_lse_merge(out, lse, group, "mailbox", owned_counts=owned)
            for _ in range(3)
        ]
        fp8 = dcp_ops.dcp_lse_merge(
            out,
            lse,
            group,
            "mailbox",
            owned_counts=owned,
            quant_dtype=torch.float8_e4m3fn,
        )
    torch.cuda.current_stream().wait_stream(side)
    for step in range(3):
        torch.manual_seed(step * world + rank)
        out.copy_(torch.randn_like(out, dtype=torch.float32))
        lse.copy_(torch.randn_like(lse) * 3)
        if rank == 1:
            out[2] = float("nan")
        graph.replay()
        ref = dcp_ops.dcp_lse_merge(out, lse, group, "a2a", owned_counts=owned)
        eager = dcp_ops.dcp_lse_merge(out, lse, group, "mailbox", owned_counts=owned)
        for g in got:
            torch.testing.assert_close(g, ref, rtol=1e-2, atol=1e-2)
            assert torch.isfinite(g).all()
        torch.testing.assert_close(eager, ref, rtol=0, atol=0)
    # a combine that does not quantize says so with scale None
    rows, scale = fp8
    if scale is None:
        assert rows.dtype == torch.bfloat16
        torch.testing.assert_close(rows, ref, rtol=1e-2, atol=1e-2)
    else:
        assert rows.dtype == torch.float8_e4m3fn and scale.shape[0] == tokens
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

    world = min(
        torch.cuda.device_count(), int(os.environ.get("DCP_MAILBOX_WORLD", "8"))
    )
    with tempfile.TemporaryDirectory() as tmp:
        results = mp.Manager().dict()
        mp.spawn(
            _worker,
            args=(world, f"file://{tmp}/init", results),
            nprocs=world,
            join=True,
        )
    assert all(results.get(r) for r in range(world))
