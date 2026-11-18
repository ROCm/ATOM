# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""The ``mailbox`` DCP merge backend (aiter's FlyDSL ``dcp_lse_combine``):
when a merge takes it, when it falls back to all-to-all, and -- across real
ranks, in and out of a CUDA graph -- that it merges what all-to-all merges."""

import os
import sys
import tempfile
from types import SimpleNamespace

import pytest
import torch

from atom.config import DCPConfig

# dcp_ops imports triton, absent on the CPU-only CI runner: guard per test so
# the config test still runs there.
try:
    from atom.model_ops import dcp_ops
except ImportError as _e:
    dcp_ops, _DCP_OPS_ERR = None, _e
else:
    _DCP_OPS_ERR = None
needs_dcp_ops = pytest.mark.skipif(
    _DCP_OPS_ERR is not None, reason=f"requires full atom import env: {_DCP_OPS_ERR}"
)


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


@needs_dcp_ops
def test_without_a_workspace_the_mailbox_merge_is_all_to_all(a2a_calls, capturing):
    group = _group()
    out = torch.zeros(2, 16, 128, dtype=torch.bfloat16)
    lse = torch.zeros(2, 16)
    assert dcp_ops.dcp_lse_merge(out, lse, group, backend="mailbox") == "a2a"
    assert a2a_calls == [None]


@needs_dcp_ops
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


@needs_dcp_ops
def test_empty_rows_are_masked_by_the_combine(monkeypatch, capturing):
    """Sparse MLA's owned_counts reach the combine as its row_counts, with the
    LSE handed over unmasked."""
    group = _group()
    key = dcp_ops._mailbox_key(group, 4, 128)
    monkeypatch.setitem(dcp_ops._MAILBOX_WORKSPACES, key, SimpleNamespace(max_tokens=8))
    seen = {}

    def combine(ws, o, l, base_e, row_counts=None):
        seen.update(lse=l, row_counts=row_counts)
        return o

    monkeypatch.setitem(
        sys.modules,
        "aiter.ops.flydsl.dcp_mailbox",
        SimpleNamespace(dcp_lse_combine=combine),
    )
    out = torch.zeros(2, 16, 128, dtype=torch.bfloat16)
    lse = torch.ones(2, 16)
    # a2a accepts a buffer longer than the step; the combine wants one per row
    owned = torch.tensor([3, 0, 7, 7], dtype=torch.int32)
    dcp_ops.dcp_lse_merge(out, lse, group, backend="mailbox", owned_counts=owned)
    assert seen["row_counts"].tolist() == [3, 0]
    assert seen["row_counts"].data_ptr() == owned.data_ptr()
    assert torch.equal(seen["lse"], lse)


@needs_dcp_ops
def test_an_eager_step_is_all_to_all(monkeypatch, a2a_calls):
    """An eager call re-traces the FlyDSL kernel on the host every time (tens
    of ms); a capture pays that once."""
    group = _group()
    key = dcp_ops._mailbox_key(group, 4, 128)
    monkeypatch.setitem(dcp_ops._MAILBOX_WORKSPACES, key, SimpleNamespace(max_tokens=8))
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    out = torch.zeros(2, 16, 128, dtype=torch.bfloat16)
    lse = torch.zeros(2, 16)
    assert dcp_ops.dcp_lse_merge(out, lse, group, backend="mailbox") == "a2a"


@needs_dcp_ops
def test_an_aiter_without_the_mailbox_fails_startup_with_the_way_out(monkeypatch):
    """No probing: a missing module raises, naming the a2a escape hatch."""
    monkeypatch.setitem(sys.modules, "aiter.ops.flydsl.dcp_mailbox", None)
    monkeypatch.setattr(dcp_ops, "_MAILBOX_WORKSPACES", {})
    group = SimpleNamespace(world_size=4, rank_in_group=0, cpu_group="gloo")
    with pytest.raises(ImportError, match='"comm_backend": "a2a"'):
        dcp_ops.prepare_dcp_mailbox(group, 16, 256, 64)


@needs_dcp_ops
def test_a_layout_too_wide_for_one_token_falls_back_once_with_a_reason(monkeypatch):
    """The layers share one attempt, and it says why it took a2a."""
    warned = []
    monkeypatch.setattr(
        dcp_ops, "logger", SimpleNamespace(warning=lambda *a: warned.append(a))
    )
    monkeypatch.setattr(dcp_ops, "_MAILBOX_WORKSPACES", {})
    group = SimpleNamespace(world_size=8, rank_in_group=0, cpu_group="gloo")
    heads = dcp_ops._MAILBOX_MAX_ELEMS // (8 * 128) + 1  # one token > the limit
    for _ in range(3):  # three MLA layers
        assert not dcp_ops.prepare_dcp_mailbox(group, heads, 128, 64)
    assert len(warned) == 1 and "exceeds" in warned[0][0]


@needs_dcp_ops
def test_a_failed_mapping_falls_back_once(monkeypatch):
    """aiter raises on every rank alike when the peer memory cannot be
    allocated or mapped; the layers share that one attempt."""
    tries = []

    class Workspace:
        def __init__(self, *a):
            tries.append(a)
            raise RuntimeError("peer memory allocation failed on another TP rank")

    monkeypatch.setitem(
        sys.modules,
        "aiter.ops.flydsl.dcp_mailbox",
        SimpleNamespace(CombineWorkspace=Workspace),
    )
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(dcp_ops, "_MAILBOX_WORKSPACES", {})
    group = SimpleNamespace(world_size=4, rank_in_group=0, cpu_group="gloo")
    for _ in range(3):  # three MLA layers
        assert not dcp_ops.prepare_dcp_mailbox(group, 16, 256, 64)
    assert len(tries) == 1


@needs_dcp_ops
@pytest.mark.parametrize(
    "ranks, heads, dim, widest, sized",
    [
        (4, 16, 256, 1024, 128),  # GLM-5.2 DCP4, 256 seqs x MTP3: 2^21 caps it
        (4, 16, 256, 16, 16),
        (8, 12, 128, 80, 80),  # Kimi-K3 DCP8, 80 seqs: under the cap (170)
        (8, 12, 128, 2048, 170),
    ],
)
def test_the_workspace_is_sized_for_merges_the_mailbox_can_take(
    monkeypatch, ranks, heads, dim, widest, sized
):
    """Merges over _MAILBOX_MAX_ELEMS take a2a, so the peer buffer stops there
    instead of growing with max_num_seqs."""
    built = []

    class Workspace:
        def __init__(self, group, device, heads, dim, max_tokens):
            built.append(max_tokens)
            self.peers = SimpleNamespace(bytes=torch.empty(0))

    monkeypatch.setitem(
        sys.modules,
        "aiter.ops.flydsl.dcp_mailbox",
        SimpleNamespace(
            CombineWorkspace=Workspace,
            dcp_lse_combine=lambda ws, out, lse, base_e=True, row_counts=None: out,
        ),
    )
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(dcp_ops, "_MAILBOX_WORKSPACES", {})
    group = SimpleNamespace(world_size=ranks, rank_in_group=0, cpu_group="gloo")
    assert dcp_ops.prepare_dcp_mailbox(group, heads, dim, widest)
    assert built == [sized]


@needs_dcp_ops
def test_max_tokens_is_the_widest_decode_step():
    spec = SimpleNamespace(num_speculative_tokens=3)
    cfg = SimpleNamespace(max_num_seqs=16, speculative_config=spec)
    assert dcp_ops.dcp_mailbox_max_tokens(cfg) == 64
    cfg.speculative_config = None
    assert dcp_ops.dcp_mailbox_max_tokens(cfg) == 16


@needs_dcp_ops
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


_METADATA = SimpleNamespace(dcp_owned_counts=torch.ones(8, dtype=torch.int32))


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


@needs_dcp_ops
def test_decode_hands_its_merge_to_the_mailbox(monkeypatch):
    seen = []
    layer = _merge_layer(monkeypatch, seen)
    layer._dcp_decode(torch.zeros(2, 16, 576), None, _METADATA, use_qrep=True)
    assert seen == [("mailbox", torch.float8_e4m3fn)]


@needs_dcp_ops
def test_prefill_keeps_all_to_all_and_its_fused_quant(monkeypatch):
    seen = []
    layer = _merge_layer(monkeypatch, seen)
    layer._dcp_sparse_prefill(torch.zeros(2, 16, 576), None, _METADATA, use_qrep=True)
    assert seen == [("a2a", torch.float8_e4m3fn)]


@needs_dcp_ops
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
    heads, dim = 4, 128
    assert dcp_ops.prepare_dcp_mailbox(group, heads, dim, max_tokens=64)
    # 8 tokens: one token a CTA; 49: multi-token CTAs ending in a partial tile
    for tokens in (8, 49):
        _check_mailbox_merge(rank, world, group, heads, dim, tokens)
    torch.cuda.synchronize()
    dist.barrier()
    results[rank] = True
    dist.destroy_process_group()


def _check_mailbox_merge(rank, world, group, heads, dim, tokens):
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
    # captured the way decode is: a mailbox merge first traced in the capture,
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


@needs_dcp_ops
@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.device_count() < 2,
    reason="needs two GPUs",
)
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
