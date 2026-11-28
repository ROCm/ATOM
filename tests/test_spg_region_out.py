# SPDX-License-Identifier: MIT
"""A prefill token-graph recording hands the KDA recurrence region an output
buffer, which it writes and returns, so a replay reads the result in place.
Kernels are replaced by references."""

from __future__ import annotations

import contextlib
import importlib.util
import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch
import torch._dynamo  # before Triton may be faked below
from aiter_stub import atom_without_aiter
from torch import nn

from atom.utils import forward_context, region_graph


@contextlib.contextmanager
def _einops():
    """The Kimi model imports einops, which the CPU runner may not have."""
    if importlib.util.find_spec("einops") is not None:
        yield
        return
    einops = ModuleType("einops")
    einops.__getattr__ = lambda name: MagicMock(name=f"einops.{name}")
    sys.modules["einops"] = einops
    try:
        yield
    finally:
        sys.modules.pop("einops", None)


with _einops(), atom_without_aiter() as _load:
    model = _load("atom.models.kimi_k3.model")
    # `_prefill_kda` imports `gather_kda_inputs` from this package at call time.
    _kimi_k3_ops = _load("atom.model_ops.kimi_k3")


@pytest.fixture(autouse=True)
def _kimi_k3_ops_importable(monkeypatch):
    """The stand-in import above is dropped from ``sys.modules`` on exit; put
    it back while a test runs so the call-time import resolves without AITER."""
    monkeypatch.setitem(sys.modules, "atom.model_ops.kimi_k3", _kimi_k3_ops)


@pytest.fixture
def step(monkeypatch):
    ctx = SimpleNamespace(
        context=None, tr_graph=None, in_hipgraph=False, attn_metadata=None
    )
    monkeypatch.setattr(forward_context, "_forward_context", ctx)
    return ctx


H, D = 2, 4


def _kda_layer(**over):
    layer = SimpleNamespace(num_local_heads=H, head_dim=D, layer_name="kda", **over)
    return layer


@pytest.mark.parametrize(
    "tr_graph, handed",
    [("record", True), (None, False)],
)
def test_a_kda_recording_hands_its_recurrence_a_buffer(
    monkeypatch, step, tr_graph, handed
):
    step.tr_graph = tr_graph
    mixed = torch.zeros(6, 3 * H * D, dtype=torch.bfloat16)
    layer = _kda_layer(
        _project_in=lambda hs, scale: (mixed, "gate_out", "beta", "gate"),
        _project_out=lambda out, gate: out,
    )
    monkeypatch.setattr(model, "_kda_recurrence", lambda *args: args)
    args = model.KimiKDAAttention._forward_prefill_split(layer, None, None)
    assert len(args) == (5 if handed else 4)
    if handed:
        assert args[-1].shape == (12, H, D) and args[-1].dtype == torch.bfloat16


def test_the_recordings_dummy_prefill_zeros_the_buffer(step):
    step.context = SimpleNamespace(is_dummy_run=True)
    layer = _kda_layer()
    mixed = torch.zeros(6, 3 * H * D, dtype=torch.bfloat16)
    out = torch.full((12, H, D), 7.0, dtype=torch.bfloat16)
    got = model.KimiKDAAttention._live_prefill_recurrence(layer, mixed, None, None, out)
    assert got.data_ptr() == out.data_ptr() and got.shape == (6, H, D)
    assert not got.any() and bool((out[6:] == 7).all())


def _kda_inputs(total):
    generator = torch.Generator().manual_seed(total)

    def rand(*shape):
        return torch.randn(1, total, *shape, generator=generator).to(torch.bfloat16)

    return rand(H, D), rand(H, D), rand(H, D), rand(H, D), rand(H)


def _run_kda(q, k, v, g, beta, initial, cu, final, out=None):
    """A reference recurrence: output a function of each token, final states
    one per sequence."""
    o = torch.empty_like(v) if out is None else out
    o.copy_(q * 2 + v)
    return o, initial + cu.shape[0]


@pytest.mark.parametrize(
    "heads, bucket",
    [
        ([(2, 3), (6, 2)], 9),  # rows and heads overrun the bucket
        ([(6, 1)], 12),  # they do not fill it: pad rows past them
    ],
)
def test_prefill_kda_writes_the_buffer_and_returns_its_rows(heads, bucket):
    num_tokens = 8
    q, k, v, g, beta = _kda_inputs(bucket)
    tokens = torch.cat(
        [torch.arange(num_tokens)] + [torch.arange(s, s + n) for s, n in heads]
    )
    kda_cu = [0, num_tokens]
    for _, n in heads:
        kda_cu.append(kda_cu[-1] + n)
    ckpt = {
        "kda_tokens": tokens,
        "kda_cu": torch.tensor(kda_cu),
        "kda_rows": torch.zeros(1 + len(heads), dtype=torch.long),
    }
    initial = torch.zeros(1, H, D, D)
    layer = _kda_layer(
        _run_kda=_run_kda,
        _kda_out_rows=lambda out, q, v, cu: out[: q.shape[1]].unsqueeze(0),
    )
    attn = model.KimiKDAAttention
    want, want_last, want_heads = attn._prefill_kda(
        layer, q, k, v, g, beta, initial, ckpt["kda_cu"], ckpt
    )
    out = torch.full((2 * bucket, H, D), 7.0, dtype=torch.bfloat16)
    got, last, head_states = attn._prefill_kda(
        layer, q, k, v, g, beta, initial, ckpt["kda_cu"], ckpt, out=out
    )
    assert got.shape == want.shape == (1, bucket, H, D)
    assert got.data_ptr() == out.data_ptr() and got.is_contiguous()
    written = min(bucket, len(tokens))
    assert torch.equal(got[:, :written], want[:, :written])
    assert torch.equal(last, want_last) and torch.equal(head_states, want_heads)


def test_prefill_kda_without_room_keeps_its_own_output():
    q, k, v, g, beta = _kda_inputs(4)
    layer = _kda_layer(_run_kda=_run_kda, _kda_out_rows=lambda *a: None)
    cu = torch.tensor([0, 2, 4])
    rows = torch.zeros(2, dtype=torch.long)
    ckpt = {"kda_tokens": torch.arange(4), "kda_cu": cu, "kda_rows": rows}
    out = torch.zeros(8, H, D, dtype=torch.bfloat16)
    got, _, _ = model.KimiKDAAttention._prefill_kda(
        layer, q, k, v, g, beta, torch.zeros(2, H, D, D), cu, ckpt, out=out
    )
    assert got.data_ptr() != out.data_ptr() and not out.any()


def test_kda_out_rows_stays_off_the_cpu():
    layer = _kda_layer(_kda_gate_lower_bound=-5.0, A_log=torch.zeros(H))
    q = torch.zeros(1, 6, H, D, dtype=torch.bfloat16)
    out = torch.zeros(12, H, D, dtype=torch.bfloat16)
    rows = model.KimiKDAAttention._kda_out_rows
    # CPU tensors never reach FlashKDA.
    assert rows(layer, out, q, q, torch.tensor([0, 3, 6])) is None
    assert rows(layer, None, q, q, torch.tensor([0, 3, 6])) is None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="asks FlashKDA")
@pytest.mark.parametrize("cu", [[0, 96], [0, 40, 96]])
def test_every_flash_kda_call_writes_the_buffer(cu):
    """One sequence goes through `chunk_kimi_delta_attn`, several (short ones,
    below the balanced segmenting) too: both reach FlashKDA, which takes it."""
    layer = _kda_layer(A_log=torch.rand(12, device="cuda"), _kda_gate_lower_bound=-5.0)
    q = torch.randn(1, 96, 12, 128, device="cuda", dtype=torch.bfloat16)
    out = torch.empty(192, 12, 128, device="cuda", dtype=torch.bfloat16)
    cu = torch.tensor(cu, device="cuda", dtype=torch.int32)
    rows = model.KimiKDAAttention._kda_out_rows(layer, out, q, q, cu)
    assert rows is not None and rows.data_ptr() == out.data_ptr()
    assert rows.shape == q.shape


@pytest.mark.parametrize("given", [False, True])
def test_the_paged_recurrence_writes_the_buffer(given):
    seen = {}

    def chunk_kda(**kwargs):
        seen.update(kwargs)
        return kwargs["out"], None

    layer = _kda_layer(
        _chunk_kda=chunk_kda,
        A_log=None,
        dt_bias=None,
        _kda_gate_lower_bound=-5.0,
    )
    q, k, v, g, beta = _kda_inputs(6)
    metadata = SimpleNamespace(
        non_spec_query_start_loc=torch.tensor([0, 6]),
        non_spec_state_indices_tensor=torch.tensor([0]),
        has_initial_state=torch.tensor([True]),
    )
    out = torch.zeros(12, H, D, dtype=torch.bfloat16) if given else None
    extra = {"out": out} if given else {}
    got = model.KimiKDAAttention._run_kda_paged(
        layer, q, k, v, g, beta, metadata, "pool", **extra
    )
    assert got.shape == v.shape and got.is_contiguous()
    assert (
        (got.data_ptr() == out.data_ptr())
        if given
        else got.data_ptr()
        not in (
            q.data_ptr(),
            v.data_ptr(),
        )
    )


# The recording pattern on a card: a segment allocates the region's buffer,
# the region writes it and returns a view at its start, so the next segment
# reads the buffer itself and a replay copies nothing.
LIVE = {"scale": 1.0, "rows": 0}


class _BufferedRegion(nn.Module):
    def __init__(self, copy_in):
        super().__init__()
        self.a = nn.Linear(64, 64, device="cuda")
        self.copy_in = copy_in

    @region_graph.eager_region()
    def region(self, x, buf):
        n = x.shape[0]
        result = x * LIVE["scale"]
        if self.copy_in:  # a kernel that could not take the buffer
            buf[:n].copy_(result)
        else:
            torch.mul(x, LIVE["scale"], out=buf[:n])
        buf[n : n + LIVE["rows"]] = 1.0  # past the rows read: head tokens
        return buf[:n]

    def forward(self, x):
        h = self.a(x)
        buf = h.new_empty((2 * h.shape[0], h.shape[1]))
        return self.region(h, buf) * 3.0 + h


@pytest.mark.skipif(not torch.cuda.is_available(), reason="records a CUDA graph")
@pytest.mark.parametrize("copy_in", [False, True])
@torch.inference_mode()
def test_a_recorded_region_writes_the_segment_buffer_in_place(monkeypatch, copy_in):
    ctx = SimpleNamespace(in_hipgraph=False, main_stream=torch.cuda.current_stream())
    monkeypatch.setattr(forward_context, "_forward_context", ctx)
    torch.manual_seed(0)
    unit = _BufferedRegion(copy_in)
    n = 48
    x = torch.randn(n, 64, device="cuda")
    unit(x)  # hipBLASLt's first GEMM, outside the capture
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        ctx.main_stream = stream
        LIVE.update(scale=1.0, rows=5)
        unit(x)
        ctx.in_hipgraph = True
        segments, _ = region_graph._record_segments(
            unit, x, pool=torch.cuda.graph_pool_handle()
        )
        ctx.in_hipgraph = False
    torch.cuda.current_stream().wait_stream(stream)
    ctx.main_stream = torch.cuda.current_stream()
    (eager_step,) = [
        s for s in segments._steps if isinstance(s, region_graph._EagerStep)
    ]
    delivered = eager_step._target
    assert delivered.data_ptr() == eager_step._arguments["buf"].data_ptr()

    copies = []
    real_copy = torch.Tensor.copy_
    monkeypatch.setattr(
        torch.Tensor,
        "copy_",
        lambda self, src, *a, **k: (
            copies.append(self.data_ptr()),
            real_copy(self, src, *a, **k),
        )[1],
    )
    for scale, rows in ((0.5, 0), (2.0, 7), (-1.0, 48)):
        LIVE.update(scale=scale, rows=rows)
        fresh = torch.randn(n, 64, device="cuda")
        real_copy(x, fresh)
        copies.clear()
        got = segments.replay().clone()
        assert copies == ([delivered.data_ptr()] if copy_in else [])
        with torch.inference_mode():
            want = unit(fresh)
        torch.testing.assert_close(got, want, rtol=0, atol=0)
    del segments
    torch.cuda.synchronize()
