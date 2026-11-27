# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""apply_attn_res must read a strided slice of a block bank exactly as its copy."""

import json

import pytest
import torch

pytest.importorskip("aiter")
pytest.importorskip("triton")

from aiter import dtypes

from atom.model_ops.kimi_k3 import attention_residual
from atom.model_ops.kimi_k3.attention_residual import _apply_attn_res_impl

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="ROCm GPU required"
)

HIDDEN = 7168
BANK = 8  # Kimi-K3's 93 layers close a block every 12


def _inputs(tokens, seed):
    generator = torch.Generator().manual_seed(seed)

    def randn(*shape, scale=1.0):
        return (torch.randn(*shape, generator=generator) * scale).bfloat16().cuda()

    return {
        "bank": randn(tokens, BANK, HIDDEN),
        "prefix": randn(tokens, HIDDEN),
        "adds": [randn(tokens, HIDDEN) for _ in range(2)],
        "score_weight": randn(HIDDEN, scale=0.01).float(),
        "out_norm": randn(HIDDEN),
    }


def _call(x, block_residual, adds, out_norm, quant, num_blocks=None):
    return _apply_attn_res_impl(
        x["prefix"],
        block_residual,
        x["score_weight"],
        1e-6,
        add_hidden=x["adds"][0] if adds >= 1 else None,
        out_norm_weight=x["out_norm"] if out_norm else None,
        add_hidden2=x["adds"][1] if adds == 2 else None,
        quant_dtype=dtypes.fp8 if quant else None,
        num_blocks=num_blocks,
    )


def _assert_bitwise(got, want, what=""):
    assert len(got) == len(want)
    for g, w in zip(got, want):
        assert g.dtype == w.dtype and g.shape == w.shape, what
        assert torch.equal(g.view(torch.uint8), w.view(torch.uint8)), what


@pytest.mark.parametrize("tokens", [1, 6, 16, 48, 130])
@pytest.mark.parametrize(
    "adds,out_norm,quant",
    [
        (0, False, False),
        (1, False, False),
        (0, True, False),
        (0, True, True),
        (2, True, False),
        (2, True, True),
    ],
)
@torch.inference_mode()
def test_bank_slice_matches_its_copy(tokens, adds, out_norm, quant):
    x = _inputs(tokens, 20261029 + tokens)
    for blocks in range(1, BANK + 1):
        view = x["bank"][:, :blocks]
        # a single token's slice is contiguous at any width
        assert view.is_contiguous() == (blocks == BANK or tokens == 1)
        want = _call(x, view.contiguous(), adds, out_norm, quant)
        _assert_bitwise(_call(x, view, adds, out_norm, quant), want, f"{blocks} blocks")
        _assert_bitwise(_call(x, x["bank"], adds, out_norm, quant, blocks), want)


@pytest.mark.parametrize("tokens", [1, 16])
@torch.inference_mode()
def test_offset_slices_match_their_copies(tokens):
    # a slice past block 0, and one past token 0, are read in place too
    x = _inputs(tokens + 1, 20261031 + tokens)
    past_t0 = {**x, "prefix": x["prefix"][1:], "adds": [a[1:] for a in x["adds"]]}
    for inputs, view in ((x, x["bank"][:, 2:5]), (past_t0, x["bank"][1:, :3])):
        assert view.storage_offset() > 0
        want = _call(inputs, view.contiguous(), 2, True, True)
        _assert_bitwise(_call(inputs, view, 2, True, True), want)


@torch.inference_mode()
def test_the_kernel_reads_the_bank_in_place(monkeypatch):
    # The launch gets the slice itself, not a copy of it.
    kernel = attention_residual._attn_res_fused_kernel
    launched = []

    class Recorder:
        def __getitem__(self, grid):
            def launch(br, *args, **kwargs):
                launched.append((br.data_ptr(), br.stride()))
                return kernel[grid](br, *args, **kwargs)

            return launch

    monkeypatch.setattr(attention_residual, "_attn_res_fused_kernel", Recorder())
    x = _inputs(16, 20261032)
    view = x["bank"][:, :3]
    _call(x, view, 2, True, False)
    assert launched == [(view.data_ptr(), view.stride())]


@pytest.mark.parametrize(
    "tokens,adds,quant",
    [
        (1, 0, False),
        (4096, 2, False),
        (16, 1, True),
        (16, 2, True),
    ],
)
@torch.inference_mode()
def test_close_block_matches_copy_and_replays(tokens, adds, quant, record_property):
    from torch._inductor import metrics

    from atom.model_ops.kimi_k3.attention_residual import apply_attn_res

    torch.cuda.reset_peak_memory_stats()
    x = _inputs(tokens, 20261124 + tokens)
    storage = torch.full(
        (tokens + 2, BANK + 2, HIDDEN),
        -7,
        dtype=torch.bfloat16,
        device="cuda",
    )
    bank = storage[1:-1, 1:-1]
    bank.copy_(x["bank"])
    blocks = 3

    def reset_bank():
        storage.fill_(-7)
        bank.copy_(x["bank"])

    def close(prefix, bank, add, add2):
        return apply_attn_res(
            prefix,
            bank,
            x["score_weight"],
            1e-6,
            add,
            x["out_norm"],
            add_hidden2=add2,
            quant_dtype=dtypes.fp8 if quant else None,
            num_blocks=blocks,
            close_block=True,
        )

    add = x["adds"][0] if adds else None
    add2 = x["adds"][1] if adds == 2 else None
    want = _call(x, bank, adds, True, quant, blocks)
    expected = storage.clone()
    expected[1:-1, 1 + blocks].copy_(want[-1])

    def check(result):
        output, prefix = result
        assert prefix is None
        _assert_bitwise(output if quant else (output,), want[:-1])
        _assert_bitwise((storage,), (expected,))

    check(close(x["prefix"], bank, add, add2))
    compiled = torch.compile(close, fullgraph=True)
    reset_bank()
    generated_before = metrics.generated_kernel_count
    check(compiled(x["prefix"], bank, add, add2))
    # The custom op is one external launch. An Inductor-generated copy
    # would reintroduce the full-bank materialization this path avoids.
    assert metrics.generated_kernel_count == generated_before

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        replay_output = close(x["prefix"], bank, add, add2)
    for _ in range(2):
        reset_bank()
        graph.replay()
        check(replay_output)

    def explicit_copy():
        mixed, prefix = apply_attn_res(
            x["prefix"],
            bank,
            x["score_weight"],
            1e-6,
            add,
            x["out_norm"],
            add_hidden2=add2,
            quant_dtype=dtypes.fp8 if quant else None,
            num_blocks=blocks,
        )
        bank[:, blocks].copy_(prefix)
        return mixed

    def fused_close():
        return close(x["prefix"], bank, add, add2)

    for _ in range(10):
        explicit_copy()
        fused_close()
    samples = {"explicit_copy_us": [], "fused_close_us": []}
    for trial in range(3):
        order = (("explicit_copy_us", explicit_copy), ("fused_close_us", fused_close))
        for name, call in (order if trial % 2 == 0 else reversed(order)):
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(
                enable_timing=True
            )
            start.record()
            for _ in range(50):
                call()
            end.record()
            end.synchronize()
            samples[name].append(start.elapsed_time(end) * 1000 / 50)
    record_property("close_latency", json.dumps(samples))
    record_property("peak_allocated_bytes", torch.cuda.max_memory_allocated())


@pytest.mark.parametrize(
    "capacity,blocks,block_major",
    [(7, 7, False), (8, 1, False), (8, 8, True)],
    ids=["contiguous-seven", "capacity-eight-live-one", "block-major-eight"],
)
@torch.inference_mode()
def test_wide_bank_offsets(capacity, blocks, block_major):
    tokens = 49152
    shape = (capacity, tokens, HIDDEN) if block_major else (tokens, capacity, HIDDEN)
    bank = torch.full(shape, blocks + 1, dtype=torch.bfloat16, device="cuda")
    if block_major:
        bank = bank.transpose(0, 1)
    assert bank.stride(-1) == 1
    assert max((tokens - 1) * bank.stride(0), (blocks - 1) * bank.stride(1)) > 2**31
    prefix = torch.zeros((tokens, HIDDEN), dtype=torch.bfloat16, device="cuda")
    score = torch.zeros(HIDDEN, dtype=torch.float32, device="cuda")

    got, prefix_out = _apply_attn_res_impl(prefix, bank, score, 1e-6, num_blocks=blocks)
    # Zero scores average the live constant blocks with the zero prefix.
    torch.testing.assert_close(got, torch.full_like(got, blocks), rtol=0, atol=0)
    assert prefix_out is prefix
