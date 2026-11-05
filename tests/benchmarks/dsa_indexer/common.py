# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Checkpoint loading, synthetic cache construction and graph timing helpers."""

import gzip
import json

import torch
import torch.distributed as dist
import triton
import triton.language as tl
from aiter.dist.parallel_state import get_tp_group
from aiter.ops.triton._triton_kernels.quant.quant import _mxfp4_quant_op
from safetensors import safe_open

MODEL = None
PREFIX = "model.layers.0.self_attn.indexer"
PAGE, GRAPH_STEPS = 128, 8


def sync():
    torch.cuda.synchronize()
    dist.barrier(group=get_tp_group().cpu_group)


def checkpoint_tensors(device):
    mapping = json.loads((MODEL / "model.safetensors.index.json").read_text())["weight_map"]
    result = {}
    for suffix in ("wq_b.weight", "k_norm.weight", "k_norm.bias", "wk.weight", "weights_proj.weight"):
        key = f"{PREFIX}.{suffix}"
        with safe_open(MODEL / mapping[key], framework="pt", device="cpu") as handle:
            result[suffix] = handle.get_tensor(key).to(device)
    return result


def make_cache(blocks, rank, device, row_bytes):
    # Independent physical pages per request, no shared-prefix cache shortcut.
    # Old keys are synthetic FP8 values. All construction is outside timing.
    generator = torch.Generator(device=device).manual_seed(31415 + rank)
    cache = torch.zeros(blocks, PAGE, row_bytes, dtype=torch.uint8, device=device)
    flat = cache.view(blocks, -1)
    for start in range(0, blocks, 256):
        count = min(256, blocks - start)
        values = torch.randn(count, PAGE, 128, generator=generator, device=device, dtype=torch.float32)
        scale = (values.abs().amax(-1, keepdim=True).clamp_min(1e-4) / 448).log2().ceil().exp2()
        packed = (values / scale).to(torch.float8_e4m3fn).view(torch.uint8)
        packed = packed.view(count, PAGE // 16, 16, 8, 16).permute(0, 1, 3, 2, 4).contiguous()
        flat[start:start + count, :PAGE * 128] = packed.reshape(count, PAGE * 128)
        flat[start:start + count, PAGE * 128:PAGE * 132] = scale.reshape(count, PAGE).view(torch.uint8)
    return cache


def capture(call):
    sync()
    graph = torch.cuda.CUDAGraph()
    with (
        get_tp_group().graph_capture() as capture_context,
        torch.cuda.graph(graph, stream=capture_context.stream),
    ):
        for _ in range(GRAPH_STEPS):
            call()
    sync()
    graph.replay()
    sync()
    return graph


def time_graph(graph, replays):
    sync()
    begin = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    begin.record()
    for _ in range(replays):
        graph.replay()
    end.record()
    end.synchronize()
    elapsed_us = begin.elapsed_time(end) * 1000 / (GRAPH_STEPS * replays)
    gathered = [None] * dist.get_world_size()
    dist.all_gather_object(gathered, elapsed_us, group=get_tp_group().cpu_group)
    return {"rank_us": gathered, "max_rank_us": max(gathered)}


def profile_graph(graph, path, replays):
    sync()
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                           torch.profiler.ProfilerActivity.CUDA]) as profiler:
        for _ in range(replays):
            graph.replay()
        torch.cuda.synchronize()
    profiler.export_chrome_trace(str(path))
    with path.open("rb") as source, gzip.open(str(path) + ".gz", "wb") as target:
        import shutil
        shutil.copyfileobj(source, target)
    path.unlink()
    sync()


@triton.jit
def _quant(X, Q, S, N: tl.constexpr):
    row = tl.program_id(0)
    x = tl.load(X + row * 128 + tl.arange(0, 128))[None, :].to(tl.float32)
    q, s = _mxfp4_quant_op(x, 128, 1, 32, SCALING_MODE=1, USE_ASM=True)
    tl.store(Q + row * 64 + tl.arange(0, 64), q.reshape(64))
    tl.store(S + row * 4 + tl.arange(0, 4), s.reshape(4))


def quant(x):
    x = x.contiguous()
    q = torch.empty((*x.shape[:-1], 64), dtype=torch.uint8, device=x.device)
    s = torch.empty((*x.shape[:-1], 4), dtype=torch.uint8, device=x.device)
    _quant[(x.numel() // 128,)](x, q, s, x.numel() // 128)
    return q, s


def scales(s, rows):
    return s.reshape(-1, rows // 32, 32, 2, 2).permute(0, 4, 2, 3, 1).reshape(-1, 2, 32, rows // 16).contiguous()
