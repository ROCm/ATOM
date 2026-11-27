# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

"""Benchmark 5 — Wgrad delay improves overlap.

**Benchmark tiers**

  * **Tier 0**: Legacy mechanism checks and low-level sanity for ``_DeferredWgrad``
    basics (``defer`` / ``execute``, stream overlap, etc.).
  * **Tier 1**: Real-module API overlap benchmarks using ``LumenColumnParallelLinear``
    and ``backward_dw()`` (not the legacy ``dwg.defer(weight, ...)`` path).
  * **Tier 2**: Megatron-style realism via ``TestMegatronStyleWgradDelay``; select
    with ``-k MegatronStyle`` under ``torchrun``.

Tests **actual Lumen implementations** only:

  * **Tier 0 — _DeferredWgrad API**: ``defer(weight, compute_fn)`` / ``execute()``
    from ``atomrl.lumen.modules.parallel_linear`` (mechanism / overhead checks).
  * **Tier 0 — stream overlap**: Deferred ``execute()`` on a secondary stream
    while the default stream runs other work — illustrates scheduling only;
    training-style overlap is benchmarked in Tier 1 / Tier 2.
  * **Tier 1 — real-module wgrad + NCCL allreduce** (multi-GPU): Overlaps the
    deferred wgrad GEMM with a real NCCL allreduce on a separate stream,
    demonstrating true hardware parallelism (compute SMs vs NIC/RDMA).
  * **Tier 1 — real-module wgrad + SDMA allreduce** (multi-GPU): Overlaps the deferred
    wgrad GEMM with Lumen's ``SdmaTpComm`` async allreduce.  SDMA uses
    dedicated DMA engines with zero SM contention — expected to yield
    higher overlap ratios than NCCL.
  * **gradient_accumulation_fusion**: The ``main_grad.add_(dw)`` path inside
    ``quantized_linear`` backward (``gradient_accumulation_fusion=True``).

Run single-GPU::

    python -m benchmarks.bench_wgrad_delay
    pytest benchmarks/bench_wgrad_delay.py -v -s

Run multi-GPU — NCCL::

    torchrun --nproc_per_node=2 -m pytest benchmarks/bench_wgrad_delay.py -v -s -k RealComm
    torchrun --nproc_per_node=8 -m pytest benchmarks/bench_wgrad_delay.py -v -s -k RealComm

Run multi-GPU — SDMA (requires mori)::

    torchrun --nproc_per_node=2 -m pytest benchmarks/bench_wgrad_delay.py -v -s -k SdmaComm
    torchrun --nproc_per_node=8 -m pytest benchmarks/bench_wgrad_delay.py -v -s -k SdmaComm

Run multi-GPU — NCCL vs SDMA comparison (requires mori)::

    torchrun --nproc_per_node=2 -m pytest benchmarks/bench_wgrad_delay.py -v -s -k NCCLvsSdma
    torchrun --nproc_per_node=8 -m pytest benchmarks/bench_wgrad_delay.py -v -s -k NCCLvsSdma

Run multi-GPU — NCCL vs SDMA wgrad scaling sweep (requires mori)::

    torchrun --nproc_per_node=8 -m pytest benchmarks/bench_wgrad_delay.py -v -s -k "wgrad_overlap_scaling_summary"

Run multi-GPU — Tier 2 Megatron-style realism (``LumenColumnParallelLinear`` /
``LumenRowParallelLinear``, sequence parallel, real TP)::

    torchrun --nproc_per_node=2 -m pytest benchmarks/bench_wgrad_delay.py -v -s -k MegatronStyle

Run profiled experiments (2+ GPUs; run each profile in a separate invocation)::

    torchrun --nproc_per_node=8 -m pytest benchmarks/bench_wgrad_delay.py -v -s -k backend_gap_experiment
    torchrun --nproc_per_node=8 -m pytest benchmarks/bench_wgrad_delay.py -v -s -k pipeline_gain_experiment

Override the active profile or individual dimensions::

    # Profiles: default | backend_gap | pipeline_gain
    export LUMEN_E2E_PROFILE=backend_gap
    export LUMEN_E2E_BATCH=4
    export LUMEN_E2E_SEQ=2048
    export LUMEN_E2E_HIDDEN=4096
    export LUMEN_E2E_FFN=28672

Run multi-GPU — all distributed tests (NCCL + SDMA + comparison, requires mori)::

    torchrun --nproc_per_node=2 -m pytest benchmarks/bench_wgrad_delay.py -v -s -k "RealComm or SdmaComm or NCCLvsSdma"
    torchrun --nproc_per_node=8 -m pytest benchmarks/bench_wgrad_delay.py -v -s -k "RealComm or SdmaComm or NCCLvsSdma"
"""

from __future__ import annotations

import math
import os
import traceback
from collections.abc import Callable
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
from torch import nn

from benchmarks.bench_utils import (
    BenchResult,
    cuda_timer,
    print_overlap_summary,
    print_report,
    print_report_with_table,
    require_cuda,
)
from benchmarks.conftest import AITER, CUDA
from benchmarks.e2e_fusion_profiles import (
    E2EFusionProfile,
    format_e2e_fusion_profile,
    get_e2e_fusion_profile,
    wgrad_delay_dims,
)

# ---------------------------------------------------------------------------
# Dimensions (Llama 3.1 8B)
# ---------------------------------------------------------------------------
M = 4096  # tokens (B * S)
K = 4096  # hidden_dim
N = 14336  # FFN intermediate
DEFAULT_PROFILE = get_e2e_fusion_profile()

# Timing parameters — overridable via LUMEN_BENCH_WARMUP / LUMEN_BENCH_ITERS
_WARMUP = 10
_ITERS = 30
_TRIM = 10.0


def _make_bench_linear_config(*, sequence_parallel: bool, tp_size: int):
    return SimpleNamespace(
        params_dtype=torch.bfloat16,
        perform_initialization=False,
        use_cpu_initialization=False,
        sequence_parallel=sequence_parallel,
        tensor_model_parallel_size=tp_size,
        expert_model_parallel_size=1,
        lumen_tp_comm_overlap=False,
    )


def _tp_sharded_out_features(out_features: int, world_size: int) -> int:
    """Return the per-rank output width for TP-style communication modeling."""
    if out_features % world_size != 0:
        raise ValueError(
            f"TP-sharded wgrad overlap requires out_features={out_features} divisible by world_size={world_size}"
        )
    return out_features // world_size


def _validate_megatron_style_world_size(world_size: int) -> None:
    if world_size < 2:
        raise ValueError(
            f"Tier 2 Megatron-style benchmark requires world_size >= 2, got {world_size}"
        )


def _make_profiled_wgrad_single_layer_inputs(
    device: torch.device,
    profile: E2EFusionProfile,
):
    tokens, hidden, ffn = wgrad_delay_dims(profile)
    x = torch.randn(tokens, hidden, device=device, dtype=torch.bfloat16)
    w = nn.Parameter(
        torch.randn(ffn, hidden, device=device, dtype=torch.bfloat16) * 0.02
    )
    w.main_grad = torch.zeros(ffn, hidden, device=device, dtype=torch.bfloat16)
    grad_out = torch.randn(tokens, ffn, device=device, dtype=torch.bfloat16)
    ar_buf = torch.randn(ffn, hidden, device=device, dtype=torch.bfloat16)
    return x, w, grad_out, ar_buf


def _make_profiled_wgrad_multi_layer_inputs(
    device: torch.device,
    profile: E2EFusionProfile,
    n_layers: int,
):
    tokens, hidden, ffn = wgrad_delay_dims(profile)
    x = torch.randn(tokens, hidden, device=device, dtype=torch.bfloat16)
    weights = [
        nn.Parameter(
            torch.randn(ffn, hidden, device=device, dtype=torch.bfloat16) * 0.02
        )
        for _ in range(n_layers)
    ]
    for w_i in weights:
        w_i.main_grad = torch.zeros_like(w_i.data)
    grad_outs = [
        torch.randn(tokens, ffn, device=device, dtype=torch.bfloat16)
        for _ in range(n_layers)
    ]
    return x, weights, grad_outs


def _profiled_wgrad_title(prefix: str, profile: E2EFusionProfile, world: int) -> str:
    return f"{prefix} ({format_e2e_fusion_profile(profile)}, world={world})"


# ---------------------------------------------------------------------------
# Tier 1 real-module API helpers (rank-local group + LumenColumnParallelLinear)
# ---------------------------------------------------------------------------

_RANK_LOCAL_GROUP = None
_RANK_LOCAL_GROUPS = None


def _get_or_create_rank_local_group():
    global _RANK_LOCAL_GROUP, _RANK_LOCAL_GROUPS
    if _RANK_LOCAL_GROUP is not None:
        return _RANK_LOCAL_GROUP, _RANK_LOCAL_GROUPS

    rank = dist.get_rank()
    world = dist.get_world_size()
    groups = []
    mine = None
    for r in range(world):
        group = dist.new_group(ranks=[r])
        groups.append(group)
        if r == rank:
            mine = group
    _RANK_LOCAL_GROUP = mine
    _RANK_LOCAL_GROUPS = groups
    return _RANK_LOCAL_GROUP, _RANK_LOCAL_GROUPS


def _build_real_api_single_layer(
    device: torch.device,
    profile: E2EFusionProfile,
    tp_group,
    *,
    tokens: int | None = None,
    hidden: int | None = None,
    out_features: int | None = None,
):
    from atomrl.lumen.modules.parallel_linear import LumenColumnParallelLinear

    t, h, ff = wgrad_delay_dims(profile)
    tok = tokens if tokens is not None else t
    hid = hidden if hidden is not None else h
    fout = out_features if out_features is not None else ff
    config = _make_bench_linear_config(sequence_parallel=False, tp_size=1)
    module = LumenColumnParallelLinear(
        hid,
        fout,
        config=config,
        init_method=lambda w: None,
        tp_group=tp_group,
    )
    module.delay_wgrad = True
    module.gradient_accumulation_fusion = True
    module.weight.main_grad = torch.zeros_like(module.weight)
    x = torch.randn(tok, hid, device=device, dtype=torch.bfloat16, requires_grad=True)
    grad_out = torch.randn(
        tok, module.weight.shape[0], device=device, dtype=torch.bfloat16
    )
    return module, x, grad_out


def _build_real_api_layer_stack(
    device: torch.device,
    profile: E2EFusionProfile,
    n_layers: int,
    *,
    tokens: int | None = None,
    hidden: int | None = None,
    out_features: int | None = None,
):
    tp_group, groups = _get_or_create_rank_local_group()
    modules = []
    xs = []
    grad_outs = []
    for _ in range(n_layers):
        module, x, grad_out = _build_real_api_single_layer(
            device,
            profile,
            tp_group,
            tokens=tokens,
            hidden=hidden,
            out_features=out_features,
        )
        modules.append(module)
        xs.append(x)
        grad_outs.append(grad_out)
    return modules, xs, grad_outs, groups


def _run_real_api_backward_then_queue(module, x, grad_out):
    y, _ = module(x)
    torch.autograd.backward(y, grad_out)
    if not module._deferred_wgrad.has_pending:
        raise AssertionError("expected pending deferred wgrad after backward")


def _run_real_api_single_layer_wgrad_queue_and_drain(
    device: torch.device,
    profile: E2EFusionProfile,
    *,
    tokens: int | None = None,
    hidden: int | None = None,
    out_features: int | None = None,
):
    """Tier 1 compute baseline: same deferred-wgrad API as overlap paths, drained immediately (no comm)."""
    tp_group, _ = _get_or_create_rank_local_group()
    module, x_m, g_m = _build_real_api_single_layer(
        device,
        profile,
        tp_group,
        tokens=tokens,
        hidden=hidden,
        out_features=out_features,
    )
    _run_real_api_backward_then_queue(module, x_m, g_m)
    module.backward_dw()


def _run_real_api_multi_layer_sequential_per_layer(
    device: torch.device,
    profile: E2EFusionProfile,
    n_layers: int,
    *,
    tokens: int | None = None,
    hidden: int | None = None,
    out_features: int | None = None,
    after_each_layer: Callable[[int, nn.Module], None] | None = None,
):
    """Per layer: forward → backward (defer wgrad) → backward_dw(); optional hook after each layer (e.g. AR)."""
    modules, xs, g_outs, _ = _build_real_api_layer_stack(
        device,
        profile,
        n_layers,
        tokens=tokens,
        hidden=hidden,
        out_features=out_features,
    )
    for i in range(n_layers):
        y, _ = modules[i](xs[i])
        torch.autograd.backward(y, g_outs[i])
        modules[i].backward_dw()
        if after_each_layer is not None:
            after_each_layer(i, modules[i])


# ---------------------------------------------------------------------------
# Tier 2 Megatron-style stack (WORLD TP + sequence parallel)
# ---------------------------------------------------------------------------


def _build_megatron_style_stack(
    device, world_size, profile, n_layers=2, use_sdma=False
):
    """Build a Column→Row→… stack on ``dist.group.WORLD`` with sequence parallel.

    Megatron-style Tier 2: ``tensor_model_parallel_size == world_size``,
    chained forward activations with independent per-layer autograd graphs
    (each next input is detached before the next layer runs).
    """
    from atomrl.lumen.modules.parallel_linear import (
        LumenColumnParallelLinear,
        LumenRowParallelLinear,
    )

    _validate_megatron_style_world_size(world_size)
    tokens, hidden, ffn = wgrad_delay_dims(profile)
    if tokens % world_size != 0:
        raise ValueError(
            f"Megatron-style stack: token count {tokens} not divisible by world_size={world_size}"
        )
    if ffn % world_size != 0:
        raise ValueError(
            f"Megatron-style stack: FFN dim {ffn} not divisible by world_size={world_size}"
        )
    config = _make_bench_linear_config(sequence_parallel=True, tp_size=world_size)
    modules = []
    for idx in range(n_layers):
        if idx % 2 == 0:
            module = LumenColumnParallelLinear(
                hidden,
                ffn,
                config=config,
                init_method=lambda w: None,
                tp_group=dist.group.WORLD,
            )
        else:
            module = LumenRowParallelLinear(
                ffn,
                hidden,
                config=config,
                init_method=lambda w: None,
                input_is_parallel=True,
                tp_group=dist.group.WORLD,
            )
        module.delay_wgrad = True
        module.gradient_accumulation_fusion = True
        module.use_sdma = use_sdma
        with torch.no_grad():
            # Tier 2 chains one layer's activations into the next, so unlike the
            # independent Tier 1 helpers we must seed finite params here instead
            # of relying on uninitialized storage.
            module.weight.fill_(0.01)
            if module.bias is not None:
                module.bias.zero_()
        module.weight.main_grad = torch.zeros_like(module.weight)
        modules.append(module)

    inputs = []
    outputs = []
    grad_outputs = []
    current = torch.randn(
        tokens // world_size, hidden, device=device, dtype=torch.bfloat16
    )
    for module in modules:
        layer_input = current.detach().requires_grad_(True)
        layer_output, _ = module(layer_input)
        inputs.append(layer_input)
        outputs.append(layer_output)
        grad_outputs.append(torch.randn_like(layer_output))
        current = layer_output
    return modules, inputs, outputs, grad_outputs


def _run_with_rank_local_diagnostics(label: str, fn):
    """Run *fn* and print a rank-local traceback on failure."""
    try:
        return fn()
    except Exception as exc:
        if dist.is_initialized():
            rank = dist.get_rank()
            world = dist.get_world_size()
        else:
            rank = -1
            world = -1
        print(
            f"\n[{label}] rank={rank}/{world} failed with {type(exc).__name__}: {exc}",
            flush=True,
        )
        traceback.print_exc()
        raise


def _validate_megatron_style_modules(modules, label: str) -> None:
    """Validate deferred-wgrad drain and main_grad writes for Megatron-style runs."""
    pending = [m._deferred_wgrad.has_pending for m in modules]
    grad_abs_sums = [float(m.weight.main_grad.abs().sum().item()) for m in modules]
    module_names = [f"{idx}:{type(m).__name__}" for idx, m in enumerate(modules)]
    rank = dist.get_rank() if dist.is_initialized() else -1

    if any(pending):
        raise AssertionError(
            f"{label}: rank={rank} still has pending deferred wgrad; "
            f"pending={pending} grad_abs_sums={grad_abs_sums} modules={module_names}"
        )
    if any((not math.isfinite(total)) or total <= 0.0 for total in grad_abs_sums):
        raise AssertionError(
            f"{label}: rank={rank} has invalid main_grad after backward_dw; "
            f"pending={pending} grad_abs_sums={grad_abs_sums} modules={module_names}"
        )


def _run_megatron_style_layerwise_backward(
    modules, outputs, grad_outputs, wgrad_stream
):
    """Layer-wise backward with deferred wgrad drained on ``wgrad_stream`` (reverse order)."""
    for idx in range(len(modules) - 1, -1, -1):
        try:
            if idx < len(modules) - 1:
                # Drain the prior layer's deferred wgrad before the next layer's
                # backward launches more TP communication. Without this handoff,
                # different ranks can diverge in SDMA collective ordering.
                torch.cuda.current_stream().wait_stream(wgrad_stream)
            torch.autograd.backward(outputs[idx], grad_outputs[idx])
            if not modules[idx]._deferred_wgrad.has_pending:
                raise AssertionError(f"layer {idx} missing deferred wgrad")
            wgrad_stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(wgrad_stream):
                modules[idx].backward_dw()
        except Exception:
            try:
                pending = [m._deferred_wgrad.has_pending for m in modules]
                module_names = [type(m).__name__ for m in modules]
                rank = dist.get_rank() if dist.is_initialized() else -1
                print(
                    f"[MegatronStyle layerwise] rank={rank} layer={idx} "
                    f"module={type(modules[idx]).__name__} "
                    f"output_shape={tuple(outputs[idx].shape)} grad_shape={tuple(grad_outputs[idx].shape)} "
                    f"pending={pending} modules={module_names}",
                    flush=True,
                )
            except (AttributeError, RuntimeError) as diag_exc:
                rank = dist.get_rank() if dist.is_initialized() else -1
                print(
                    f"[MegatronStyle layerwise] rank={rank} layer={idx} "
                    f"failed to collect diagnostics: {diag_exc}",
                    flush=True,
                )
            raise


# ---------------------------------------------------------------------------
# 1. Tier 0 — Lumen _DeferredWgrad API (mechanism / overhead)
# ---------------------------------------------------------------------------


@CUDA
class TestDeferredWgradAPI:
    """Tier 0: ``_DeferredWgrad`` defer/execute overhead and accumulation paths."""

    # Expected: Deferred wgrad (defer + execute) should have nearly identical
    # total latency to eager wgrad. The purpose of deferral is NOT to speed up
    # a single wgrad, but to allow scheduling it on a secondary stream later,
    # overlapping with other work. The defer() call itself is near-zero cost
    # (just stores a closure); execute() triggers the same GEMM.
    def test_defer_and_execute_latency(self):
        """Measure defer + execute vs eager wgrad."""
        from atomrl.lumen.modules.parallel_linear import _DeferredWgrad

        x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
        w = nn.Parameter(torch.randn(N, K, device="cuda", dtype=torch.bfloat16) * 0.02)
        grad_out = torch.randn(M, N, device="cuda", dtype=torch.bfloat16)

        def _eager():
            dw = grad_out.T @ x
            w.grad = dw

        r_eager = cuda_timer(_eager, label="eager wgrad (dW = gO^T @ X)")

        dwg = _DeferredWgrad()

        def _deferred():
            dwg.defer(w, lambda: grad_out.T @ x)
            dwg.execute()

        r_deferred = cuda_timer(_deferred, label="deferred wgrad (defer + execute)")

        print_report("_DeferredWgrad: Eager vs Deferred", [r_eager, r_deferred])

    # Expected: Near-zero latency (~0.001 ms). defer() only captures a Python
    # closure (weight reference + compute lambda) without launching any GPU
    # work. This confirms the deferral mechanism adds negligible overhead to
    # the backward pass critical path.
    def test_defer_only_latency(self):
        """Measure just the defer call (no execute) — should be near-zero."""
        from atomrl.lumen.modules.parallel_linear import _DeferredWgrad

        x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
        w = nn.Parameter(torch.randn(N, K, device="cuda", dtype=torch.bfloat16) * 0.02)
        grad_out = torch.randn(M, N, device="cuda", dtype=torch.bfloat16)
        dwg = _DeferredWgrad()

        r = cuda_timer(
            lambda: dwg.defer(w, lambda: grad_out.T @ x),
            label="defer only (closure capture)",
        )
        print_report("_DeferredWgrad: Defer Only", [r])
        print(f"  Defer overhead: {r.avg_ms:.4f} ms (should be ~0)")

    # Expected: _DeferredWgrad.execute() accumulates into main_grad when it
    # exists (line 58-59 of parallel_linear.py). After execute(), the weight's
    # main_grad should contain the computed gradient, and _pending_wgrad should
    # be cleared. This validates the accumulation path that the distributed
    # optimizer relies on.
    def test_main_grad_accumulation(self):
        """Verify execute() writes to main_grad when present."""
        from atomrl.lumen.modules.parallel_linear import _DeferredWgrad

        x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
        w = nn.Parameter(torch.randn(N, K, device="cuda", dtype=torch.bfloat16) * 0.02)
        w.main_grad = torch.zeros(N, K, device="cuda", dtype=torch.bfloat16)
        grad_out = torch.randn(M, N, device="cuda", dtype=torch.bfloat16)

        dwg = _DeferredWgrad()
        dwg.defer(w, lambda: grad_out.T @ x)

        assert dwg.has_pending
        dwg.execute()
        assert not dwg.has_pending

        assert (
            w.main_grad.abs().sum() > 0
        ), "main_grad should be non-zero after execute()"
        print(f"\n  main_grad norm: {w.main_grad.norm().item():.4f}")


# ---------------------------------------------------------------------------
# 2. Tier 0 — _DeferredWgrad stream overlap (scheduling demos)
# ---------------------------------------------------------------------------


@CUDA
class TestDeferredWgradStreamOverlap:
    """Tier 0: ``execute()`` on a secondary stream vs other stream work.

    Micro-benchmarks for the deferral mechanism; module-level training overlap
    is covered in Tier 1 / Tier 2.
    """

    # NOTE: Two large GEMMs on the same GPU typically saturate HBM bandwidth,
    # so stream-level overlap yields little speedup (overlap_ratio ≈ 0).
    # This test establishes the baseline. See test_overlap_with_simulated_comm
    # for the realistic scenario where wgrad overlaps with communication.
    def test_overlap_vs_sequential(self):
        from atomrl.lumen.modules.parallel_linear import _DeferredWgrad

        x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
        w = nn.Parameter(torch.randn(N, K, device="cuda", dtype=torch.bfloat16) * 0.02)
        w.main_grad = torch.zeros(N, K, device="cuda", dtype=torch.bfloat16)
        w_next = torch.randn(K, N, device="cuda", dtype=torch.bfloat16) * 0.02
        grad_out = torch.randn(M, N, device="cuda", dtype=torch.bfloat16)

        dwg = _DeferredWgrad()
        wgrad_stream = torch.cuda.Stream()

        # Sequential: execute deferred wgrad, then next-layer forward
        def _sequential():
            w.main_grad.zero_()
            dwg.defer(w, lambda: grad_out.T @ x)
            dwg.execute()
            torch.cuda.synchronize()
            _ = x @ w_next
            torch.cuda.synchronize()

        r_seq = cuda_timer(_sequential, label="sequential: execute() then fwd")

        # Measure individual components for overlap ratio
        r_dw = cuda_timer(lambda: grad_out.T @ x, label="dW alone")
        r_fwd = cuda_timer(lambda: x @ w_next, label="next-layer fwd alone")

        # Overlapped: execute deferred wgrad on wgrad_stream, fwd on compute
        def _overlapped():
            w.main_grad.zero_()
            dwg.defer(w, lambda: grad_out.T @ x)
            with torch.cuda.stream(wgrad_stream):
                dwg.execute()
            _ = x @ w_next  # concurrent on default stream
            wgrad_stream.synchronize()

        r_ovl = cuda_timer(_overlapped, label="overlapped: execute() || fwd")

        speedup = r_seq.avg_ms / max(r_ovl.avg_ms, 1e-6)
        T_parts = r_dw.avg_ms + r_fwd.avg_ms
        overlap_ratio = 1 - (r_ovl.avg_ms / max(T_parts, 1e-6))
        r_ovl.extra["speedup"] = round(speedup, 2)
        r_ovl.extra["overlap_ratio"] = round(overlap_ratio, 3)
        print_report(
            "_DeferredWgrad Stream Overlap",
            [r_dw, r_fwd, r_seq, r_ovl],
        )
        print(f"  Overlap ratio: {overlap_ratio:.3f}")

    # Expected: overlap_ratio > 0.3. Under distributed training (Tier 1/2),
    # _DeferredWgrad.execute() (a GEMM on compute SMs) can run concurrently with
    # communication (allreduce / reduce-scatter on SDMA/NIC hardware). Since they
    # use different hardware resources, true parallelism is achieved. We simulate the communication
    # latency with torch.cuda._sleep() which blocks the stream without using
    # compute SMs or HBM bandwidth, mimicking SDMA/NIC behaviour.
    def test_overlap_with_simulated_comm(self):
        from atomrl.lumen.modules.parallel_linear import _DeferredWgrad

        x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
        w = nn.Parameter(torch.randn(N, K, device="cuda", dtype=torch.bfloat16) * 0.02)
        w.main_grad = torch.zeros(N, K, device="cuda", dtype=torch.bfloat16)
        grad_out = torch.randn(M, N, device="cuda", dtype=torch.bfloat16)

        dwg = _DeferredWgrad()
        comm_stream = torch.cuda.Stream()

        # Calibrate: measure wgrad GEMM latency, then set sleep to ~same duration
        r_dw = cuda_timer(
            lambda: grad_out.T @ x,
            label="dW alone",
            warmup=_WARMUP,
            iters=_ITERS,
            trim_pct=_TRIM,
        )
        dw_ns = int(r_dw.avg_ms * 1e6)
        # _sleep() takes GPU clock cycles; approximate 1 ns ≈ 1-2 cycles at ~1.5 GHz
        sleep_cycles = max(dw_ns * 2, 100_000)

        r_comm = cuda_timer(
            lambda: torch.cuda._sleep(sleep_cycles),
            label="simulated comm (sleep)",
            warmup=_WARMUP,
            iters=_ITERS,
            trim_pct=_TRIM,
        )

        # Sequential: wgrad then comm
        def _sequential():
            w.main_grad.zero_()
            dwg.defer(w, lambda: grad_out.T @ x)
            dwg.execute()
            torch.cuda.synchronize()
            torch.cuda._sleep(sleep_cycles)
            torch.cuda.synchronize()

        r_seq = cuda_timer(
            _sequential,
            label="sequential: wgrad then comm",
            warmup=_WARMUP,
            iters=_ITERS,
            trim_pct=_TRIM,
        )

        # Overlapped: wgrad on default stream, comm on comm_stream
        def _overlapped():
            w.main_grad.zero_()
            dwg.defer(w, lambda: grad_out.T @ x)
            with torch.cuda.stream(comm_stream):
                torch.cuda._sleep(sleep_cycles)
            dwg.execute()
            torch.cuda.current_stream().wait_stream(comm_stream)

        r_ovl = cuda_timer(
            _overlapped,
            label="overlapped: wgrad || comm",
            warmup=_WARMUP,
            iters=_ITERS,
            trim_pct=_TRIM,
        )

        T_parts = r_dw.avg_ms + r_comm.avg_ms
        overlap_ratio = 1 - (r_ovl.avg_ms / max(T_parts, 1e-6))
        speedup = r_seq.avg_ms / max(r_ovl.avg_ms, 1e-6)
        r_ovl.extra["speedup"] = round(speedup, 2)
        r_ovl.extra["overlap_ratio"] = round(overlap_ratio, 3)
        print_report(
            "_DeferredWgrad + Simulated Comm Overlap",
            [r_dw, r_comm, r_seq, r_ovl],
        )
        print_overlap_summary(
            t_compute=r_dw.avg_ms,
            t_comm=r_comm.avg_ms,
            t_seq=r_seq.avg_ms,
            t_ovl=r_ovl.avg_ms,
            compute_label="wgrad",
            comm_label="sim comm",
        )

    # Expected: Running execute() on a secondary stream for each of N layers
    # in a pipeline should show cumulative speedup. Each layer's deferred dW
    # overlaps with the next layer's dX, hiding N-1 dW computations from
    # the critical path. The total time approaches T(all_dX) + T(one_dW)
    # instead of T(all_dX) + T(all_dW).
    def test_multi_layer_pipeline(self):
        from atomrl.lumen.modules.parallel_linear import _DeferredWgrad

        n_layers = 4
        weights = [
            nn.Parameter(torch.randn(N, K, device="cuda", dtype=torch.bfloat16) * 0.02)
            for _ in range(n_layers)
        ]
        for w in weights:
            w.main_grad = torch.zeros_like(w.data)

        x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
        grad_outs = [
            torch.randn(M, N, device="cuda", dtype=torch.bfloat16)
            for _ in range(n_layers)
        ]

        dwg = _DeferredWgrad()
        wgrad_stream = torch.cuda.Stream()

        def _eager_pipeline():
            for i in range(n_layers):
                weights[i].main_grad.zero_()
                _ = grad_outs[i] @ weights[i]  # dX
                weights[i].main_grad.add_(grad_outs[i].T @ x)  # dW

        def _deferred_pipeline():
            for i in range(n_layers):
                weights[i].main_grad.zero_()
                if dwg.has_pending:
                    with torch.cuda.stream(wgrad_stream):
                        dwg.execute()
                _ = grad_outs[i] @ weights[i]  # dX on compute stream
                dwg.defer(weights[i], lambda i=i: grad_outs[i].T @ x)
            with torch.cuda.stream(wgrad_stream):
                dwg.execute()
            wgrad_stream.synchronize()

        r_eager = cuda_timer(_eager_pipeline, label=f"{n_layers}-layer eager")
        r_deferred = cuda_timer(_deferred_pipeline, label=f"{n_layers}-layer deferred")

        speedup = r_eager.avg_ms / max(r_deferred.avg_ms, 1e-6)
        r_deferred.extra["speedup"] = round(speedup, 2)
        print_report(
            f"{n_layers}-Layer Pipeline: Eager vs _DeferredWgrad",
            [r_eager, r_deferred],
        )


# ---------------------------------------------------------------------------
# 3. Tier 1 — Multi-GPU: real-module deferred wgrad + NCCL allreduce overlap
# ---------------------------------------------------------------------------


def _init_dist():
    if dist.is_initialized():
        return
    if "RANK" not in os.environ:
        return
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    dist.init_process_group(
        backend="cpu:gloo,cuda:nccl",
        device_id=torch.device(f"cuda:{local_rank}"),
    )


_DIST = pytest.mark.skipif(
    "RANK" not in os.environ,
    reason="Multi-GPU — run with torchrun --nproc_per_node=N",
)


@_DIST
class TestDeferredWgradRealComm:
    """Overlap real ``LumenColumnParallelLinear`` deferred wgrad (``backward_dw``) with NCCL allreduce.

    After each layer's backward, the weight gradient is deferred inside the
    module and then drained via ``backward_dw()`` while the main stream
    launches an allreduce for the *previous* layer's gradient.  NCCL
    allreduce uses NIC/RDMA hardware while the wgrad path uses compute SMs.

    Run::

        torchrun --nproc_per_node=2 benchmarks/bench_wgrad_delay.py
        torchrun --nproc_per_node=8 -m pytest benchmarks/bench_wgrad_delay.py -v -s -k RealComm
    """

    @pytest.fixture(autouse=True)
    def _setup(self):
        _init_dist()
        self.rank = dist.get_rank()
        self.world = dist.get_world_size()
        self.device = torch.device(f"cuda:{self.rank}")
        yield
        dist.barrier()

    # Expected: The deferred wgrad work (real LumenColumnParallelLinear path,
    # drained via backward_dw) runs on compute SMs concurrently with NCCL
    # allreduce (on NIC/RDMA hardware).
    # Total time ≈ max(T_wgrad, T_allreduce) instead of their sum.
    # When allreduce >> wgrad (common with small TP-local matrices),
    # the overlap_ratio formula yields a low value because the hidden
    # wgrad is a small fraction of T_total. The key metric is:
    #   speedup = T_seq / T_ovl  (should be > 1.0)
    #   hidden_ms = T_seq - T_ovl  (should be close to T_wgrad)
    def test_wgrad_overlap_with_allreduce(self):
        profile = DEFAULT_PROFILE
        _, _, _, ar_buf = _make_profiled_wgrad_single_layer_inputs(self.device, profile)

        comm_stream = torch.cuda.Stream(device=self.device)

        # Warmup
        for _ in range(3):
            dist.all_reduce(ar_buf.clone())
            _run_real_api_single_layer_wgrad_queue_and_drain(self.device, profile)
        torch.cuda.synchronize()
        dist.barrier()

        # Measure components individually
        r_wgrad = cuda_timer(
            lambda: _run_real_api_single_layer_wgrad_queue_and_drain(
                self.device, profile
            ),
            label="real module: wgrad (queue+drain, no comm)",
            warmup=_WARMUP,
            iters=_ITERS,
            trim_pct=_TRIM,
        )

        def _allreduce():
            buf = ar_buf.clone()
            dist.all_reduce(buf)

        r_comm = cuda_timer(
            _allreduce,
            label="allreduce alone",
            warmup=_WARMUP,
            iters=_ITERS,
            trim_pct=_TRIM,
            dist_barrier=True,
        )

        # Sequential: wgrad then allreduce
        def _sequential():
            tp_group, _ = _get_or_create_rank_local_group()
            module, x_m, g_m = _build_real_api_single_layer(
                self.device, profile, tp_group
            )
            _run_real_api_backward_then_queue(module, x_m, g_m)
            module.backward_dw()
            buf = ar_buf.clone()
            dist.all_reduce(buf)

        r_seq = cuda_timer(
            _sequential,
            label="sequential: real module wgrad then allreduce",
            warmup=_WARMUP,
            iters=_ITERS,
            trim_pct=_TRIM,
            dist_barrier=True,
        )

        # Overlapped: allreduce on comm_stream, backward_dw on default stream
        def _overlapped():
            tp_group, _ = _get_or_create_rank_local_group()
            module, x_m, g_m = _build_real_api_single_layer(
                self.device, profile, tp_group
            )
            _run_real_api_backward_then_queue(module, x_m, g_m)
            with torch.cuda.stream(comm_stream):
                buf = ar_buf.clone()
                dist.all_reduce(buf)
            module.backward_dw()
            torch.cuda.current_stream().wait_stream(comm_stream)

        r_ovl = cuda_timer(
            _overlapped,
            label="overlapped: real module wgrad || allreduce",
            warmup=_WARMUP,
            iters=_ITERS,
            trim_pct=_TRIM,
            dist_barrier=True,
        )

        T_parts = r_wgrad.avg_ms + r_comm.avg_ms
        overlap_ratio = 1 - (r_ovl.avg_ms / max(T_parts, 1e-6))
        speedup = r_seq.avg_ms / max(r_ovl.avg_ms, 1e-6)
        r_ovl.extra["speedup"] = round(speedup, 2)
        r_ovl.extra["overlap_ratio"] = round(overlap_ratio, 3)

        if self.rank == 0:
            print_report_with_table(
                _profiled_wgrad_title(
                    "Real module wgrad-delay + NCCL AllReduce", profile, self.world
                ),
                [r_wgrad, r_comm, r_seq, r_ovl],
            )
            print_overlap_summary(
                t_compute=r_wgrad.avg_ms,
                t_comm=r_comm.avg_ms,
                t_seq=r_seq.avg_ms,
                t_ovl=r_ovl.avg_ms,
                compute_label="wgrad (real API)",
                comm_label="allreduce",
            )

    # Expected: In a 4-layer pipeline, each layer's deferred wgrad overlaps
    # with the allreduce of the previous layer's gradient. Cumulative
    # savings ≈ N × T_wgrad (all wgrads hidden behind allreduces).
    def test_multi_layer_pipeline_with_allreduce(self):
        profile = DEFAULT_PROFILE
        n_layers = 4
        _, _, _, ar_ref = _make_profiled_wgrad_single_layer_inputs(self.device, profile)

        comm_stream = torch.cuda.Stream(device=self.device)

        # Warmup
        for _ in range(3):
            dist.all_reduce(ar_ref.clone())
            _run_real_api_single_layer_wgrad_queue_and_drain(self.device, profile)
        torch.cuda.synchronize()
        dist.barrier()

        # Sequential baseline: real module per layer (queue+drain) + allreduce (no cross-layer overlap)
        def _eager():
            _run_real_api_multi_layer_sequential_per_layer(
                self.device,
                profile,
                n_layers,
                after_each_layer=lambda _i, m: dist.all_reduce(m.weight.main_grad),
            )

        r_eager = cuda_timer(
            _eager,
            label=f"{n_layers}L sequential real-module+NCCL AR",
            warmup=_WARMUP,
            iters=_ITERS,
            trim_pct=_TRIM,
            dist_barrier=True,
        )

        # Deferred: overlap wgrad with allreduce of previous layer
        def _deferred():
            modules, xs, g_outs, _ = _build_real_api_layer_stack(
                self.device, profile, n_layers
            )
            pending_ar = None
            for i in range(n_layers):
                if i > 0 and modules[i - 1]._deferred_wgrad.has_pending:
                    modules[i - 1].backward_dw()

                if pending_ar is not None:
                    torch.cuda.current_stream().wait_stream(comm_stream)

                y, _ = modules[i](xs[i])
                torch.autograd.backward(y, g_outs[i])

                if i > 0:
                    comm_stream.wait_stream(torch.cuda.current_stream())
                    with torch.cuda.stream(comm_stream):
                        dist.all_reduce(modules[i - 1].weight.main_grad)
                    pending_ar = i - 1

            if modules[n_layers - 1]._deferred_wgrad.has_pending:
                modules[n_layers - 1].backward_dw()
            comm_stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(comm_stream):
                dist.all_reduce(modules[n_layers - 1].weight.main_grad)
            torch.cuda.current_stream().wait_stream(comm_stream)

        r_deferred = cuda_timer(
            _deferred,
            label=f"{n_layers}L pipelined deferred wgrad+NCCL AR",
            warmup=_WARMUP,
            iters=_ITERS,
            trim_pct=_TRIM,
            dist_barrier=True,
        )

        speedup = r_eager.avg_ms / max(r_deferred.avg_ms, 1e-6)
        r_deferred.extra["speedup"] = round(speedup, 2)

        saved_ms = r_eager.avg_ms - r_deferred.avg_ms

        if self.rank == 0:
            print_report_with_table(
                _profiled_wgrad_title(
                    f"{n_layers}-Layer Pipeline: Sequential vs Pipelined (real module + NCCL)",
                    profile,
                    self.world,
                ),
                [r_eager, r_deferred],
            )
            print(f"  Speedup:  {speedup:.2f}x")
            print(
                f"  Saved:    {saved_ms:.3f} ms  "
                f"(≈ {n_layers} × {saved_ms / n_layers:.3f} ms per layer)"
            )
            print()


# ---------------------------------------------------------------------------
# SDMA / mori helpers: MegatronStyle SDMA test + Tier 1-style SdmaComm below
# ---------------------------------------------------------------------------


def _sdma_available():
    try:
        os.environ.setdefault("MORI_ENABLE_SDMA", "1")
        import mori  # noqa: F401

        return True
    except ImportError:
        return False


_SDMA = pytest.mark.skipif(
    "RANK" not in os.environ or not _sdma_available(),
    reason="Multi-GPU + mori SDMA required",
)


# ---------------------------------------------------------------------------
# 4. Tier 2 — Megatron-style wgrad-delay realism (-k MegatronStyle)
# ---------------------------------------------------------------------------


@_DIST
class TestMegatronStyleWgradDelay:
    """Tier 2 Megatron-style wgrad-delay realism: Column/Row TP + sequence parallel.

    Uses real ``LumenColumnParallelLinear`` / ``LumenRowParallelLinear`` on
    ``dist.group.WORLD`` with ``sequence_parallel=True``.  Select with pytest
    ``-k MegatronStyle``.
    """

    @pytest.fixture(autouse=True)
    def _setup(self, request):
        # Match other SDMA benchmarks: MORI must be enabled before process-group init.
        if request.node.name == "test_megatron_style_pipeline_sdma":
            os.environ["MORI_ENABLE_SDMA"] = "1"
        _init_dist()
        self.rank = dist.get_rank()
        self.world = dist.get_world_size()
        self.device = torch.device(f"cuda:{self.rank}")
        yield
        dist.barrier()

    def test_megatron_style_pipeline_nccl(self):
        profile = DEFAULT_PROFILE
        torch.cuda.synchronize()
        dist.barrier()
        wgrad_stream = torch.cuda.Stream(device=self.device)

        def _run_once():
            modules, _inputs, outputs, grad_outputs = _build_megatron_style_stack(
                self.device,
                self.world,
                profile,
                n_layers=2,
                use_sdma=False,
            )
            _run_megatron_style_layerwise_backward(
                modules, outputs, grad_outputs, wgrad_stream
            )
            torch.cuda.synchronize()
            return modules

        modules = _run_once()
        assert all(not m._deferred_wgrad.has_pending for m in modules)
        assert all(m.weight.main_grad.abs().sum() > 0 for m in modules)

        r = cuda_timer(
            _run_once,
            label="Megatron-style NCCL realism",
            warmup=_WARMUP,
            iters=_ITERS,
            trim_pct=_TRIM,
            dist_barrier=True,
        )
        if self.rank == 0:
            print_report_with_table(
                _profiled_wgrad_title(
                    "Tier 2 Megatron-style NCCL", profile, self.world
                ),
                [r],
            )

    @_SDMA
    def test_megatron_style_pipeline_sdma(self):
        profile = DEFAULT_PROFILE
        torch.cuda.synchronize()
        dist.barrier()
        wgrad_stream = torch.cuda.Stream(device=self.device)
        run_counter = {"value": 0}
        validate_each_call = (
            os.environ.get("LUMEN_MEGATRON_VALIDATE_EACH_CALL", "0") == "1"
        )

        def _run_once():
            run_counter["value"] += 1

            def _body():
                modules, _inputs, outputs, grad_outputs = _build_megatron_style_stack(
                    self.device,
                    self.world,
                    profile,
                    n_layers=2,
                    use_sdma=True,
                )
                _run_megatron_style_layerwise_backward(
                    modules, outputs, grad_outputs, wgrad_stream
                )
                torch.cuda.synchronize()
                if run_counter["value"] == 1 or validate_each_call:
                    _validate_megatron_style_modules(
                        modules,
                        f"Megatron-style SDMA call={run_counter['value']}",
                    )
                return modules

            return _run_with_rank_local_diagnostics(
                f"Megatron-style SDMA call={run_counter['value']}",
                _body,
            )

        _run_once()

        r = cuda_timer(
            _run_once,
            label="Megatron-style SDMA realism",
            warmup=_WARMUP,
            iters=_ITERS,
            trim_pct=_TRIM,
            dist_barrier=True,
        )
        if self.rank == 0:
            print_report_with_table(
                _profiled_wgrad_title(
                    "Tier 2 Megatron-style SDMA", profile, self.world
                ),
                [r],
            )


# ---------------------------------------------------------------------------
# 5. Multi-GPU: real-module deferred wgrad + SDMA allreduce overlap
# ---------------------------------------------------------------------------


@_SDMA
class TestDeferredWgradSdmaComm:
    """Overlap real ``LumenColumnParallelLinear`` deferred wgrad with SDMA allreduce.

    SDMA uses dedicated hardware DMA engines that are often independent of
    compute SMs.  This test validates comm-compute overlap using the same
    module/autograd path as training, with wgrad drained via ``backward_dw()``.

    Run::

        torchrun --nproc_per_node=2 -m pytest benchmarks/bench_wgrad_delay.py -v -s -k SdmaComm
        torchrun --nproc_per_node=8 -m pytest benchmarks/bench_wgrad_delay.py -v -s -k SdmaComm
    """

    @pytest.fixture(autouse=True)
    def _setup(self):
        os.environ["MORI_ENABLE_SDMA"] = "1"
        _init_dist()
        self.rank = dist.get_rank()
        self.world = dist.get_world_size()
        self.device = torch.device(f"cuda:{self.rank}")
        yield
        dist.barrier()

    def test_wgrad_overlap_with_sdma_allreduce(self):
        """Single-layer and multi-layer SDMA wgrad overlap.

        Runs all SDMA benchmarks in a single test method to keep exactly
        one ``AllreduceSdma`` handle alive.  Mori's SDMA transport
        allocates KFD queues during handle construction; creating
        multiple handles across separate test methods (without
        ``shmem_finalize`` in between) exhausts the queue pool and
        causes AllGather timeouts.
        """
        from atomrl.lumen.modules.sdma_comm import SdmaTpComm

        profile = DEFAULT_PROFILE
        torch.cuda.synchronize()
        dist.barrier()
        comm = SdmaTpComm.get(dist.group.WORLD)

        # ── Part 1: single-layer overlap ─────────────────────────
        _, _, _, ar_buf = _make_profiled_wgrad_single_layer_inputs(self.device, profile)

        sdma_stream = torch.cuda.Stream(device=self.device)

        # Warmup both NCCL and SDMA paths
        for _ in range(3):
            dist.all_reduce(ar_buf.clone())
            comm.allreduce_sum(ar_buf.clone())
            _run_real_api_single_layer_wgrad_queue_and_drain(self.device, profile)
        torch.cuda.synchronize()
        dist.barrier()

        r_wgrad = cuda_timer(
            lambda: _run_real_api_single_layer_wgrad_queue_and_drain(
                self.device, profile
            ),
            label="real module: wgrad (queue+drain, no comm)",
            warmup=_WARMUP,
            iters=_ITERS,
            trim_pct=_TRIM,
        )

        def _nccl_ar():
            buf = ar_buf.clone()
            dist.all_reduce(buf)

        r_nccl_comm = cuda_timer(
            _nccl_ar,
            label="NCCL allreduce alone (reference)",
            warmup=_WARMUP,
            iters=_ITERS,
            trim_pct=_TRIM,
            dist_barrier=True,
        )

        def _sdma_ar_sync():
            buf = ar_buf.clone()
            comm.allreduce_sum_inplace(buf)

        r_sdma_comm = cuda_timer(
            _sdma_ar_sync,
            label="SDMA allreduce alone (sync)",
            warmup=_WARMUP,
            iters=_ITERS,
            trim_pct=_TRIM,
            dist_barrier=True,
        )

        def _sequential():
            tp_group, _ = _get_or_create_rank_local_group()
            module, x_m, g_m = _build_real_api_single_layer(
                self.device, profile, tp_group
            )
            _run_real_api_backward_then_queue(module, x_m, g_m)
            module.backward_dw()
            buf = ar_buf.clone()
            dist.all_reduce(buf)

        r_seq = cuda_timer(
            _sequential,
            label="sequential: real module wgrad then NCCL AR",
            warmup=_WARMUP,
            iters=_ITERS,
            trim_pct=_TRIM,
            dist_barrier=True,
        )

        torch.cuda.synchronize()
        dist.barrier()

        _ovl_put_done = torch.cuda.Event()

        def _overlapped():
            tp_group, _ = _get_or_create_rank_local_group()
            module, x_m, g_m = _build_real_api_single_layer(
                self.device, profile, tp_group
            )
            _run_real_api_backward_then_queue(module, x_m, g_m)
            buf = ar_buf.clone()
            sdma_stream.wait_stream(torch.cuda.current_stream())
            comm.allreduce_sum_async(buf, stream=sdma_stream)
            _ovl_put_done.record(sdma_stream)
            torch.cuda.current_stream().wait_event(_ovl_put_done)
            module.backward_dw()
            comm.wait_allreduce_sum(stream=sdma_stream)
            torch.cuda.current_stream().wait_stream(sdma_stream)

        r_ovl = cuda_timer(
            _overlapped,
            label="overlapped: real module wgrad || SDMA AR",
            warmup=_WARMUP,
            iters=_ITERS,
            trim_pct=_TRIM,
            dist_barrier=True,
        )

        T_parts = r_wgrad.avg_ms + r_sdma_comm.avg_ms
        overlap_ratio = 1 - (r_ovl.avg_ms / max(T_parts, 1e-6))
        speedup = r_seq.avg_ms / max(r_ovl.avg_ms, 1e-6)
        r_ovl.extra["speedup"] = round(speedup, 2)
        r_ovl.extra["overlap_ratio"] = round(overlap_ratio, 3)

        if self.rank == 0:
            print_report_with_table(
                _profiled_wgrad_title(
                    "Real module wgrad-delay + SDMA AllReduce", profile, self.world
                ),
                [r_wgrad, r_nccl_comm, r_sdma_comm, r_seq, r_ovl],
            )
            print_overlap_summary(
                t_compute=r_wgrad.avg_ms,
                t_comm=r_sdma_comm.avg_ms,
                t_seq=r_seq.avg_ms,
                t_ovl=r_ovl.avg_ms,
                compute_label="wgrad (real API)",
                comm_label="SDMA AR (sync alone)",
            )

        sdma_stream.synchronize()
        torch.cuda.synchronize()
        dist.barrier()

        # ── Part 2: multi-layer pipeline ─────────────────────────
        n_layers = 4
        _, _, _, ar_ml = _make_profiled_wgrad_single_layer_inputs(self.device, profile)

        # Re-prime the SDMA handle after the idle gap (NCCL-only Part 1
        # baselines).  The mori ccl reference tests always barrier+sync
        # around every SDMA operation; a long idle gap can leave the
        # SDMA transport in an inconsistent state across peers.
        for _ in range(3):
            dist.all_reduce(ar_ml.clone())
            comm.allreduce_sum(ar_ml.clone())
            _run_real_api_single_layer_wgrad_queue_and_drain(self.device, profile)
        torch.cuda.synchronize()
        dist.barrier()

        def _nccl_sequential():
            _run_real_api_multi_layer_sequential_per_layer(
                self.device,
                profile,
                n_layers,
                after_each_layer=lambda _i, m: dist.all_reduce(m.weight.main_grad),
            )

        r_baseline = cuda_timer(
            _nccl_sequential,
            label=f"{n_layers}L sequential real-module+NCCL AR",
            warmup=_WARMUP,
            iters=_ITERS,
            trim_pct=_TRIM,
            dist_barrier=True,
        )

        torch.cuda.synchronize()
        dist.barrier()

        _ml_put_done = torch.cuda.Event()

        def _deferred():
            modules, xs, g_outs, _ = _build_real_api_layer_stack(
                self.device, profile, n_layers
            )
            pending_ar = False
            for i in range(n_layers):
                if i > 0 and modules[i - 1]._deferred_wgrad.has_pending:
                    modules[i - 1].backward_dw()

                if pending_ar:
                    comm.wait_allreduce_sum(stream=sdma_stream)
                    torch.cuda.current_stream().wait_stream(sdma_stream)

                y, _ = modules[i](xs[i])
                torch.autograd.backward(y, g_outs[i])

                if i > 0:
                    sdma_stream.wait_stream(torch.cuda.current_stream())
                    comm.allreduce_sum_async(
                        modules[i - 1].weight.main_grad,
                        stream=sdma_stream,
                    )
                    _ml_put_done.record(sdma_stream)
                    torch.cuda.current_stream().wait_event(_ml_put_done)
                    pending_ar = True

            if modules[n_layers - 1]._deferred_wgrad.has_pending:
                modules[n_layers - 1].backward_dw()
            if pending_ar:
                comm.wait_allreduce_sum(stream=sdma_stream)
            sdma_stream.wait_stream(torch.cuda.current_stream())
            comm.allreduce_sum_async(
                modules[n_layers - 1].weight.main_grad, stream=sdma_stream
            )
            comm.wait_allreduce_sum(stream=sdma_stream)
            torch.cuda.current_stream().wait_stream(sdma_stream)

        r_deferred = cuda_timer(
            _deferred,
            label=f"{n_layers}L pipelined deferred wgrad+SDMA AR",
            warmup=_WARMUP,
            iters=_ITERS,
            trim_pct=_TRIM,
            dist_barrier=True,
        )

        ml_speedup = r_baseline.avg_ms / max(r_deferred.avg_ms, 1e-6)
        r_deferred.extra["speedup"] = round(ml_speedup, 2)

        saved_ms = r_baseline.avg_ms - r_deferred.avg_ms

        if self.rank == 0:
            print_report_with_table(
                _profiled_wgrad_title(
                    f"{n_layers}-Layer Pipeline: sequential vs pipelined (real module + SDMA)",
                    profile,
                    self.world,
                ),
                [r_baseline, r_deferred],
            )
            print(f"  Speedup:  {ml_speedup:.2f}x")
            print(
                f"  Saved:    {saved_ms:.3f} ms  "
                f"(≈ {n_layers} × {saved_ms / n_layers:.3f} ms per layer)"
            )
            print()

        sdma_stream.synchronize()
        torch.cuda.synchronize()
        dist.barrier()


# ---------------------------------------------------------------------------
# 6. Direct comparison: NCCL vs SDMA wgrad overlap (+ profile experiment selectors)
# ---------------------------------------------------------------------------


@_SDMA
class TestNCCLvsSdmaWgradDelay:
    """Head-to-head comparison of NCCL vs SDMA for wgrad-delay overlap.

    Both backends overlap the deferred wgrad GEMM with an allreduce using
    identical tensor shapes and timing parameters.  The key metric is the
    overlap speedup (T_seq / T_ovl): SDMA should achieve higher overlap
    because its DMA engines have zero SM contention with the wgrad GEMM.

    Run::

        torchrun --nproc_per_node=2 -m pytest benchmarks/bench_wgrad_delay.py -v -s -k NCCLvsSdma
        torchrun --nproc_per_node=8 -m pytest benchmarks/bench_wgrad_delay.py -v -s -k NCCLvsSdma
    """

    @pytest.fixture(autouse=True)
    def _setup(self):
        os.environ["MORI_ENABLE_SDMA"] = "1"
        _init_dist()
        self.rank = dist.get_rank()
        self.world = dist.get_world_size()
        self.device = torch.device(f"cuda:{self.rank}")
        yield
        dist.barrier()

    def _run_single_layer_overlap_comparison(
        self, profile: E2EFusionProfile = DEFAULT_PROFILE
    ):
        """Single-layer wgrad + allreduce: NCCL vs SDMA side by side."""
        from atomrl.lumen.modules.sdma_comm import SdmaTpComm

        _, _, _, ar_buf = _make_profiled_wgrad_single_layer_inputs(self.device, profile)

        torch.cuda.synchronize()
        dist.barrier()
        sdma_comm = SdmaTpComm.get(dist.group.WORLD)
        nccl_stream = torch.cuda.Stream(device=self.device)
        sdma_stream = torch.cuda.Stream(device=self.device)

        for _ in range(3):
            dist.all_reduce(ar_buf.clone())
            _run_real_api_single_layer_wgrad_queue_and_drain(self.device, profile)
        torch.cuda.synchronize()
        dist.barrier()
        for _ in range(3):
            sdma_comm.allreduce_sum(ar_buf.clone())
        torch.cuda.synchronize()
        dist.barrier()

        # Shared: real-module wgrad path (queue + immediate drain), no comm
        r_wgrad = cuda_timer(
            lambda: _run_real_api_single_layer_wgrad_queue_and_drain(
                self.device, profile
            ),
            label="real module: wgrad (queue+drain, no comm)",
            warmup=_WARMUP,
            iters=_ITERS,
            trim_pct=_TRIM,
        )

        # ── NCCL ──────────────────────────────────────────────

        def _nccl_ar():
            buf = ar_buf.clone()
            dist.all_reduce(buf)

        r_nccl_comm = cuda_timer(
            _nccl_ar,
            label="NCCL allreduce alone",
            warmup=_WARMUP,
            iters=_ITERS,
            trim_pct=_TRIM,
            dist_barrier=True,
        )

        def _nccl_seq():
            tp_group, _ = _get_or_create_rank_local_group()
            module, x_m, g_m = _build_real_api_single_layer(
                self.device, profile, tp_group
            )
            _run_real_api_backward_then_queue(module, x_m, g_m)
            module.backward_dw()
            buf = ar_buf.clone()
            dist.all_reduce(buf)

        r_nccl_seq = cuda_timer(
            _nccl_seq,
            label="NCCL sequential (real module wgrad + AR)",
            warmup=_WARMUP,
            iters=_ITERS,
            trim_pct=_TRIM,
            dist_barrier=True,
        )

        def _nccl_ovl():
            tp_group, _ = _get_or_create_rank_local_group()
            module, x_m, g_m = _build_real_api_single_layer(
                self.device, profile, tp_group
            )
            _run_real_api_backward_then_queue(module, x_m, g_m)
            with torch.cuda.stream(nccl_stream):
                buf = ar_buf.clone()
                dist.all_reduce(buf)
            module.backward_dw()
            torch.cuda.current_stream().wait_stream(nccl_stream)

        r_nccl_ovl = cuda_timer(
            _nccl_ovl,
            label="NCCL overlapped (real module)",
            warmup=_WARMUP,
            iters=_ITERS,
            trim_pct=_TRIM,
            dist_barrier=True,
        )

        nccl_speedup = r_nccl_seq.avg_ms / max(r_nccl_ovl.avg_ms, 1e-6)
        nccl_overlap = 1 - (
            r_nccl_ovl.avg_ms / max(r_wgrad.avg_ms + r_nccl_comm.avg_ms, 1e-6)
        )
        r_nccl_ovl.extra["speedup"] = round(nccl_speedup, 2)
        r_nccl_ovl.extra["overlap"] = round(nccl_overlap, 3)

        # ── SDMA ──────────────────────────────────────────────
        # Drain all NCCL work before switching to SDMA backend
        torch.cuda.synchronize()
        dist.barrier()

        def _sdma_ar():
            buf = ar_buf.clone()
            sdma_comm.allreduce_sum_inplace(buf)

        r_sdma_comm = cuda_timer(
            _sdma_ar,
            label="SDMA allreduce alone",
            warmup=_WARMUP,
            iters=_ITERS,
            trim_pct=_TRIM,
            dist_barrier=True,
        )

        def _sdma_seq():
            tp_group, _ = _get_or_create_rank_local_group()
            module, x_m, g_m = _build_real_api_single_layer(
                self.device, profile, tp_group
            )
            _run_real_api_backward_then_queue(module, x_m, g_m)
            module.backward_dw()
            buf = ar_buf.clone()
            sdma_comm.allreduce_sum_inplace(buf)

        r_sdma_seq = cuda_timer(
            _sdma_seq,
            label="SDMA sequential (real module wgrad + AR)",
            warmup=_WARMUP,
            iters=_ITERS,
            trim_pct=_TRIM,
            dist_barrier=True,
        )

        # ── SDMA async overlap ─────────────────────────────────
        # The ReduceScatter kernel uses max_blocks CUs (304) with blocks
        # 1-N spinning on barrier->flag. Overlapping a large GEMM on
        # the default stream causes CU contention that can deadlock the
        # RS/AGPut kernels.  Record a CUDA event after start_async so the
        # default stream waits for RS+AGPut to finish before launching the
        # wgrad GEMM.  The SDMA DMA hardware transfers still overlap with
        # the GEMM; only the CU-bound kernel dispatch is serialized.
        torch.cuda.synchronize()
        dist.barrier()

        sdma_comm.reset_allreduce_flags()
        _sdma_put_done = torch.cuda.Event()

        def _sdma_ovl():
            tp_group, _ = _get_or_create_rank_local_group()
            module, x_m, g_m = _build_real_api_single_layer(
                self.device, profile, tp_group
            )
            _run_real_api_backward_then_queue(module, x_m, g_m)
            buf = ar_buf.clone()
            sdma_stream.wait_stream(torch.cuda.current_stream())
            sdma_comm.allreduce_sum_async(buf, stream=sdma_stream)
            _sdma_put_done.record(sdma_stream)
            torch.cuda.current_stream().wait_event(_sdma_put_done)
            module.backward_dw()
            sdma_comm.wait_allreduce_sum(stream=sdma_stream)
            torch.cuda.current_stream().wait_stream(sdma_stream)

        r_sdma_ovl = cuda_timer(
            _sdma_ovl,
            label="SDMA overlapped (real module)",
            warmup=_WARMUP,
            iters=_ITERS,
            trim_pct=_TRIM,
            dist_barrier=True,
        )

        sdma_speedup = r_sdma_seq.avg_ms / max(r_sdma_ovl.avg_ms, 1e-6)
        sdma_overlap = 1 - (
            r_sdma_ovl.avg_ms / max(r_wgrad.avg_ms + r_sdma_comm.avg_ms, 1e-6)
        )
        r_sdma_ovl.extra["speedup"] = round(sdma_speedup, 2)
        r_sdma_ovl.extra["overlap"] = round(sdma_overlap, 3)

        # ── Head-to-head summary ──────────────────────────────
        sdma_vs_nccl = r_nccl_ovl.avg_ms / max(r_sdma_ovl.avg_ms, 1e-6)

        if self.rank == 0:
            print_report_with_table(
                _profiled_wgrad_title(
                    "NCCL vs SDMA real-module wgrad-delay overlap", profile, self.world
                ),
                [
                    r_wgrad,
                    r_nccl_comm,
                    r_nccl_seq,
                    r_nccl_ovl,
                    r_sdma_comm,
                    r_sdma_seq,
                    r_sdma_ovl,
                ],
            )
            print(f"  NCCL overlap speedup:  {nccl_speedup:.2f}x")
            print(f"  SDMA overlap speedup:  {sdma_speedup:.2f}x")
            print(f"  SDMA vs NCCL:          {sdma_vs_nccl:.2f}x")
            print()

        torch.cuda.synchronize()
        dist.barrier()

    def test_single_layer_overlap_comparison(self):
        self._run_single_layer_overlap_comparison()

    def test_multi_layer_pipeline_comparison(self):
        """4-layer pipeline: NCCL vs SDMA deferred-wgrad overlap."""
        from atomrl.lumen.modules.sdma_comm import SdmaTpComm

        profile = DEFAULT_PROFILE
        torch.cuda.synchronize()
        dist.barrier()
        sdma_comm = SdmaTpComm.get(dist.group.WORLD)
        n_layers = 4
        _, _, _, ar_ref = _make_profiled_wgrad_single_layer_inputs(self.device, profile)

        nccl_stream = torch.cuda.Stream(device=self.device)
        sdma_stream = torch.cuda.Stream(device=self.device)

        for _ in range(3):
            dist.all_reduce(ar_ref.clone())
            _run_real_api_single_layer_wgrad_queue_and_drain(self.device, profile)
        torch.cuda.synchronize()
        dist.barrier()
        for _ in range(3):
            sdma_comm.allreduce_sum(ar_ref.clone())
        torch.cuda.synchronize()
        dist.barrier()

        # Sequential baseline: real module per layer + NCCL AR (no pipeline overlap)
        def _eager_nccl():
            _run_real_api_multi_layer_sequential_per_layer(
                self.device,
                profile,
                n_layers,
                after_each_layer=lambda _i, m: dist.all_reduce(m.weight.main_grad),
            )

        r_eager_nccl = cuda_timer(
            _eager_nccl,
            label=f"{n_layers}L sequential real-module+NCCL",
            warmup=_WARMUP,
            iters=_ITERS,
            trim_pct=_TRIM,
            dist_barrier=True,
        )

        # Deferred NCCL pipeline
        def _deferred_nccl():
            modules, xs, g_outs, _ = _build_real_api_layer_stack(
                self.device, profile, n_layers
            )
            pending_ar = None
            for i in range(n_layers):
                if i > 0 and modules[i - 1]._deferred_wgrad.has_pending:
                    modules[i - 1].backward_dw()
                if pending_ar is not None:
                    torch.cuda.current_stream().wait_stream(nccl_stream)
                y, _ = modules[i](xs[i])
                torch.autograd.backward(y, g_outs[i])
                if i > 0:
                    nccl_stream.wait_stream(torch.cuda.current_stream())
                    with torch.cuda.stream(nccl_stream):
                        dist.all_reduce(modules[i - 1].weight.main_grad)
                    pending_ar = i - 1
            if modules[n_layers - 1]._deferred_wgrad.has_pending:
                modules[n_layers - 1].backward_dw()
            nccl_stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(nccl_stream):
                dist.all_reduce(modules[n_layers - 1].weight.main_grad)
            torch.cuda.current_stream().wait_stream(nccl_stream)

        r_deferred_nccl = cuda_timer(
            _deferred_nccl,
            label=f"{n_layers}L pipelined deferred+NCCL",
            warmup=_WARMUP,
            iters=_ITERS,
            trim_pct=_TRIM,
            dist_barrier=True,
        )

        nccl_speedup = r_eager_nccl.avg_ms / max(r_deferred_nccl.avg_ms, 1e-6)
        r_deferred_nccl.extra["speedup"] = round(nccl_speedup, 2)

        # Drain all GPU work before switching to SDMA
        torch.cuda.synchronize()
        dist.barrier()

        # Sequential baseline: real module per layer + synchronous SDMA AR
        def _eager_sdma():
            _run_real_api_multi_layer_sequential_per_layer(
                self.device,
                profile,
                n_layers,
                after_each_layer=lambda _i, m: sdma_comm.allreduce_sum_inplace(
                    m.weight.main_grad
                ),
            )

        r_eager_sdma = cuda_timer(
            _eager_sdma,
            label=f"{n_layers}L sequential real-module+SDMA",
            warmup=_WARMUP,
            iters=_ITERS,
            trim_pct=_TRIM,
            dist_barrier=True,
        )

        # Drain synchronous SDMA ops before starting async pipeline
        torch.cuda.synchronize()
        dist.barrier()
        sdma_comm.reset_allreduce_flags()

        _ds_put_done = torch.cuda.Event()

        def _deferred_sdma():
            modules, xs, g_outs, _ = _build_real_api_layer_stack(
                self.device, profile, n_layers
            )
            pending_ar = False
            for i in range(n_layers):
                if i > 0 and modules[i - 1]._deferred_wgrad.has_pending:
                    modules[i - 1].backward_dw()
                if pending_ar:
                    sdma_comm.wait_allreduce_sum(stream=sdma_stream)
                    torch.cuda.current_stream().wait_stream(sdma_stream)
                y, _ = modules[i](xs[i])
                torch.autograd.backward(y, g_outs[i])
                if i > 0:
                    sdma_stream.wait_stream(torch.cuda.current_stream())
                    sdma_comm.allreduce_sum_async(
                        modules[i - 1].weight.main_grad,
                        stream=sdma_stream,
                    )
                    _ds_put_done.record(sdma_stream)
                    torch.cuda.current_stream().wait_event(_ds_put_done)
                    pending_ar = True
            if modules[n_layers - 1]._deferred_wgrad.has_pending:
                modules[n_layers - 1].backward_dw()
            if pending_ar:
                sdma_comm.wait_allreduce_sum(stream=sdma_stream)
            sdma_stream.wait_stream(torch.cuda.current_stream())
            sdma_comm.allreduce_sum_async(
                modules[n_layers - 1].weight.main_grad, stream=sdma_stream
            )
            sdma_comm.wait_allreduce_sum(stream=sdma_stream)
            torch.cuda.current_stream().wait_stream(sdma_stream)

        r_deferred_sdma = cuda_timer(
            _deferred_sdma,
            label=f"{n_layers}L pipelined deferred+SDMA",
            warmup=_WARMUP,
            iters=_ITERS,
            trim_pct=_TRIM,
            dist_barrier=True,
        )

        sdma_speedup = r_eager_sdma.avg_ms / max(r_deferred_sdma.avg_ms, 1e-6)
        r_deferred_sdma.extra["speedup"] = round(sdma_speedup, 2)

        # Head-to-head
        sdma_vs_nccl = r_deferred_nccl.avg_ms / max(r_deferred_sdma.avg_ms, 1e-6)

        if self.rank == 0:
            print_report_with_table(
                _profiled_wgrad_title(
                    f"NCCL vs SDMA {n_layers}-Layer Wgrad Pipeline",
                    profile,
                    self.world,
                ),
                [r_eager_nccl, r_deferred_nccl, r_eager_sdma, r_deferred_sdma],
            )
            nccl_saved = r_eager_nccl.avg_ms - r_deferred_nccl.avg_ms
            sdma_saved = r_eager_sdma.avg_ms - r_deferred_sdma.avg_ms
            print(f"  NCCL:  speedup={nccl_speedup:.2f}x  saved={nccl_saved:.3f} ms")
            print(f"  SDMA:  speedup={sdma_speedup:.2f}x  saved={sdma_saved:.3f} ms")
            print(f"  SDMA vs NCCL (deferred):  {sdma_vs_nccl:.2f}x")
            print()

        torch.cuda.synchronize()
        dist.barrier()

    @pytest.mark.parametrize(
        "gemm_n",
        [1024, 4096, 7168, 14336, 28672, 57344],
        ids=lambda n: f"N={n}",
    )
    def test_wgrad_overlap_scaling(self, gemm_n):
        """4-layer pipeline sweep: vary gemm_n to measure effective overlap.

        Uses a realistic 4-layer deferred-wgrad pipeline where layer i's
        allreduce overlaps with layer i+1's real-module wgrad (queued in
        backward, drained via ``backward_dw()``).

        Baselines use the same ``LumenColumnParallelLinear`` stack: a
        compute-only timer (per-layer fwd + bwd queue + drain, no AR) and a
        sequential timer (+ NCCL AR each layer), matching
        ``test_multi_layer_pipeline_comparison``. ``gemm_n`` is the global
        FFN width; each rank uses the TP-sharded local width for module
        compute and communication size.
        """
        from atomrl.lumen.modules.sdma_comm import SdmaTpComm

        n_layers = 4
        profile = DEFAULT_PROFILE
        torch.cuda.synchronize()
        dist.barrier()
        sdma_comm = SdmaTpComm.get(dist.group.WORLD)
        local_n = _tp_sharded_out_features(gemm_n, self.world)

        # Warm up the same per-rank shard geometry used by the timed paths.
        ar_buf = torch.randn(local_n, K, device=self.device, dtype=torch.bfloat16)

        nccl_stream = torch.cuda.Stream(device=self.device)
        sdma_stream = torch.cuda.Stream(device=self.device)

        for _ in range(3):
            dist.all_reduce(ar_buf.clone())
            _run_real_api_single_layer_wgrad_queue_and_drain(
                self.device, profile, tokens=M, hidden=K, out_features=local_n
            )
        torch.cuda.synchronize()
        dist.barrier()
        for _ in range(3):
            sdma_comm.allreduce_sum(ar_buf.clone())
        torch.cuda.synchronize()
        dist.barrier()

        # ── Compute-only baseline: real module, no AR (fwd + bwd queue + drain per layer) ──
        def _wgrad_only():
            _run_real_api_multi_layer_sequential_per_layer(
                self.device,
                profile,
                n_layers,
                tokens=M,
                hidden=K,
                out_features=local_n,
            )

        r_wgrad = cuda_timer(
            _wgrad_only,
            label=f"4L real-module compute N={gemm_n}",
            warmup=_WARMUP,
            iters=_ITERS,
            trim_pct=_TRIM,
            dist_barrier=True,
        )

        # ── Sequential baseline: real module + NCCL AR per layer ──
        def _eager():
            _run_real_api_multi_layer_sequential_per_layer(
                self.device,
                profile,
                n_layers,
                tokens=M,
                hidden=K,
                out_features=local_n,
                after_each_layer=lambda _i, m: dist.all_reduce(m.weight.main_grad),
            )

        r_eager = cuda_timer(
            _eager,
            label=f"4L sequential real-module+NCCL N={gemm_n}",
            warmup=_WARMUP,
            iters=_ITERS,
            trim_pct=_TRIM,
            dist_barrier=True,
        )

        torch.cuda.synchronize()
        dist.barrier()

        # ── NCCL deferred pipeline ──
        def _deferred_nccl():
            modules, xs_m, g_outs, _ = _build_real_api_layer_stack(
                self.device,
                profile,
                n_layers,
                tokens=M,
                hidden=K,
                out_features=local_n,
            )
            pending_ar = None
            for i in range(n_layers):
                if i > 0 and modules[i - 1]._deferred_wgrad.has_pending:
                    modules[i - 1].backward_dw()
                if pending_ar is not None:
                    torch.cuda.current_stream().wait_stream(nccl_stream)
                y, _ = modules[i](xs_m[i])
                torch.autograd.backward(y, g_outs[i])
                if i > 0:
                    nccl_stream.wait_stream(torch.cuda.current_stream())
                    with torch.cuda.stream(nccl_stream):
                        dist.all_reduce(modules[i - 1].weight.main_grad)
                    pending_ar = i - 1
            if modules[n_layers - 1]._deferred_wgrad.has_pending:
                modules[n_layers - 1].backward_dw()
            nccl_stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(nccl_stream):
                dist.all_reduce(modules[n_layers - 1].weight.main_grad)
            torch.cuda.current_stream().wait_stream(nccl_stream)

        r_nccl = cuda_timer(
            _deferred_nccl,
            label=f"4L NCCL N={gemm_n}",
            warmup=_WARMUP,
            iters=_ITERS,
            trim_pct=_TRIM,
            dist_barrier=True,
        )

        torch.cuda.synchronize()
        dist.barrier()
        sdma_comm.reset_allreduce_flags()

        _ds2_put_done = torch.cuda.Event()

        def _deferred_sdma():
            modules, xs_m, g_outs, _ = _build_real_api_layer_stack(
                self.device,
                profile,
                n_layers,
                tokens=M,
                hidden=K,
                out_features=local_n,
            )
            pending_ar = False
            for i in range(n_layers):
                if i > 0 and modules[i - 1]._deferred_wgrad.has_pending:
                    modules[i - 1].backward_dw()
                if pending_ar:
                    sdma_comm.wait_allreduce_sum(stream=sdma_stream)
                    torch.cuda.current_stream().wait_stream(sdma_stream)
                y, _ = modules[i](xs_m[i])
                torch.autograd.backward(y, g_outs[i])
                if i > 0:
                    sdma_stream.wait_stream(torch.cuda.current_stream())
                    sdma_comm.allreduce_sum_async(
                        modules[i - 1].weight.main_grad,
                        stream=sdma_stream,
                    )
                    _ds2_put_done.record(sdma_stream)
                    torch.cuda.current_stream().wait_event(_ds2_put_done)
                    pending_ar = True
            if modules[n_layers - 1]._deferred_wgrad.has_pending:
                modules[n_layers - 1].backward_dw()
            if pending_ar:
                sdma_comm.wait_allreduce_sum(stream=sdma_stream)
            sdma_stream.wait_stream(torch.cuda.current_stream())
            sdma_comm.allreduce_sum_async(
                modules[n_layers - 1].weight.main_grad, stream=sdma_stream
            )
            sdma_comm.wait_allreduce_sum(stream=sdma_stream)
            torch.cuda.current_stream().wait_stream(sdma_stream)

        r_sdma = cuda_timer(
            _deferred_sdma,
            label=f"4L SDMA N={gemm_n}",
            warmup=_WARMUP,
            iters=_ITERS,
            trim_pct=_TRIM,
            dist_barrier=True,
        )

        nccl_speedup = r_eager.avg_ms / max(r_nccl.avg_ms, 1e-6)
        sdma_speedup = r_eager.avg_ms / max(r_sdma.avg_ms, 1e-6)
        sdma_vs_nccl = r_nccl.avg_ms / max(r_sdma.avg_ms, 1e-6)
        ar_total = r_eager.avg_ms - r_wgrad.avg_ms
        nccl_hidden = r_eager.avg_ms - r_nccl.avg_ms
        sdma_hidden = r_eager.avg_ms - r_sdma.avg_ms

        r_wgrad.extra["flops"] = f"{n_layers * 2 * M * K * gemm_n / 1e9:.1f}G"
        r_nccl.extra["speedup"] = round(nccl_speedup, 2)
        r_nccl.extra["ovl_eff"] = f"{nccl_hidden / max(ar_total, 1e-6) * 100:.0f}%"
        r_sdma.extra["speedup"] = round(sdma_speedup, 2)
        r_sdma.extra["ovl_eff"] = f"{sdma_hidden / max(ar_total, 1e-6) * 100:.0f}%"
        r_sdma.extra["vs_nccl"] = round(sdma_vs_nccl, 2)

        if self.rank == 0:
            print_report_with_table(
                f"4L Wgrad Pipeline N={gemm_n} (world={self.world})",
                [r_wgrad, r_eager, r_nccl, r_sdma],
            )

        torch.cuda.synchronize()
        dist.barrier()

    def test_wgrad_overlap_scaling_summary(self):
        """All GEMM sizes in a single 4-layer pipeline summary table.

        The reported ``gemm_n`` values are global FFN widths; each rank uses
        the TP-sharded local width for module compute and communication size.
        """
        from atomrl.lumen.modules.sdma_comm import SdmaTpComm

        n_layers = 4
        profile = DEFAULT_PROFILE
        torch.cuda.synchronize()
        dist.barrier()
        sdma_comm = SdmaTpComm.get(dist.group.WORLD)
        gemm_sizes = [1024, 4096, 7168, 14336, 28672, 57344]
        all_results: list[BenchResult] = []

        nccl_stream = torch.cuda.Stream(device=self.device)
        sdma_stream = torch.cuda.Stream(device=self.device)

        for gemm_n in gemm_sizes:
            local_n = _tp_sharded_out_features(gemm_n, self.world)
            ar_buf = torch.randn(local_n, K, device=self.device, dtype=torch.bfloat16)

            for _ in range(3):
                dist.all_reduce(ar_buf.clone())
                _run_real_api_single_layer_wgrad_queue_and_drain(
                    self.device, profile, tokens=M, hidden=K, out_features=local_n
                )
            torch.cuda.synchronize()
            dist.barrier()
            for _ in range(3):
                sdma_comm.allreduce_sum(ar_buf.clone())
            torch.cuda.synchronize()
            dist.barrier()

            # ── Real-module compute only (for overlap efficiency vs +NCCL) ──
            def _wgrad_only(local_n=local_n):
                _run_real_api_multi_layer_sequential_per_layer(
                    self.device,
                    profile,
                    n_layers,
                    tokens=M,
                    hidden=K,
                    out_features=local_n,
                )

            r_wgrad = cuda_timer(
                _wgrad_only,
                label=f"real-module compute N={gemm_n}",
                warmup=_WARMUP,
                iters=_ITERS,
                trim_pct=_TRIM,
                dist_barrier=True,
            )

            # ── Sequential real module + NCCL AR per layer ──
            def _eager(local_n=local_n):
                _run_real_api_multi_layer_sequential_per_layer(
                    self.device,
                    profile,
                    n_layers,
                    tokens=M,
                    hidden=K,
                    out_features=local_n,
                    after_each_layer=lambda _i, m: dist.all_reduce(m.weight.main_grad),
                )

            r_eager = cuda_timer(
                _eager,
                label=f"sequential real-module+NCCL N={gemm_n}",
                warmup=_WARMUP,
                iters=_ITERS,
                trim_pct=_TRIM,
                dist_barrier=True,
            )

            torch.cuda.synchronize()
            dist.barrier()

            # ── NCCL deferred pipeline ──
            def _deferred_nccl(local_n=local_n):
                modules, xs_m, g_outs, _ = _build_real_api_layer_stack(
                    self.device,
                    profile,
                    n_layers,
                    tokens=M,
                    hidden=K,
                    out_features=local_n,
                )
                pending_ar = None
                for i in range(n_layers):
                    if i > 0 and modules[i - 1]._deferred_wgrad.has_pending:
                        modules[i - 1].backward_dw()
                    if pending_ar is not None:
                        torch.cuda.current_stream().wait_stream(nccl_stream)
                    y, _ = modules[i](xs_m[i])
                    torch.autograd.backward(y, g_outs[i])
                    if i > 0:
                        nccl_stream.wait_stream(torch.cuda.current_stream())
                        with torch.cuda.stream(nccl_stream):
                            dist.all_reduce(modules[i - 1].weight.main_grad)
                        pending_ar = i - 1
                if modules[n_layers - 1]._deferred_wgrad.has_pending:
                    modules[n_layers - 1].backward_dw()
                nccl_stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(nccl_stream):
                    dist.all_reduce(modules[n_layers - 1].weight.main_grad)
                torch.cuda.current_stream().wait_stream(nccl_stream)

            r_nccl = cuda_timer(
                _deferred_nccl,
                label=f"NCCL N={gemm_n}",
                warmup=_WARMUP,
                iters=_ITERS,
                trim_pct=_TRIM,
                dist_barrier=True,
            )

            torch.cuda.synchronize()
            dist.barrier()
            sdma_comm.reset_allreduce_flags()

            _ds3_put_done = torch.cuda.Event()

            def _deferred_sdma(_ds3_put_done=_ds3_put_done, local_n=local_n):
                modules, xs_m, g_outs, _ = _build_real_api_layer_stack(
                    self.device,
                    profile,
                    n_layers,
                    tokens=M,
                    hidden=K,
                    out_features=local_n,
                )
                pending_ar = False
                for i in range(n_layers):
                    if i > 0 and modules[i - 1]._deferred_wgrad.has_pending:
                        modules[i - 1].backward_dw()
                    if pending_ar:
                        sdma_comm.wait_allreduce_sum(stream=sdma_stream)
                        torch.cuda.current_stream().wait_stream(sdma_stream)
                    y, _ = modules[i](xs_m[i])
                    torch.autograd.backward(y, g_outs[i])
                    if i > 0:
                        sdma_stream.wait_stream(torch.cuda.current_stream())
                        sdma_comm.allreduce_sum_async(
                            modules[i - 1].weight.main_grad,
                            stream=sdma_stream,
                        )
                        _ds3_put_done.record(sdma_stream)
                        torch.cuda.current_stream().wait_event(_ds3_put_done)
                        pending_ar = True
                if modules[n_layers - 1]._deferred_wgrad.has_pending:
                    modules[n_layers - 1].backward_dw()
                if pending_ar:
                    sdma_comm.wait_allreduce_sum(stream=sdma_stream)
                sdma_stream.wait_stream(torch.cuda.current_stream())
                sdma_comm.allreduce_sum_async(
                    modules[n_layers - 1].weight.main_grad, stream=sdma_stream
                )
                sdma_comm.wait_allreduce_sum(stream=sdma_stream)
                torch.cuda.current_stream().wait_stream(sdma_stream)

            r_sdma = cuda_timer(
                _deferred_sdma,
                label=f"SDMA N={gemm_n}",
                warmup=_WARMUP,
                iters=_ITERS,
                trim_pct=_TRIM,
                dist_barrier=True,
            )

            nccl_speedup = r_eager.avg_ms / max(r_nccl.avg_ms, 1e-6)
            sdma_speedup = r_eager.avg_ms / max(r_sdma.avg_ms, 1e-6)
            sdma_vs_nccl = r_nccl.avg_ms / max(r_sdma.avg_ms, 1e-6)
            ar_total = r_eager.avg_ms - r_wgrad.avg_ms
            nccl_hidden = r_eager.avg_ms - r_nccl.avg_ms
            sdma_hidden = r_eager.avg_ms - r_sdma.avg_ms

            r_nccl.extra["speedup"] = f"{nccl_speedup:.2f}x"
            r_nccl.extra["ovl_eff"] = f"{nccl_hidden / max(ar_total, 1e-6) * 100:.0f}%"
            r_nccl.extra["vs_sdma"] = f"{1 / max(sdma_vs_nccl, 1e-6):.2f}x"
            r_sdma.extra["speedup"] = f"{sdma_speedup:.2f}x"
            r_sdma.extra["ovl_eff"] = f"{sdma_hidden / max(ar_total, 1e-6) * 100:.0f}%"
            r_sdma.extra["vs_nccl"] = f"{sdma_vs_nccl:.2f}x"

            all_results.extend([r_nccl, r_sdma])

            torch.cuda.synchronize()
            dist.barrier()

        if self.rank == 0:
            print_report_with_table(
                f"4L Wgrad Pipeline Scaling Summary (world={self.world})",
                all_results,
            )

        torch.cuda.synchronize()
        dist.barrier()


@_SDMA
class TestWgradDelayProfileExperiments(TestNCCLvsSdmaWgradDelay):
    test_single_layer_overlap_comparison = None
    test_multi_layer_pipeline_comparison = None
    test_wgrad_overlap_scaling = None
    test_wgrad_overlap_scaling_summary = None

    @pytest.fixture(autouse=True)
    def _setup(self):
        os.environ["MORI_ENABLE_SDMA"] = "1"
        _init_dist()
        self.rank = dist.get_rank()
        self.world = dist.get_world_size()
        self.device = torch.device(f"cuda:{self.rank}")

        from atomrl.lumen.modules.sdma_comm import SdmaTpComm

        max_profile = get_e2e_fusion_profile("pipeline_gain")
        _, _, _, ar_buf = _make_profiled_wgrad_single_layer_inputs(
            self.device, max_profile
        )
        comm = SdmaTpComm.get(dist.group.WORLD)
        comm.allreduce_sum(ar_buf)
        torch.cuda.synchronize()
        dist.barrier()

        yield
        torch.cuda.synchronize()
        dist.barrier()

    def test_backend_gap_experiment_single_layer_overlap_comparison(self):
        self._run_single_layer_overlap_comparison(
            profile=get_e2e_fusion_profile("backend_gap")
        )

    def test_pipeline_gain_experiment_single_layer_overlap_comparison(self):
        self._run_single_layer_overlap_comparison(
            profile=get_e2e_fusion_profile("pipeline_gain")
        )


# ---------------------------------------------------------------------------
# 7. gradient_accumulation_fusion via quantized_linear backward
# ---------------------------------------------------------------------------


@CUDA
@AITER
class TestGradAccumFusionQuantizedLinear:
    """Test the gradient_accumulation_fusion flag in Lumen's quantized_linear.

    When gradient_accumulation_fusion=True, the backward pass of
    quantized_linear directly does weight.main_grad.add_(grad_weight)
    instead of returning grad_weight. This avoids a separate accumulation
    kernel in multi-microbatch training.
    """

    # Expected: With gradient_accumulation_fusion=True, the backward should
    # accumulate directly into main_grad (grad_weight returned as None).
    # With =False, backward returns grad_weight as a separate tensor.
    # Latency should be similar, but the fused path saves one kernel launch
    # (the separate .add_ call) and one tensor allocation per microbatch.
    def test_fused_vs_separate_accumulation(self):
        from atomrl.lumen.ops.quantize.linear import quantized_linear
        from atomrl.lumen.quantize.config import _get_float8_e4m3

        fp8_dtype = _get_float8_e4m3()
        hidden = K
        ffn = 1024

        x = torch.randn(
            M, hidden, device="cuda", dtype=torch.bfloat16, requires_grad=True
        )
        w = nn.Parameter(
            torch.randn(ffn, hidden, device="cuda", dtype=torch.bfloat16) * 0.02
        )
        w.main_grad = torch.zeros_like(w.data)

        def _without_fusion():
            x_in = x.detach().requires_grad_(True)
            out = quantized_linear(
                x_in,
                w,
                scaling_type="none",
                fp8_dtype=fp8_dtype,
                gradient_accumulation_fusion=False,
            )
            out.sum().backward()

        def _with_fusion():
            w.main_grad.zero_()
            x_in = x.detach().requires_grad_(True)
            out = quantized_linear(
                x_in,
                w,
                scaling_type="none",
                fp8_dtype=fp8_dtype,
                gradient_accumulation_fusion=True,
            )
            out.sum().backward()

        r_no_fuse = cuda_timer(_without_fusion, label="grad_accum_fusion=False")
        r_fuse = cuda_timer(_with_fusion, label="grad_accum_fusion=True")

        print_report(
            "quantized_linear: gradient_accumulation_fusion",
            [r_no_fuse, r_fuse],
        )

    # Expected: After backward with gradient_accumulation_fusion=True,
    # w.grad should be None (not returned) and w.main_grad should contain
    # the gradient. This verifies the fused path works correctly.
    def test_fusion_writes_to_main_grad(self):
        from atomrl.lumen.ops.quantize.linear import quantized_linear
        from atomrl.lumen.quantize.config import _get_float8_e4m3

        fp8_dtype = _get_float8_e4m3()
        hidden = 256
        ffn = 512

        x = torch.randn(
            32, hidden, device="cuda", dtype=torch.bfloat16, requires_grad=True
        )
        w = nn.Parameter(
            torch.randn(ffn, hidden, device="cuda", dtype=torch.bfloat16) * 0.02
        )
        w.main_grad = torch.zeros(ffn, hidden, device="cuda", dtype=torch.bfloat16)

        out = quantized_linear(
            x,
            w,
            scaling_type="none",
            fp8_dtype=fp8_dtype,
            gradient_accumulation_fusion=True,
        )
        out.sum().backward()

        assert w.main_grad.abs().sum() > 0, "main_grad should be non-zero"
        print(
            f"\n  main_grad norm after fused backward: {w.main_grad.norm().item():.4f}"
        )

    # Expected: With gradient_accumulation_fusion=True under FP8 scaling modes,
    # the backward should still correctly accumulate into main_grad. FP8
    # quantization in backward adds quant/dequant steps but the final
    # main_grad.add_(grad_weight) path should work identically.
    def test_fusion_with_fp8_scaling(self):
        from atomrl.lumen.ops.quantize.linear import quantized_linear
        from atomrl.lumen.quantize.config import _get_float8_e4m3

        fp8_dtype = _get_float8_e4m3()
        hidden = 256
        ffn = 512

        results: list[BenchResult] = []
        for scaling in ["none", "delayed", "dynamic"]:
            x = torch.randn(
                32, hidden, device="cuda", dtype=torch.bfloat16, requires_grad=True
            )
            w = nn.Parameter(
                torch.randn(ffn, hidden, device="cuda", dtype=torch.bfloat16) * 0.02
            )
            w.main_grad = torch.zeros(ffn, hidden, device="cuda", dtype=torch.bfloat16)

            def _run(s=scaling, w=w, x=x):
                w.main_grad.zero_()
                x_in = x.detach().requires_grad_(True)
                out = quantized_linear(
                    x_in,
                    w,
                    scaling_type=s,
                    fp8_dtype=fp8_dtype,
                    gradient_accumulation_fusion=True,
                )
                out.sum().backward()

            r = cuda_timer(_run, label=f"fused backward scaling={scaling}")
            results.append(r)

        print_report("gradient_accumulation_fusion across FP8 modes", results)


# ---------------------------------------------------------------------------
# Standalone runner
# ---------------------------------------------------------------------------


def main():
    require_cuda()

    from atomrl.lumen.modules.parallel_linear import _DeferredWgrad

    results: list[BenchResult] = []

    x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
    w = nn.Parameter(torch.randn(N, K, device="cuda", dtype=torch.bfloat16) * 0.02)
    w.main_grad = torch.zeros(N, K, device="cuda", dtype=torch.bfloat16)
    grad_out = torch.randn(M, N, device="cuda", dtype=torch.bfloat16)

    # 1. Eager vs deferred
    r_eager = cuda_timer(lambda: grad_out.T @ x, label="eager wgrad")
    results.append(r_eager)

    dwg = _DeferredWgrad()

    def _deferred():
        w.main_grad.zero_()
        dwg.defer(w, lambda: grad_out.T @ x)
        dwg.execute()

    r_def = cuda_timer(_deferred, label="Lumen _DeferredWgrad")
    results.append(r_def)

    # 2. Stream overlap: wgrad || simulated comm (shows true overlap)
    comm_stream = torch.cuda.Stream()
    r_dw = cuda_timer(
        lambda: grad_out.T @ x,
        label="dW alone",
        warmup=_WARMUP,
        iters=_ITERS,
        trim_pct=_TRIM,
    )
    dw_ns = int(r_dw.avg_ms * 1e6)
    sleep_cycles = max(dw_ns * 2, 100_000)

    r_comm = cuda_timer(
        lambda: torch.cuda._sleep(sleep_cycles),
        label="simulated comm",
        warmup=_WARMUP,
        iters=_ITERS,
        trim_pct=_TRIM,
    )

    def _seq_comm():
        w.main_grad.zero_()
        dwg.defer(w, lambda: grad_out.T @ x)
        dwg.execute()
        torch.cuda.synchronize()
        torch.cuda._sleep(sleep_cycles)
        torch.cuda.synchronize()

    r_seq = cuda_timer(
        _seq_comm,
        label="sequential: wgrad then comm",
        warmup=_WARMUP,
        iters=_ITERS,
        trim_pct=_TRIM,
    )

    def _ovl_comm():
        w.main_grad.zero_()
        dwg.defer(w, lambda: grad_out.T @ x)
        with torch.cuda.stream(comm_stream):
            torch.cuda._sleep(sleep_cycles)
        dwg.execute()
        torch.cuda.current_stream().wait_stream(comm_stream)

    r_ovl = cuda_timer(
        _ovl_comm,
        label="_DeferredWgrad || comm",
        warmup=_WARMUP,
        iters=_ITERS,
        trim_pct=_TRIM,
    )

    T_parts = r_dw.avg_ms + r_comm.avg_ms
    overlap_ratio = 1 - (r_ovl.avg_ms / max(T_parts, 1e-6))
    r_ovl.extra["speedup"] = round(r_seq.avg_ms / max(r_ovl.avg_ms, 1e-6), 2)
    r_ovl.extra["overlap_ratio"] = round(overlap_ratio, 3)
    results.extend([r_dw, r_comm, r_seq, r_ovl])

    print_report("Lumen Wgrad Delay Benchmarks", results)


if __name__ == "__main__":
    main()
