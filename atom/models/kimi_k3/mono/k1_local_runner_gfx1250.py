# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Product host adapter for the rank-local gfx1250 K1 pipeline."""

from __future__ import annotations

from dataclasses import dataclass, fields

import torch

from atom.models.kimi_k3.mono.kernels.attention.kda_builder_gfx1250 import (
    HIDDEN,
    K1_ABI_NAMES,
    K1BuildResult,
    K1LocalBuildSpec,
    local_scratch_layout,
)
from atom.models.kimi_k3.mono.kernels.execution_plan import (
    Gfx1250KimiBuildPlan,
)


@dataclass(frozen=True)
class K1LocalRuntime:
    """One product K1 invocation in the local pipeline's named order."""

    prefix: torch.Tensor
    delta: torch.Tensor
    blocks: torch.Tensor
    blk_sm: int
    blk_sr: int
    ares_nw: torch.Tensor
    ares_qk: torch.Tensor
    in_nw: torch.Tensor
    w_in: torch.Tensor
    w_fb: torch.Tensor
    conv_w: torch.Tensor
    conv_st: torch.Tensor
    cs_seq: int
    cs_dim: int
    cs_tok: int
    a_log: torch.Tensor
    dt_bias: torch.Tensor
    on_w: torch.Tensor
    rstate: torch.Tensor
    rs_seq: int
    st_idx: torch.Tensor
    st_stride: int
    num_acc: torch.Tensor
    core_out: torch.Tensor
    x_dbg: torch.Tensor
    proj_dbg: torch.Tensor
    layer: int
    tl: torch.Tensor

    def __post_init__(self) -> None:
        names = tuple(field.name for field in fields(self))
        expected = tuple(name for name in K1_ABI_NAMES if name != "scratch")
        if names != expected:
            raise TypeError("K1LocalRuntime fields drifted from the K1 ABI")


class Gfx1250K1LocalRunner:
    """Own scratch and launch a complete local K1 on the caller's stream."""

    def __init__(
        self,
        plan: Gfx1250KimiBuildPlan,
        spec: K1LocalBuildSpec,
        build_result: K1BuildResult,
    ) -> None:
        spec.validate(plan)
        if not build_result.runnable or not build_result.kernel.complete:
            raise ValueError("local K1 runner requires a complete build result")
        if build_result.declaration.tokens != plan.tokens:
            raise ValueError("local K1 build rows disagree with plan")
        self.plan = plan
        self.spec = spec
        self.build_result = build_result
        self.entrypoint = build_result.kernel
        self.layout = local_scratch_layout(build_result.declaration)
        self._scratch: torch.Tensor | None = None

    @property
    def launch_count(self) -> int:
        return self.entrypoint.launch_count

    def _validate(self, runtime: K1LocalRuntime) -> torch.device:
        tensors = [
            getattr(runtime, field.name)
            for field in fields(runtime)
            if isinstance(getattr(runtime, field.name), torch.Tensor)
        ]
        device = runtime.prefix.device
        if any(tensor.device != device for tensor in tensors):
            raise ValueError("local K1 tensors must share one device")
        # The production KDA cache may expose [request, channel, token] as a
        # transpose of its allocated pool.  K1 carries all three conv strides
        # explicitly, so requiring that one tensor to be contiguous would
        # reject a layout the kernel already supports.
        if any(
            not tensor.is_contiguous()
            for tensor in tensors
            if tensor is not runtime.conv_st
        ):
            raise ValueError("local K1 tensors other than conv_st must be contiguous")
        rows = self.plan.tokens
        if tuple(runtime.prefix.shape) != (rows, HIDDEN):
            raise ValueError(f"prefix must have shape {(rows, HIDDEN)}")
        if tuple(runtime.delta.shape) != (rows, HIDDEN):
            raise ValueError(f"delta must have shape {(rows, HIDDEN)}")
        if (
            runtime.blocks.ndim != 3
            or runtime.blocks.shape[0] != rows
            or runtime.blocks.shape[2] != HIDDEN
        ):
            raise ValueError("blocks must have shape [rows, blocks, 7168]")
        if runtime.prefix.dtype != torch.bfloat16:
            raise ValueError("local K1 activations must be bfloat16")
        for name in (
            "delta",
            "blocks",
            "ares_nw",
            "ares_qk",
            "in_nw",
            "w_in",
            "w_fb",
            "conv_st",
            "on_w",
            "core_out",
            "x_dbg",
            "proj_dbg",
        ):
            if getattr(runtime, name).dtype != torch.bfloat16:
                raise ValueError(f"local K1 {name} must be bfloat16")
        for name in ("conv_w", "a_log", "dt_bias", "rstate"):
            if getattr(runtime, name).dtype != torch.float32:
                raise ValueError(f"local K1 {name} must be float32")
        for name in ("st_idx", "num_acc"):
            if getattr(runtime, name).dtype != torch.int32:
                raise ValueError(f"local K1 {name} must be int32")
        if runtime.tl.dtype != torch.int64:
            raise ValueError("local K1 timeline storage must be int64")
        if runtime.blk_sm < HIDDEN or runtime.blk_sr < HIDDEN:
            raise ValueError("local K1 block strides must span one hidden row")
        if (
            min(
                runtime.cs_seq,
                runtime.cs_dim,
                runtime.cs_tok,
                runtime.rs_seq,
                runtime.st_stride,
            )
            <= 0
        ):
            raise ValueError("local K1 runtime strides must be positive")
        if runtime.layer < 0:
            raise ValueError("local K1 layer tag must be nonnegative")
        return device

    def _scratch_for(self, device: torch.device) -> torch.Tensor:
        if self._scratch is None:
            size = max(offset + length for offset, length in self.layout.values())
            self._scratch = torch.zeros(size, dtype=torch.uint8, device=device)
        elif self._scratch.device != device:
            raise ValueError("local K1 runner cannot change devices")
        return self._scratch

    def abi_values(self, runtime: K1LocalRuntime) -> dict[str, int]:
        device = self._validate(runtime)
        scratch = self._scratch_for(device)
        values = {}
        for name in K1_ABI_NAMES:
            if name == "scratch":
                values[name] = scratch.data_ptr()
                continue
            value = getattr(runtime, name)
            values[name] = (
                value.data_ptr() if isinstance(value, torch.Tensor) else value
            )
        return values

    def __call__(
        self,
        runtime: K1LocalRuntime,
        *,
        stream=None,
    ) -> torch.Tensor:
        values = self.abi_values(runtime)
        launcher = self.entrypoint.launcher
        arguments = [values[name] for name in K1_ABI_NAMES]
        if stream is None:
            launcher(*arguments)
        else:
            launcher(*arguments, stream=stream)
        return runtime.core_out
