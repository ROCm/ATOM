# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Product host adapter for the rank-local gfx1250 K1 pipeline."""

from __future__ import annotations

from dataclasses import dataclass, fields

import torch

from atom.models.kimi_k3.mono.kernels.attention.kda_builder_gfx1250 import (
    HEAD_DIM,
    HEADS,
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
    delta: torch.Tensor | None
    blocks: torch.Tensor
    blk_sm: int
    blk_sr: int
    ares_nw: torch.Tensor
    ares_qk: torch.Tensor
    in_nw: torch.Tensor
    w_in: torch.Tensor
    w_in_scale: torch.Tensor | None
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
    core_q: torch.Tensor | None
    core_scale: torch.Tensor | None

    def __post_init__(self) -> None:
        names = tuple(
            field.name for field in fields(self) if field.name != "core_scale"
        )
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
        if build_result.launch_count != 1 or build_result.kernel.launch_count != 1:
            raise ValueError("gfx1250 resident K1 must contain exactly one launch")
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
        if self.spec.delta and tuple(runtime.delta.shape) != (rows, HIDDEN):
            raise ValueError(f"delta must have shape {(rows, HIDDEN)}")
        if not self.spec.delta and runtime.delta is not None:
            raise ValueError("delta is only read when K1 applies the residual delta")
        if (
            runtime.blocks.ndim != 3
            or runtime.blocks.shape[0] != rows
            or runtime.blocks.shape[2] != HIDDEN
        ):
            raise ValueError("blocks must have shape [rows, blocks, 7168]")
        needed_blocks = self.build_result.declaration.nblocks
        if self.spec.write_idx >= 0:
            needed_blocks = max(needed_blocks, self.spec.write_idx + 1)
        if runtime.blocks.shape[1] < needed_blocks:
            raise ValueError(
                f"blocks must expose at least {needed_blocks} residual rows"
            )
        if runtime.prefix.dtype != torch.bfloat16:
            raise ValueError("local K1 activations must be bfloat16")
        for name in (
            "delta",
            "blocks",
            "ares_nw",
            "ares_qk",
            "in_nw",
            "w_fb",
            "conv_st",
            "core_out",
        ):
            if getattr(runtime, name) is None and name == "delta":
                continue
            if getattr(runtime, name).dtype != torch.bfloat16:
                raise ValueError(f"local K1 {name} must be bfloat16")
        if self.spec.fp8_projection:
            w_in, scale = runtime.w_in, runtime.w_in_scale
            if w_in.dtype != torch.float8_e4m3fn or w_in.shape[1] != HIDDEN:
                raise ValueError("local K1 FP8 w_in must be float8_e4m3fn [N, 7168]")
            if (
                scale is None
                or scale.element_size() != 1
                or w_in.shape[0] % 128
                or tuple(scale.shape) != (w_in.shape[0] // 128, HIDDEN // 128)
            ):
                raise ValueError(
                    "local K1 w_in_scale must be the E8M0 [N / 128, 56] block scales"
                )
        else:
            if runtime.w_in.dtype != torch.bfloat16:
                raise ValueError("local K1 w_in must be bfloat16")
            if runtime.w_in_scale is not None:
                raise ValueError("a BF16 local K1 in-projection takes no w_in_scale")
        if self.spec.quant_core:
            if runtime.core_q is None or tuple(runtime.core_q.shape) != (
                rows,
                HEADS * HEAD_DIM,
            ):
                raise ValueError(
                    f"local K1 core_q must have shape {(rows, HEADS * HEAD_DIM)}"
                )
            if runtime.core_q.element_size() != 1:
                raise ValueError("local K1 core_q must use an 8-bit FP8 container")
            if (
                runtime.core_scale is None
                or tuple(runtime.core_scale.shape) != (rows, 1)
                or runtime.core_scale.dtype != torch.float32
            ):
                raise ValueError(f"local K1 core_scale must be float32 {(rows, 1)}")
            if (
                runtime.core_scale.data_ptr()
                != runtime.core_q.data_ptr() + rows * HEADS * HEAD_DIM
            ):
                raise ValueError(
                    "local K1 core_scale must immediately follow packed core_q"
                )
        elif runtime.core_q is not None or runtime.core_scale is not None:
            raise ValueError("unquantized local K1 must not receive quant outputs")
        for name in ("conv_w", "a_log", "dt_bias", "on_w", "rstate"):
            if getattr(runtime, name).dtype != torch.float32:
                raise ValueError(f"local K1 {name} must be float32")
        for name in ("st_idx", "num_acc"):
            if getattr(runtime, name).dtype != torch.int32:
                raise ValueError(f"local K1 {name} must be int32")
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
        return device

    def _scratch_for(self, device: torch.device) -> torch.Tensor:
        if self._scratch is None:
            size = max(offset + length for offset, length in self.layout.values())
            self._scratch = torch.zeros(size, dtype=torch.uint8, device=device)
        elif self._scratch.device != device:
            raise ValueError("local K1 runner cannot change devices")
        return self._scratch

    def prepare(self, device: torch.device) -> Gfx1250K1LocalRunner:
        """Allocate stable scratch before warmup/capture enters ``__call__``."""

        self._scratch_for(device)
        return self

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
                value.data_ptr()
                if isinstance(value, torch.Tensor)
                else (0 if value is None else value)
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
