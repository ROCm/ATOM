# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

"""JSON-serializable integrity checks for weight-sync tensors."""

from __future__ import annotations

import hashlib
from collections.abc import Generator, Iterable
from typing import Any

import torch

from atomrl.engine.inference.vllm_fp8_utils import fingerprint_tensor


def _float8_stats(
    value: torch.Tensor,
) -> tuple[int, int, int, float | None, float | None]:
    if "fnuz" in str(value.dtype):
        raw = value.reshape(-1).view(torch.uint8)
        # ROCm FNUZ formats reserve the negative-zero encoding (0x80) for NaN
        # and do not encode infinities.
        nan_count = int((raw == 0x80).sum().item())
        finite_count = int(value.numel()) - nan_count
        sample = value.reshape(-1)[:4096].float()
        finite_sample = sample[torch.isfinite(sample)]
        finite_min = (
            float(finite_sample.min().item()) if finite_sample.numel() else None
        )
        finite_max = (
            float(finite_sample.max().item()) if finite_sample.numel() else None
        )
        return finite_count, nan_count, 0, finite_min, finite_max

    finite_count = 0
    nan_count = 0
    inf_count = 0
    finite_min: float | None = None
    finite_max: float | None = None
    flat = value.reshape(-1)
    chunk_size = 4 * 1024 * 1024
    for start in range(0, flat.numel(), chunk_size):
        chunk = flat[start : start + chunk_size].float()
        finite = torch.isfinite(chunk)
        chunk_finite = int(finite.sum().item())
        finite_count += chunk_finite
        nan_count += int(torch.isnan(chunk).sum().item())
        inf_count += int(torch.isinf(chunk).sum().item())
        if chunk_finite:
            values = chunk if chunk_finite == chunk.numel() else chunk[finite]
            current_min = float(values.min().item())
            current_max = float(values.max().item())
            finite_min = (
                current_min if finite_min is None else min(finite_min, current_min)
            )
            finite_max = (
                current_max if finite_max is None else max(finite_max, current_max)
            )
    return finite_count, nan_count, inf_count, finite_min, finite_max


@torch.no_grad()
def tensor_integrity(
    name: str,
    tensor: torch.Tensor,
    *,
    include_checksum: bool = True,
) -> dict[str, Any]:
    value = tensor.detach()
    if value.is_meta:
        return {
            "name": str(name),
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "numel": int(value.numel()),
            "checksum": None,
            "finite_count": 0,
            "nan_count": 0,
            "inf_count": 0,
            "finite_min": None,
            "finite_max": None,
            "materialized": False,
            "all_finite": False,
        }
    report: dict[str, Any] = {
        "name": str(name),
        "shape": list(value.shape),
        "dtype": str(value.dtype),
        "numel": int(value.numel()),
        "checksum": (fingerprint_tensor(value).checksum if include_checksum else None),
        "materialized": True,
    }
    if not (value.is_floating_point() or value.is_complex()):
        report.update(
            {
                "finite_count": int(value.numel()),
                "nan_count": 0,
                "inf_count": 0,
                "finite_min": None,
                "finite_max": None,
                "all_finite": True,
            }
        )
        return report

    if "float8" in str(value.dtype):
        (
            finite_count,
            nan_count,
            inf_count,
            finite_min,
            finite_max,
        ) = _float8_stats(value)
    else:
        finite = torch.isfinite(value)
        finite_count = int(finite.sum().item())
        nan_count = int(torch.isnan(value).sum().item())
        inf_count = int(torch.isinf(value).sum().item())
        finite_min: float | None = None
        finite_max: float | None = None
        if finite_count:
            finite_values = value if finite_count == value.numel() else value[finite]
            finite_min = float(finite_values.min().item())
            finite_max = float(finite_values.max().item())
    report.update(
        {
            "finite_count": finite_count,
            "nan_count": nan_count,
            "inf_count": inf_count,
            "finite_min": finite_min,
            "finite_max": finite_max,
            "all_finite": finite_count == value.numel(),
        }
    )
    return report


@torch.no_grad()
def scan_named_tensors(
    named_tensors: Iterable[tuple[str, torch.Tensor]],
    *,
    stop_on_first_bad: bool = True,
) -> dict[str, Any]:
    tensor_count = 0
    total_numel = 0
    first_bad: dict[str, Any] | None = None
    for name, tensor in named_tensors:
        report = tensor_integrity(name, tensor)
        tensor_count += 1
        total_numel += int(report["numel"])
        if not report["all_finite"] and first_bad is None:
            first_bad = report
            if stop_on_first_bad:
                break
    return {
        "tensor_count": tensor_count,
        "total_numel": total_numel,
        "all_finite": first_bad is None,
        "first_bad": first_bad,
    }


@torch.no_grad()
def sampled_tensor_integrity(
    name: str,
    tensor: torch.Tensor,
    *,
    max_samples: int = 4096,
) -> dict[str, Any]:
    value = tensor.detach()
    if value.is_meta or value.numel() <= max_samples:
        report = tensor_integrity(name, value)
        report["sample_count"] = int(value.numel())
        return report
    flat = value.reshape(-1)
    first_count = max_samples // 2
    sample = torch.cat((flat[:first_count], flat[-(max_samples - first_count) :]))
    report = tensor_integrity(name, sample, include_checksum=False)
    sample_bytes = sample.contiguous().view(torch.uint8).cpu().numpy().tobytes()
    report.update(
        {
            "shape": list(value.shape),
            "numel": int(value.numel()),
            "sample_count": int(sample.numel()),
            "checksum": hashlib.blake2b(
                sample_bytes,
                digest_size=16,
            ).hexdigest(),
        }
    )
    return report


@torch.no_grad()
def scan_fp8_scales(
    named_tensors: Iterable[tuple[str, torch.Tensor]],
) -> dict[str, Any]:
    scale_tensor_count = 0
    scale_numel = 0
    nonfinite_count = 0
    nonpositive_count = 0
    scale_min: float | None = None
    scale_max: float | None = None
    first_bad: str | None = None
    for name, tensor in named_tensors:
        if not name.endswith("weight_scale_inv") or tensor.is_meta:
            continue
        value = tensor.detach()
        finite = torch.isfinite(value)
        current_nonfinite = int((~finite).sum().item())
        current_nonpositive = int(((value <= 0) & finite).sum().item())
        scale_tensor_count += 1
        scale_numel += int(value.numel())
        nonfinite_count += current_nonfinite
        nonpositive_count += current_nonpositive
        if (current_nonfinite or current_nonpositive) and first_bad is None:
            first_bad = name
        if finite.any():
            finite_values = value[finite]
            current_min = float(finite_values.min().item())
            current_max = float(finite_values.max().item())
            scale_min = (
                current_min if scale_min is None else min(scale_min, current_min)
            )
            scale_max = (
                current_max if scale_max is None else max(scale_max, current_max)
            )
    return {
        "scale_tensor_count": scale_tensor_count,
        "scale_numel": scale_numel,
        "nonfinite_count": nonfinite_count,
        "nonpositive_count": nonpositive_count,
        "scale_min": scale_min,
        "scale_max": scale_max,
        "all_valid": nonfinite_count == 0 and nonpositive_count == 0,
        "first_bad": first_bad,
    }


def require_finite_stream(
    named_tensors: Iterable[tuple[str, torch.Tensor]],
    *,
    stage: str,
) -> Generator[tuple[str, torch.Tensor], None, None]:
    for name, tensor in named_tensors:
        report = sampled_tensor_integrity(name, tensor)
        if not report["all_finite"]:
            raise FloatingPointError(
                f"non-finite weight at stage={stage} tensor={name}: {report}"
            )
        yield name, tensor
