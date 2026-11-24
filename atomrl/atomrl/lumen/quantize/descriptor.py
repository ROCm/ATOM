# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

import os
from dataclasses import dataclass, field

import torch

__all__ = ["FP8Descriptor"]

_USE_FAST_TRANSPOSE = os.environ.get("LUMEN_FUSED_CAST_TRANSPOSE_V2", "0") == "1"
_TRANSPOSE_CACHE_ENABLED = os.environ.get("LUMEN_TRANSPOSE_CACHE", "1") != "0"
_fast_transpose_fn = None


def _get_fast_transpose():
    global _fast_transpose_fn
    if _fast_transpose_fn is not None:
        return _fast_transpose_fn
    try:
        from atomrl.lumen.ops.quantize.fast_transpose import fast_transpose_fp8

        _fast_transpose_fn = fast_transpose_fp8
    except (ImportError, OSError):
        _fast_transpose_fn = False
    return _fast_transpose_fn


@dataclass
class FP8Descriptor:
    """Lightweight FP8 data bundle with lazy transpose cache.

    NOT a Tensor subclass — just a plain Python object that keeps
    FP8 data, its scale, and an optional cached transpose together.
    """

    data: torch.Tensor
    scale: torch.Tensor
    fp8_dtype: torch.dtype
    unpadded_shape: torch.Size | None = None
    _transpose: torch.Tensor | None = field(default=None, repr=False)
    _scale_f32_1x1: torch.Tensor | None = field(default=None, repr=False)

    @property
    def scale_f32_1x1(self) -> torch.Tensor:
        """Return the scale as a ``(1, 1)`` float32 tensor, cached after first access."""
        if self._scale_f32_1x1 is not None:
            return self._scale_f32_1x1
        s = self.scale
        if (
            isinstance(s, torch.Tensor)
            and s.dtype == torch.float32
            and s.shape == (1, 1)
        ):
            self._scale_f32_1x1 = s
        elif isinstance(s, torch.Tensor):
            self._scale_f32_1x1 = s.float().reshape(1, 1)
        else:
            self._scale_f32_1x1 = torch.tensor(
                [[s]], dtype=torch.float32, device=self.data.device
            )
        return self._scale_f32_1x1

    @property
    def transpose_cached(self) -> torch.Tensor:
        if not _TRANSPOSE_CACHE_ENABLED:
            if _USE_FAST_TRANSPOSE and self.data.dim() == 2 and self.data.is_cuda:
                fn = _get_fast_transpose()
                if fn:
                    return fn(self.data)
            return self.data.t().contiguous()
        if self._transpose is None:
            if _USE_FAST_TRANSPOSE and self.data.dim() == 2 and self.data.is_cuda:
                fn = _get_fast_transpose()
                if fn:
                    self._transpose = fn(self.data)
                    return self._transpose
            self._transpose = self.data.t().contiguous()
        return self._transpose

    def invalidate_transpose(self) -> None:
        self._transpose = None

    def tensors(self) -> tuple[torch.Tensor, torch.Tensor]:
        return self.data, self.scale

    @classmethod
    def from_tensors(
        cls,
        data: torch.Tensor,
        scale: torch.Tensor,
        fp8_dtype: torch.dtype,
    ) -> "FP8Descriptor":
        return cls(data=data, scale=scale, fp8_dtype=fp8_dtype)
