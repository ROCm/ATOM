"""Grouped Gemma RMSNorm shared by Qwen3.8-Flash-Next's HyperConnection and PLE.

Both normalize a `[..., hc_count * hidden_size]` tensor one `hidden_size`
slice at a time while keeping one affine weight per element of the wide
layout, and both use Gemma's `x * (1 + w)` scaling.
"""

import torch
from torch import nn

from atom.model_ops.qwen3_8_flash_next.kernels.grouped_gemma_rmsnorm import (
    grouped_gemma_rmsnorm,
)
from atom.model_ops.utils import atom_parameter


class Qwen3_8FlashNextGroupedGemmaRMSNorm(nn.Module):
    """RMSNorm with per-`group_size` variance and a full-width Gemma weight."""

    def __init__(
        self, hidden_size: int, eps: float, group_size: int | None = None
    ) -> None:
        super().__init__()
        if group_size is not None and hidden_size % group_size:
            raise ValueError(
                f"hidden_size ({hidden_size}) must be divisible by "
                f"group_size ({group_size})"
            )
        self.eps = eps
        self.group_size = group_size
        self.weight = atom_parameter(torch.zeros(hidden_size))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Fused when grouped, which is every use in this model.

        `forward_native` stays the definition of the math -- it is what the
        parity test checks bitwise against the reference -- and this only
        replaces the eight-kernel eager form with one launch.
        """
        if self.group_size is None or hidden_states.numel() == 0:
            return self.forward_native(hidden_states)
        return grouped_gemma_rmsnorm(
            hidden_states, self.weight, self.eps, self.group_size
        )

    def forward_native(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        x = hidden_states.float()
        if self.group_size is None:
            variance = x.square().mean(dim=-1, keepdim=True)
            normalized = x * torch.rsqrt(variance + self.eps)
        else:
            grouped = x.unflatten(-1, (x.shape[-1] // self.group_size, self.group_size))
            variance = grouped.square().mean(dim=-1, keepdim=True)
            normalized = (grouped * torch.rsqrt(variance + self.eps)).flatten(-2)
        return (normalized * (1.0 + self.weight.float())).to(input_dtype)
