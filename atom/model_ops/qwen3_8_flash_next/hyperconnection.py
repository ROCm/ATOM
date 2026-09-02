"""Qwen3.8-Flash-Next Hyper-Connections: `hc_count` parallel residual streams.

Port of `qwen3_8_flash_next/common/hyperconnection.py:GatedResidualSimple`,
verified bitwise against it (`tests/qwen3_8_flash_next/test_hyperconnection_parity.py`).

There is NO `input_layernorm` / `post_attention_layernorm` in this checkpoint.
The hyper-connection carries the norm (`hc_norm`) and replaces the classic
residual outright, so the tensor threaded between layers is the FLAT
`[tokens, hc_count * hidden_size]` stream bundle (HC outer, hidden inner --
the checkpoint-native layout), not `[tokens, hidden_size]`.

    mixed, residual = hc.mix(hidden)             # [T, hc*H] -> [T, H]
    hidden = hc.combine(sublayer_out, residual)  # -> [T, hc*H]

`combine` reuses the NORMALIZED input that the paired `mix` produced, so the
two calls must be paired per sub-layer. The final `hyper_connection_mixer` is
built with `has_block_inject=False` and only `mix()` is ever called on it; the
checkpoint's `block_inject_weight` for that module is skipped at load.

Parameters are replicated, not TP-sharded: ~13 MB per module.
"""

import torch
import torch.nn.functional as F
from torch import nn

from atom.model_ops.linear import ReplicatedLinear
from atom.model_ops.qwen3_8_flash_next.kernels.hyperconnection_ops import (
    combine_inject,
    mix_gated_mean,
)
from atom.model_ops.qwen3_8_flash_next.norm import Qwen3_8FlashNextGroupedGemmaRMSNorm


class Qwen3_8FlashNextHyperConnection(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        hc_count: int,
        hc_lowrank: int,
        has_block_inject: bool = True,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.hc_count = hc_count
        self.hc_dim = hc_count * hidden_size
        self.hc_norm = Qwen3_8FlashNextGroupedGemmaRMSNorm(
            self.hc_dim, eps=eps, group_size=hidden_size
        )
        self.input_mix_weight_down = ReplicatedLinear(
            self.hc_dim, hc_lowrank, bias=False, prefix="input_mix_weight_down"
        )
        self.input_mix_weight_up = ReplicatedLinear(
            hc_lowrank, self.hc_dim, bias=False, prefix="input_mix_weight_up"
        )
        self.block_inject_weight = (
            ReplicatedLinear(
                self.hc_dim, hc_count, bias=False, prefix="block_inject_weight"
            )
            if has_block_inject
            else None
        )

    def mix(
        self, hyper_input: torch.Tensor
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """RMSNorm -> low-rank silu/sigmoid gate -> gated mean over streams.

        Only the two low-rank projections are GEMMs; the sigmoid, the
        broadcast multiply and the mean over streams are one fused pass.
        """
        normed = self.hc_norm(hyper_input)
        gate = F.silu(
            F.linear(normed, self.input_mix_weight_down.weight) / self.hc_count
        )
        gate = F.linear(gate, self.input_mix_weight_up.weight)
        mixed = mix_gated_mean(normed, gate, self.hc_count)
        return mixed.to(hyper_input.dtype), (hyper_input, normed)

    def mix_native(
        self, hyper_input: torch.Tensor
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """Eager reference for `mix`; the parity test pins the math to this."""
        normed = self.hc_norm.forward_native(hyper_input)
        gate = F.silu(
            F.linear(normed, self.input_mix_weight_down.weight) / self.hc_count
        )
        gate = torch.sigmoid(F.linear(gate, self.input_mix_weight_up.weight)).unflatten(
            -1, (self.hc_count, self.hidden_size)
        )
        mixed = (gate * normed.unflatten(-1, (self.hc_count, self.hidden_size))).mean(
            dim=-2
        )
        return mixed.to(hyper_input.dtype), (hyper_input, normed)

    def combine(
        self,
        block_output: torch.Tensor,
        residuals: tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        """Inject the sub-layer output into every stream with a learned gate."""
        if self.block_inject_weight is None:
            raise RuntimeError("combine was disabled for this hyper-connection")
        hyper_input, normed = residuals
        raw = F.linear(normed, self.block_inject_weight.weight) / self.hc_count
        return combine_inject(hyper_input, block_output, raw, self.hc_count).to(
            hyper_input.dtype
        )

    def combine_native(
        self,
        block_output: torch.Tensor,
        residuals: tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        """Eager reference for `combine`."""
        if self.block_inject_weight is None:
            raise RuntimeError("combine was disabled for this hyper-connection")
        hyper_input, normed = residuals
        residual = hyper_input.unflatten(-1, (self.hc_count, self.hidden_size))
        injection = 2.0 * torch.sigmoid(
            F.linear(normed, self.block_inject_weight.weight) / self.hc_count
        )
        out = residual + block_output.unsqueeze(-2) * injection.unsqueeze(-1)
        return out.flatten(-2).to(hyper_input.dtype)
