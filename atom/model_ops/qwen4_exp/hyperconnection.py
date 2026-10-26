"""Qwen3.8-Flash-Next Hyper-Connections: `hc_count` parallel residual streams.

Parity-tested against Transformers' Qwen4ExpTextGatedResidual.

There is NO `input_layernorm` / `post_attention_layernorm` in this checkpoint.
The hyper-connection carries the norm (`hc_norm`) and replaces the classic
residual outright, so the tensor threaded between layers is the FLAT
`[tokens, hc_count * hidden_size]` stream bundle (HC outer, hidden inner --
the checkpoint-native layout), not `[tokens, hidden_size]`.

    mixed, residual = hc.mix(hidden)             # [T, hc*H] -> [T, H]
    hidden = hc.combine(sublayer_out, residual)  # -> [T, hc*H]

`combine` uses the NORMALIZED input for its injection gate and adds the result
to the ORIGINAL residual; the two calls must be paired per sub-layer. The final `hyper_connection_mixer` is
built with `has_block_inject=False` and only `mix()` is ever called on it; the
checkpoint's `block_inject_weight` for that module is skipped at load.

Parameters are replicated, not TP-sharded: ~13 MB per module.
"""

import torch
from aiter.tuned_gemm import tgemm
from torch import nn

from atom.model_ops.linear import ReplicatedLinear
from atom.model_ops.qwen4_exp.ops.gated import (
    combine_inject,
    mix_gated_mean,
    scaled_silu,
)
from atom.model_ops.qwen4_exp.ops.hc_fused import (
    hc_combine_norm,
    hc_gated_mean,
    hc_rows,
)

# The fused path is the only one the served model takes: `process_weights_after_loading`
# always builds the `[down | inject]` weight and the decoder layer always defers the
# combine into the next mix. `mix`/`combine` stay for the MTP drafter, which mixes and
# combines in two separate calls.
# Up to this many tokens the two fused kernels beat norm + GEMMs; each row
# re-reads the (cache resident) weights, so the cost grows with the batch.
HC_ROWS_MAX = 2
from atom.model_ops.triton_gemma_rmsnorm import gemma_rmsnorm_triton
from atom.model_ops.utils import atom_parameter


class Qwen4ExpGroupedRMSNorm(nn.Module):
    """Per-HC-stream Gemma norm, with checkpoint-native full-width weights."""

    def __init__(self, hidden_size: int, group_size: int, eps: float = 1e-6):
        super().__init__()
        if group_size <= 0 or hidden_size % group_size:
            raise ValueError("group_size must divide hidden_size")
        self.weight = atom_parameter(torch.zeros(hidden_size))
        self.group_size = group_size
        self.variance_epsilon = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return gemma_rmsnorm_triton(
            x, self.weight, self.variance_epsilon, None, self.group_size
        )


class Qwen4ExpHyperConnection(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        hc_count: int,
        hc_lowrank: int,
        has_block_inject: bool = True,
        eps: float = 1e-6,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.hc_count = hc_count
        self.hc_dim = hc_count * hidden_size
        self.hc_norm = Qwen4ExpGroupedRMSNorm(
            self.hc_dim, group_size=hidden_size, eps=eps
        )

        # Parented, because `prefix` is the key `get_layer_quant_config` looks
        # a layer up by and the string every GEMM error message prints. Bare
        # names would make all 97 of these modules the same three layers.
        def name(leaf: str) -> str:
            return f"{prefix}.{leaf}" if prefix else leaf

        self.input_mix_weight_down = ReplicatedLinear(
            self.hc_dim, hc_lowrank, bias=False, prefix=name("input_mix_weight_down")
        )
        self.input_mix_weight_up = ReplicatedLinear(
            hc_lowrank, self.hc_dim, bias=False, prefix=name("input_mix_weight_up")
        )
        self.block_inject_weight = (
            ReplicatedLinear(
                self.hc_dim, hc_count, bias=False, prefix=name("block_inject_weight")
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
        gate = scaled_silu(
            self.input_mix_weight_down(normed, otype=normed.dtype), self.hc_count
        )
        gate = self.input_mix_weight_up(gate, otype=gate.dtype)
        mixed = mix_gated_mean(normed, gate, self.hc_count)
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
        raw = self.block_inject_weight(normed, otype=normed.dtype)
        return combine_inject(hyper_input, block_output, raw, self.hc_count).to(
            hyper_input.dtype
        )

    def process_weights_after_loading(self) -> None:
        """Concatenate `[down | inject]` into one GEMM weight.

        The source parameters become row views of it, so the weight is stored
        once and `mix` / `combine` (the MTP drafter's unfused path) still work.
        """
        rank = self.input_mix_weight_down.weight.shape[0]
        parts = [self.input_mix_weight_down.weight.data]
        if self.block_inject_weight is not None:
            parts.append(self.block_inject_weight.weight.data)
        fused = torch.cat(parts, 0).contiguous()
        self.input_mix_weight_down.weight.data = fused[:rank]
        if self.block_inject_weight is not None:
            self.block_inject_weight.weight.data = fused[rank:]
        self.fused_w_cat = fused
        self.fused_rank = rank

    def mix_fused(
        self,
        hyper_input: torch.Tensor,
        pending: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor | None]]:
        """`combine` of the previous sub-layer (deferred as `pending`) + `mix`.

        `pending` is `(block_output, inject_logits)` from the previous
        hyper-connection. Returns the mixed activation and `(streams,
        inject_logits)`; the logits are None when there is no block injection.
        """
        block_out, raw_prev = pending if pending is not None else (None, None)
        has_inject = self.block_inject_weight is not None
        w_up = self.input_mix_weight_up.weight
        eps = self.hc_norm.variance_epsilon
        if hyper_input.shape[0] <= HC_ROWS_MAX:
            h, mixed, raw = hc_rows(
                hyper_input,
                self.hc_norm.weight,
                self.fused_w_cat,
                w_up,
                eps,
                self.hc_count,
                has_inject,
                block_out,
                raw_prev,
            )
            return mixed, (h, raw)
        h, normed = hc_combine_norm(
            hyper_input, self.hc_norm.weight, eps, self.hc_count, block_out, raw_prev
        )
        rank = self.fused_rank
        d = tgemm.mm(normed, self.fused_w_cat, otype=normed.dtype)
        gate = scaled_silu(d[:, :rank], self.hc_count)
        up = tgemm.mm(gate, w_up, otype=gate.dtype)
        mixed = hc_gated_mean(normed, up, self.hc_count)
        raw = d[:, rank : rank + self.hc_count] if has_inject else None
        return mixed, (h, raw)

    def apply_pending(
        self, hyper_input: torch.Tensor, pending: tuple[torch.Tensor, torch.Tensor]
    ) -> torch.Tensor:
        """Materialize a deferred combine without mixing."""
        block_out, raw = pending
        return combine_inject(hyper_input, block_out, raw, self.hc_count)
