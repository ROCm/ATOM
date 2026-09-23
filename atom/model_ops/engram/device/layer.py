# SPDX-License-Identifier: MIT
"""Engram projection and FP32 residual gating, derived from PR #2185.

Checkpoint I/O, table lookups and request staging are owned by separate
modules.
"""

import torch
from torch import nn

from atom.config import QuantizationConfig
from atom.model_ops.engram.device.gate import engram_post_wkv
from atom.model_ops.linear import ReplicatedLinear
from atom.model_ops.utils import atom_parameter


class EngramOp(nn.Module):
    """One Engram module: gate a host-supplied memory read into the residual.

    `forward` consumes provider-supplied embeddings and returns the complete
    updated residual; this module has no dependency on the provider's table
    residency.
    """

    def __init__(
        self,
        layer_id: int,
        hidden_size: int = 5120,
        engram_hidden_size: int = 6144,
        hc_mult: int = 4,
        norm_eps: float = 1e-20,
        quant_config: QuantizationConfig | None = None,
    ):
        super().__init__()
        self.layer_id = layer_id
        self.hidden_size = hidden_size
        self.engram_hidden_size = engram_hidden_size
        self.hc_mult = hc_mult
        self.norm_eps = norm_eps

        # One fused projection, laid out as the checkpoint stores it: the
        # hc_mult key projections first, the single shared value projection
        # last. Built here, not taken as a module, so the shape is derived
        # once from the arguments that already fix it.
        self.wkv = ReplicatedLinear(
            engram_hidden_size,
            (hc_mult + 1) * hidden_size,
            bias=False,
            quant_config=quant_config,
        )
        self.register_buffer("gate_weight", None, persistent=False)
        self.k_weight = atom_parameter(torch.ones(hc_mult, hidden_size))
        self.q_weight = atom_parameter(torch.ones(hc_mult, hidden_size))

    @property
    def key_rows(self) -> int:
        return self.hc_mult * self.hidden_size

    def forward(
        self,
        hidden_states: torch.Tensor,
        embeddings: torch.Tensor,
        token_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return the updated residual, with no BF16 gate or intermediate addition.

        The token dimension is flat, matching the rest of ATOM: a model's
        residual stream is [num_tokens, hc, dim], not [batch, seq, ...]. Leading
        dimensions are otherwise free, so a [B, T, hc, H] caller also works.

        Context-aware gating: each branch scores its own key against its own
        slice of the residual, and the shared value is admitted in proportion to
        that score.
        """
        if embeddings.shape[-1] != self.engram_hidden_size:
            raise ValueError(
                f"engram embeddings are {embeddings.shape[-1]} wide, expected "
                f"{self.engram_hidden_size}"
            )
        if hidden_states.shape[-2] != self.hc_mult:
            raise ValueError(
                f"hidden states carry {hidden_states.shape[-2]} branches, "
                f"expected hc_mult={self.hc_mult}"
            )
        if hidden_states.shape[:-2] != embeddings.shape[:-1]:
            raise ValueError(
                f"hidden states cover {tuple(hidden_states.shape[:-2])} tokens "
                f"but embeddings cover {tuple(embeddings.shape[:-1])}"
            )
        kv = self.wkv(embeddings)
        return engram_post_wkv(
            hidden_states, kv, self.gate_weight, token_mask, self.norm_eps
        )

    @torch.no_grad()
    def process_weights_after_loading(self):
        self.gate_weight = self.q_weight.float() * self.k_weight.float()

    @torch.no_grad()
    def load_checkpoint_weights(
        self,
        wkv: torch.Tensor,
        k_weight: torch.Tensor,
        q_weight: torch.Tensor,
        wkv_scale: torch.Tensor | None = None,
        block: int = 32,
    ) -> None:
        """Copy projection/gate tensors, preserving an injected native projection.

        Shapes are checked rather than reshaped into submission: a silently
        transposed or mis-split wkv produces plausible numbers and a wrong model.
        """
        expected = ((self.hc_mult + 1) * self.hidden_size, self.engram_hidden_size)
        if tuple(wkv.shape) != expected:
            raise ValueError(f"wkv is {tuple(wkv.shape)}, expected {expected}")
        for name, tensor in (("k_weight", k_weight), ("q_weight", q_weight)):
            if tuple(tensor.shape) != (self.hc_mult, self.hidden_size):
                raise ValueError(
                    f"{name} is {tuple(tensor.shape)}, expected "
                    f"{(self.hc_mult, self.hidden_size)}"
                )
        native_scale = getattr(self.wkv, "weight_scale", None)
        if native_scale is not None:
            if (
                wkv_scale is None
                or wkv.dtype != self.wkv.weight.dtype
                or wkv_scale.dtype != native_scale.dtype
                or wkv_scale.shape != native_scale.shape
            ):
                raise ValueError(
                    "Native Engram projection requires matching weight and scale layout"
                )
            self.wkv.weight.copy_(wkv)
            native_scale.copy_(wkv_scale)
            self.k_weight.copy_(k_weight)
            self.q_weight.copy_(q_weight)
            self.process_weights_after_loading()
            return
        if wkv_scale is not None:
            # `.float()` decodes 2**(code-127) only for a float8 E8M0 dtype; a
            # raw uint8 exponent-code table would multiply by ~127 instead. Fail
            # loud rather than silently mis-scale the projection.
            if not wkv_scale.is_floating_point():
                raise ValueError(
                    f"wkv scale must be a float8 (E8M0) dtype, got {wkv_scale.dtype}"
                )
            rows, cols = wkv.shape
            if tuple(wkv_scale.shape) != (rows // block, cols // block):
                raise ValueError(
                    f"wkv scale is {tuple(wkv_scale.shape)}, expected "
                    f"{(rows // block, cols // block)} for {block}x{block} blocks"
                )
            wkv = (
                wkv.float().reshape(rows // block, block, cols // block, block)
                * wkv_scale.float().reshape(rows // block, 1, cols // block, 1)
            ).reshape(rows, cols)
        self.wkv.weight.copy_(wkv.to(self.wkv.weight.dtype))
        self.k_weight.copy_(k_weight.to(self.k_weight.dtype))
        self.q_weight.copy_(q_weight.to(self.q_weight.dtype))
        self.process_weights_after_loading()
