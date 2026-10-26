# SPDX-License-Identifier: MIT
"""Engram projection and FP32 residual gating, derived from PR #2185.

Checkpoint I/O, table lookups and request staging are owned by separate
modules.
"""

import torch
from aiter.dist.parallel_state import get_tp_group
from torch import nn

from atom.config import QuantizationConfig
from atom.model_ops.engram.device.gate import engram_post_wkv, engram_post_wkv_tp
from atom.model_ops.linear import ReplicatedLinear
from atom.model_ops.utils import atom_parameter


class _EngramWKV(ReplicatedLinear):
    """Keep complete local key branches and the shared value in checkpoint order."""

    def __init__(self, input_size, hidden_size, hc_mult, quant_config=None):
        super().__init__(
            input_size, (hc_mult + 1) * hidden_size, quant_config=quant_config
        )
        self.hc_mult = self.local_hc = hc_mult
        self.hidden_size = hidden_size
        if (
            self.native_a8_group_rows == 32
            and self.weight.dtype == torch.float8_e4m3fn
            and self.tp_size in (2, 4)
            and hc_mult % self.tp_size == 0
            and hidden_size % 32 == 0
        ):
            self.local_hc = hc_mult // self.tp_size
            self.output_size = (self.local_hc + 1) * hidden_size
            self.output_partition_sizes = [self.output_size]
            # Preserve Parameter identities and their loader callbacks. Fresh
            # storage releases the full allocation, including on meta devices.
            self.weight.data = self.weight.new_empty((self.output_size, input_size))
            self.weight_scale.data = self.weight_scale.new_empty(
                (self.output_size // 32, input_size // 32)
            )

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        if self.local_hc != self.hc_mult:
            group = 32 if param is self.weight_scale else 1
            branch_rows = self.hidden_size // group
            expected = (
                (self.hc_mult + 1) * branch_rows,
                self.input_size // (32 if group == 32 else 1),
            )
            if tuple(loaded_weight.shape) != expected:
                name = "weight_scale" if param is self.weight_scale else "weight"
                raise ValueError(
                    f"Engram {name} is {tuple(loaded_weight.shape)}, expected "
                    f"full checkpoint shape {expected}"
                )
            rows = self.local_hc * branch_rows
            start = self.tp_rank * rows
            # Select before Linear's dtype/shape processing and before its
            # normal child post-load hook shuffles this compact local weight.
            loaded_weight = torch.cat(
                (
                    loaded_weight[start : start + rows],
                    loaded_weight[self.hc_mult * branch_rows :],
                )
            )
        super().weight_loader(param, loaded_weight)


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
        self.wkv = _EngramWKV(
            engram_hidden_size,
            hidden_size,
            hc_mult,
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
        if self.wkv.local_hc != self.hc_mult:
            return engram_post_wkv_tp(
                hidden_states,
                kv,
                self.gate_weight,
                get_tp_group(),
                token_mask,
                self.norm_eps,
            )
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
            group_rows = self.wkv.native_a8_group_rows
            expected_scale = (
                (expected[0] // group_rows, expected[1] // 32)
                if group_rows is not None
                else tuple(native_scale.shape)
            )
            if (
                wkv_scale is None
                or wkv.dtype != self.wkv.weight.dtype
                or wkv_scale.dtype != native_scale.dtype
                or tuple(wkv_scale.shape) != expected_scale
            ):
                raise ValueError(
                    "Native Engram projection requires matching weight and scale layout"
                )
            self.wkv.weight_loader(self.wkv.weight, wkv)
            self.wkv.weight_loader(native_scale, wkv_scale)
            self.k_weight.copy_(k_weight)
            self.q_weight.copy_(q_weight)
            self.process_weights_after_loading()
            # The production loader visits the linear child independently.
            # This direct helper owns that visit when loading without it.
            self.wkv.process_weights_after_loading()
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
