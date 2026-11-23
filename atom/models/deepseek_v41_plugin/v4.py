# SPDX-License-Identifier: MIT
"""The V4 MoE and shared expert as the plugin model builds them.

ATOM's own `atom.models.deepseek_v4` stays as it is. These subclasses add, for
the plugin model only:

- the shared expert's activation as the `v41.moe.shared_act` seam, with the
  act..down span one opaque op under a planned owner (ROCm/ATOM#3529);
- the shared expert reading the FFN mHC's MXFP8 pair for its gate_up_proj, and
  the one-launch act + quant (`model_options.ffn_mhc_fuse_quant` /
  `shared_act_quant`, ROCm/ATOM#3645). The fused act + quant is the seam's
  *native*, so a planned `v41.moe.shared_act` owner still serves first;
- a static expert placement (`model_options.expert_placement`) applied to the
  routed experts and the router rows alike;
- MegaMoE (`--moe-backend mega`) kept single-stream: its fused kernels hold
  every CTA resident while they wait on peers, and a side-stream shared expert
  can deadlock the group (ROCm/ATOM#3187).
"""

import logging
import os

import torch

import atom.models.deepseek_v4 as _v4
from atom import plugin_host
from atom.config import get_current_atom_config
from atom.distributed.ulysses_sp import sp_is_enabled
from atom.model_ops.deepseek_v41.shared_act import (
    shared_act_quant,
    shared_act_quant_enabled,
)

logger = logging.getLogger("atom")


class PluginExpert(_v4.Expert):
    """`deepseek_v4.Expert` with the shared-act seam and the MXFP8 input pair."""

    def _plugin_init(self, prefix: str) -> None:
        self.use_shared_act_quant = (
            not self.use_fused_clamp_act_mul
            and not self.w2.runs_bf16
            and shared_act_quant_enabled(
                self.w2,
                self.swiglu_limit,
                plugin_host.model_options().shared_act_quant,
            )
        )
        self._act_seam = plugin_host.bind(
            "v41.moe.shared_act", self._act, capability=self._act_capability
        )
        self.act_down = plugin_host.span_op(
            "v41.moe.shared_act", prefix, self._act_down, self._act_down_fake
        )
        # gate_up + act (+ quant) as one span; bound to the native itself
        # unless a plan (or a hot-plug boot) may own it.
        self._expert_seam = plugin_host.bind(
            "v41.moe.shared_expert",
            self._gate_up_act,
            capability=self._expert_capability,
        )

    def forward(self, x, weights=None, x_quant=None):
        # ``x_quant``: `quantize_fp8(x)`'s bytes from the FFN mHC; a group-32
        # gate_up_proj reads them instead of quantizing the rows again.
        if x_quant is not None and self.gate_up_proj.native_a8_group_rows is not None:
            combined = self.gate_up_proj(x_quant[0], x_scale=x_quant[1])
            return self.act_down(combined, weights)
        if not torch.compiler.is_compiling() and plugin_host.owns(
            "v41.moe.shared_expert"
        ):
            act, act_scale = self._expert_seam(
                x,
                weights,
                self.gate_up_proj.weight,
                getattr(self.gate_up_proj, "weight_scale", None),
            )
            return self.w2(act, x_scale=act_scale)
        return self.act_down(self.gate_up_proj(x), weights)

    def _gate_up_act(self, x, weights, w13_w, w13_ws):
        """`v41.moe.shared_expert`'s native: gate_up, then the act seam."""
        return self._act_seam(self.gate_up_proj(x), weights)

    def _expert_capability(self, x=None, weights=None, *rest, **kwargs):
        cap = self._act_capability(None, weights)
        w = self.gate_up_proj.weight
        cap["flags"] = {
            **cap["flags"],
            "gate_up_n": int(w.shape[0]),
            "gate_up_k": int(w.shape[-1]),
            "gate_up_group_rows": self.gate_up_proj.native_a8_group_rows,
            "module": self,
        }
        return cap

    def _act_down(self, combined, weights):
        act, act_scale = self._act_seam(combined, weights)
        return self.w2(act, x_scale=act_scale)

    def _act_down_fake(self, combined, weights):
        return combined.new_empty(
            (combined.shape[0], self.w2.output_size), dtype=_v4.dtypes.bf16
        )

    def _act_capability(self, combined=None, weights=None, **kwargs):
        """What a `v41.moe.shared_act` plugin may rely on."""
        group32 = self.w2.native_a8_group_rows == 32
        fused = self.use_fused_clamp_act_mul or (
            self.use_shared_act_quant and weights is None
        )
        return {
            "quant": plugin_host.MX if group32 else None,
            "flags": {"swiglu_limit": self.swiglu_limit, "native_fused_act": fused},
        }

    def _act(self, combined, weights):
        """w2's input: the quantized pair when fused, else bf16 with no scale."""
        if self.use_shared_act_quant and weights is None:
            return shared_act_quant(combined, self.swiglu_limit)
        dtype = combined.dtype
        scale_kwargs = _v4.fused_scale_kwargs(
            self.w2.native_a8_group_rows,
            self.w2.quant_type.value == _v4.QuantType.per_1x128.value,
        )
        if self.use_fused_clamp_act_mul and scale_kwargs and not self.w2.runs_bf16:
            native_mx = self.w2.native_a8_group_rows is not None
            x_fp8, x_scale = _v4.fused_clamp_act_mul(
                combined,
                swiglu_limit=self.swiglu_limit,
                activation="silu",
                weights=weights,
                dtype_quant=_v4.dtypes.fp8,
                **scale_kwargs,
            )
            if native_mx:
                x_scale = x_scale.view(torch.float8_e8m0fnu)
            return x_fp8, x_scale
        out = torch.empty(
            (combined.shape[0], combined.shape[-1] // 2),
            dtype=dtype,
            device=combined.device,
        )
        _v4.aiter_silu_and_mul(out, combined, self.swiglu_limit)
        if weights is not None:
            out = weights.to(dtype) * out
        return out, None


class PluginV4MoE(_v4.MoE):
    """`deepseek_v4.MoE` with the plugin model's options (module docstring)."""

    def __init__(self, layer_id, args, prefix="", alt_stream=None):
        super().__init__(layer_id, args, prefix=prefix, alt_stream=alt_stream)
        # The FFN mHC's MXFP8 pair, parked by the block for the one call that
        # reads it (`Block._ffn_pair`); never read outside that call.
        self._pending_x_quant = None
        if self.shared_experts is not None:
            # Same parameters; only the forward and its seam differ.
            self.shared_experts.__class__ = PluginExpert
            self.shared_experts._plugin_init(f"{prefix}.shared_experts")
        if get_current_atom_config().moe_backend == "mega" and not sp_is_enabled():
            self._use_dual_stream = False
        options = plugin_host.model_options()
        self._static_placement_applied = False
        placement = options.placement_spec()
        if placement is not None:
            self.experts.set_static_placement(placement)

    def process_weights_after_loading(self):
        """Static expert placement: relabel the router's per-expert tensors the
        way FusedMoE relabeled the expert weights it loaded. The loader visits
        a parent before its children, so this runs on the gate's checkpoint
        layout."""
        perm = getattr(self.experts, "_static_placement", None)
        if perm is None or self._static_placement_applied:
            return
        n = len(perm)
        inverse = torch.empty(n, dtype=torch.long)
        inverse[torch.tensor(perm, dtype=torch.long)] = torch.arange(n)
        for name in ("weight", "e_score_correction_bias", "bias_vl"):
            tensor = getattr(self.gate, name, None)
            if tensor is None:
                continue
            data = tensor.data
            if data.shape[0] != n:
                raise RuntimeError(
                    f"{self.prefix}: gate.{name} has {data.shape[0]} rows, "
                    f"the placement has {n}"
                )
            data.copy_(data.index_select(0, inverse.to(data.device)))
        tid2eid = getattr(self.gate, "tid2eid", None)
        if tid2eid is not None:
            table = torch.tensor(perm, dtype=tid2eid.dtype, device=tid2eid.device)
            tid2eid.data.copy_(table[tid2eid.data.long()])
        self._static_placement_applied = True
        if self.layer_id == 0:
            moved = sum(1 for e, p in enumerate(perm) if e != p)
            logger.info(
                "[PLACEMENT] static expert placement (%s): layer 0 moves %d of %d experts",
                os.path.basename(str(plugin_host.model_options().expert_placement))[
                    :64
                ],
                moved,
                n,
            )

    def single_stream_moe_forward(self, x, x_quant=None):
        """Sequential: shared_experts -> routed_experts -> combine. ``x_quant``
        (or the pair the block parked for this call) reaches only the shared
        expert."""
        if x_quant is None:
            pending = self._pending_x_quant
            if pending is not None and pending[0].shape[:-1].numel() == x.shape[0]:
                rows = x.shape[0]
                x_quant = (pending[0].reshape(rows, -1), pending[1].reshape(rows, -1))
        shared = (
            self.shared_experts(x, x_quant=x_quant)
            if self.shared_experts is not None
            else None
        )
        routed, is_complete = self.routed_expert_forward(x, shared_partial=shared)
        if is_complete:
            return routed
        return self.combine_outputs(
            routed, shared, prefix=f"{self.prefix}.combine_outputs"
        )

    def forward(self, x, x_quant=None):
        assert (
            x.dim() == 2 and x.shape[-1] == self.dim
        ), f"MoE expects 2D [num_tokens, {self.dim}], got {tuple(x.shape)}"
        if self._use_dual_stream or self._use_comm_fused_dispatch:
            # The op carries one tensor: the shared expert quantizes `x` itself
            # (`accepts_x_quant` says so beforehand).
            return torch.ops.aiter.maybe_dual_stream_forward(x, self.prefix)
        return self.single_stream_moe_forward(x, x_quant)

    @property
    def accepts_x_quant(self) -> bool:
        """Whether `forward` uses an ``x_quant`` pair: a standalone shared
        expert whose gate_up_proj reads group-32 MXFP8, on the direct
        (single-stream, not comm-fused) path."""
        return (
            self.shared_experts is not None
            and self.shared_experts.gate_up_proj.native_a8_group_rows is not None
            and not (self._use_dual_stream or self._use_comm_fused_dispatch)
        )
