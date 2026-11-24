# SPDX-License-Identifier: MIT
"""V4.1 routed experts: V4's MoE with image-specific routing bias.

The two models share this layer. Routing is `sqrtsoftplus` with a
selection-only per-expert bias, renormalized top-k and a `routed_scaling_factor`;
the experts use the inherited V4 quantization, clamped SwiGLU and whole-expert
ownership. The routed FP4 experts take MXFP8 activations, as the checkpoint's
reference runs them: aiter's gfx950 default for them is MXFP4, which put the
routed output ~12% off the reference and cost ~1.4pp of GSM8K. Activation
format, routing-weight placement and kernel dispatch are owned by V4/FusedMoE.
`DeepseekV4Args` reads all of it off the V4.1 config by its HF names, so V4's
`MoE` constructs directly. vLLM's ROCm V4.1 reuses the V4 MoE the same way.

The gate's GEMM at decode widths (`router_supported`) is the fixed-order router
GEMV of `atom.model_ops.deepseek_v41.router`, the one the mono decode's MoE
kernel runs, so the two decode paths route alike bit for bit; other widths and
shapes keep the gate's own GEMM.

`bias_vl` is the one V4.1-only tensor: a second routing bias for image sentinel
tokens. Mixed image/text prefills use the shared fused two-bias router;
expert dispatch, quantization, shared overlap and combination stay in V4.

The shared expert is its own module unless it has the routed experts' format:
an FP4PTPC checkpoint's per-channel FP8 one is fused into the routed experts
that online quantization makes per-channel FP8 (FusedMoE's fused shared slot).
"""

import torch
from aiter.jit.utils.torch_guard import torch_compile_guard

from atom.model_ops.deepseek_v41.router import (
    router_logits as fixed_order_router_logits,
)
from atom.model_ops.deepseek_v41.router import router_supported
from atom.model_ops.moe import FusedMoE
from atom.model_ops.topK import mm_topk, topk_meta_rows
from atom.model_ops.utils import atom_parameter
from atom.models.deepseek_v4 import DeepseekV4Args
from atom.models.deepseek_v4 import MoE as V4MoE
from atom.models.deepseek_v41.config import FP4PTPC_RECIPE
from atom.utils.forward_context import get_forward_context, pass_scheduled_tokens_gpu

from . import seg_fuse


@torch_compile_guard(mutates_args=["output"], gen_fake=lambda output: None)
def v41_record_tbo_expert_output(output: torch.Tensor) -> None:
    # Keep this runtime stream dependency in compiled execution as well.
    from atom.utils.tbo.ubatching import tbo_active

    if tbo_active():
        output.record_stream(torch.cuda.current_stream())


class MoE(V4MoE):
    # ATOM_DSV41_SEG_FUSE (default empty: every field keeps the value here and
    # the forward is the path it was; see `seg_fuse`). Class defaults, so a
    # layer assembled without `__init__` has them too.
    # Item 1: `gate_up_proj` reads the FFN seam's MXFP8 pair.
    seg_fuse_input_pair = False
    # Item 7: SiLU-mul + `w2`'s input quant in one launch (the op, or None).
    _seg_fuse_act_op = None
    # Item 10: `w2` and the top-k selection in one launch (the op, or None),
    # and the selection (logits, weights, ids) it made for this forward's
    # routed call.
    _seg_fuse_tail_op = None
    _seg_fuse_sel = None

    def __init__(
        self,
        config,
        layer_id: int,
        prefix: str = "",
        *,
        quant_config,
        alt_stream: torch.cuda.Stream | None = None,
    ):
        args = DeepseekV4Args.from_hf_config(config)
        args.quant_config = quant_config
        args.moe_fp8_activations = True
        super().__init__(layer_id, args, prefix=prefix, alt_stream=alt_stream)
        if config.fp4ptpc and not self._fuse_shared_into_routed:
            # Standalone, the per-channel FP8 shared expert has no GEMM at TP4.
            raise ValueError(f"{prefix}: {FP4PTPC_RECIPE}")
        self.gate.bias_vl = atom_parameter(
            torch.empty(args.n_routed_experts, dtype=torch.float32)
        )

        self.experts.custom_routing_function = self._topk
        if seg_fuse.ITEMS:
            self._init_seg_fuse(layer_id, getattr(config, "num_hidden_layers", None))

    @property
    def seg_fuse_act_quant(self) -> bool:
        return self._seg_fuse_act_op is not None

    def _init_seg_fuse(self, layer_id: int, layers: int | None):
        """Which `ATOM_DSV41_SEG_FUSE` items this layer's shared expert takes
        where it runs on the layer's own stream (`single_stream_moe_forward`):
        its two GEMMs must be native MXFP8, its activation the unfused one."""
        shared = self.shared_experts
        if (
            shared is not None
            and not shared.use_fused_clamp_act_mul
            and shared.gate_up_proj.native_a8_group_rows is not None
            and shared.w2.native_a8_group_rows is not None
            and not shared.gate_up_proj.runs_bf16
            and not shared.w2.runs_bf16
        ):
            # The seam's pair comes in through `forward`, which hands only the
            # bf16 rows to the dual-stream / comm-fused op.
            self.seg_fuse_input_pair = seg_fuse.enabled(1) and not (
                self._use_dual_stream or self._use_comm_fused_dispatch
            )
            if seg_fuse.enabled(7):
                self._seg_fuse_act_op = seg_fuse.act_quant_op(shared.w2.input_size)
        if seg_fuse.enabled(10) and seg_fuse.tail_layer(self, layer_id, layers):
            from atom.model_ops.deepseek_v41.seg_fuse_ops import v41_down_topk

            self._seg_fuse_tail_op = v41_down_topk

    def _seg_fuse_shared(self, x, x_fp8, x_scale):
        """`Expert.forward` of the shared expert with the fused launches."""
        shared = self.shared_experts
        if x_fp8 is None:
            combined = shared.gate_up_proj(x)
        else:
            combined = shared.gate_up_proj(x_fp8, x_scale=x_scale)
        if self._seg_fuse_act_op is None:
            return shared.act_down(combined, dtype=x.dtype)
        # Item 10: the router GEMM goes in front of the activation, and `w2`
        # carries the selection in its launch. Not on an image step (two-bias
        # routing) nor on a `w2` that is not preshuffled.
        tail = (
            self._seg_fuse_tail_op is not None
            and getattr(shared.w2.weight, "is_shuffled", False)
            and getattr(get_forward_context().attn_metadata, "image_mask", None) is None
        )
        if tail:
            logits = self.router_logits(x)
        act_fp8, act_scale = self._seg_fuse_act_op(combined, shared.swiglu_limit)
        if not tail:
            return shared.w2(act_fp8, x_scale=act_scale)
        out, weights, ids = self._seg_fuse_tail_op(
            act_fp8,
            act_scale,
            shared.w2.weight,
            shared.w2.weight_scale,
            logits,
            self.gate.e_score_correction_bias,
            self.experts.top_k,
            self.experts.renormalize,
            self.routed_scaling_factor,
        )
        self._seg_fuse_sel = (logits, weights, ids)
        return out

    def single_stream_moe_forward(self, x, x_fp8=None, x_scale=None):
        """V4's, with `_seg_fuse_shared` as its shared expert where an
        `ATOM_DSV41_SEG_FUSE` item applies; `x_fp8` / `x_scale` are the FFN
        seam's MXFP8 pair of `x`, or None. Every single-stream caller lands
        here: `forward`, and the dual-stream op above its row threshold."""
        if x_fp8 is None and self._seg_fuse_act_op is None:
            return super().single_stream_moe_forward(x)
        return self._single_stream_with_shared(
            x, self._seg_fuse_shared(x, x_fp8, x_scale)
        )

    def _topk(self, hidden_states, gating_output, topk, renormalize):
        sel = self._seg_fuse_sel
        if (
            sel is not None
            and gating_output.data_ptr() == sel[0].data_ptr()
            and gating_output.shape == sel[0].shape
            and topk == sel[2].shape[1]
        ):
            # Item 10: these logits were selected in `w2`'s launch.
            return sel[1], sel[2]
        image_mask = getattr(get_forward_context().attn_metadata, "image_mask", None)
        num_shared = self.experts.num_fused_shared_experts
        if image_mask is None:
            return FusedMoE.select_experts(
                hidden_states=hidden_states,
                router_logits=gating_output,
                top_k=topk,
                use_grouped_topk=False,
                renormalize=renormalize,
                scoring_func="sqrtsoftplus",
                e_score_correction_bias=self.gate.e_score_correction_bias,
                num_fused_shared_experts=num_shared,
                routed_scaling_factor=self.routed_scaling_factor,
            )
        rows = gating_output.shape[0]
        if num_shared:
            # Routed columns first; the shared ones are prefilled.
            weights, ids = topk_meta_rows(rows)
        else:
            ids = torch.empty(
                (rows, topk), dtype=torch.int32, device=gating_output.device
            )
            weights = torch.empty(
                (rows, topk), dtype=torch.float32, device=gating_output.device
            )
        mm_topk(
            ids=None,
            gating_output=gating_output,
            bias=self.gate.e_score_correction_bias,
            bias_alt=self.gate.bias_vl,
            hash_table=None,
            vocab_size=0,
            renormalize=renormalize,
            scaling=self.routed_scaling_factor,
            out_ids=ids[:, :topk],
            out_weights=weights[:, :topk],
            image_mask=image_mask.flatten(),
            num_valid_rows=pass_scheduled_tokens_gpu(rows),
        )
        return weights, ids

    def router_logits(self, x: torch.Tensor) -> torch.Tensor:
        """The gate: at decode widths the fixed-order router GEMV that the mono
        decode's MoE kernel runs too, else the gate's own GEMM."""
        if router_supported(x, self.gate.weight):
            return fixed_order_router_logits(x, self.gate.weight)
        return self.gate(x)

    def routed_expert_forward(
        self,
        x: torch.Tensor,
        shared_partial: torch.Tensor | None = None,
        before_stage2_for_rows=None,
        before_shared_add=None,
        stage2_stream: torch.cuda.Stream | None = None,
        shared_ready: torch.cuda.Event | None = None,
    ) -> tuple[torch.Tensor, bool]:
        if self._seg_fuse_sel is None:
            routed, is_complete = self.experts.forward_maybe_comm_fused(
                x,
                self.router_logits(x),
                shared_partial,
                before_stage2_for_rows=before_stage2_for_rows,
                before_shared_add=before_shared_add,
                stage2_stream=stage2_stream,
                shared_ready=shared_ready,
            )
        else:
            # ATOM_DSV41_SEG_FUSE item 10: the logits `_seg_fuse_shared` made
            # (and selected: `_topk` hands the selection over). Cleared
            # whatever happens, so no later forward can meet a stale one.
            try:
                routed, is_complete = self.experts.forward_maybe_comm_fused(
                    x,
                    self._seg_fuse_sel[0],
                    shared_partial,
                    before_stage2_for_rows=before_stage2_for_rows,
                    before_shared_add=before_shared_add,
                    stage2_stream=stage2_stream,
                    shared_ready=shared_ready,
                )
            finally:
                self._seg_fuse_sel = None
        # create_comm_fused_moe_backend excludes TBO (and DP > 1), so TBO
        # returns the fallback routed output before shared combine/mHC.
        # This marker protects consumers after dispatch returns; it cannot
        # fence an internal combine in a backend returning is_complete=True.
        v41_record_tbo_expert_output(routed)
        return routed, is_complete

    def forward(self, hidden, hidden_fp8=None, hidden_scale=None):
        if hidden_fp8 is not None:
            # ATOM_DSV41_SEG_FUSE item 1 (`seg_fuse_input_pair`: never the
            # dual-stream / comm-fused op): the pair goes with its rows.
            return self.single_stream_moe_forward(
                hidden.reshape(-1, hidden.shape[-1]),
                hidden_fp8.reshape(-1, hidden_fp8.shape[-1]),
                hidden_scale.reshape(-1, hidden_scale.shape[-1]),
            ).view_as(hidden)
        return super().forward(hidden.reshape(-1, hidden.shape[-1])).view_as(hidden)
