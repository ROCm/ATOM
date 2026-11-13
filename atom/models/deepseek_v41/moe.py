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

The shared expert stays its own module rather than taking a routed slot: both
are scaled per 1x32, but the shared one is FP8 where the routed ones are FP4,
and one buffer holds one dtype. V4's checkpoints are the same pair, so there is
no fused variant to inherit. With per-channel FP8 routed experts,
`_SharedExpertFold` folds it in after loading.
"""

import weakref

import torch
import triton
import triton.language as tl
from aiter import QuantType, dtypes, get_hip_quant
from aiter.jit.utils.torch_guard import torch_compile_guard
from aiter.ops.shuffle import shuffle_weight
from torch import nn

from atom.model_ops.deepseek_v41.router import (
    router_logits as fixed_order_router_logits,
)
from atom.model_ops.deepseek_v41.router import router_supported
from atom.model_ops.moe import FusedMoE
from atom.model_ops.topK import mm_topk
from atom.model_ops.utils import atom_parameter
from atom.models.deepseek_v4 import DeepseekV4Args
from atom.models.deepseek_v4 import MoE as V4MoE
from atom.utils.forward_context import get_forward_context, pass_scheduled_tokens_gpu


@triton.jit
def _append_shared_kernel(
    IDS, WTS, IDS_OUT, WTS_OUT, stride_i, stride_w,
    TOPK: tl.constexpr, SHARED: tl.constexpr,
):  # fmt: skip
    # The row's routed top-k, then the folded shared expert at weight 1.
    row = tl.program_id(0).to(tl.int64)
    KP: tl.constexpr = triton.next_power_of_2(TOPK + 1)
    k = tl.arange(0, KP)
    routed = k < TOPK
    ids = tl.load(IDS + row * stride_i + k, mask=routed, other=SHARED)
    wts = tl.load(WTS + row * stride_w + k, mask=routed, other=1.0)
    tl.store(IDS_OUT + row * (TOPK + 1) + k, ids, mask=k <= TOPK)
    tl.store(WTS_OUT + row * (TOPK + 1) + k, wts, mask=k <= TOPK)


def _with_shared_slot(weights, ids, shared):
    """Append expert `shared` at weight 1 to each row's top-k: [M, topk + 1]."""
    rows, topk = ids.shape
    out_ids = torch.empty(rows, topk + 1, dtype=torch.int32, device=ids.device)
    out_wts = torch.empty(rows, topk + 1, dtype=torch.float32, device=ids.device)
    if rows:
        _append_shared_kernel[(rows,)](
            ids, weights, out_ids, out_wts, ids.stride(0), weights.stride(0),
            TOPK=topk, SHARED=shared, num_warps=1,
        )  # fmt: skip
    return out_wts, out_ids


class _SharedExpertFold(nn.Module):
    """Fold the shared expert into the routed experts' FP8 buffers as expert E,
    taken by every token at weight 1: one MoE instead of four launches.

    Registered after `shared_experts`, so post-loading reaches it once both
    are processed.
    """

    def __init__(self, moe):
        super().__init__()
        object.__setattr__(self, "_moe", weakref.ref(moe))

    def process_weights_after_loading(self):
        moe = self._moe()
        experts, shared = moe.experts, moe.shared_experts
        if shared is None or not getattr(experts.quant_method, "ptpc_clamp", False):
            return
        w13, w2 = shared.gate_up_proj.weight.data, shared.w2.weight.data
        num_experts, n1, hidden = experts.w13_weight.shape
        if (
            w13.dtype != torch.bfloat16
            or w2.dtype != torch.bfloat16
            or tuple(w13.shape) != (n1, hidden)
            or tuple(w2.shape) != tuple(experts.w2_weight.shape[1:])
        ):
            return
        quant = get_hip_quant(QuantType.per_Token)
        q13, s13 = quant(w13, quant_dtype=dtypes.fp8)
        q2, s2 = quant(w2, quant_dtype=dtypes.fp8)
        assert q13.dtype == experts.w13_weight.dtype == experts.w2_weight.dtype
        # Same per-expert (16, 16) preshuffle as the routed experts.
        shuffled = getattr(experts.w13_weight, "is_shuffled", False)
        if shuffled:
            q13 = shuffle_weight(q13, layout=(16, 16))
            q2 = shuffle_weight(q2, layout=(16, 16))
        experts.w13_weight = atom_parameter(
            torch.cat([experts.w13_weight.data, q13[None]])
        )
        experts.w13_weight_scale = atom_parameter(
            torch.cat([experts.w13_weight_scale.data, s13.reshape(1, -1).float()])
        )
        experts.w2_weight = atom_parameter(
            torch.cat([experts.w2_weight.data, q2[None]])
        )
        experts.w13_weight.is_shuffled = experts.w2_weight.is_shuffled = shuffled
        experts.w2_weight_scale = atom_parameter(
            torch.cat([experts.w2_weight_scale.data, s2.reshape(1, -1).float()])
        )
        # Every row takes it as one more top-k slot.
        route = experts.custom_routing_function

        def route_with_shared(**kwargs):
            return _with_shared_slot(*route(**kwargs), num_experts)

        experts.custom_routing_function = route_with_shared
        moe.shared_experts = None
        # Nothing left to overlap on `alt_stream`.
        moe._use_dual_stream = False


@torch_compile_guard(mutates_args=["output"], gen_fake=lambda output: None)
def v41_record_tbo_expert_output(output: torch.Tensor) -> None:
    # Keep this runtime stream dependency in compiled execution as well.
    from atom.utils.tbo.ubatching import tbo_active

    if tbo_active():
        output.record_stream(torch.cuda.current_stream())


class MoE(V4MoE):
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
        # After `shared_experts` (see `_SharedExpertFold`).
        if getattr(self, "shared_experts", None) is not None:
            self.shared_fold = _SharedExpertFold(self)
        self.gate.bias_vl = atom_parameter(
            torch.empty(args.n_routed_experts, dtype=torch.float32)
        )

        self.experts.custom_routing_function = self._topk

    def _topk(self, hidden_states, gating_output, topk, renormalize):
        image_mask = getattr(get_forward_context().attn_metadata, "image_mask", None)
        if image_mask is None:
            return FusedMoE.select_experts(
                hidden_states=hidden_states,
                router_logits=gating_output,
                top_k=topk,
                use_grouped_topk=False,
                renormalize=renormalize,
                scoring_func="sqrtsoftplus",
                e_score_correction_bias=self.gate.e_score_correction_bias,
                routed_scaling_factor=self.routed_scaling_factor,
            )
        rows = gating_output.shape[0]
        ids = torch.empty((rows, topk), dtype=torch.int32, device=gating_output.device)
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
            out_ids=ids,
            out_weights=weights,
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
    ) -> tuple[torch.Tensor, bool]:
        routed, is_complete = self.experts.forward_maybe_comm_fused(
            x,
            self.router_logits(x),
            shared_partial,
            before_stage2_for_rows=before_stage2_for_rows,
            before_shared_add=before_shared_add,
            stage2_stream=stage2_stream,
        )
        # create_comm_fused_moe_backend excludes TBO (and DP > 1), so TBO
        # returns the fallback routed output before shared combine/mHC.
        # This marker protects consumers after dispatch returns; it cannot
        # fence an internal combine in a backend returning is_complete=True.
        v41_record_tbo_expert_output(routed)
        return routed, is_complete

    def forward(self, hidden):
        return super().forward(hidden.reshape(-1, hidden.shape[-1])).view_as(hidden)
