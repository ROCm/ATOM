# SPDX-License-Identifier: MIT
"""V4.1 routed experts: V4's MoE with image-specific routing bias.

The two models share this layer. Routing is `sqrtsoftplus` with a
selection-only per-expert bias, renormalized top-k and a `routed_scaling_factor`;
the experts use the inherited V4 quantization, clamped SwiGLU and whole-expert
ownership. Activation format, routing-weight placement and kernel dispatch
are owned by V4/FusedMoE. `DeepseekV4Args`
reads all of it off the V4.1 config by its HF names, so V4's `MoE` constructs
directly. vLLM's ROCm V4.1 reuses the V4 MoE the same way.

`bias_vl` is the one V4.1-only tensor: a second routing bias for image sentinel
tokens. Mixed image/text prefills use the shared fused two-bias router;
expert dispatch, quantization, shared overlap and combination stay in V4.

The shared expert stays its own module rather than taking a routed slot: both
are scaled per 1x32, but the shared one is FP8 where the routed ones are FP4,
and one buffer holds one dtype. V4's checkpoints are the same pair, so there is
no fused variant to inherit.
"""

import torch

from atom.model_ops.moe import FusedMoE
from atom.model_ops.topK import mm_topk
from atom.model_ops.utils import atom_parameter
from atom.models.deepseek_v4 import DeepseekV4Args
from atom.models.deepseek_v4 import MoE as V4MoE
from atom.utils.forward_context import get_forward_context


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
        super().__init__(layer_id, args, prefix=prefix, alt_stream=alt_stream)
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
        )
        return weights, ids

    def forward(self, hidden):
        return super().forward(hidden.reshape(-1, hidden.shape[-1])).view_as(hidden)
