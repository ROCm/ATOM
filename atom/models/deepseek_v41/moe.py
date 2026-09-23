# SPDX-License-Identifier: MIT
"""V4.1 routed experts: V4's MoE, unchanged.

The two models share this layer. Routing is `sqrtsoftplus` with a
selection-only per-expert bias, renormalized top-k and a `routed_scaling_factor`;
the experts use the inherited V4 quantization, clamped SwiGLU and whole-expert
ownership. Activation format, routing-weight placement and kernel dispatch
are owned by V4/FusedMoE. `DeepseekV4Args`
reads all of it off the V4.1 config by its HF names, so V4's `MoE` constructs
directly. vLLM's ROCm V4.1 reuses the V4 MoE the same way.

`bias_vl` is the one V4.1-only tensor: a second routing bias for image sentinel
tokens. It is declared so the checkpoint loads and left unread, matching the
text-only admission.

The shared expert stays its own module rather than taking a routed slot: both
are scaled per 1x32, but the shared one is FP8 where the routed ones are FP4,
and one buffer holds one dtype. V4's checkpoints are the same pair, so there is
no fused variant to inherit.
"""

import torch

from atom.model_ops.utils import atom_parameter
from atom.models.deepseek_v4 import DeepseekV4Args
from atom.models.deepseek_v4 import MoE as V4MoE


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

    def forward(self, hidden, image_mask=None):
        if image_mask is not None:
            raise NotImplementedError(
                "V4.1 image-sentinel routing is not part of the text-only admission"
            )
        flat = hidden.reshape(-1, hidden.shape[-1])
        return super().forward(flat).view_as(hidden)
