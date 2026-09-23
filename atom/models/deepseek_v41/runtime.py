# SPDX-License-Identifier: MIT
"""ModelRunner interface over the accepted V4.1 text backbone."""

import torch

from atom.utils.forward_context import get_forward_context

from .multimodal import DeepseekV41MultimodalModel


class DeepseekV41RuntimeModel(DeepseekV41MultimodalModel):
    # Weights arrive through the shared loader, the way V4's do, so the
    # renames, the packed projections and the expert mapping are declared once
    # as tables on `DeepseekV41ForCausalLM` and inherited here rather than
    # restated per model. Engram's mmap tables still come from
    # `model_loader.deepseek_v41.engram_tables`, which `engram_runtime` imports
    # directly and does not route through here.

    def __init__(self, config):
        super().__init__(
            config.hf_config,
            max_length=config.max_model_len,
            online_quant_config=config.online_quant_config,
        )

    @torch.inference_mode()
    def forward(self, input_ids, positions, inputs_embeds=None):
        """Pure tensor work: every row handed in, and no state of its own.

        Nothing here reads a Python request or writes one back, which is what
        lets the whole forward be one captured graph. The step's padding rows
        are computed like any other and dropped by the caller; making them a
        narrower forward is what a replay cannot do.
        """
        metadata = get_forward_context().attn_metadata
        step = metadata.step
        step.begin_forward()
        if not step.requests:
            # Nothing scheduled here, so no layer has anything to read or
            # write. The caller still wants its rows back.
            return self.embed.weight.new_zeros(
                (input_ids.numel(), self.config.hidden_size)
            )
        if input_ids.numel() != step.width:
            raise ValueError("Token rows disagree with the width this step declared")
        return self.forward_hidden(
            input_ids.unsqueeze(0),
            metadata.cache,
            step,
            metadata.engram_embeddings,
            inputs_embeds=(
                None if inputs_embeds is None else inputs_embeds.unsqueeze(0)
            ),
            image_mask=metadata.image_mask,
        ).squeeze(0)

    def compute_logits(self, hidden):
        return self.head.get_logits(self.norm(hidden))
