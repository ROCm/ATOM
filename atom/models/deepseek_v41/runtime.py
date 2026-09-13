# SPDX-License-Identifier: MIT
"""ModelRunner interface over the accepted V4.1 text backbone."""

import torch
import torch.nn.functional as F

from atom.utils.forward_context import get_forward_context

from .model import DeepseekV41ForCausalLM


class DeepseekV41RuntimeModel(DeepseekV41ForCausalLM):
    checkpoint_loader = "atom.model_loader.deepseek_v41.load_checkpoint"

    def __init__(self, config):
        super().__init__(config.hf_config, max_length=config.max_model_len)

    @torch.inference_mode()
    def forward(self, input_ids, positions):
        metadata = get_forward_context().attn_metadata
        step = metadata.step
        if step.length:
            hidden = self.forward_hidden(
                input_ids[: step.length].unsqueeze(0),
                metadata.cache,
                step,
                metadata.engram_embeddings,
            ).squeeze(0)
            metadata.cache.finish_step(step, metadata.next_histories)
        else:
            hidden = self.embed.weight.new_empty((0, self.config.hidden_size))
        return F.pad(hidden, (0, 0, 0, input_ids.numel() - step.length))

    def compute_logits(self, hidden):
        context = get_forward_context()
        if context.context.is_prefill:
            hidden = hidden[context.attn_metadata.cu_seqlens_q[1:].long() - 1]
        return self.head(self.norm(hidden))
