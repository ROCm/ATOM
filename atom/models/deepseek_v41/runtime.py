# SPDX-License-Identifier: MIT
"""ModelRunner interface over the accepted V4.1 text backbone."""

from functools import partial

import torch
import torch.nn.functional as F

from atom.config import CUDAGraphMode
from atom.utils.forward_context import get_forward_context

from .execution import DenseGraphExecutor
from .multimodal import DeepseekV41MultimodalModel


class DeepseekV41RuntimeModel(DeepseekV41MultimodalModel):
    # No `checkpoint_loader`: V4.1 loads through the shared path, the way V4
    # does, so the renames, the packed projections and the expert mapping are
    # declared once as tables on `DeepseekV41ForCausalLM` and inherited here
    # rather than restated per model. Engram's mmap tables still come from
    # `model_loader.deepseek_v41.engram_tables`, which `engram_runtime` imports
    # directly and does not route through here.

    def __init__(self, config):
        super().__init__(
            config.hf_config,
            max_length=config.max_model_len,
            online_quant_config=config.online_quant_config,
        )
        self.dense_graphs = None if config.enforce_eager else DenseGraphExecutor()

    @torch.inference_mode()
    def forward(self, input_ids, positions, inputs_embeds=None):
        context = get_forward_context()
        metadata = context.attn_metadata
        execution = None
        if (
            self.dense_graphs is not None
            and context.cudagraph_runtime_mode == CUDAGraphMode.PIECEWISE
        ):
            execution = partial(
                self.dense_graphs.run,
                bucket=context.batch_descriptor.num_tokens,
                capture=context.in_hipgraph,
            )
        step = metadata.step
        if step.length:
            hidden = self.forward_hidden(
                input_ids[: step.length].unsqueeze(0),
                metadata.cache,
                step,
                metadata.engram_embeddings,
                execution=execution,
                inputs_embeds=(
                    None
                    if inputs_embeds is None
                    else inputs_embeds[: step.length].unsqueeze(0)
                ),
                image_mask=metadata.image_mask,
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
