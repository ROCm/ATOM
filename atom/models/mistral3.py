# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Inference-only Mistral3 / Ministral 3 model (text path).

Architecture: `Mistral3ForConditionalGeneration` is the multimodal HF wrapper around
a Pixtral vision encoder + a Ministral text backbone. The text backbone is
architecturally identical to Llama (GQA, RMSNorm, RoPE, SwiGLU MLP), so we reuse
`atom.models.llama.LlamaForCausalLM` and add only the multimodal weight-mapping
glue needed to load `Mistral3ForConditionalGeneration` checkpoints text-only.
"""

import copy
from typing import Optional

import torch
from torch import nn

from atom.config import Config
from atom.models.llama import LlamaForCausalLM
from atom.models.utils import IntermediateTensors, PPMissingLayer


def _get_text_atom_config(atom_config: Config) -> Config:
    """Return an atom_config view whose hf_config is the inner text sub-config.

    The HF Mistral3Config wraps text_config (Ministral3) + vision_config (Pixtral).
    LlamaForCausalLM reads attributes off atom_config.hf_config directly
    (vocab_size, hidden_size, etc.), so we hand it the text sub-config.
    """
    if not hasattr(atom_config.hf_config, "text_config"):
        return atom_config
    text_atom_config = copy.copy(atom_config)
    text_atom_config.hf_config = atom_config.hf_config.text_config
    return text_atom_config


class Mistral3ForCausalLM(LlamaForCausalLM):
    """Text backbone of Mistral3 / Ministral 3. Same compute graph as Llama."""

    def __init__(self, atom_config: Config, prefix: str = ""):
        super().__init__(_get_text_atom_config(atom_config), prefix=prefix)


class Mistral3TextOnly(nn.Module):
    """Loads only the text path of a Mistral3ForConditionalGeneration checkpoint.

    The HF checkpoint stores text weights under model.language_model.* and
    vision weights under model.vision_tower.* / model.multi_modal_projector.*.
    The text weights are remapped to match our language_model.model.* layout;
    the vision and projector shards are skipped entirely.
    """

    packed_modules_mapping = LlamaForCausalLM.packed_modules_mapping

    # Mistral3 checkpoints store text weights flat under language_model.* (no
    # outer model. prefix), and our wrapper exposes the same path via
    # self.language_model.* — so no name rewriting is needed for the text path.
    weights_mapping = {}
    quant_exclude_name_mapping = {
        "language_model.": "",
    }
    skip_weight_prefixes = [
        "model.vision_tower.",
        "model.multi_modal_projector.",
        "vision_tower.",
        "multi_modal_projector.",
    ]

    def __init__(self, atom_config: Config, prefix: str = ""):
        super().__init__()
        self.config = atom_config.hf_config
        self.vision_tower = PPMissingLayer()
        self.multi_modal_projector = PPMissingLayer()
        self.language_model = Mistral3ForCausalLM(atom_config=atom_config, prefix="")
        self.make_empty_intermediate_tensors = (
            self.language_model.make_empty_intermediate_tensors
        )

    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.language_model.model.get_input_embeddings(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        **_: object,
    ):
        return self.language_model(
            input_ids, positions, intermediate_tensors, inputs_embeds
        )

    def compute_logits(self, hidden_states: torch.Tensor):
        return self.language_model.compute_logits(hidden_states)
