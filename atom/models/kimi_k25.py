# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

"""Inference-only Kimi-K2.5 model (text-only backbone).

Kimi-K2.5 is a multimodal model whose language backbone is a DeepseekV3-style
MoE transformer with MLA attention.  For text-only serving we load only the
``language_model.*`` weights and delegate to the existing
:class:`DeepseekV2ForCausalLM` implementation.

Vision encoder and multimodal projector weights are skipped during loading
via :pyattr:`skip_weight_prefixes`.
"""

from typing import Optional, Union

import torch
from torch import nn

from atom.config import Config
from atom.models.deepseek_v2 import DeepseekV2ForCausalLM
from atom.models.utils import IntermediateTensors


class KimiK25ForCausalLM(nn.Module):
    """Kimi-K2.5 text-only wrapper around :class:`DeepseekV2ForCausalLM`.

    The HuggingFace checkpoint stores the LLM weights under the
    ``language_model.*`` prefix.  By placing the underlying model as
    ``self.language_model``, PyTorch's parameter naming automatically
    matches the checkpoint layout so no explicit prefix stripping is needed.

    Vision tower and multimodal projector weights are excluded via
    :pyattr:`skip_weight_prefixes` which the model loader respects.
    """

    # Weight prefixes that should be silently skipped during loading
    # (these belong to the vision encoder / MM projector that we don't use).
    skip_weight_prefixes = [
        "vision_tower.",
        "mm_projector.",
    ]

    def __init__(
        self,
        atom_config: Config,
        prefix: str = "",
    ):
        super().__init__()
        self.config = atom_config.hf_config

        # The underlying LLM – named ``language_model`` so that its parameter
        # names match the ``language_model.*`` keys in the checkpoint.
        self.language_model = DeepseekV2ForCausalLM(
            atom_config=atom_config,
            prefix="",
        )

        self.make_empty_intermediate_tensors = (
            self.language_model.make_empty_intermediate_tensors
        )

    # ---- properties forwarded to the inner model ----

    @property
    def packed_modules_mapping(self):
        return self.language_model.packed_modules_mapping

    # ---- forward / inference API ----

    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.language_model.get_input_embeddings(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, IntermediateTensors]:
        return self.language_model(
            input_ids, positions, intermediate_tensors, inputs_embeds
        )

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        return self.language_model.compute_logits(hidden_states)

    def get_expert_mapping(self) -> list[tuple[str, str, int, str]]:
        return self.language_model.get_expert_mapping()
