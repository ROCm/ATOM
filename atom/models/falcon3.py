# SPDX-License-Identifier: Apache-2.0
"""Falcon3 model support.

Falcon3 (tiiuae/Falcon3-*) uses the Llama architecture internally
(architectures = ["LlamaForCausalLM"], model_type = "llama").
This module provides a thin alias so that ATOM can also be loaded
with an explicit "Falcon3ForCausalLM" architecture key if future
checkpoints adopt one.
"""

from atom.models.llama import LlamaForCausalLM


class Falcon3ForCausalLM(LlamaForCausalLM):
    """Falcon3 reuses the Llama architecture unchanged."""

    pass
