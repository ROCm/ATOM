# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Copyright 2025 Google Inc. and The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Gemma 4 text model configuration."""

from transformers.configuration_utils import PretrainedConfig


class Gemma4TextConfig(PretrainedConfig):
    """Configuration for the text-only component of Gemma 4.

    This config corresponds to the ``text_config`` sub-dict inside the root
    ``gemma4`` config.json.  ATOM extracts it as the primary HF config for
    text-only inference.
    """

    model_type = "gemma4_text"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        vocab_size: int = 262144,
        hidden_size: int = 5376,
        intermediate_size: int = 21504,
        num_hidden_layers: int = 60,
        num_attention_heads: int = 32,
        # Sliding-window (local) attention KV heads
        num_key_value_heads: int = 16,
        # Full (global) attention KV heads
        num_global_key_value_heads: int = 4,
        # Sliding-window attention head dim
        head_dim: int = 256,
        # Full attention head dim
        global_head_dim: int = 512,
        hidden_activation: str = "gelu_pytorch_tanh",
        max_position_embeddings: int = 262144,
        initializer_range: float = 0.02,
        rms_norm_eps: float = 1e-6,
        use_cache: bool = True,
        tie_word_embeddings: bool = True,
        # rope_parameters is a dict keyed by layer type:
        #   {"full_attention": {...}, "sliding_attention": {...}}
        rope_parameters: dict | None = None,
        attention_bias: bool = False,
        attention_dropout: float = 0.0,
        # Per-layer attention type list; length must equal num_hidden_layers.
        layer_types: list | None = None,
        sliding_window: int = 1024,
        final_logit_softcapping: float = 30.0,
        pad_token_id: int = 0,
        bos_token_id: int = 2,
        eos_token_id: int = 1,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.num_global_key_value_heads = num_global_key_value_heads
        self.head_dim = head_dim
        self.global_head_dim = global_head_dim
        self.hidden_activation = hidden_activation
        self.max_position_embeddings = max_position_embeddings
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        self.sliding_window = sliding_window
        self.final_logit_softcapping = final_logit_softcapping
        self.rope_parameters = rope_parameters or {}

        self.layer_types = layer_types
        if self.layer_types is None:
            # Default pattern from Gemma 4 31B: 5 sliding + 1 full, repeated.
            self.layer_types = [
                "full_attention" if (i + 1) % 6 == 0 else "sliding_attention"
                for i in range(self.num_hidden_layers)
            ]

        super().__init__(**kwargs)
        # Set token IDs AFTER super().__init__() to prevent transformers v4
        # PretrainedConfig from overwriting them with its own defaults.
        self.pad_token_id = pad_token_id
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.tie_word_embeddings = tie_word_embeddings


__all__ = ["Gemma4TextConfig"]
