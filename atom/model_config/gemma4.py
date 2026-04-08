# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Google LLC and the ATOM contributors.
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
"""Gemma 4 model configuration for ATOM."""

from transformers.configuration_utils import PretrainedConfig


class Gemma4TextConfig(PretrainedConfig):
    model_type = "gemma4_text"
    keys_to_ignore_at_inference = ["past_key_values"]

    base_model_tp_plan = {
        "layers.*.self_attn.q_proj": "colwise",
        "layers.*.self_attn.k_proj": "colwise",
        "layers.*.self_attn.v_proj": "colwise",
        "layers.*.self_attn.o_proj": "rowwise",
        "layers.*.mlp.gate_proj": "colwise",
        "layers.*.mlp.up_proj": "colwise",
        "layers.*.mlp.down_proj": "rowwise",
    }
    base_config_key = "text_config"

    def __init__(
        self,
        vocab_size=262144,
        hidden_size=5376,
        intermediate_size=21504,
        num_hidden_layers=60,
        num_attention_heads=32,
        num_key_value_heads=16,
        num_global_key_value_heads=4,
        head_dim=256,
        global_head_dim=512,
        hidden_activation="gelu_pytorch_tanh",
        max_position_embeddings=262144,
        initializer_range=0.02,
        rms_norm_eps=1e-6,
        use_cache=True,
        tie_word_embeddings=True,
        rope_parameters=None,
        attention_bias=False,
        attention_dropout=0.0,
        attention_k_eq_v=True,
        sliding_window=1024,
        final_logit_softcapping=30.0,
        layer_types=None,
        use_bidirectional_attention="vision",
        enable_moe_block=False,
        num_experts=None,
        top_k_experts=None,
        moe_intermediate_size=None,
        use_double_wide_mlp=False,
        hidden_size_per_layer_input=0,
        vocab_size_per_layer_input=262144,
        num_kv_shared_layers=0,
        pad_token_id=0,
        bos_token_id=2,
        eos_token_id=1,
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
        self.attention_k_eq_v = attention_k_eq_v
        self.sliding_window = sliding_window
        self.final_logit_softcapping = final_logit_softcapping
        self.use_bidirectional_attention = use_bidirectional_attention
        self.rope_parameters = rope_parameters

        self.enable_moe_block = enable_moe_block
        self.num_experts = num_experts
        self.top_k_experts = top_k_experts
        self.moe_intermediate_size = moe_intermediate_size
        self.use_double_wide_mlp = use_double_wide_mlp
        self.hidden_size_per_layer_input = hidden_size_per_layer_input
        self.vocab_size_per_layer_input = vocab_size_per_layer_input
        self.num_kv_shared_layers = num_kv_shared_layers

        self.layer_types = layer_types
        if self.layer_types is None:
            self.layer_types = [
                (
                    "full_attention"
                    if (i + 1) % 6 == 0
                    else "sliding_attention"
                )
                for i in range(self.num_hidden_layers)
            ]

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )


class Gemma4VisionConfig(PretrainedConfig):
    model_type = "gemma4_vision"
    base_config_key = "vision_config"

    def __init__(
        self,
        hidden_size=1152,
        intermediate_size=4304,
        num_hidden_layers=27,
        num_attention_heads=16,
        num_key_value_heads=16,
        head_dim=72,
        global_head_dim=72,
        hidden_activation="gelu_pytorch_tanh",
        patch_size=16,
        pooling_kernel_size=3,
        position_embedding_size=10240,
        default_output_length=280,
        max_position_embeddings=131072,
        rms_norm_eps=1e-6,
        attention_bias=False,
        attention_dropout=0.0,
        initializer_range=0.02,
        standardize=True,
        use_clipped_linears=False,
        rope_parameters=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.global_head_dim = global_head_dim
        self.hidden_activation = hidden_activation
        self.patch_size = patch_size
        self.pooling_kernel_size = pooling_kernel_size
        self.position_embedding_size = position_embedding_size
        self.default_output_length = default_output_length
        self.max_position_embeddings = max_position_embeddings
        self.rms_norm_eps = rms_norm_eps
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        self.initializer_range = initializer_range
        self.standardize = standardize
        self.use_clipped_linears = use_clipped_linears
        self.rope_parameters = rope_parameters


class Gemma4Config(PretrainedConfig):
    model_type = "gemma4"
    sub_configs = {
        "text_config": Gemma4TextConfig,
        "vision_config": Gemma4VisionConfig,
    }
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        text_config=None,
        vision_config=None,
        audio_config=None,
        image_token_id=258880,
        video_token_id=258884,
        audio_token_id=258881,
        boi_token_id=255999,
        eoi_token_id=258882,
        boa_token_id=256000,
        eoa_token_id=258883,
        vision_soft_tokens_per_image=280,
        tie_word_embeddings=True,
        **kwargs,
    ):
        if isinstance(text_config, dict):
            self.text_config = self.sub_configs["text_config"](**text_config)
        elif text_config is None:
            self.text_config = self.sub_configs["text_config"]()
        else:
            self.text_config = text_config

        if isinstance(vision_config, dict):
            self.vision_config = self.sub_configs["vision_config"](**vision_config)
        elif vision_config is None:
            self.vision_config = self.sub_configs["vision_config"]()
        else:
            self.vision_config = vision_config

        self.audio_config = audio_config
        self.image_token_id = image_token_id
        self.video_token_id = video_token_id
        self.audio_token_id = audio_token_id
        self.boi_token_id = boi_token_id
        self.eoi_token_id = eoi_token_id
        self.boa_token_id = boa_token_id
        self.eoa_token_id = eoa_token_id
        self.vision_soft_tokens_per_image = vision_soft_tokens_per_image

        super().__init__(tie_word_embeddings=tie_word_embeddings, **kwargs)


__all__ = ["Gemma4Config", "Gemma4TextConfig", "Gemma4VisionConfig"]
