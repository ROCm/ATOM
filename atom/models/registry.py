# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Checkpoint architecture (``hf_config.architectures[0]``) -> implementation.

Dotted paths, so an implementation is imported only when its architecture is
selected.
"""

support_model_arch_dict = {
    "Qwen3ForCausalLM": "atom.models.qwen3.Qwen3ForCausalLM",
    "Qwen3MoeForCausalLM": "atom.models.qwen3_moe.Qwen3MoeForCausalLM",
    "LlamaForCausalLM": "atom.models.llama.LlamaForCausalLM",
    "MixtralForCausalLM": "atom.models.mixtral.MixtralForCausalLM",
    "DeepseekV3ForCausalLM": "atom.models.deepseek_v2.DeepseekV2ForCausalLM",
    "DeepseekV32ForCausalLM": "atom.models.deepseek_v2.DeepseekV2ForCausalLM",
    "DeepseekV41ForCausalLM": "atom.models.deepseek_v41.runtime.DeepseekV41RuntimeModel",
    "DeepseekV4ForCausalLM": "atom.models.deepseek_v4.DeepseekV4ForCausalLM",
    "GptOssForCausalLM": "atom.models.gpt_oss.GptOssForCausalLM",
    "GlmMoeDsaForCausalLM": "atom.models.deepseek_v2.GlmMoeDsaForCausalLM",
    "Glm4MoeForCausalLM": "atom.models.glm4_moe.Glm4MoeForCausalLM",
    "Qwen3NextForCausalLM": "atom.models.qwen3_next.Qwen3NextForCausalLM",
    "Qwen4ExpForConditionalGeneration": (
        "atom.models.qwen4_exp.Qwen4ExpForConditionalGeneration"
    ),
    "Qwen3_5ForConditionalGeneration": "atom.models.qwen3_5.Qwen3_5MultimodalModel",
    "Qwen3_5MoeForConditionalGeneration": "atom.models.qwen3_5.Qwen3_5MoeMultimodalModel",
    "Qwen3_5MoeForCausalLM": "atom.models.qwen3_5.Qwen3_5MoeForCausalLM",
    "KimiK25ForConditionalGeneration": "atom.models.kimi_k25.KimiK25ForCausalLM",
    "KimiK3ForConditionalGeneration": (
        "atom.models.kimi_k3.KimiK3ForConditionalGeneration"
    ),
    "MiniMaxM2ForCausalLM": "atom.models.minimax_m2.MiniMaxM2ForCausalLM",
    "MiMoV2ForCausalLM": "atom.models.mimo_v2.MiMoV2ForCausalLM",
    "MiMoV2FlashForCausalLM": "atom.models.mimo_v2.MiMoV2ForCausalLM",
    "Mistral3ForConditionalGeneration": "atom.models.mistral3.Mistral3TextOnly",
    "MistralForCausalLM": "atom.models.mistral3.Mistral3ForCausalLM",
    "MiniMaxM3SparseForCausalLM": "atom.models.minimax_m3.MiniMaxM3SparseForCausalLM",
    "MiniMaxM3SparseForConditionalGeneration": (
        "atom.models.minimax_m3.vl.MiniMaxM3SparseForConditionalGenerationVL"
    ),
    "Glm5NextForConditionalGeneration": (
        "atom.models.glm5_next.Glm5NextForConditionalGeneration"
    ),
}
# Model-specific Mono setup runs after the drafter arms the original layers.
# An installer may wrap the model or finish binding routing already inside it.
mono_decode_installers = {
    "DeepseekV4ForCausalLM": (
        "atom.models.deepseek_v4.mono.dispatch.install_mono_decode"
    ),
    "DeepseekV41ForCausalLM": (
        "atom.models.deepseek_v41.mono.dispatch.install_mono_decode"
    ),
    "KimiK3ForConditionalGeneration": (
        "atom.models.kimi_k3.mono.dispatch.install_mono_decode"
    ),
    "MiniMaxM3SparseForCausalLM": (
        "atom.models.minimax_m3.mono.dispatch.install_mono_decode"
    ),
    "MiniMaxM3SparseForConditionalGeneration": (
        "atom.models.minimax_m3.mono.dispatch.install_mono_decode"
    ),
}
