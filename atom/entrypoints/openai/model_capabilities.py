# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Discovery metadata for the running deployment, not the hosted offering."""


def deployment_metadata(config, encoder):
    hf = getattr(config, "hf_config", None)
    text = getattr(hf, "text_config", None) or hf
    limit = getattr(config, "max_model_len", None)
    declared = getattr(text, "max_position_embeddings", None)
    limits = [v for v in (limit, declared) if type(v) is int and v > 0]
    result = {"output_modalities": ["text"]}
    if limits:
        result["context_window"] = min(limits)
        result["max_output_tokens"] = min(limits)
    method = getattr(getattr(config, "speculative_config", None), "method", None)
    if (method or getattr(config, "method", None)) == "dspark":
        result["input_modalities"] = ["text"]
    if getattr(encoder, "name", None) == "encoding_dsv41":
        result["name"] = "DeepSeek-V4.1-Flash"
        result["effort"] = {
            "supported_levels": ["low", "high", "max"],
            "default_level": "high",
        }
        # Match the accepted output cap; context plus prompt validation may
        # reduce the tokens available to a particular request further.
        result["max_output_tokens"] = min(
            result.get("max_output_tokens", 393216), 393216
        )
    return result
