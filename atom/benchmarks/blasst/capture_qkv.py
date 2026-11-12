# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Capture per-layer Q/K/V activations for BLASST benchmarking.

Block skipping depends on the attention score distribution, so the benchmark
needs real activations rather than random tensors.

Hooks HuggingFace directly rather than running the activations through the
kernel under test, which would be circular.

This module holds the benchmark's only dependency on `transformers` and on a
model being present on disk.
"""

import logging

import torch

logger = logging.getLogger("atom")


def capture_model_qkv(model_name, text, device="cuda", layers=None):
    """Run one forward pass and capture Q/K/V per layer.

    Args:
        model_name: HuggingFace model id or local path.
        text: Prompt to run. Its token count sets the captured sequence length.
        device: Device to load the model on.
        layers: Optional set of layer indices to keep. Keeping all layers of a
            long-context prompt can exhaust HBM, so pass the few being replayed.

    Returns:
        dict: layer_idx -> {"Q", "K", "V", "num_kv_groups"}, with Q/K/V shaped
        (batch, heads, seq, head_dim) exactly as HF's attention interface sees
        them -- post-RoPE, pre-softmax, GQA heads unexpanded.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # Load to CPU then move, rather than device_map=: device_map pulls in
    # `accelerate` for no benefit on a single-GPU capture. `torch_dtype` was
    # renamed `dtype` in transformers 5.x, so accept either.
    try:
        model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.bfloat16)
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.bfloat16
        )
    model = model.to(device)
    model.eval()

    impl = model.config._attn_implementation
    original_fn = ALL_ATTENTION_FUNCTIONS[impl]
    captured = {}

    def hook(module, query, key, value, attention_mask, *args, **kwargs):
        if layers is None or module.layer_idx in layers:
            captured[module.layer_idx] = {
                "Q": query.detach(),
                "K": key.detach(),
                "V": value.detach(),
                "num_kv_groups": getattr(module, "num_key_value_groups", 1),
            }
        # Delegate so the model still runs correctly and later layers see real
        # inputs; capturing without delegating would poison every later layer.
        return original_fn(module, query, key, value, attention_mask, *args, **kwargs)

    ALL_ATTENTION_FUNCTIONS[impl] = hook
    ids = tokenizer.encode(text, return_tensors="pt").to(device)
    logger.info("capturing %d tokens from %s", ids.shape[-1], model_name)
    # Run the base transformer, not the LM head. We only want attention
    # activations, and projecting a 32K-token sequence to a 150K vocab costs
    # ~10 GB we would immediately discard (it OOMs an otherwise-fine capture).
    inner = getattr(model, "model", None)
    try:
        with torch.no_grad():
            if inner is not None:
                inner(ids)
            else:
                model(ids)
    finally:
        ALL_ATTENTION_FUNCTIONS[impl] = original_fn

    del model
    torch.cuda.empty_cache()
    return captured
