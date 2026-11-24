# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

"""Export AtomRL Eagle3 v2 checkpoint to HuggingFace safetensors format.

Architecture aligned with lightseekorg/kimi-k2.5-eagle3:
- Trained weights in float16
- Base model weights (embed_tokens, lm_head) in bfloat16
- No bias in any layer
- config: attention_bias=false, rms_norm_eps=1e-6, rope_theta=1000000, rope_scaling=YaRN(64)

Usage:
    python3 tools/export_eagle3_hf_phase2_mixed.py --ckpt <phase2 checkpoint .pt> \
        --base-model <Kimi-K2.5-BF16 dir> --output <HF dir>
"""

import argparse
import json
import os

import torch
from safetensors.torch import save_file

_parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
_parser.add_argument(
    "--ckpt",
    required=True,
    help="Phase 2 (phase2_mixed.yaml) checkpoint file, e.g. .../kimi_k25_eagle3_v2_phase2/checkpoint_65000.pt",
)
_parser.add_argument(
    "--base-model",
    required=True,
    help="Kimi-K2.5 BF16 checkpoint directory (embed_tokens and lm_head are read from it)",
)
_parser.add_argument(
    "--output", required=True, help="output directory for the HF export"
)
_args = _parser.parse_args()
CKPT_PATH = _args.ckpt
BASE_MODEL_DIR = _args.base_model
OUTPUT_DIR = _args.output

os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- 1. Load our trained draft checkpoint ---
print("Loading checkpoint...")
ckpt = torch.load(CKPT_PATH, map_location="cpu", weights_only=False)
msd = ckpt["state_dict"]["model_state_dict"]
step = ckpt["state_dict"].get("step", ckpt.get("step", "unknown"))
print(f"Checkpoint step: {step}")

# --- 2. Load embed_tokens and lm_head from base model ---
print("Loading embed_tokens and lm_head from base model...")
from safetensors import safe_open

base_shard = os.path.join(BASE_MODEL_DIR, "model-00062-of-000064.safetensors")
with safe_open(base_shard, framework="pt", device="cpu") as f:
    embed_tokens = f.get_tensor("language_model.model.embed_tokens.weight")
    lm_head = f.get_tensor("language_model.lm_head.weight")
print(f"  embed_tokens: {list(embed_tokens.shape)} {embed_tokens.dtype}")
print(f"  lm_head: {list(lm_head.shape)} {lm_head.dtype}")

# --- 3. Map AtomRL keys to HF keys ---
# v2 architecture: no bias, separate Q/K/V, RMSNorm, 3×hidden fc
# AtomRL keys → HF keys (direct mapping, layers.0 → midlayer)

hf_state_dict = {}

# embed_tokens and lm_head (from base model, keep bfloat16)
hf_state_dict["embed_tokens.weight"] = embed_tokens
hf_state_dict["lm_head.weight"] = lm_head

# fc: 3×hidden projection (float16)
hf_state_dict["fc.weight"] = msd["fc.weight"].to(torch.float16)

# decoder layer (layers.0 → midlayer)
hf_state_dict["midlayer.hidden_norm.weight"] = msd["layers.0.hidden_norm.weight"].to(
    torch.float16
)
hf_state_dict["midlayer.input_layernorm.weight"] = msd[
    "layers.0.input_layernorm.weight"
].to(torch.float16)

# attention (separate Q/K/V, no bias)
hf_state_dict["midlayer.self_attn.q_proj.weight"] = msd[
    "layers.0.self_attn.q_proj.weight"
].to(torch.float16)
hf_state_dict["midlayer.self_attn.k_proj.weight"] = msd[
    "layers.0.self_attn.k_proj.weight"
].to(torch.float16)
hf_state_dict["midlayer.self_attn.v_proj.weight"] = msd[
    "layers.0.self_attn.v_proj.weight"
].to(torch.float16)
hf_state_dict["midlayer.self_attn.o_proj.weight"] = msd[
    "layers.0.self_attn.o_proj.weight"
].to(torch.float16)

# post-attention norm
hf_state_dict["midlayer.post_attention_layernorm.weight"] = msd[
    "layers.0.post_attention_layernorm.weight"
].to(torch.float16)

# MLP: gate_proj, up_proj, down_proj
hf_state_dict["midlayer.mlp.gate_proj.weight"] = msd[
    "layers.0.mlp.gate_proj.weight"
].to(torch.float16)
hf_state_dict["midlayer.mlp.up_proj.weight"] = msd["layers.0.mlp.up_proj.weight"].to(
    torch.float16
)
hf_state_dict["midlayer.mlp.down_proj.weight"] = msd[
    "layers.0.mlp.down_proj.weight"
].to(torch.float16)

# output norm
hf_state_dict["norm.weight"] = msd["out_norm.weight"].to(torch.float16)

# --- 4. Split into 2 shards ---
shard1_keys = [k for k in hf_state_dict if k != "lm_head.weight"]
shard2_keys = ["lm_head.weight"]

shard1 = {k: hf_state_dict[k] for k in shard1_keys}
shard2 = {k: hf_state_dict[k] for k in shard2_keys}

shard1_file = "model-00001-of-00002.safetensors"
shard2_file = "model-00002-of-00002.safetensors"

print(f"Saving {shard1_file} ({len(shard1)} tensors)...")
save_file(shard1, os.path.join(OUTPUT_DIR, shard1_file))
print(f"Saving {shard2_file} ({len(shard2)} tensors)...")
save_file(shard2, os.path.join(OUTPUT_DIR, shard2_file))

# --- 5. model.safetensors.index.json ---
total_params = sum(v.numel() for v in hf_state_dict.values())
total_size = sum(v.numel() * v.element_size() for v in hf_state_dict.values())

weight_map = {}
for k in shard1_keys:
    weight_map[k] = shard1_file
for k in shard2_keys:
    weight_map[k] = shard2_file

index = {
    "metadata": {
        "total_parameters": total_params,
        "total_size": total_size,
    },
    "weight_map": dict(sorted(weight_map.items())),
}
with open(os.path.join(OUTPUT_DIR, "model.safetensors.index.json"), "w") as f:
    json.dump(index, f, indent=2)
print(f"Total parameters: {total_params:,}")
print(f"Total size: {total_size / 1e9:.2f} GB")

# --- 6. config.json (matching reference lightseekorg/kimi-k2.5-eagle3) ---
config = {
    "architectures": ["LlamaForCausalLMEagle3"],
    "attention_bias": False,
    "attention_dropout": 0.0,
    "bos_token_id": 163584,
    "draft_vocab_size": 163840,
    "dtype": "bfloat16",
    "eos_token_id": 163585,
    "head_dim": 128,
    "hidden_act": "silu",
    "hidden_size": 7168,
    "initializer_range": 0.02,
    "intermediate_size": 12288,
    "max_position_embeddings": 262144,
    "mlp_bias": False,
    "model_type": "llama",
    "num_attention_heads": 64,
    "num_hidden_layers": 1,
    "num_key_value_heads": 64,
    "pretraining_tp": 1,
    "rms_norm_eps": 1e-06,
    "rope_scaling": {
        "type": "yarn",
        "factor": 64.0,
        "original_max_position_embeddings": 4096,
        "beta_fast": 32.0,
        "beta_slow": 1.0,
        "mscale": 1.0,
        "mscale_all_dim": 1.0,
    },
    "rope_theta": 1000000.0,
    "sliding_window": None,
    "tie_word_embeddings": False,
    "use_cache": True,
    "use_sliding_window": False,
    "vocab_size": 163840,
}
with open(os.path.join(OUTPUT_DIR, "config.json"), "w") as f:
    json.dump(config, f, indent=2)

# --- 7. .gitattributes ---
gitattributes = "*.safetensors filter=lfs diff=lfs merge=lfs -text\n"
with open(os.path.join(OUTPUT_DIR, ".gitattributes"), "w") as f:
    f.write(gitattributes)

# --- 8. README.md ---
readme = f"""---
license: mit
base_model: moonshotai/Kimi-K2.5
tags:
- safetensors
- llama
- speculative-decoding
- eagle3
- draft-model
- kimi-k2.5
---

# Kimi-K2.5 Eagle3 Draft Model v2 (AtomRL, MI350)

Eagle3 MTP draft model for accelerating inference of [Kimi-K2.5](https://huggingface.co/moonshotai/Kimi-K2.5),
trained with [AtomRL](https://github.com/ROCm/ATOM/tree/main/atomrl) on 8x AMD MI350 GPUs.

Architecture aligned with [lightseekorg/kimi-k2.5-eagle3](https://huggingface.co/lightseekorg/kimi-k2.5-eagle3).

## Training Setup

- **Hardware**: 8x AMD MI350 (4 train + 4 inference)
- **Training**: FSDP2 (float16), GPUs 0-3
- **Inference**: vLLM (BF16), GPUs 4-7, TP=4
- **Transfer**: Mooncake TCP
- **Draft**: Eagle3, 1-layer Transformer (hidden_size=7168, intermediate_size=12288, num_heads=64, head_dim=128)

## Training Phases

- **Phase 1 (Foundation)**: perfectblend subset, 111K steps (3 epochs)
- **Phase 2 (Mixed Domain)**: VL + Chinese + tool-call + agent + writing, 67,826 steps (3 epochs)

Final checkpoint at step {step}.

## Architecture (matches reference)

- attention_bias: false
- rms_norm_eps: 1e-6
- rope_theta: 1000000.0
- rope_scaling: YaRN (factor=64)
- dtype: float16 (trained weights), bfloat16 (embed_tokens, lm_head)
"""
with open(os.path.join(OUTPUT_DIR, "README.md"), "w") as f:
    f.write(readme)

# Summary
print("\n=== Export complete ===")
for fn in sorted(os.listdir(OUTPUT_DIR)):
    fpath = os.path.join(OUTPUT_DIR, fn)
    size = os.path.getsize(fpath)
    if size > 1e9:
        print(f"  {fn}: {size/1e9:.2f} GB")
    elif size > 1e6:
        print(f"  {fn}: {size/1e6:.1f} MB")
    else:
        print(f"  {fn}: {size/1e3:.1f} KB")
