# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

"""Export AtomRL Eagle3 v2 Phase 2 checkpoint to HuggingFace safetensors format.

Uses the final checkpoint from phase2 training. Same architecture as phase1.

Usage:
    python3 tools/export_eagle3_hf_phase2_atom.py --ckpt <phase2_atom checkpoint dir> \
        --base-model <Kimi-K2.5 dir> --output <HF dir>
"""

import argparse
import glob
import json
import os

import torch
from safetensors import safe_open
from safetensors.torch import save_file

_parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
_parser.add_argument(
    "--ckpt",
    required=True,
    help="Phase 2 (phase2_atom.yaml) checkpoint directory; the latest checkpoint_*.pt in it is exported",
)
_parser.add_argument(
    "--base-model",
    required=True,
    help="Kimi-K2.5 checkpoint directory the teacher was served from (embed_tokens and lm_head are read from it)",
)
_parser.add_argument(
    "--output", required=True, help="output directory for the HF export"
)
_args = _parser.parse_args()
BASE_MODEL_DIR = _args.base_model
CKPT_DIR = _args.ckpt
OUTPUT_DIR = _args.output

# Find the latest checkpoint
ckpt_files = sorted(glob.glob(os.path.join(CKPT_DIR, "checkpoint_*.pt")))
if not ckpt_files:
    raise RuntimeError(f"No checkpoints found in {CKPT_DIR}")
CKPT_PATH = ckpt_files[-1]
print(f"Using checkpoint: {CKPT_PATH}")

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("Loading checkpoint...")
ckpt = torch.load(CKPT_PATH, map_location="cpu", weights_only=False)
msd = ckpt["state_dict"]["model_state_dict"]
step = ckpt["state_dict"].get("step", ckpt.get("step", "unknown"))
print(f"Checkpoint step: {step}")

print("Loading embed_tokens and lm_head from base model...")
base_shard = os.path.join(BASE_MODEL_DIR, "model-00062-of-000064.safetensors")
with safe_open(base_shard, framework="pt", device="cpu") as f:
    embed_tokens = f.get_tensor("language_model.model.embed_tokens.weight")
    lm_head = f.get_tensor("language_model.lm_head.weight")
print(f"  embed_tokens: {list(embed_tokens.shape)} {embed_tokens.dtype}")
print(f"  lm_head: {list(lm_head.shape)} {lm_head.dtype}")

hf_state_dict = {}
hf_state_dict["embed_tokens.weight"] = embed_tokens
hf_state_dict["lm_head.weight"] = lm_head.to(torch.float16)
hf_state_dict["fc.weight"] = msd["fc.weight"].to(torch.float16)
hf_state_dict["midlayer.hidden_norm.weight"] = msd["layers.0.hidden_norm.weight"].to(
    torch.float16
)
hf_state_dict["midlayer.input_layernorm.weight"] = msd[
    "layers.0.input_layernorm.weight"
].to(torch.float16)
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
hf_state_dict["midlayer.post_attention_layernorm.weight"] = msd[
    "layers.0.post_attention_layernorm.weight"
].to(torch.float16)
hf_state_dict["midlayer.mlp.gate_proj.weight"] = msd[
    "layers.0.mlp.gate_proj.weight"
].to(torch.float16)
hf_state_dict["midlayer.mlp.up_proj.weight"] = msd["layers.0.mlp.up_proj.weight"].to(
    torch.float16
)
hf_state_dict["midlayer.mlp.down_proj.weight"] = msd[
    "layers.0.mlp.down_proj.weight"
].to(torch.float16)
hf_state_dict["norm.weight"] = msd["out_norm.weight"].to(torch.float16)

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

total_params = sum(v.numel() for v in hf_state_dict.values())
total_size = sum(v.numel() * v.element_size() for v in hf_state_dict.values())
weight_map = {}
for k in shard1_keys:
    weight_map[k] = shard1_file
for k in shard2_keys:
    weight_map[k] = shard2_file
index = {
    "metadata": {"total_parameters": total_params, "total_size": total_size},
    "weight_map": dict(sorted(weight_map.items())),
}
with open(os.path.join(OUTPUT_DIR, "model.safetensors.index.json"), "w") as f:
    json.dump(index, f, indent=2)

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
    "max_window_layers": 36,
    "mlp_bias": False,
    "model_type": "llama",
    "num_attention_heads": 64,
    "num_hidden_layers": 1,
    "num_key_value_heads": 64,
    "pretraining_tp": 1,
    "rms_norm_eps": 1e-06,
    "rope_scaling": {
        "beta_fast": 32.0,
        "beta_slow": 1.0,
        "factor": 64.0,
        "mscale": 1.0,
        "mscale_all_dim": 1.0,
        "original_max_position_embeddings": 4096,
        "type": "yarn",
    },
    "rope_theta": 50000.0,
    "sliding_window": None,
    "tie_word_embeddings": False,
    "use_cache": True,
    "use_sliding_window": False,
    "torch_dtype": "float16",
    "transformers_version": "4.51.0",
    "vocab_size": 163840,
    "_torchspec_version": "0.0.1",
}
with open(os.path.join(OUTPUT_DIR, "config.json"), "w") as f:
    json.dump(config, f, indent=2)

gitattributes = "*.safetensors filter=lfs diff=lfs merge=lfs -text\n"
with open(os.path.join(OUTPUT_DIR, ".gitattributes"), "w") as f:
    f.write(gitattributes)

print(f"\nTotal parameters: {total_params:,}")
print(f"Total size: {total_size / 1e9:.2f} GB")
print(f"\n=== Export complete to {OUTPUT_DIR} ===")
for fn in sorted(os.listdir(OUTPUT_DIR)):
    fpath = os.path.join(OUTPUT_DIR, fn)
    size = os.path.getsize(fpath)
    if size > 1e9:
        print(f"  {fn}: {size/1e9:.2f} GB")
    elif size > 1e6:
        print(f"  {fn}: {size/1e6:.1f} MB")
    else:
        print(f"  {fn}: {size/1e3:.1f} KB")
