# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

"""Convert legacy Megatron checkpoint key format to mcore format.

Legacy format:  model.language_model.{embedding,encoder,output_layer}
  encoder.layers.N.self_attention.query_key_value.weight
  encoder.layers.N.self_attention.dense.weight
  encoder.layers.N.mlp.dense_h_to_4h.weight
  encoder.layers.N.mlp.dense_4h_to_h.weight
  encoder.layers.N.input_norm.weight
  encoder.layers.N.post_attention_norm.weight
  encoder.final_norm.weight

Mcore format:   model.{embedding,decoder,output_layer}
  decoder.layers.N.self_attention.linear_qkv.weight
  decoder.layers.N.self_attention.linear_proj.weight
  decoder.layers.N.mlp.linear_fc1.weight
  decoder.layers.N.mlp.linear_fc2.weight
  decoder.layers.N.input_layernorm.weight
  decoder.layers.N.pre_mlp_layernorm.weight
  decoder.final_layernorm.weight
"""

import argparse
import glob
import os
import sys

import torch

LAYER_KEY_MAP = {
    "self_attention.query_key_value.weight": "self_attention.linear_qkv.weight",
    "self_attention.dense.weight": "self_attention.linear_proj.weight",
    "mlp.dense_h_to_4h.weight": "mlp.linear_fc1.weight",
    "mlp.dense_4h_to_h.weight": "mlp.linear_fc2.weight",
    "input_norm.weight": "input_layernorm.weight",
    "post_attention_norm.weight": "pre_mlp_layernorm.weight",
}


def convert_state_dict(legacy_sd):
    """Convert a legacy state dict to mcore format."""
    mcore_sd = {}
    lm = legacy_sd.get("language_model", legacy_sd)

    # Embedding
    emb = lm.get("embedding", {})
    if isinstance(emb, dict):
        for k, v in _flatten(emb):
            mcore_sd[f"embedding.{k}"] = v

    # Encoder -> Decoder
    enc = lm.get("encoder", {})
    if isinstance(enc, dict):
        for k, v in _flatten(enc):
            new_k = _remap_encoder_key(k)
            mcore_sd[f"decoder.{new_k}"] = v

    # Output layer
    out = lm.get("output_layer", {})
    if isinstance(out, dict):
        for k, v in _flatten(out):
            mcore_sd[f"output_layer.{k}"] = v

    return mcore_sd


def _flatten(d, prefix=""):
    """Flatten nested dicts into dotted-key, tensor pairs."""
    for k, v in d.items():
        key = f"{prefix}{k}" if prefix else k
        if isinstance(v, dict):
            yield from _flatten(v, key + ".")
        elif isinstance(v, torch.Tensor):
            yield key, v


def _remap_encoder_key(key):
    """Remap a single encoder key to decoder key."""
    if key == "final_norm.weight":
        return "final_layernorm.weight"

    parts = key.split(".", 2)
    if len(parts) >= 3 and parts[0] == "layers":
        layer_idx = parts[1]
        suffix = parts[2]
        new_suffix = LAYER_KEY_MAP.get(suffix, suffix)
        return f"layers.{layer_idx}.{new_suffix}"

    return key


def main():
    parser = argparse.ArgumentParser(
        description="Convert legacy Megatron ckpt to mcore format"
    )
    parser.add_argument(
        "--ckpt-dir", required=True, help="Path to checkpoint dir (parent of iter_*)"
    )
    args = parser.parse_args()

    iter_dirs = sorted(glob.glob(os.path.join(args.ckpt_dir, "iter_*")))
    if not iter_dirs:
        print(f"No iter_* directories found in {args.ckpt_dir}")
        sys.exit(1)

    for iter_dir in iter_dirs:
        rank_dirs = sorted(glob.glob(os.path.join(iter_dir, "mp_rank_*")))
        if not rank_dirs:
            print(f"No mp_rank_* directories found in {iter_dir}")
            continue

        print(f"Converting {iter_dir} ({len(rank_dirs)} ranks)...")
        for rank_dir in rank_dirs:
            ckpt_file = os.path.join(rank_dir, "model_optim_rng.pt")
            if not os.path.isfile(ckpt_file):
                print(f"  Skipping {rank_dir}: no checkpoint file")
                continue

            print(f"  Processing {os.path.basename(rank_dir)}...", end=" ", flush=True)
            ckpt = torch.load(ckpt_file, map_location="cpu", weights_only=False)

            if "language_model" in ckpt.get("model", {}):
                old_keys = list(_flatten_count(ckpt["model"]["language_model"]))
                mcore_sd = convert_state_dict(ckpt["model"])
                ckpt["model"] = mcore_sd
                new_keys = list(mcore_sd.keys())
                print(f"converted {len(old_keys)} -> {len(new_keys)} keys")
            else:
                print("already in mcore format, skipping")
                continue

            torch.save(ckpt, ckpt_file)

    print("Done!")


def _flatten_count(d):
    """Count tensors in nested dict."""
    for k, v in d.items():
        if isinstance(v, dict):
            yield from _flatten_count(v)
        elif isinstance(v, torch.Tensor):
            yield k


if __name__ == "__main__":
    main()
