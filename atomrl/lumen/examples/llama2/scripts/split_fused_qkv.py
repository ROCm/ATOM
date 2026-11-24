# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

"""Split fused-QKV LLaMA model into standard separate Q/K/V format.

Converts regisss/llama2-70b-fused-qkv-mlperf (qkv_proj) to standard
LlamaForCausalLM format (q_proj, k_proj, v_proj).

Usage:
    python split_fused_qkv.py --src /data1/lumen/model --dst /data1/lumen/model-standard
"""

import argparse
import glob
import json
import os
import shutil
from collections import OrderedDict

from safetensors.torch import load_file, save_file


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", required=True, help="Fused-QKV model directory")
    parser.add_argument("--dst", required=True, help="Output standard LLaMA directory")
    args = parser.parse_args()

    os.makedirs(args.dst, exist_ok=True)

    with open(os.path.join(args.src, "config.json")) as config_file:
        cfg = json.load(config_file)
    num_heads = cfg["num_attention_heads"]
    num_kv_heads = cfg["num_key_value_heads"]
    hidden = cfg["hidden_size"]
    head_dim = hidden // num_heads
    q_size = num_heads * head_dim
    k_size = num_kv_heads * head_dim
    v_size = num_kv_heads * head_dim

    print(f"  Model: {cfg.get('_name_or_path', 'unknown')}")
    print(
        f"  heads={num_heads}, kv_heads={num_kv_heads}, hidden={hidden}, head_dim={head_dim}"
    )
    print(f"  Q={q_size}, K={k_size}, V={v_size}, total={q_size+k_size+v_size}")

    files = sorted(glob.glob(os.path.join(args.src, "*.safetensors")))
    for i, fn in enumerate(files):
        sd = load_file(fn)
        new_sd = OrderedDict()
        for k, v in sd.items():
            if "qkv_proj" in k:
                assert v.shape[0] == q_size + k_size + v_size, f"{k}: {v.shape}"
                q = v[:q_size, :]
                kk = v[q_size : q_size + k_size, :]
                vv = v[q_size + k_size :, :]
                new_sd[k.replace("qkv_proj.weight", "q_proj.weight")] = q
                new_sd[k.replace("qkv_proj.weight", "k_proj.weight")] = kk
                new_sd[k.replace("qkv_proj.weight", "v_proj.weight")] = vv
            else:
                new_sd[k] = v
        out = os.path.join(args.dst, os.path.basename(fn))
        save_file(new_sd, out)
        print(
            f"  [{i+1}/{len(files)}] {os.path.basename(fn)}: {len(sd)} -> {len(new_sd)} tensors"
        )

    cfg["model_type"] = "llama"
    cfg["architectures"] = ["LlamaForCausalLM"]
    cfg.pop("auto_map", None)
    with open(os.path.join(args.dst, "config.json"), "w") as config_file:
        json.dump(cfg, config_file, indent=2)

    for f in [
        "generation_config.json",
        "tokenizer.json",
        "tokenizer.model",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "model.safetensors.index.json",
    ]:
        src_f = os.path.join(args.src, f)
        if os.path.exists(src_f):
            shutil.copy2(src_f, os.path.join(args.dst, f))

    idx_file = os.path.join(args.dst, "model.safetensors.index.json")
    if os.path.exists(idx_file):
        with open(idx_file) as index_file:
            idx = json.load(index_file)
        new_map = {}
        for k, v in idx.get("weight_map", {}).items():
            if "qkv_proj" in k:
                new_map[k.replace("qkv_proj.weight", "q_proj.weight")] = v
                new_map[k.replace("qkv_proj.weight", "k_proj.weight")] = v
                new_map[k.replace("qkv_proj.weight", "v_proj.weight")] = v
            else:
                new_map[k] = v
        idx["weight_map"] = new_map
        with open(idx_file, "w") as index_file:
            json.dump(idx, index_file, indent=2)

    print("Split complete.")


if __name__ == "__main__":
    main()
