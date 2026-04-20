#!/usr/bin/env python3
"""
Simulate the loader's behavior for visual weights to check if they'd be loaded correctly.
"""
import re

MODEL_PATH = "/data/models/Qwen3.5-35B-A3B-FP8"

# Packed modules mapping from Qwen3_5MoeMultimodalModel
packed_modules_mapping = {
    "q_proj": ("qkv_proj", "q"),
    "k_proj": ("qkv_proj", "k"),
    "v_proj": ("qkv_proj", "v"),
    "gate_proj": ("gate_up_proj", 0),
    "up_proj": ("gate_up_proj", 1),
    "gate_up_proj": ["gate_proj", "up_proj"],
    "in_proj_qkv": ("in_proj_qkvz", (0, 1, 2)),
    "in_proj_z": ("in_proj_qkvz", 3),
    "in_proj_b": ("in_proj_ba", 0),
    "in_proj_a": ("in_proj_ba", 1),
    ".gate.": (".gate.", 0),
    "shared_expert_gate": ("gate", 1),
}

# Weight mapping: checkpoint → model
hf_to_atom_prefix = {
    "model.visual.": "visual.",
    "lm_head.": "language_model.lm_head.",
    "model.language_model.": "language_model.model.",
}


def map_name(name):
    for prefix, new_prefix in hf_to_atom_prefix.items():
        if name.startswith(prefix):
            return name.replace(prefix, new_prefix, 1)
    return name


def main():
    import safetensors.torch
    import glob
    import os

    # Collect all checkpoint weight names
    all_names = []
    for f in sorted(glob.glob(os.path.join(MODEL_PATH, "*.safetensors"))):
        with safetensors.torch.safe_open(f, framework="pt") as sf:
            all_names.extend(sf.keys())

    visual_names = [n for n in all_names if "visual" in n]
    print(f"Total checkpoint weights: {len(all_names)}")
    print(f"Visual checkpoint weights: {len(visual_names)}")

    # Simulate the loader path for each visual weight
    loaded_via_packed = 0
    loaded_via_expert = 0
    loaded_via_fallback = 0
    skipped_layer = 0
    skipped_other = 0
    issues = []

    for orig_name in visual_names:
        name = map_name(orig_name)

        # Layer filter check
        layerId_ = re.search(r"model\.layers\.(\d+)\.", name)
        layerId = int(layerId_.group(1)) if layerId_ else 0
        num_hidden_layers = 40
        if num_hidden_layers and layerId >= num_hidden_layers:
            skipped_layer += 1
            continue

        # Packed modules check
        matched_packed = False
        for k in packed_modules_mapping:
            if k in name:
                matched_packed = True
                break

        if matched_packed:
            loaded_via_packed += 1
            issues.append(f"  PACKED MATCH: {orig_name} → {name} matched key '{k}'")
            continue

        # Expert mapping check
        if "experts" in name:
            loaded_via_expert += 1
            continue

        # Fallback (direct load)
        loaded_via_fallback += 1

    print(f"\nVisual weight loading simulation:")
    print(f"  Loaded via packed: {loaded_via_packed}")
    print(f"  Loaded via expert: {loaded_via_expert}")
    print(f"  Loaded via fallback: {loaded_via_fallback}")
    print(f"  Skipped (layer): {skipped_layer}")
    print(f"  Skipped (other): {skipped_other}")

    if issues:
        print(f"\n⚠ Issues ({len(issues)}):")
        for issue in issues[:20]:
            print(issue)


if __name__ == "__main__":
    main()
