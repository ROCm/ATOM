#!/usr/bin/env python3
"""Compare ATOM native vision encoder forward pass vs HuggingFace reference.

Loads only the vision encoder weights (CPU), runs both implementations
on the same synthetic input, and checks output equivalence.
"""

import torch
import numpy as np

MODEL_PATH = "/data/models/Qwen3.5-35B-A3B-FP8"


def main():
    from transformers import AutoConfig
    from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (
        Qwen3_5MoeVisionModel,
    )
    from atom.models.qwen3_5_vl import Qwen3VisionTransformer

    # Load config
    config = AutoConfig.from_pretrained(MODEL_PATH)
    vision_config = config.vision_config

    print(f"Vision config:")
    print(f"  hidden_size={vision_config.hidden_size}")
    print(f"  num_heads={vision_config.num_heads}")
    print(f"  head_dim={vision_config.hidden_size // vision_config.num_heads}")
    print(f"  depth={vision_config.depth}")
    print(f"  out_hidden_size={vision_config.out_hidden_size}")
    print(f"  patch_size={vision_config.patch_size}")
    print(f"  temporal_patch_size={vision_config.temporal_patch_size}")
    print(f"  spatial_merge_size={vision_config.spatial_merge_size}")
    print(f"  in_channels={vision_config.in_channels}")
    print(f"  intermediate_size={vision_config.intermediate_size}")
    print(f"  num_position_embeddings={vision_config.num_position_embeddings}")
    print(f"  hidden_act={getattr(vision_config, 'hidden_act', 'NOT FOUND')}")

    # Build HF model (CPU)
    print("\nBuilding HF vision model...")
    hf_model = Qwen3_5MoeVisionModel(vision_config)
    hf_model.eval()

    # Build ATOM model (CPU)
    print("Building ATOM vision model...")
    atom_model = Qwen3VisionTransformer(
        vision_config, norm_eps=getattr(config, "rms_norm_eps", 1e-6)
    )
    atom_model.eval()

    # Copy all weights from HF to ATOM
    print("\nCopying weights from HF → ATOM...")
    hf_state = hf_model.state_dict()
    atom_state = atom_model.state_dict()

    print(f"  HF params: {len(hf_state)}")
    print(f"  ATOM params: {len(atom_state)}")

    # Check for mismatches
    missing_in_atom = set(hf_state.keys()) - set(atom_state.keys())
    missing_in_hf = set(atom_state.keys()) - set(hf_state.keys())
    if missing_in_atom:
        print(f"  In HF but not ATOM: {missing_in_atom}")
    if missing_in_hf:
        print(f"  In ATOM but not HF: {missing_in_hf}")

    # Copy matching weights
    copied = 0
    for name in atom_state:
        if name in hf_state:
            if atom_state[name].shape == hf_state[name].shape:
                atom_state[name].copy_(hf_state[name])
                copied += 1
            else:
                print(f"  Shape mismatch: {name}: ATOM={atom_state[name].shape}, HF={hf_state[name].shape}")
    atom_model.load_state_dict(atom_state)
    print(f"  Copied {copied} weight tensors")

    # Create synthetic input
    # For a 224x224 image with patch_size=16, temporal_patch_size=2:
    # grid is (1, 14, 14) but temporal needs at least 2 frames -> actually for single image,
    # the processor handles this by repeating.
    # Let's use a small grid to keep it fast: (1, 8, 8) -> 64 patches before merge
    t, h, w = 1, 8, 8
    num_patches = t * h * w  # 64
    in_channels = vision_config.in_channels
    temp_ps = vision_config.temporal_patch_size
    ps = vision_config.patch_size
    patch_dim = in_channels * temp_ps * ps * ps  # 3 * 2 * 16 * 16 = 1536

    torch.manual_seed(42)
    pixel_values = torch.randn(num_patches, patch_dim, dtype=torch.float32)
    grid_thw = torch.tensor([[t, h, w]])

    # Run HF
    print(f"\nInput: pixel_values={pixel_values.shape}, grid_thw={grid_thw.tolist()}")
    with torch.no_grad():
        hf_result = hf_model(pixel_values, grid_thw)
        hf_pre_merger = hf_result.last_hidden_state
        hf_out = hf_result.pooler_output  # post-merger

    print(f"HF pre-merger: {hf_pre_merger.shape}, mean={hf_pre_merger.float().mean():.6f}, std={hf_pre_merger.float().std():.6f}")
    print(f"HF post-merger: {hf_out.shape}, mean={hf_out.float().mean():.6f}, std={hf_out.float().std():.6f}")

    # Run ATOM
    with torch.no_grad():
        atom_out = atom_model(pixel_values, grid_thw)

    print(f"ATOM output: {atom_out.shape}, mean={atom_out.float().mean():.6f}, std={atom_out.float().std():.6f}")

    # Compare
    if hf_out.shape != atom_out.shape:
        print(f"\n✗ Shape mismatch: HF={hf_out.shape}, ATOM={atom_out.shape}")
        return

    diff = (hf_out.float() - atom_out.float()).abs()
    print(f"\nDifference: max={diff.max():.6e}, mean={diff.mean():.6e}")

    match = torch.allclose(hf_out.float(), atom_out.float(), atol=1e-3, rtol=1e-3)
    if match:
        print("✓ Outputs MATCH (atol=1e-3, rtol=1e-3)")
    else:
        print("✗ Outputs DO NOT MATCH")
        # Find worst positions
        flat_diff = diff.flatten()
        worst_indices = flat_diff.argsort(descending=True)[:5]
        for idx in worst_indices:
            row = idx.item() // diff.shape[-1]
            col = idx.item() % diff.shape[-1]
            print(f"  [{row},{col}]: HF={hf_out[row, col]:.6f}, ATOM={atom_out[row, col]:.6f}, diff={diff[row, col]:.6e}")


if __name__ == "__main__":
    main()
