#!/usr/bin/env python3
"""
Test ATOM vision encoder with real model weights (not engine, just module).
Compare against HF reference model on the same real image input.
"""

import torch
from PIL import Image
from transformers import AutoProcessor, AutoConfig
from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (
    Qwen3_5MoeVisionModel,
)

MODEL_PATH = "/data/models/Qwen3.5-35B-A3B-FP8"
IMAGE_PATH = "/app/ATOM/recipes/atom_vllm/dog.png"


def main():
    # Load config and processor
    config = AutoConfig.from_pretrained(MODEL_PATH)
    vision_config = config.vision_config
    processor = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True)

    # Load real image
    image = Image.open(IMAGE_PATH).convert("RGB")
    print(f"Image size: {image.size}")

    # Process image with processor
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": "Describe this image."},
            ],
        }
    ]
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = processor(text=[text], images=[image], return_tensors="pt")

    pixel_values = inputs["pixel_values"]
    image_grid_thw = inputs["image_grid_thw"]
    input_ids = inputs["input_ids"][0]

    print(f"pixel_values: shape={pixel_values.shape}, dtype={pixel_values.dtype}")
    print(f"  mean={pixel_values.float().mean():.6f}, std={pixel_values.float().std():.6f}")
    print(f"  min={pixel_values.min():.6f}, max={pixel_values.max():.6f}")
    print(f"image_grid_thw: {image_grid_thw}")
    print(f"input_ids length: {len(input_ids)}")

    # Count image tokens
    image_token_id = getattr(config, "image_token_id", 248056)
    num_image_tokens = (input_ids == image_token_id).sum().item()
    print(f"image_token_id={image_token_id}, count in input_ids={num_image_tokens}")

    # Build HF vision model (CPU, with real weights from checkpoint)
    print("\nLoading HF vision model with real weights...")
    import safetensors.torch
    import glob
    import os

    # Load all visual weights from checkpoint
    visual_weights = {}
    for f in sorted(glob.glob(os.path.join(MODEL_PATH, "*.safetensors"))):
        tensors = safetensors.torch.load_file(f, device="cpu")
        for name, tensor in tensors.items():
            if name.startswith("model.visual."):
                # Strip "model.visual." prefix to match HF model parameter names
                hf_name = name[len("model.visual."):]
                visual_weights[hf_name] = tensor
        del tensors

    print(f"Loaded {len(visual_weights)} visual weights from checkpoint")

    # Build HF model and load weights
    hf_model = Qwen3_5MoeVisionModel(vision_config)
    hf_model.eval()

    hf_state = hf_model.state_dict()
    loaded = 0
    for name in hf_state:
        if name in visual_weights:
            if hf_state[name].shape == visual_weights[name].shape:
                hf_state[name].copy_(visual_weights[name])
                loaded += 1
            else:
                print(f"  Shape mismatch: {name}: model={hf_state[name].shape}, ckpt={visual_weights[name].shape}")
        else:
            print(f"  Missing in checkpoint: {name}")
    hf_model.load_state_dict(hf_state)
    print(f"Loaded {loaded} weights into HF model")

    # Build ATOM model and load same weights
    from atom.models.qwen3_5_vl import Qwen3VisionTransformer

    print("\nBuilding ATOM vision model...")
    atom_model = Qwen3VisionTransformer(
        vision_config, norm_eps=getattr(config, "rms_norm_eps", 1e-6)
    )
    atom_model.eval()

    atom_state = atom_model.state_dict()
    loaded = 0
    for name in atom_state:
        if name in visual_weights:
            if atom_state[name].shape == visual_weights[name].shape:
                atom_state[name].copy_(visual_weights[name])
                loaded += 1
            else:
                print(f"  Shape mismatch: {name}: model={atom_state[name].shape}, ckpt={visual_weights[name].shape}")
        else:
            print(f"  Missing in checkpoint: {name}")
    atom_model.load_state_dict(atom_state)
    print(f"Loaded {loaded} weights into ATOM model")

    # Run both models on real image data
    print("\nRunning HF vision model...")
    with torch.no_grad():
        hf_result = hf_model(pixel_values, image_grid_thw)
        hf_out = hf_result.pooler_output  # post-merger
        hf_pre = hf_result.last_hidden_state  # pre-merger

    print(f"HF pre-merger: shape={hf_pre.shape}, mean={hf_pre.float().mean():.6f}, std={hf_pre.float().std():.6f}")
    print(f"HF post-merger: shape={hf_out.shape}, mean={hf_out.float().mean():.6f}, std={hf_out.float().std():.6f}")

    print("\nRunning ATOM vision model...")
    with torch.no_grad():
        atom_out = atom_model(pixel_values, image_grid_thw)

    print(f"ATOM output: shape={atom_out.shape}, mean={atom_out.float().mean():.6f}, std={atom_out.float().std():.6f}")

    # Compare
    if hf_out.shape == atom_out.shape:
        diff = (hf_out.float() - atom_out.float()).abs()
        print(f"\nDifference: max={diff.max():.6e}, mean={diff.mean():.6e}")
        match = torch.allclose(hf_out.float(), atom_out.float(), atol=1e-2, rtol=1e-2)
        if match:
            print("✓ Outputs MATCH (atol=1e-2, rtol=1e-2)")
        else:
            print("✗ Outputs DO NOT MATCH")
        # Also check tight tolerance
        tight = torch.allclose(hf_out.float(), atom_out.float(), atol=1e-4, rtol=1e-4)
        print(f"  Tight match (atol=1e-4): {tight}")
    else:
        print(f"✗ Shape mismatch: HF={hf_out.shape}, ATOM={atom_out.shape}")

    # Expected number of merged patches
    t, h, w = image_grid_thw[0].tolist()
    merge = vision_config.spatial_merge_size
    expected_merged = t * (h // merge) * (w // merge)
    print(f"\nExpected merged patches: {expected_merged} (from grid {t}x{h}x{w}, merge={merge})")
    print(f"num_image_tokens in input_ids: {num_image_tokens}")
    print(f"Match: {expected_merged == num_image_tokens}")


if __name__ == "__main__":
    main()
