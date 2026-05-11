# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

import argparse

import torch
from PIL import Image
from transformers import AutoProcessor

from atom import SamplingParams
from atom.model_engine.arg_utils import EngineArgs


parser = argparse.ArgumentParser(
    formatter_class=argparse.RawTextHelpFormatter,
    description="Qwen3.5 multimodal offline inference (native ATOM engine)",
)

EngineArgs.add_cli_args(parser)

parser.add_argument(
    "--image", type=str, required=True, help="Path to input image file"
)
parser.add_argument(
    "--prompt",
    type=str,
    default="Describe this image in detail.",
    help="Text prompt to accompany the image",
)
parser.add_argument(
    "--temperature", type=float, default=0.6, help="Temperature for sampling"
)
parser.add_argument(
    "--max-tokens", type=int, default=512, help="Max tokens to generate"
)


def main():
    args = parser.parse_args()

    # Force eager mode and single-batch cudagraph sizes for simplicity
    args.cudagraph_capture_sizes = "[1]"

    # Load processor (handles image preprocessing + chat template)
    processor = AutoProcessor.from_pretrained(
        args.model, trust_remote_code=True
    )

    # Load and preprocess image
    image = Image.open(args.image).convert("RGB")

    # Build chat messages with image
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": args.prompt},
            ],
        }
    ]

    # Apply chat template to get the text with image placeholders
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    print(f"Formatted prompt (first 500 chars): {text[:500]}")

    # Process text + image to get input_ids, pixel_values, image_grid_thw
    inputs = processor(
        text=[text],
        images=[image],
        return_tensors="pt",
    )

    input_ids = inputs["input_ids"][0].tolist()
    print(f"Input token count: {len(input_ids)}")

    # Build multimodal data dict
    multimodal_data = {
        "pixel_values": inputs["pixel_values"],
        "image_grid_thw": inputs["image_grid_thw"],
    }

    print(f"pixel_values shape: {multimodal_data['pixel_values'].shape}")
    print(f"image_grid_thw: {multimodal_data['image_grid_thw']}")

    # Create engine
    engine_args = EngineArgs.from_cli_args(args)
    llm = engine_args.create_engine()

    sampling_params = SamplingParams(
        temperature=args.temperature, max_tokens=args.max_tokens
    )

    # Run multimodal generation
    print("\nStarting multimodal inference...")
    outputs = llm.generate_multimodal(
        [input_ids],
        sampling_params,
        [multimodal_data],
    )

    # Print results
    for output in outputs:
        print("\n" + "=" * 70)
        print(f"Generated text:\n{output['text']}")
        print(f"\nInput tokens: {output['num_tokens_input']}")
        print(f"Output tokens: {output['num_tokens_output']}")
        print(f"Latency: {output['latency']:.2f}s")
        print(f"TTFT: {output['ttft']:.3f}s")
        print(f"TPOT: {output['tpot']:.3f}s")
        print(f"Finish reason: {output['finish_reason']}")
        print("=" * 70)

    llm.close()


if __name__ == "__main__":
    main()
