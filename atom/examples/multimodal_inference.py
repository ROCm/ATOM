# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

import argparse
import json

from PIL import Image
from transformers import AutoProcessor

from atom import SamplingParams
from atom.model_engine.arg_utils import EngineArgs
from atom.model_engine.multimodal import build_multimodal_inputs
from atom.utils.arg_parser import FlexibleArgumentParser

parser = FlexibleArgumentParser(
    formatter_class=argparse.RawTextHelpFormatter,
    description=(
        "Generic image+text multimodal offline inference using the native ATOM engine.\n"
        "Validated with Qwen3.5 and Kimi-K3. The script relies on the model's\n"
        "Hugging Face processor and chat template, plus the architecture-specific\n"
        "input builders in atom.model_engine.multimodal for processors that do not\n"
        "follow the Qwen convention."
    ),
)

EngineArgs.add_cli_args(parser)

parser.add_argument(
    "--image",
    type=str,
    action="append",
    required=True,
    help="Path to an input image file. Repeat for multi-image prompts.",
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
parser.add_argument(
    "--chat-template-kwargs",
    type=str,
    default="{}",
    help="JSON kwargs passed to processor.apply_chat_template, e.g. '{\"enable_thinking\": false}'",
)


def main():
    args = parser.parse_args()
    chat_template_kwargs = json.loads(args.chat_template_kwargs)

    # Force eager mode and single-batch cudagraph sizes for simplicity
    args.cudagraph_capture_sizes = "[1]"

    # Load processor (handles media preprocessing and chat template)
    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)

    images = [Image.open(path).convert("RGB") for path in args.image]

    # Build chat messages. Image parts intentionally precede text so Qwen-style
    # templates emit image placeholders in the same order as the media tensors.
    messages = [
        {
            "role": "user",
            "content": [
                *({"type": "image", "image": image} for image in images),
                {"type": "text", "text": args.prompt},
            ],
        }
    ]

    # Create the engine first: the architecture-specific input builders are
    # selected from its resolved config.
    engine_args = EngineArgs.from_cli_args(args)
    llm = engine_args.create_engine()

    built = build_multimodal_inputs(
        llm.io_processor.config,
        processor,
        messages,
        images,
        chat_template_kwargs,
    )
    if built is not None:
        input_ids, multimodal_data = built
    else:
        # Default (Qwen-style) path: the template expands image placeholders
        # itself, so tokenize the rendered text together with the images.
        text = processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            **chat_template_kwargs,
        )
        print(f"Formatted prompt (first 500 chars): {text[:500]}")
        inputs = processor(text=[text], images=images, return_tensors="pt")
        input_ids = inputs["input_ids"][0].tolist()
        multimodal_data = {
            "pixel_values": inputs["pixel_values"],
            "image_grid_thw": inputs["image_grid_thw"],
        }

    print(f"Input token count: {len(input_ids)}")
    print(f"pixel_values shape: {multimodal_data['pixel_values'].shape}")
    print(f"image_grid_thw: {multimodal_data['image_grid_thw']}")

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
