# SPDX-License-Identifier: Apache-2.0
"""Example: Generate images with Flux."""

import argparse
from PIL import Image
from atom.model_engine.arg_utils import EngineArgs
from atom.model_engine.diffusion_runner import DiffusionModelRunner


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--prompt", default="A beautiful sunset over mountains")
    parser.add_argument("--output", default="output.png")
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--guidance", type=float, default=3.5)
    EngineArgs.add_cli_args(parser)
    args = parser.parse_args()

    config = EngineArgs.from_cli_args(args).create_atom_config()
    runner = DiffusionModelRunner(config)
    images = runner.generate(
        [args.prompt], args.height, args.width, args.steps, args.guidance
    )

    img = Image.fromarray(
        (images[0].permute(1, 2, 0).cpu().numpy() * 255).astype("uint8")
    )
    img.save(args.output)
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
