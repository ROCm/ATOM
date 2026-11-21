# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

"""Download LLaMA 3.1 model checkpoint from HuggingFace.

Thin CLI wrapper around :func:`atomrl.lumen.models.utils.download_hf_model`.

Usage:
    python download_model.py --model_name meta-llama/Llama-3.1-8B --output_dir /data/model
    python download_model.py --output_dir /data/model --verify
"""

import argparse
import logging

from atomrl.lumen.models.utils import download_hf_model

logging.basicConfig(level=logging.INFO, format="%(message)s")


def main():
    parser = argparse.ArgumentParser(
        description="Download LLaMA 3.1 model from HuggingFace"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="meta-llama/Llama-3.1-8B",
        help="HuggingFace model name (e.g. meta-llama/Llama-3.1-8B, "
        "meta-llama/Llama-3.1-8B-Instruct)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/data/model",
        help="Directory to save the downloaded model",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        default=False,
        help="Compute and print SHA-256 hashes of downloaded weight files.",
    )
    args = parser.parse_args()
    download_hf_model(args.model_name, args.output_dir, verify=args.verify)


if __name__ == "__main__":
    main()
