# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

"""Download a HuggingFace model checkpoint.

Thin CLI wrapper around :func:`atomrl.lumen.models.utils.download_hf_model`.

Usage:
    python download_model.py --model_name Qwen/Qwen3-8B --output_dir /data/model
    python download_model.py --output_dir /data/model --verify
"""

import argparse
import logging

from atomrl.lumen.models.utils import download_hf_model

logging.basicConfig(level=logging.INFO, format="%(message)s")

DEFAULT_MODEL = "Qwen/Qwen3-8B"
DEFAULT_REVISION = "b968826d9c46dd6066d109eabc6255188de91218"


def main():
    parser = argparse.ArgumentParser(description="Download model from HuggingFace")
    parser.add_argument(
        "--model_name",
        type=str,
        default=DEFAULT_MODEL,
        help="HuggingFace model name; custom repositories require --revision",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/data/model",
        help="Directory to save the downloaded model",
    )
    parser.add_argument(
        "--revision",
        default=None,
        help="Immutable HuggingFace commit",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        default=False,
        help="Compute and print SHA-256 hashes of downloaded weight files.",
    )
    args = parser.parse_args()
    revision = args.revision
    if revision is None:
        if args.model_name != DEFAULT_MODEL:
            parser.error("--revision is required with a custom --model_name")
        revision = DEFAULT_REVISION
    download_hf_model(
        args.model_name,
        args.output_dir,
        verify=args.verify,
        revision=revision,
    )


if __name__ == "__main__":
    main()
