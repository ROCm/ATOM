# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

"""Download alpaca-style dataset for Qwen3-30B-A3B SFT.

Usage:
    python download_dataset.py --output_dir <dataset-directory>
"""

import argparse
import logging

from atomrl.lumen.models.utils import download_hf_dataset

logging.basicConfig(level=logging.INFO, format="%(message)s")


def main():
    parser = argparse.ArgumentParser(description="Download HuggingFace dataset")
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="tatsu-lab/alpaca",
    )
    parser.add_argument("--subset", type=str, default=None)
    parser.add_argument("--verify", action="store_true", default=False)
    args = parser.parse_args()
    download_hf_dataset(
        args.dataset_name, args.output_dir, subset=args.subset, verify=args.verify
    )


if __name__ == "__main__":
    main()
