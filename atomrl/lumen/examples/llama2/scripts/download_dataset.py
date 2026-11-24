# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

"""Download a HuggingFace dataset and save splits as jsonl.

Thin CLI wrapper around :func:`atomrl.lumen.models.utils.download_hf_dataset`.

Usage:
    python download_dataset.py --output_dir /data/gov_report
    python download_dataset.py --output_dir /data/gov_report --verify
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
        default="/data/gov_report",
        help="Directory to save the dataset",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="tau/scrolls",
        help="HuggingFace dataset name",
    )
    parser.add_argument(
        "--subset",
        type=str,
        default="gov_report",
        help="Dataset subset name",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        default=False,
        help="Compute and print SHA-256 hashes of output files for verification.",
    )
    args = parser.parse_args()
    download_hf_dataset(
        args.dataset_name, args.output_dir, subset=args.subset, verify=args.verify
    )


if __name__ == "__main__":
    main()
