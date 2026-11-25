# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

"""Download Qwen3-30B-A3B from HuggingFace.

Usage:
    DATA_ROOT=/data python download_model.py
    python download_model.py --model_name Qwen/Qwen3-30B-A3B --output_dir /data/model
"""

import argparse
import logging
import os
from pathlib import Path

from atomrl.lumen.models.utils import download_hf_model

logging.basicConfig(level=logging.INFO, format="%(message)s")

DEFAULT_MODEL = "Qwen/Qwen3-30B-A3B"
DEFAULT_REVISION = "ad44e777bcd18fa416d9da3bd8f70d33ebb85d39"


def main():
    parser = argparse.ArgumentParser(description="Download Qwen3-30B-A3B model")
    parser.add_argument(
        "--model_name",
        type=str,
        default=DEFAULT_MODEL,
        help="HuggingFace model name; custom repositories require --revision",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=str(
            Path(os.environ.get("DATA_ROOT", "/data")) / "models" / "Qwen3-30B-A3B"
        ),
        help="Directory to save the downloaded model",
    )
    parser.add_argument(
        "--revision",
        default=None,
        help="Immutable HuggingFace commit",
    )
    parser.add_argument("--verify", action="store_true", default=False)
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
