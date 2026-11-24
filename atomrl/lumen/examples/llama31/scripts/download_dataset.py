# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

"""Download C4 pretraining dataset from HuggingFace.

Downloads the English split of the C4 dataset (allenai/c4) and saves it as
jsonl files for use with :class:`PretrainTextDataset`.

Usage:
    # Download the default English streaming subset
    python download_dataset.py --output_dir /data/c4

    # Download a specific number of train/validation samples
    python download_dataset.py --output_dir /data/c4 \
        --max_train_samples 1000000 --max_val_samples 10000
"""

import argparse
import json
import logging
import os

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def download_c4(
    output_dir: str,
    max_train_samples: int = 0,
    max_val_samples: int = 0,
    verify: bool = False,
):
    """Download C4 English dataset and save as jsonl."""
    from datasets import load_dataset

    os.makedirs(output_dir, exist_ok=True)

    splits = {
        "train": ("train", max_train_samples),
        "validation": ("validation", max_val_samples),
    }

    for split_name, (hf_split, max_samples) in splits.items():
        output_path = os.path.join(output_dir, f"{split_name}.jsonl")
        logger.info("Downloading C4 %s split ...", split_name)

        ds = load_dataset(
            "allenai/c4",
            "en",
            split=hf_split,
            streaming=True,
            trust_remote_code=True,
        )

        count = 0
        with open(output_path, "w", encoding="utf-8") as f:
            for example in ds:
                text = example.get("text", "")
                if not text.strip():
                    continue
                f.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
                count += 1
                if max_samples > 0 and count >= max_samples:
                    break
                if count % 100000 == 0:
                    logger.info("  %s: %d samples written ...", split_name, count)

        logger.info("  %s: %d samples -> %s", split_name, count, output_path)

        if verify:
            from atomrl.lumen.models.utils import sha256_file

            digest = sha256_file(output_path)
            print(f"    SHA-256: {digest}")

    logger.info("Download complete.")


def main():
    parser = argparse.ArgumentParser(
        description="Download C4 pretraining dataset from HuggingFace"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/data/c4",
        help="Directory to save the dataset",
    )
    parser.add_argument(
        "--max_train_samples",
        type=int,
        default=0,
        help="Max number of training samples to download (0 = all)",
    )
    parser.add_argument(
        "--max_val_samples",
        type=int,
        default=0,
        help="Max number of validation samples to download (0 = all)",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        default=False,
        help="Compute and print SHA-256 hashes of output files.",
    )
    args = parser.parse_args()
    download_c4(
        args.output_dir,
        max_train_samples=args.max_train_samples,
        max_val_samples=args.max_val_samples,
        verify=args.verify,
    )


if __name__ == "__main__":
    main()
