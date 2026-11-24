# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

"""Convert pretraining dataset to the format expected by PretrainTextDataset.

PretrainTextDataset reads either:
  - plain text (.txt): one document per line
  - jsonl (.jsonl): each line is {"text": "..."}

This script handles two common source formats:

1. **HuggingFace C4 jsonl** (already in the right format — validates and
   optionally filters/samples).
2. **Raw text directory** — concatenates all .txt files in a directory into
   a single jsonl file (one document = one line).

Usage:
    # Validate / subsample existing C4 jsonl
    python convert_dataset.py --input_dir /data/c4 --output_dir /data \
        --max_train_samples 500000

    # Convert a directory of .txt files to jsonl
    python convert_dataset.py --input_dir /data/raw_text --output_dir /data \
        --input_format txt
"""

import argparse
import json
import logging
import os
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def convert_jsonl(input_path: str, output_path: str, max_samples: int = 0) -> int:
    """Validate / subsample a jsonl file with {"text": ...} entries."""
    count = 0
    skipped = 0

    with (
        open(input_path, "r", encoding="utf-8") as fin,
        open(output_path, "w", encoding="utf-8") as fout,
    ):
        for line in fin:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                skipped += 1
                continue

            text = obj.get("text", "")
            if not text.strip():
                skipped += 1
                continue

            fout.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
            count += 1
            if max_samples > 0 and count >= max_samples:
                break

    logger.info("  %s: %d samples (skipped %d)", output_path, count, skipped)
    return count


def convert_txt_dir(input_dir: str, output_path: str, max_samples: int = 0) -> int:
    """Convert a directory of .txt files to a single jsonl file."""
    txt_files = sorted(Path(input_dir).glob("*.txt"))
    if not txt_files:
        logger.warning("  No .txt files found in %s", input_dir)
        return 0

    count = 0
    with open(output_path, "w", encoding="utf-8") as fout:
        for txt_file in txt_files:
            with open(txt_file, "r", encoding="utf-8") as fin:
                for line in fin:
                    text = line.strip()
                    if not text:
                        continue
                    fout.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
                    count += 1
                    if max_samples > 0 and count >= max_samples:
                        break
            if max_samples > 0 and count >= max_samples:
                break

    logger.info("  %s: %d samples from %d files", output_path, count, len(txt_files))
    return count


def main():
    parser = argparse.ArgumentParser(
        description="Convert pretraining data to PretrainTextDataset format"
    )
    parser.add_argument(
        "--input_dir", type=str, required=True, help="Directory containing source data"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/data",
        help="Directory to write output jsonl files",
    )
    parser.add_argument(
        "--input_format",
        type=str,
        default="jsonl",
        choices=["jsonl", "txt"],
        help="Input format: 'jsonl' (C4-style) or 'txt' (raw text dir)",
    )
    parser.add_argument(
        "--max_train_samples",
        type=int,
        default=0,
        help="Max training samples (0 = all)",
    )
    parser.add_argument(
        "--max_val_samples",
        type=int,
        default=0,
        help="Max validation samples (0 = all)",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    total = 0

    if args.input_format == "jsonl":
        split_map = {
            "train": ("train.jsonl", args.max_train_samples),
            "validation": ("validation.jsonl", args.max_val_samples),
        }
        for split_name, (filename, max_s) in split_map.items():
            in_path = os.path.join(args.input_dir, filename)
            out_path = os.path.join(args.output_dir, filename)
            if not os.path.exists(in_path):
                logger.info("  Skipping %s (not found: %s)", split_name, in_path)
                continue
            if in_path == out_path and max_s == 0:
                logger.info("  Skipping %s (input == output, no sampling)", split_name)
                continue
            n = convert_jsonl(in_path, out_path, max_s)
            total += n
    else:
        out_path = os.path.join(args.output_dir, "train.jsonl")
        total = convert_txt_dir(args.input_dir, out_path, args.max_train_samples)

    logger.info("Conversion complete. Total samples: %d", total)


if __name__ == "__main__":
    main()
