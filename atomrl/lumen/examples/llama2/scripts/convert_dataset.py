# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

"""Convert Gov Report dataset to SFT jsonl format for Megatron training.

Reads the raw Gov Report dataset (scrolls format) and produces jsonl files
with {"input": ..., "output": ...} entries suitable for LLaMA2SFTDataset.

Usage:
    python convert_dataset.py \
        --input_dir /data/gov_report \
        --output_dir /data \
        --max_input_length 7168 \
        --max_output_length 1024
"""

import argparse
import json
import os

SUMMARIZE_PROMPT = (
    "Below is a government report. Write a concise summary.\n\n"
    "Report:\n{document}\n\n"
    "Summary:"
)


def convert_split(
    input_path: str, output_path: str, max_input_len: int, max_output_len: int
):
    """Convert one split from scrolls format to SFT jsonl."""
    samples = []
    skipped = 0

    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)

            document = row.get("input", "")
            summary = row.get("output", "")

            if not document or not summary:
                skipped += 1
                continue

            if max_input_len > 0:
                words = document.split()
                if len(words) > max_input_len:
                    document = " ".join(words[:max_input_len])

            if max_output_len > 0:
                words = summary.split()
                if len(words) > max_output_len:
                    summary = " ".join(words[:max_output_len])

            samples.append(
                {
                    "input": SUMMARIZE_PROMPT.format(document=document),
                    "output": summary,
                }
            )

    with open(output_path, "w", encoding="utf-8") as f:
        f.writelines(json.dumps(s, ensure_ascii=False) + "\n" for s in samples)

    print(f"  {output_path}: {len(samples)} samples (skipped {skipped})")
    return len(samples)


def main():
    parser = argparse.ArgumentParser(description="Convert Gov Report to SFT jsonl")
    parser.add_argument("--input_dir", type=str, default="/data/gov_report")
    parser.add_argument("--output_dir", type=str, default="/data")
    parser.add_argument(
        "--max_input_length",
        type=int,
        default=7168,
        help="Max number of words in the input document (truncated if longer)",
    )
    parser.add_argument(
        "--max_output_length",
        type=int,
        default=1024,
        help="Max number of words in the output summary",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    split_map = {
        "train": "train.jsonl",
        "validation": "validation.jsonl",
        "test": "test.jsonl",
    }

    total = 0
    for split_name, out_filename in split_map.items():
        in_path = os.path.join(args.input_dir, f"{split_name}.jsonl")
        if not os.path.exists(in_path):
            print(f"  Skipping {split_name} (not found: {in_path})")
            continue
        out_path = os.path.join(args.output_dir, out_filename)
        n = convert_split(
            in_path, out_path, args.max_input_length, args.max_output_length
        )
        total += n

    print(f"Conversion complete. Total samples: {total}")


if __name__ == "__main__":
    main()
