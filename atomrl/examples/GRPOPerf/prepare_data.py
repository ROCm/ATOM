# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

"""Build NeMo-RL v0.8's OpenMathInstruct-2 GRPO data as AtomRL parquet.

Same split and prompt as NeMo-RL's ``openmathinstruct2`` dataset: ``train_1M`` of
``nvidia/OpenMathInstruct-2`` (pinned revision), ``train_test_split(test_size=0.05,
seed=42)``, the user turn is ``examples/prompts/cot.txt`` formatted with the
problem, and the answer is ``expected_answer``. The trainer applies the chat
template at rollout time.

Run inside the release image (it has datasets, huggingface_hub, transformers):

    python3 atomrl/examples/GRPOPerf/prepare_data.py \
        --out $DATA_ROOT/data/omi2_train_1M --tokenizer $DATA_ROOT/models/Qwen3-8B
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

REPO = "nvidia/OpenMathInstruct-2"
REVISION = "469216e3f46f4dacf476b382e192485ea51a143e"
# NeMo-RL v0.8.0 examples/prompts/cot.txt, as a str.format template.
COT = (
    "Think step-by-step to solve the following problem. "
    "Output your answer inside of \\\\boxed{{}} tags.:\n{}\n\nLet's think step-by-step"
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--tokenizer", required=True, help="for the prompt-length stats")
    ap.add_argument("--cache", default=None, help="where the raw parquet files go")
    ap.add_argument("--stats-sample", type=int, default=20000)
    args = ap.parse_args()

    import datasets
    from huggingface_hub import snapshot_download
    from transformers import AutoTokenizer

    out = Path(args.out)
    if all((out / f).exists() for f in ("train.parquet", "val.parquet", "stats.json")):
        print(f"{out}: already prepared")
        return
    out.mkdir(parents=True, exist_ok=True)
    raw = Path(args.cache or out / "raw")
    snapshot_download(
        REPO,
        repo_type="dataset",
        revision=REVISION,
        allow_patterns=["data/train_1M-*.parquet"],
        local_dir=str(raw),
    )
    files = sorted(str(p) for p in (raw / "data").glob("train_1M-*.parquet"))
    assert len(files) == 3, files
    ds = datasets.load_dataset("parquet", data_files=files, split="train")

    def fmt(row):
        return {
            "prompt": [{"role": "user", "content": COT.format(row["problem"])}],
            "reward_model": {"ground_truth": row["expected_answer"], "style": "rule"},
            "data_source": "openmathinstruct2",
        }

    ds = ds.map(fmt, remove_columns=ds.column_names, num_proc=32)
    split = ds.train_test_split(test_size=0.05, seed=42)
    split["train"].to_parquet(str(out / "train.parquet"))
    split["test"].to_parquet(str(out / "val.parquet"))

    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    n = min(args.stats_sample, len(split["train"]))
    lens = sorted(
        len(
            tok(
                tok.apply_chat_template(
                    r["prompt"], tokenize=False, add_generation_prompt=True
                ),
                add_special_tokens=False,
            )["input_ids"]
        )
        for r in split["train"].select(range(n))
    )
    stats = {
        "repo": REPO,
        "revision": REVISION,
        "train_rows": len(split["train"]),
        "val_rows": len(split["test"]),
        "prompt_tokens_mean": sum(lens) / n,
        "prompt_tokens_p99": lens[int(n * 0.99)],
        "prompt_tokens_max": lens[-1],
    }
    (out / "stats.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
