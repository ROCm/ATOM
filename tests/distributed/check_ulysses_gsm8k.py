# SPDX-License-Identifier: MIT
"""Offline sampled GSM8K smoke check with identical few-shot prompts.

This is not the official lm-eval score. The final numeric answer is extracted
from each generation; retain the individual outputs to review disagreements.
Use the same --limit and --batch-size for TP/SP, and disable prefix caching.
"""

import argparse
import hashlib
import json
import re
from pathlib import Path

from datasets import load_dataset
from transformers import AutoTokenizer

from atom import SamplingParams
from atom.model_engine.arg_utils import EngineArgs
from atom.utils.arg_parser import FlexibleArgumentParser


def answer(text):
    boxed = re.findall(r"\\boxed\{([^{}]+)\}", text)
    values = re.findall(r"-?\d[\d,]*(?:\.\d+)?", boxed[-1] if boxed else text)
    return values[-1].replace(",", "").rstrip(".") if values else None


def main():
    p = FlexibleArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    EngineArgs.add_cli_args(p)
    p.add_argument("--result-file", required=True)
    p.add_argument("--limit", type=int, default=64)
    p.add_argument("--batch-size", type=int, default=16)
    a = p.parse_args()
    if a.enable_prefix_caching:
        p.error("pass --no-enable_prefix_caching")
    if a.limit < 1 or a.batch_size < 1:
        p.error("limit and batch size must be positive")
    tok = AutoTokenizer.from_pretrained(a.model, trust_remote_code=True)
    train = load_dataset("openai/gsm8k", "main", split="train")
    test = load_dataset("openai/gsm8k", "main", split="test")
    examples = []
    for row in train.select(range(5)):
        examples += [
            {"role": "user", "content": row["question"]},
            {"role": "assistant", "content": row["answer"]},
        ]
    selected = list(test.select(range(a.limit)))
    prompts = []
    for row in selected:
        messages = (
            [
                {
                    "role": "system",
                    "content": "Solve each math problem. Give concise reasoning and put the final numeric answer after ####.",
                }
            ]
            + examples
            + [{"role": "user", "content": row["question"]}]
        )
        prompts.append(
            tok.apply_chat_template(
                messages,
                tokenize=True,
                return_dict=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        )
    llm = EngineArgs.from_cli_args(a).create_engine()
    records = []
    try:
        for start in range(0, a.limit, a.batch_size):
            outputs = llm.generate(
                prompts[start : start + a.batch_size],
                SamplingParams(temperature=0, max_tokens=1024),
            )
            if len(outputs) != len(prompts[start : start + a.batch_size]):
                raise RuntimeError("engine did not complete every evaluation sample")
            for row, ids, out in zip(
                selected[start : start + a.batch_size],
                prompts[start : start + a.batch_size],
                outputs,
            ):
                expected = answer(row["answer"])
                actual = answer(out["text"])
                records.append(
                    {
                        "expected": expected,
                        "actual": actual,
                        "correct": actual == expected,
                        "prompt_sha256": hashlib.sha256(
                            json.dumps(ids).encode()
                        ).hexdigest(),
                        "output": out,
                    }
                )
            print(
                "SCORED", len(records), sum(r["correct"] for r in records), flush=True
            )
    finally:
        llm.close()
    Path(a.result_file).write_text(
        json.dumps(
            {
                "args": vars(a),
                "samples": records,
                "correct": sum(r["correct"] for r in records),
                "count": len(records),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
