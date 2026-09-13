# SPDX-License-Identifier: MIT
"""Independent real-checkpoint numerical validation (not a throughput test).

Run with torchrun -m tests.models.deepseek_v41.validate_checkpoint. Full model
quality is evaluated separately using the lm-evaluation-harness adapter.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch
from aiter.dist.parallel_state import (
    destroy_distributed_environment,
    destroy_model_parallel,
    get_tp_group,
)

from atom.examples.deepseek_v41_offline import (
    initialize_parallel,
    load_offline_model,
    prepare_engram,
)

from .checkpoint_reference import checkpoint_reference
from .reference import FIXTURES, MANIFEST


def compare_logits(actual, expected, labels):
    actual, expected = actual.float(), expected.float()
    if not (torch.isfinite(actual).all() and torch.isfinite(expected).all()):
        raise AssertionError("Non-finite checkpoint logits")
    left, right = actual.log_softmax(-1), expected.log_softmax(-1)
    top = expected.topk(2, dim=-1)
    chosen = actual.argmax(-1)
    mismatch = chosen != top.indices[..., 0]
    margin = top.values[..., 0] - top.values[..., 1]
    regret = top.values[..., 0] - expected.gather(-1, chosen.unsqueeze(-1)).squeeze(-1)
    kl = (right.exp() * (right - left)).sum(-1)
    count = labels.numel()
    selected = labels.reshape(1, -1, 1)
    centered_a = actual - actual.mean(-1, keepdim=True)
    centered_b = expected - expected.mean(-1, keepdim=True)
    return {
        "tokens": count,
        "target_nll_sum": -left[:, :count].gather(-1, selected).sum().item(),
        "reference_nll_sum": -right[:, :count].gather(-1, selected).sum().item(),
        "positions": chosen.numel(),
        "top1_matches": (~mismatch).sum().item(),
        "kl_sum": kl.sum().item(),
        "kl_max": kl.max().item(),
        "logits_relative_l2": ((actual - expected).norm() / expected.norm()).item(),
        "centered_logits_relative_l2": (
            (centered_a - centered_b).norm() / centered_b.norm()
        ).item(),
        "mismatches": [
            {
                "position": int(i),
                "reference": int(top.indices[0, i, 0]),
                "target": int(chosen[0, i]),
                "margin": float(margin[0, i]),
                "regret": float(regret[0, i]),
                "kl": float(kl[0, i]),
            }
            for i in mismatch[0].nonzero().flatten().tolist()
        ],
    }


def summarize_records(records):
    """Aggregate token-weighted errors without hiding individual mismatches."""
    tokens = sum(row["tokens"] for row in records)
    positions = sum(row["positions"] for row in records)
    if not tokens or not positions:
        raise ValueError("Numerical validation needs scored tokens and positions")
    target = sum(row["target_nll_sum"] for row in records) / tokens
    reference = sum(row["reference_nll_sum"] for row in records) / tokens
    return {
        "tokens": tokens,
        "positions": positions,
        "target_mean_nll": target,
        "reference_mean_nll": reference,
        "mean_nll_delta": target - reference,
        "top1_agreement": sum(row["top1_matches"] for row in records) / positions,
        "mean_kl": sum(row["kl_sum"] for row in records) / positions,
        "max_kl": max(row["kl_max"] for row in records),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gsm8k-samples", type=int, default=32)
    parser.add_argument("--long-length", type=int, default=2049)
    parser.add_argument("--repeatability-runs", type=int, default=4)
    parser.add_argument("--max-nll-increase", type=float, default=0.01)
    args = parser.parse_args()
    if not math.isfinite(args.max_nll_increase) or args.max_nll_increase < 0:
        parser.error("--max-nll-increase must be finite and nonnegative")
    if args.repeatability_runs < 1:
        parser.error("--repeatability-runs must be positive")
    rank = initialize_parallel()
    torch.set_num_threads(4)
    records = []
    try:
        with load_offline_model(args.model, max(4096, args.long_length + 4)) as (
            target,
            tokenizer,
            mapping,
            host,
        ):
            if rank == 0:
                print(
                    json.dumps(
                        {
                            "event": "target_loaded",
                            "hbm_gib": torch.cuda.memory_allocated() / 2**30,
                        }
                    ),
                    flush=True,
                )
            with checkpoint_reference(
                args.model, target.config, target.max_length
            ) as reference:
                if rank == 0:
                    print(
                        json.dumps(
                            {
                                "event": "reference_loaded",
                                "hbm_gib": torch.cuda.memory_allocated() / 2**30,
                            }
                        ),
                        flush=True,
                    )
                cases = json.loads((FIXTURES / "text_inputs.json").read_text())
                if args.gsm8k_samples:
                    from datasets import load_dataset

                    dataset = load_dataset("openai/gsm8k", "main", split="test")
                    for index in range(args.gsm8k_samples):
                        sample = dataset[index]
                        text = f"Question: {sample['question']}\nAnswer: {sample['answer']}"
                        cases.append(
                            {
                                "text": f"gsm8k/test/{index}",
                                "input_ids": tokenizer.encode(
                                    text, add_special_tokens=True
                                ),
                            }
                        )
                if args.long_length:
                    base = tokenizer.encode(
                        "A cache stores previous keys. Each request has a private sliding window; global keys may be shared by layers.\n",
                        add_special_tokens=False,
                    )
                    cases.append(
                        {
                            "text": "long-cache-contract",
                            "input_ids": (
                                [tokenizer.bos_token_id]
                                + base * (args.long_length // len(base) + 1)
                            )[: args.long_length],
                        }
                    )
                for case in cases:
                    tokens = case["input_ids"]
                    cache = target.new_cache(1)
                    history = np.full((1, 3), -1, dtype=np.int64)
                    for position, length in [(0, len(tokens) - 3)] + [
                        (i, 1) for i in range(len(tokens) - 3, len(tokens))
                    ]:
                        chunk = tokens[position : position + length]
                        embeddings, history = prepare_engram(
                            chunk, position, history, mapping, host
                        )
                        ids = torch.tensor([chunk], device="cuda")
                        with torch.inference_mode():
                            actual = target(ids, cache, embeddings, full_logits=True)
                            if position == 0:
                                for _ in range(args.repeatability_runs - 1):
                                    repeated_cache = target.new_cache(1)
                                    repeated = target(
                                        ids,
                                        repeated_cache,
                                        embeddings,
                                        full_logits=True,
                                    )
                                    torch.testing.assert_close(
                                        repeated,
                                        actual,
                                        rtol=0,
                                        atol=0,
                                        msg="Identical checkpoint prefill is not repeatable",
                                    )
                                    del repeated, repeated_cache
                            expected = reference(
                                ids, position, embeddings, full_logits=True
                            )
                            labels = torch.tensor(
                                tokens[position + 1 : position + length + 1],
                                device="cuda",
                                dtype=torch.int64,
                            )
                            row = {
                                "case": case["text"],
                                "input_sha256": hashlib.sha256(
                                    json.dumps(tokens).encode()
                                ).hexdigest(),
                                "position": position,
                                "length": length,
                                **compare_logits(actual, expected, labels),
                            }
                        records.append(row)
                        if rank == 0:
                            print(json.dumps({"event": "step", **row}), flush=True)
                            args.output.write_text(
                                json.dumps(
                                    {
                                        "reference": MANIFEST,
                                        "tensor_parallel_size": get_tp_group().world_size,
                                        "kernel_alignment": False,
                                        "collective_substitution": False,
                                        "target_custom_all_reduce": False,
                                        "repeatability_runs": args.repeatability_runs,
                                        "completed": False,
                                        "expected_records": 4 * len(cases),
                                        "records": records,
                                    },
                                    indent=2,
                                    allow_nan=False,
                                )
                            )
                        del actual, expected
                summary = summarize_records(records)
                passed = summary["mean_nll_delta"] <= args.max_nll_increase
                if rank == 0:
                    report = json.loads(args.output.read_text())
                    report.update(
                        completed=True,
                        summary=summary,
                        max_nll_increase=args.max_nll_increase,
                        passed=passed,
                    )
                    args.output.write_text(
                        json.dumps(report, indent=2, allow_nan=False)
                    )
                    print(
                        json.dumps({"event": "complete", "passed": passed, **summary}),
                        flush=True,
                    )
                if not passed:
                    raise AssertionError(
                        f"Mean NLL increase {summary['mean_nll_delta']:.6f} exceeds "
                        f"{args.max_nll_increase:.6f} nats/token"
                    )
    finally:
        destroy_model_parallel()
        destroy_distributed_environment()


if __name__ == "__main__":
    main()
