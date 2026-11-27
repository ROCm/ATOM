# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

"""NeMo-RL v0.8's performance-table metrics and correctness gate from an AtomRL log.

NeMo-RL (``nemo_rl/algorithms/utils.py::print_performance_metrics``):
  * Tokens/sec/GPU = total tokens of the step (prompt + response, every sample)
    / total_step_time / every GPU of the cluster;
  * G-Average Seq len = mean generated tokens per sample;
  * the table averages steps 3-8 of a 10-step run.
Gate of its perf suite (``tests/test_suites/llm/performance/grpo-qwen3-30ba3b-4n8g.sh``):
  median(token_mult_prob_error) < 1.1 and token_mult_prob_error at the last step < 1.1.

Input is the trainer log, or the ``step=N k=v ...`` lines copied from it:

    python3 metrics.py <trainer.log> --gpus 16 [--first 3 --last 8] [--json out.json]
"""

from __future__ import annotations

import argparse
import json
import math
import re
import statistics
import sys

STEP_RE = re.compile(r"step=(\d+) (.*)$")
KV_RE = re.compile(r"(\S+?)=(\S+)")

TIMING_KEYS = {
    "gen": "timing_s/gen",
    "old_logprob": "timing_s/old_log_prob",
    "ref_logprob": "timing_s/ref",
    "reward": "timing_s/reward",
    "adv": "timing_s/adv",
    "train": "timing_s/update_actor",
    "weight_sync": "timing_s/update_weights",
    "validation": "timing_s/testing",
}


def parse_steps(lines) -> dict[int, dict[str, float]]:
    steps: dict[int, dict[str, float]] = {}
    for line in lines:
        m = STEP_RE.search(line)
        if not m:
            continue
        vals = {}
        for k, v in KV_RE.findall(m.group(2)):
            try:
                vals[k] = float(v)
            except ValueError:
                continue
        if "timing_s/step" in vals:
            steps[int(m.group(1))] = vals
    return steps


def summarize(steps, gpus: int, first: int, last: int) -> dict:
    window = [s for s in range(first, last + 1) if s in steps]
    if not window:
        raise SystemExit(f"no steps in [{first}, {last}]; have {sorted(steps)}")

    def mean(key, ss=window):
        vals = [steps[s][key] for s in ss if key in steps[s]]
        return statistics.fmean(vals) if vals else None

    per_step_tps = [
        steps[s]["perf/total_num_tokens"] / steps[s]["timing_s/step"] / gpus
        for s in window
    ]
    tokens = sum(steps[s]["perf/total_num_tokens"] for s in window)
    seconds = sum(steps[s]["timing_s/step"] for s in window)
    samples = [
        steps[s]["perf/total_num_tokens"]
        / (steps[s]["prompt_length/mean"] + steps[s]["response_length/mean"])
        for s in window
    ]
    out = {
        "window": [first, last],
        "steps_present": sorted(steps),
        "gpus": gpus,
        "g_average_seq_len": mean("response_length/mean"),
        "prompt_len_mean": mean("prompt_length/mean"),
        "tokens_per_sec_per_gpu": statistics.fmean(per_step_tps),
        "tokens_per_sec_per_gpu_ratio_of_sums": tokens / seconds / gpus,
        "total_step_time_s": mean("timing_s/step"),
        "samples_per_step": statistics.fmean(samples),
        "breakdown_s": {name: mean(key) for name, key in TIMING_KEYS.items()},
        "gen_tokens_per_sec_per_gpu": None,
        "reward_accuracy": mean("reward/accuracy"),
    }
    gen_t = out["breakdown_s"]["gen"]
    if gen_t:
        out["gen_tokens_per_sec_per_gpu"] = (
            statistics.fmean(
                [
                    steps[s]["response_length/mean"] * out["samples_per_step"]
                    for s in window
                ]
            )
            / gen_t
            / gpus
        )
    return out


def gate(steps) -> dict:
    ordered = sorted(steps)
    mpe = {
        s: steps[s]["rollout_corr/token_mult_prob_error"]
        for s in ordered
        if "rollout_corr/token_mult_prob_error" in steps[s]
    }
    finite = all(
        math.isfinite(v)
        for s in ordered
        for k, v in steps[s].items()
        if k.rsplit("/", 1)[-1] in ("grad_norm", "loss")
    )
    checks = {
        "token_mult_prob_error_present": bool(mpe),
        "token_mult_prob_error_median_lt_1.1": bool(mpe)
        and statistics.median(mpe.values()) < 1.1,
        "token_mult_prob_error_last_lt_1.1": bool(mpe) and mpe[max(mpe)] < 1.1,
        "loss_and_grad_norm_finite": finite,
    }
    return {
        "checks": checks,
        "pass": all(checks.values()),
        "token_mult_prob_error": mpe,
        "rollout_corr_k3_kl": {
            s: steps[s]["rollout_corr/k3_kl"]
            for s in ordered
            if "rollout_corr/k3_kl" in steps[s]
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("log")
    ap.add_argument("--gpus", type=int, required=True, help="every GPU of the run")
    ap.add_argument("--first", type=int, default=3)
    ap.add_argument("--last", type=int, default=8)
    ap.add_argument("--json")
    args = ap.parse_args()

    with open(args.log, encoding="utf-8", errors="replace") as f:
        steps = parse_steps(f)
    if not steps:
        sys.exit(f"no step lines in {args.log}")
    last = min(args.last, max(steps))
    first = min(args.first, last)
    result = {
        "summary": summarize(steps, args.gpus, first, last),
        "gate": gate(steps),
    }
    text = json.dumps(result, indent=2, default=str)
    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            f.write(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
