# SPDX-License-Identifier: MIT
"""Paired task scores and full-vocabulary teacher-forced numerical differences."""

import argparse
import json
from pathlib import Path

import torch

from .compare_lm_eval import paired_summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    baseline = json.loads(args.baseline.read_text())
    candidate = json.loads(args.candidate.read_text())
    for path, result in ((args.baseline, baseline), (args.candidate, candidate)):
        if not result.get("completed"):
            raise ValueError(f"Incomplete quality run: {path}")
        if (
            "gsm8k" in result["samples"]
            and (path.parent / "generation_invalid.json").exists()
        ):
            raise ValueError(f"Invalid generation evidence: {path}")
    if not baseline["baseline"] or candidate["baseline"]:
        raise ValueError("Expected non-speculative baseline and DSpark candidate")
    if baseline.get("chat") != candidate.get("chat"):
        raise ValueError("Generation prompt formats differ")
    report = {"tasks": paired_summary(candidate, baseline)}
    left = {p.name: p for p in (args.baseline.parent / "logits").glob("*.pt")}
    right = {p.name: p for p in (args.candidate.parent / "logits").glob("*.pt")}
    if left.keys() != right.keys():
        raise ValueError("Teacher-forced request identities differ")
    rows = []
    for name, path in left.items():
        a, b = torch.load(path, weights_only=True), torch.load(
            right[name], weights_only=True
        )
        if a["prompt"] != b["prompt"] or a["labels"] != b["labels"]:
            raise ValueError("Teacher-forced token inputs differ")
        x, y = a["logits"].double(), b["logits"].double()
        if x.shape != y.shape or not (
            torch.isfinite(x).all() and torch.isfinite(y).all()
        ):
            raise ValueError("Invalid paired logits")
        labels = torch.tensor(a["labels"], dtype=torch.long)
        logp, logq = x.log_softmax(-1), y.log_softmax(-1)
        p = logp.exp()
        delta = (logp - logq).gather(-1, labels[:, None]).squeeze(-1)
        kl = (p * (logp - logq)).sum(-1)
        top = x.topk(2, dim=-1)
        same = x.argmax(-1) == y.argmax(-1)
        rows.append(
            {
                "request": name,
                "tokens": len(labels),
                "baseline_nll_sum": float(-logp.gather(-1, labels[:, None]).sum()),
                "candidate_nll_sum": float(-logq.gather(-1, labels[:, None]).sum()),
                "delta_nll_sum": float(delta.sum()),
                "kl_sum": float(kl.sum()),
                "kl_max": float(kl.max()),
                "logit_error_max": float((x - y).abs().max()),
                "top1_matches": int(same.sum()),
                "margin_gt_01_mismatches": int(
                    ((top.values[:, 0] - top.values[:, 1] > 0.1) & ~same).sum()
                ),
            }
        )
    if rows:
        tokens = sum(row["tokens"] for row in rows)
        report["numerical"] = {
            "requests": len(rows),
            "tokens": tokens,
            **{
                name.removesuffix("_sum"): sum(row[name] for row in rows) / tokens
                for name in (
                    "baseline_nll_sum",
                    "candidate_nll_sum",
                    "delta_nll_sum",
                    "kl_sum",
                )
            },
            "top1_agreement": sum(row["top1_matches"] for row in rows) / tokens,
            "margin_gt_01_mismatches": sum(
                row["margin_gt_01_mismatches"] for row in rows
            ),
            "worst_kl_requests": sorted(
                rows, key=lambda row: row["kl_max"], reverse=True
            )[:10],
        }
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(
        json.dumps(
            {key: value for key, value in report.items() if key != "numerical"},
            indent=2,
        )
    )
    if rows:
        print(
            json.dumps(
                {
                    key: value
                    for key, value in report["numerical"].items()
                    if key != "worst_kl_requests"
                },
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
