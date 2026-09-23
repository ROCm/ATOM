# SPDX-License-Identifier: MIT
"""Report full-logit differences, including each configuration's A/A variation.

Use directories produced by validate_prefill_parallel. No accuracy threshold
is inferred from these measurements; task evaluation is a separate check.
"""

import argparse
import itertools
import json
from collections import defaultdict
from pathlib import Path

import torch


def read_probes(directory):
    outputs = json.loads((directory / "outputs.json").read_text())
    prefills = []
    for path in sorted(directory.glob("logits-*.pt")):
        saved = torch.load(path, map_location="cpu", weights_only=True)
        if saved["prefill_tokens"]:
            prefills.append(saved)
    if len(prefills) != len(outputs):
        raise ValueError(f"{directory}: prefill dumps do not match request count")
    probes = defaultdict(list)
    for saved, output in zip(prefills, outputs):
        length = output["length"]
        if saved["prefill_tokens"] != length:
            raise ValueError(f"{directory}: use an unchunked validation run")
        logits = saved["logits"].float()
        if not torch.isfinite(logits).all():
            raise ValueError(f"{directory}: nonfinite logits at length {length}")
        probes[length].append((output.get("prompt_sha256"), logits))
    return probes


def difference(a, b):
    hash_a, x = a
    hash_b, y = b
    if hash_a and hash_b and hash_a != hash_b:
        raise ValueError("cannot compare logits for different input tokens")
    if x.shape != y.shape:
        raise ValueError("vocabulary or batch size differs")
    delta = x - y
    log_p, log_q = x.log_softmax(-1), y.log_softmax(-1)
    return {
        "prompt_hash_verified": bool(hash_a and hash_b),
        "max_abs": delta.abs().max().item(),
        "rms": delta.square().mean().sqrt().item(),
        "kl_a_to_b": (log_p.exp() * (log_p - log_q)).sum(-1).mean().item(),
        "total_variation": (log_p.exp() - log_q.exp()).abs().sum(-1).mean().item() / 2,
        "argmax_a": x.argmax(-1).tolist(),
        "argmax_b": y.argmax(-1).tolist(),
        "exact_equal": torch.equal(x, y),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--result-file", type=Path, required=True)
    args = parser.parse_args()
    reference, candidate = read_probes(args.reference), read_probes(args.candidate)
    if reference.keys() != candidate.keys():
        parser.error("validation lengths differ")
    result = {"reference": str(args.reference), "candidate": str(args.candidate)}
    result["lengths"] = {
        length: {
            "reference_aa": [
                difference(a, b)
                for a, b in itertools.combinations(reference[length], 2)
            ],
            "candidate_aa": [
                difference(a, b)
                for a, b in itertools.combinations(candidate[length], 2)
            ],
            "cross": [
                difference(a, b)
                for a, b in itertools.product(reference[length], candidate[length])
            ],
        }
        for length in sorted(reference)
    }
    args.result_file.write_text(json.dumps(result, indent=2) + "\n")
    print(args.result_file)


if __name__ == "__main__":
    main()
