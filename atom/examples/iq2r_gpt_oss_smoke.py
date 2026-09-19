# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Run a deterministic GPT-OSS IQ2R load and generation smoke test."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from atom import SamplingParams
from atom.model_engine.arg_utils import EngineArgs
from atom.utils.arg_parser import FlexibleArgumentParser


def _max_logprob_delta(first: list[float] | None, second: list[float] | None) -> float:
    if first is None or second is None:
        raise RuntimeError(
            "ATOM did not return requested sampled-token log probabilities"
        )
    if len(first) != len(second):
        return float("inf")
    return max((abs(a - b) for a, b in zip(first, second, strict=True)), default=0.0)


def main() -> int:
    parser = FlexibleArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description=__doc__,
    )
    EngineArgs.add_cli_args(parser)
    parser.add_argument(
        "--prompt",
        default="In one sentence, explain why the sky appears blue.",
    )
    parser.add_argument("--max-tokens", type=int, default=16)
    parser.add_argument("--repetitions", type=int, default=2)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if os.environ.get("HIP_VISIBLE_DEVICES") != "6":
        raise RuntimeError("GPT-OSS IQ2R qualification requires HIP_VISIBLE_DEVICES=6")
    if args.repetitions < 2:
        raise ValueError("--repetitions must be at least 2")

    engine_args = EngineArgs.from_cli_args(args)
    engine = engine_args.create_engine()
    try:
        sampling = SamplingParams(
            temperature=0.0,
            max_tokens=args.max_tokens,
            ignore_eos=True,
            logprobs=True,
        )
        outputs = [
            engine.generate([args.prompt], sampling)[0] for _ in range(args.repetitions)
        ]
    finally:
        engine.close()

    reference = outputs[0]
    token_ids_match = all(
        output["token_ids"] == reference["token_ids"] for output in outputs[1:]
    )
    max_logprob_delta = max(
        _max_logprob_delta(reference["logprobs"], output["logprobs"])
        for output in outputs[1:]
    )
    result = {
        "model": args.model,
        "hip_visible_devices": os.environ["HIP_VISIBLE_DEVICES"],
        "execution": "eager" if args.enforce_eager else "graph",
        "prompt": args.prompt,
        "max_tokens": args.max_tokens,
        "repetitions": args.repetitions,
        "token_ids_match": token_ids_match,
        "max_sampled_logprob_delta": max_logprob_delta,
        "outputs": outputs,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    temporary.replace(args.output)
    print(json.dumps(result, sort_keys=True))

    if not token_ids_match or max_logprob_delta != 0.0:
        raise RuntimeError("repeated greedy IQ2R runs were not bitwise deterministic")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
