#!/usr/bin/env python3
"""Apply benchmark-only forced acceptance to vLLM speculative arguments."""

import json
import math
import shlex
import sys


def configure(role: str, phase: str, length: str, server_args: str) -> str:
    args = shlex.split(server_args)
    if not length:
        return shlex.join(args)
    if role not in ("prefill", "decode") or phase not in ("benchmark", "eval"):
        raise ValueError("Forced acceptance requires separate benchmark/eval phases")

    positions = [
        i
        for i, arg in enumerate(args)
        if arg == "--speculative-config" or arg.startswith("--speculative-config=")
    ]
    if len(positions) != 1:
        raise ValueError("Forced acceptance requires exactly one --speculative-config")
    index = positions[0]
    inline = args[index].startswith("--speculative-config=")
    if not inline and index + 1 == len(args):
        raise ValueError("Missing --speculative-config value")
    raw = args[index].split("=", 1)[1] if inline else args[index + 1]
    config = json.loads(raw)
    if not isinstance(config, dict) or config.get("method") != "dspark":
        raise ValueError("Forced acceptance is supported here only for DSpark")
    count = config.get("num_speculative_tokens")
    if type(count) is not int or count <= 0:
        raise ValueError("DSpark requires a positive integer num_speculative_tokens")
    target = float(length)
    if not math.isfinite(target) or not 1 <= target <= count + 1:
        raise ValueError(f"Acceptance length must be finite and in [1, {count + 1}]")

    config.pop("synthetic_acceptance_rates", None)
    config.pop("synthetic_acceptance_length", None)
    forced = role == "decode" and phase == "benchmark"
    config["rejection_sample_method"] = "synthetic" if forced else "standard"
    if forced:
        config["synthetic_acceptance_length"] = target
    value = json.dumps(config, separators=(",", ":"), allow_nan=False)
    if inline:
        args[index] = "--speculative-config=" + value
    else:
        args[index + 1] = value
    print(
        f"[vllm][spec] phase={phase} role={role} "
        f"rejection={config['rejection_sample_method']} "
        f"acceptance_length={target if forced else 'natural'}",
        file=sys.stderr,
    )
    return shlex.join(args)


def main() -> int:
    if len(sys.argv) != 5:
        print(
            "usage: pd_vllm_spec.py ROLE PHASE ACCEPTANCE_LENGTH SERVER_ARGS",
            file=sys.stderr,
        )
        return 2
    try:
        result = configure(*sys.argv[1:])
    except (ValueError, TypeError) as exc:
        print(f"[vllm][spec][FAIL] {exc}", file=sys.stderr)
        return 2
    print(result)
    return 0


if __name__ == "__main__":
    sys.exit(main())
