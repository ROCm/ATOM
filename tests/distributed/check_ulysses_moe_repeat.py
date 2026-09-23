# SPDX-License-Identifier: MIT
"""Test-only A/A check of native BF16 MoE with identical layer inputs.

Accepts validate_prefill_parallel arguments. Set MOE_REPEAT_RESULT to a JSONL
path. Disables compilation so the wrapper runs for each real forward, skips
dummy/large batches, and compares the first six MoE calls on rank zero. No SP
communication runs between each pair of identical native fused_moe calls.
"""

import json
import os
from pathlib import Path

import torch

import atom.model_ops.moe as module
from atom.benchmarks.validate_prefill_parallel import main
from atom.utils.forward_context import get_forward_context

original = module.fused_moe
counter = 0


def checked(*args, **kwargs):
    global counter
    x = kwargs.get("hidden_states", args[0] if args else None)
    context = get_forward_context().context
    if (
        counter >= 6
        or context is None
        or context.is_dummy_run
        or not context.is_prefill
        or x.shape[0] > 2048
    ):
        return original(*args, **kwargs)
    first = original(*args, **kwargs).clone()
    second = original(*args, **kwargs)
    delta = (first.float() - second.float()).abs()
    if torch.distributed.get_rank() == 0:
        item = {
            "call": counter,
            "shape": list(x.shape),
            "max_abs": delta.max().item(),
            "mean_abs": delta.mean().item(),
            "different": int(torch.count_nonzero(delta).item()),
            "elements": first.numel(),
        }
        with Path(os.environ["MOE_REPEAT_RESULT"]).open("a") as f:
            f.write(json.dumps(item) + "\n")
    counter += 1
    return first


module.fused_moe = checked
if __name__ == "__main__":
    import sys

    sys.argv += ["--level", "0"]
    main()
