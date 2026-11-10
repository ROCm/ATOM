# SPDX-License-Identifier: MIT
"""Output-buffer capability detection for supported AITER API versions."""

import inspect
from functools import cache

import torch


@cache
@torch.compiler.assume_constant_result
def _accepts_output(kernel):
    # AITER's guard exposes (*args, **kwargs), not the registered op's schema.
    # New guards retain the body; unknown/older wrappers use allocate-and-copy.
    body = getattr(kernel, "_eager_body", kernel)
    try:
        parameter = inspect.signature(body).parameters.get("out")
    except (TypeError, ValueError):
        return False
    return parameter is not None and parameter.kind in (
        inspect.Parameter.POSITIONAL_OR_KEYWORD,
        inspect.Parameter.KEYWORD_ONLY,
    )


def call_with_output(kernel, args, out, **kwargs):
    """Detect each callable once; kernel TypeErrors always propagate unchanged."""
    if out is None:
        return kernel(*args, **kwargs)
    if _accepts_output(kernel):
        return kernel(*args, out=out, **kwargs)
    out.copy_(kernel(*args, **kwargs))
    return out
