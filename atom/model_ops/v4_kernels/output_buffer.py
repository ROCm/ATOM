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


def call_with_output(
    kernel,
    args,
    out,
    *,
    output_strides=None,
    supports_strided_output=False,
    supports_output=None,
    **kwargs,
):
    """Write directly only when the destination satisfies the backend's layout.

    The default requires contiguous storage. Backends sharing Q/O strides
    supply ``output_strides``; their temporary must preserve those strides too.
    Backends with independent output strides set ``supports_strided_output``.
    ``supports_output`` may be set for APIs whose output argument is guaranteed
    (native wrappers and the existing BF16 OPUS API). Other AITER APIs are
    inspected once. Kernel TypeErrors always propagate unchanged.
    """
    if out is None:
        return kernel(*args, **kwargs)
    accepts_output = (
        _accepts_output(kernel) if supports_output is None else supports_output
    )
    if accepts_output:
        compatible = (
            supports_strided_output or out.is_contiguous()
            if output_strides is None
            else out.stride() == output_strides
        )
        if compatible:
            return kernel(*args, out=out, **kwargs)
        if output_strides is not None:
            # empty_like can compact a row-strided Q, which is unsafe when
            # the kernel uses Q's strides for output stores as well.
            temporary = torch.empty_strided(
                out.shape, output_strides, dtype=out.dtype, device=out.device
            )
            return out.copy_(kernel(*args, out=temporary, **kwargs))
    out.copy_(kernel(*args, **kwargs))
    return out
