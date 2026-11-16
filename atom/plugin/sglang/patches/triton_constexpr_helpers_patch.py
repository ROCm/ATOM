"""Restore Triton's in-kernel ``next_power_of_2`` after SGLang replaces it.

``sglang.srt.utils.common`` runs ``setattr(triton, "next_power_of_2", ...)``
with a plain Python function at import time. Triton 3.8 defines it as a
``constexpr_function`` so kernels may call it in their bodies; the plain
replacement makes every such kernel fail its dependency hash with
``Unsupported function referenced: <function next_power_of_2>``. AITER's
DeepSeek-V4.1 mHC kernel (``mhc_fused_post_pre_delayed_rmsnorm``) is one.
A constexpr function is also callable on the host, so SGLang's own callers
keep working; the replacement keeps their result for ``n <= 0`` as well.
"""

import logging

logger = logging.getLogger("atom")


def apply_triton_constexpr_helpers_patch() -> None:
    import triton

    constexpr_function = getattr(triton, "constexpr_function", None)
    if constexpr_function is None:
        # Older Triton: next_power_of_2 is host-only there anyway.
        return
    from triton.runtime.jit import ConstexprFunction

    current = triton.next_power_of_2
    if isinstance(current, ConstexprFunction):
        return

    @constexpr_function
    def next_power_of_2(n: int):
        # SGLang's host semantics (1 for n <= 0); Triton's for every n > 0.
        return 1 << (n - 1).bit_length() if n > 0 else 1

    triton.next_power_of_2 = next_power_of_2
    logger.info(
        "ATOM plugin: restored Triton's constexpr next_power_of_2 (replaced by %s)",
        getattr(current, "__module__", current),
    )
