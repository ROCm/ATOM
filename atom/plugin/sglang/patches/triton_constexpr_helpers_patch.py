"""Restore Triton's constexpr ``next_power_of_2`` after SGLang replaces it.

SGLang swaps it for a plain function at import; Triton 3.8 kernels that call
it in their body (AITER's V4.1 mHC kernel) then fail their dependency hash.
DeepSeek-V4.1 only (`prepare_model`): a host call to the constexpr one costs
about 1.56 us against SGLang's 75 ns.
"""

import logging

import triton
from triton.runtime.jit import ConstexprFunction

logger = logging.getLogger("atom")


def apply_triton_constexpr_helpers_patch() -> bool:
    """Install the constexpr ``next_power_of_2``; False if it already is one."""
    current = triton.next_power_of_2
    if isinstance(current, ConstexprFunction):
        return False

    @triton.constexpr_function
    def next_power_of_2(n: int):
        # SGLang's host semantics (1 for n <= 0); Triton's for every n > 0.
        return 1 << (n - 1).bit_length() if n > 0 else 1

    triton.next_power_of_2 = next_power_of_2
    logger.info(
        "ATOM plugin: restored Triton's constexpr next_power_of_2 (replaced by %s)",
        current.__module__,
    )
    return True
