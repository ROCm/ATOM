# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""The tail of one DSpark Markov draft step on gfx950, FlyDSL: the Gumbel combine of the per-block partials (vLLM's
``_gumbel_combine_kernel``), the draft-id store, the ``prev`` update and the next step's ``markov_w1`` row (in
``gemm/bf16_n200064_k256``'s packed input layout) in one launch (vocab 200064, Markov rank 256). The torch contract is
``reference.py``."""

import functools


@functools.cache
def available() -> bool:
    """gfx950 with FlyDSL importable."""
    import torch

    if not torch.cuda.is_available() or "gfx950" not in torch.cuda.get_device_properties(0).gcnArchName:
        return False
    try:
        import flydsl  # noqa: F401
    except ImportError:
        return False
    return True


def __getattr__(name):
    if name in __all__ and name != "available":
        from . import host

        return getattr(host, name)
    raise AttributeError(name)


__all__ = ["available", "gumbel_combine_markov_embed_packed", "warmup_gumbel_combine_markov_embed"]
