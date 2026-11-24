# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

###############################################################################
#
# See LICENSE for license information.
###############################################################################

"""C++ FP8 quantization dispatch — Python wrapper.

Loads the ``atomrl.lumen.csrc._fp8_quant_dispatch`` C++ extension and provides
a probe function and factory for ``FP8QuantDispatcher``.

The C++ dispatcher replaces the Python hot path:
  get_scale() → _compute_scale_kernel → static_quant_with_amax → update_amax
with a single C++ call that owns amax history, computes scale inline,
launches one HIP kernel, and updates history.

Enabled via ``LUMEN_CPP_QUANT_DISPATCH=1``.
"""

from __future__ import annotations

import functools
import logging
import os

import torch

logger = logging.getLogger(__name__)


@functools.lru_cache(maxsize=1)
def _load_module():
    """Load the C++ extension (pre-built or JIT-compiled)."""
    # Try pre-built extension first
    try:
        import importlib

        return importlib.import_module("atomrl.lumen.csrc._fp8_quant_dispatch")
    except ImportError:
        pass

    # Fall back to JIT compilation
    from torch.utils.cpp_extension import load

    src_dir = os.path.join(os.path.dirname(__file__), "..", "..", "csrc")
    src_file = os.path.join(src_dir, "fp8_quant_dispatch.cu")
    if not os.path.exists(src_file):
        raise ImportError(f"C++ source not found: {src_file}")

    logger.info("JIT-compiling C++ FP8 quant dispatch extension...")
    return load(
        name="_fp8_quant_dispatch",
        sources=[src_file],
        extra_cuda_cflags=["-O3"],
        verbose=False,
    )


@functools.lru_cache(maxsize=1)
def _probe_cpp_quant_dispatch() -> bool:
    """Return True if the C++ FP8 quant dispatch extension is functional."""
    try:
        m = _load_module()
        # Smoke test: create dispatcher, quantize a tiny tensor
        d = m.FP8QuantDispatcher(240.0, 57344.0, 0, 4)
        x = torch.randn(2, 4, device="cuda", dtype=torch.bfloat16)
        fp8, scale = d.quantize("_probe", x, False)
        assert fp8.shape == x.shape, f"Shape mismatch: {fp8.shape} vs {x.shape}"
        assert scale.numel() == 1, f"Scale should be scalar, got {scale.shape}"
        d.reset()
        return True
    except (ImportError, OSError, RuntimeError) as e:
        logger.warning("C++ FP8 quant dispatch unavailable: %s", e)
        return False


def get_cpp_dispatcher(
    fp8_max_fwd: float, fp8_max_bwd: float, margin: int, history_len: int
):
    """Create a C++ FP8QuantDispatcher if the extension is available.

    Returns None if the extension cannot be loaded.
    """
    if not _probe_cpp_quant_dispatch():
        return None
    m = _load_module()
    return m.FP8QuantDispatcher(fp8_max_fwd, fp8_max_bwd, margin, history_len)
