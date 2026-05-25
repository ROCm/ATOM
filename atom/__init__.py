# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

# When ATOM_USE_TRITON_PA_REDUCE=1, force aiter's paged_attention_decode_v2
# reduce path to use the PURE Triton fallback kernel
# (paged_attention_decode_ps_reduce_kernel) instead of the C++ HIP variant
# (pa_decode_ps_reduce_hip_kernel).
#
# The wrapper at pa_decode_gluon.py:5086-5163 has three paths:
#   1. C++ HIP — taken if CXX_PS_REDUCE_AVAILABLE is True (default)
#   2. FlyDSL Triton — has a compiler bug (NameError sink_rsrc in __else_7)
#   3. Pure Triton fallback — runs only if FlyDSL raises ImportError
#
# We disable (1) and force (2) to raise ImportError so the wrapper falls
# through to (3), the pure Triton kernel.
#
# WARNING: the pure Triton fallback is CORRECT but ~75× slower than HIP
# (TPOT 4.75s vs 0.063s on gpt-oss-120b). Leave this env OFF for any
# performance-oriented run; turn it on only when full-Triton coverage is
# required regardless of speed.
import os as _os

# When ATOM_USE_TRITON_BF16_DENSE=1, also redirect aiter.tuned_gemm's auto-
# tuned "torch" libtype fallback to use the Triton gemm_a16w16 path. The
# auto-tuner picks libtype="torch" for certain small/odd shapes (e.g. the
# Tensile MT64x16x128 kernels that fire from non-LinearBase callsites), and
# the torch fallback calls F.linear → aten::mm → rocBLAS. Re-pointing the
# dispatcher's "torch" slot at `triton_gemm` keeps those shapes on Triton too.
if _os.getenv("ATOM_USE_TRITON_BF16_DENSE", "0") == "1":
    try:
        import aiter.tuned_gemm as _tg
        if "torch" in _tg.solMap and "triton" in _tg.solMap:
            _tg.solMap["torch"] = _tg.solMap["triton"]
    except Exception:
        pass

if _os.getenv("ATOM_USE_TRITON_PA_REDUCE", "0") == "1":
    try:
        import aiter.ops.triton.gluon.pa_decode_gluon as _pa_decode_mod
        _pa_decode_mod.CXX_PS_REDUCE_AVAILABLE = False

        def _force_pure_triton_fallback(*_a, **_kw):
            raise ImportError(
                "ATOM: forcing pure Triton paged_attention_decode_ps_reduce_kernel "
                "fallback (FlyDSL variant blocked by sink_rsrc closure-capture bug)"
            )

        _pa_decode_mod.launch_pa_decode_ps_reduce_flydsl = _force_pure_triton_fallback
    except Exception:
        pass

from atom.model_engine.llm_engine import LLMEngine
from atom.sampling_params import SamplingParams

# interface for upper framework to construct the model from ATOM
from atom.plugin import prepare_model

__all__ = [
    "LLMEngine",
    "SamplingParams",
    "prepare_model",
]
