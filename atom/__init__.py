# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

# When ATOM_USE_TRITON_PA_REDUCE=1, force aiter's paged_attention_decode_v2
# reduce path to use the FlyDSL Triton kernel by disabling the C++ HIP
# variant. Only safe for models with attention sinks (gpt-oss); models without
# hit a NameError in the FlyDSL kernel's no-sinks branch.
import os as _os
if _os.getenv("ATOM_USE_TRITON_PA_REDUCE", "0") == "1":
    try:
        import aiter.ops.triton.gluon.pa_decode_gluon as _pa_decode_mod
        _pa_decode_mod.CXX_PS_REDUCE_AVAILABLE = False
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
