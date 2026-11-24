# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""The dense decoder layers around their attention on four TP ranks, MXFP8 linears, hidden 6144, I 3072 per rank:
the split input fc, the embedding + norm + quant, the all-reduce + norm (+ quant) tails, the MLP block, the layer
stack around the caller's attention and the context K / V chain. ``DenseDecoderRanks4`` holds the communicators; the
MLP and the norm + GEMM are plain entries."""

from .host import MAX_ROWS, DenseDecoderRanks4, DenseLayer, context_kv_blocker, stack_blocker
from .mlp import MXFP8_SWIGLU_FORMS, mlp_supported, mxfp8_swiglu_oai_mlp
from .norm_linear import gemma_rmsnorm_mxfp8_linear

__all__ = ["DenseDecoderRanks4", "DenseLayer", "MAX_ROWS", "MXFP8_SWIGLU_FORMS", "context_kv_blocker",
           "gemma_rmsnorm_mxfp8_linear", "mlp_supported", "mxfp8_swiglu_oai_mlp", "stack_blocker"]
