# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""``gemm1_prefill_h6144_i3072_persistent`` holds helpers the I 768 kernels import (no kernel of its own): the
row-stationary FP4 MFMA class, the e2m1 / e8m0 converters, the SwiGLU-OAI epilogue, the per-XCD claim helpers and
``interleave_w13_scales``."""
