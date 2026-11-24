# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""MXFP8 GEMM: the dispatcher and the kernels every shape shares (``gemm``); shapes with kernels of their own
live in ``gemm/mxfp8_n<N>_k<K>`` and register with ``gemm._SHAPE_HOSTS``."""
