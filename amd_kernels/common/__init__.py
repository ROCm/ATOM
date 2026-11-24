# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""FlyDSL helpers the kernels share, taken from AITER's ``aiter/ops/flydsl/kernels`` (amd-aiter 0.1.21.post2) so
that a change in AITER does not change our kernels: ``buffer_ops`` (buffer descriptors, loads and stores),
``tensor_shim`` (``ptr_arg``, ``_run_compiled``), ``mfma_preshuffle_pipeline`` (``xcd_remap_bx_by``); and
``hip_runtime`` (HIP runtime calls through ctypes). The AITER files keep their licence headers; updating one from a
newer AITER is a deliberate change with its own tests.
"""
