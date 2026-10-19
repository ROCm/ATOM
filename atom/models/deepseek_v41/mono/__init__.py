# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""DeepSeek-V4.1 mono decode: a decode step of up to 48 rows (a DSpark verify,
each request's anchor and drafted tokens, or a token a request without a drafter)
as two fused kernels a layer, on the shared mechanisms in ``atom.mono``.

The original model is untouched: ``dispatch.install_mono_decode`` wraps it after
loading, and a step the wrapper does not route runs the original forward.
"""
