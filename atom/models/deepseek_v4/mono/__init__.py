# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""DeepSeek-V4 fused per-layer decode: each decoder layer one launch of the layer
kernel (``kernels/``), up to ``config.MAX_TOKENS`` tokens a step; entered through
``dispatch.install_mono_decode``."""
