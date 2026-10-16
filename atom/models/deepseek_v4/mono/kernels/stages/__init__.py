# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""The layer kernel's stages (``layer``), a module per group: each takes the
kernel's context dict (the build's constants, the kernel arguments and the values
earlier stages returned) and returns the values it defines."""
