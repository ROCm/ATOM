# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""``on_gfx950`` as in ``vllm.platforms.rocm``: device 0's architecture."""

import functools

import torch


@functools.cache
def on_gfx950() -> bool:
    if not torch.cuda.is_available():
        return False
    return "gfx950" in torch.cuda.get_device_properties(0).gcnArchName
