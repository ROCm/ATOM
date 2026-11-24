# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Device check."""

import functools

import torch


@functools.cache
def on_gfx950() -> bool:
    return (
        torch.cuda.is_available()
        and "gfx950" in torch.cuda.get_device_properties(0).gcnArchName
    )
