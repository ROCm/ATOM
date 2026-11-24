# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""The current HIP stream's raw handle for the launches' ``fx.Stream`` argument: ``torch.cuda.current_stream()``
builds a Stream object per call, which host-bound eager prefill calls feel."""

import torch


def current_stream() -> int:
    """The current device's current stream (the capture stream inside a graph capture), as the raw handle
    ``torch.cuda.current_stream().cuda_stream``."""
    return torch._C._cuda_getCurrentRawStream(torch._C._cuda_getDevice())
