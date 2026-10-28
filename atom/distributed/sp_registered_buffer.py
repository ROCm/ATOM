# SPDX-License-Identifier: MIT
"""Producer views over the SP communicator's existing registered input pool.

Views are scratch, not persistent activations. A producer and its consuming
collective must run consecutively on the communicator's current stream. The
collective's end synchronization completes remote reads before pool reuse.
"""

import math

import torch


def registered_input_view(ca, shape, dtype):
    """Return a view of ca's registered input pool, or None if it cannot hold one."""
    pool = ca._pool["input"]
    # Raw HIP allocations have no PyTorch owner/view. Keep the established
    # copy-in path for those configurations rather than manufacturing ownership.
    storage = getattr(pool, "_buffer", None)
    if storage is None:
        return None
    nbytes = math.prod(shape) * dtype.itemsize
    if nbytes == 0 or nbytes > pool.max_size or nbytes % 16:
        return None
    result = storage[:nbytes].view(dtype).view(shape)
    if not ca.should_custom_ag(result.view(torch.bfloat16)):
        return None
    return result
