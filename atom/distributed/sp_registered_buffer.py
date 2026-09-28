# SPDX-License-Identifier: MIT
"""Producer views over the SP communicator's existing registered input pool.

Views are scratch, not persistent activations. A producer and its consuming
collective must run consecutively on the communicator's current stream. The
collective's end synchronization completes remote reads before pool reuse.
"""

import math

import torch

from atom.utils import envs


def registered_input_view(group, shape, dtype):
    """Return (view, communicator) or None for unsupported pool configurations."""
    if not envs.ATOM_USE_CUSTOM_ALL_GATHER:
        return None
    ca = getattr(getattr(group, "device_communicator", None), "ca_comm", None)
    if ca is None or ca.disabled or group.world_size != 4:
        return None
    if getattr(ca, "_is_gfx1250", False):
        return None
    if getattr(ca, "_pool", None) is None:
        return None
    pool = ca._pool["input"]
    # Raw HIP allocations have no PyTorch owner/view. Keep the established
    # copy-in path for those configurations rather than manufacturing ownership.
    storage = getattr(pool, "_buffer", None)
    if storage is None:
        return None
    elements = math.prod(shape)
    itemsize = dtype.itemsize
    nbytes = elements * itemsize
    if nbytes == 0 or nbytes > pool.max_size or nbytes % 16:
        return None
    result = storage[:nbytes].view(dtype).view(shape)
    transport = result.view(torch.bfloat16)
    if not ca.should_custom_ag(transport):
        return None
    return result, ca
