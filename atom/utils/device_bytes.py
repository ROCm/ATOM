# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""A non-owning torch view of raw device memory."""


class DeviceBytes:
    """``__cuda_array_interface__`` of ``nbytes`` bytes at a raw device address, for a
    non-owning torch view."""

    def __init__(self, ptr: int, nbytes: int):
        self.__cuda_array_interface__ = {
            "shape": (nbytes,),
            "typestr": "|u1",
            "data": (ptr, False),
            "version": 3,
        }
