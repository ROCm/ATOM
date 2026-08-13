# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Module alias for the DSV4 hybrid connector implementation."""

import sys

from atom.kv_transfer.offload.hybrid.dsv4 import connector as _connector

sys.modules[__name__] = _connector
