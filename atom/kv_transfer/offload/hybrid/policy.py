# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Module alias for the consolidated DSV4 offload policy."""

import sys

from atom.kv_transfer.offload.hybrid.dsv4 import policy as _policy

sys.modules[__name__] = _policy
