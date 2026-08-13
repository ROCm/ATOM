# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Module alias for the consolidated DSV4 checkpoint codec.

Using the same module object also preserves legacy tests/tools that monkeypatch
private AOS1 helpers during the migration.
"""

import sys

from atom.kv_transfer.offload.hybrid.dsv4 import codec as _codec

sys.modules[__name__] = _codec
