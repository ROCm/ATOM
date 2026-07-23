# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

"""DSV4 terminal-checkpoint offload (one opaque offload unit per 128-aligned
boundary). Registered as the ``dsv4_offload`` backend; see the parent package's
``_offload_common`` for machinery shared with the chunked ``lmcache_offload``."""
