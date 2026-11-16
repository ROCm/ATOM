# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Data-parallel request placement.

- `route`: a placement (`Route`) and the routing decision / metric names.
- `policies`: the placement policies `CoreManager` drives (stateless, session
  affinity, prefix spread) and their request-lifecycle hooks.
- `topology`: whether prefix routing is active for a configuration.
- `prefix_spread`: the prefix index and placement rule behind
  `--dp-load-balance prefix_spread`.
"""
