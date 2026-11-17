# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""The DSA indexer's decode as one mono launch, for any model whose layers
carry the DeepSeek-V3.2-style ``Indexer``: ``plan`` its shape and per-row
decisions, ``kernel`` the launch, ``select`` its scoring and distributed
top-k, ``bind`` a model's indexers and their per-step arguments."""
