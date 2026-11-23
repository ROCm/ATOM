# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Paged FP4 Mono indexer backend with optional candidate selection.

Model adapters supply shuffled projection weights, paged index keys and
the ABI's metadata/buffer views. FULL and candidate-bounded REINDEX share
the scorer and selection trees; this package imports no model modules.
The current score tiles support 32 heads of 128 dimensions and Top512.
"""
