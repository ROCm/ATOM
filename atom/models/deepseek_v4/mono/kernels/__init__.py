# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""The DeepSeek-V4 mono layer kernel: ``layer`` builds the one launch from the
stages (``stages/``: ``hc``, ``qkv``, ``indexer``, ``attention``, ``ffn``) over the
device helpers ``common`` binds to it (``helpers``, ``ops``)."""
