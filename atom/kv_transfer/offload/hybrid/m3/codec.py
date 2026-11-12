# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""The codec id for M3's complete-PAGE bytes.

The bytes are built by ``pure_page_codec``. Publication order is pool -> field
-> layer, then the draft's ``merge_pages``. Bump the id when that order or the
region set changes, even if the total byte count stays the same: existing
Store objects outlive the servers that wrote them.
"""

M3_CODEC_ID = "m3-page-regions"
