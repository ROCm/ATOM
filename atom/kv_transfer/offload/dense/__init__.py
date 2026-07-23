# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

"""Chunked layout family: LMCache token-chunk incremental offload (DSV2/V3 MLA,
MHA). ``DenseOffloadConnector`` / ``DenseOffloadScheduler`` drive
``CacheEngine.store/retrieve`` with ATOM's raw-byte codec + GPU connector.
Selected by the ``lmcache_offload`` shell when the model is not a hybrid family.
"""
