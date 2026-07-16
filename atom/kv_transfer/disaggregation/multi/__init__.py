# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

"""Composite KV connector that fans out to several sub-connectors.

Lets a single engine run e.g. moriio (P/D RDMA transfer) and lmcache_offload
(CPU/NVMe KV cache) at the same time. Registered under the name ``"multi"``;
see :mod:`atom.kv_transfer.disaggregation.multi.multi_connector`.
"""
