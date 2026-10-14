# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""KV-cache offload straight to a Mooncake Store (``mooncake_store``).

Prefill workers put and get 256-token KV chunks to and from a Mooncake Store
whose memory belongs to separate owner processes -- on the prefill node, and
across two nodes on the decode node too -- with no cache tier of their own in
between. Scheduling, the KV byte layout and the completion protocol are the
dense offload's; this package adds the transport:

* ``config`` -- the ``mooncake_store.*`` settings, per-NIC pools included;
* ``keys`` -- the layout namespace, the prompt's chunk hash chain, Store keys;
* ``nic`` -- one PCI-local RDMA device per worker;
* ``client`` -- a pure zero-copy Store client;
* ``pool`` -- the registered transfer slots, in HBM;
* ``scheduler`` -- Store lookups and the requests the workers receive;
* ``worker`` -- windowed put/get through the pool, and the startup probe.

Registered with the KV connector factory in ``atom.kv_transfer.offload``.
"""
