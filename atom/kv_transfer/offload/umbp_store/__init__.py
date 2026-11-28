# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

"""KV-cache offload to a Mori UMBP standalone server (``umbp_store``).

Prefill workers put and get KV chunks to and from a ``umbp_standalone_server``
that runs on the node, outside ATOM, and copies between its own pool and the
workers' registered GPU memory (HIP IPC). The keys, the layouts, the transfer
pool, the windowed put/get loop and the scheduler are the store-agnostic
:mod:`atom.kv_transfer.offload.kv_store` layer's, shared with
``mooncake_store``; this package adds what is UMBP's own:

* ``config`` -- the ``umbp_store.*`` settings, the layouts UMBP moves (dense
  and M3 PAGE) and its ``atomumbp1-`` key namespace;
* ``client`` -- the UMBP client, with UMBP's own bool-per-key answers;
* ``taint`` -- which servers this process no longer loads from (UMBP has no
  pinned object a shared mark could be);
* ``scheduler`` -- lookups against the one server, which hold nothing;
* ``worker`` -- settling a call by its answer, and a startup probe that
  cannot remove its object (UMBP has no remove by key and no groups).

Registered with the KV connector factory in ``atom.kv_transfer.offload``.
"""
