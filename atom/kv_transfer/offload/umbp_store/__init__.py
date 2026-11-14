# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""KV-cache offload to a Mori UMBP standalone server (``umbp_store``).

This package was copied from the Mooncake Store offload connector and
rewritten for UMBP; a shared layer for the two is planned as a follow-up.

Prefill workers put and get KV chunks to and from a ``umbp_standalone_server``
that runs on the node, outside ATOM, and copies between its own pool and the
workers' registered GPU memory (HIP IPC). Scheduling, the KV byte layout and
the completion protocol are the dense offload's; this package adds the
transport:

* ``config`` -- the ``umbp_store.*`` settings;
* ``layouts`` -- which layouts this connector moves, and the codec for each;
* ``keys`` -- the layout namespace, the prompt's chunk hash chain, object keys;
* ``client`` -- the UMBP client, with UMBP's own bool-per-key answers;
* ``pool`` -- the registered transfer slots, in HBM;
* ``taint`` -- which servers this process no longer loads from;
* ``scheduler`` -- server lookups and the requests the workers receive;
* ``worker`` -- windowed put/get through the pool, and the startup probe.

Registered with the KV connector factory in ``atom.kv_transfer.offload``.
"""
