# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""The store-agnostic half of the KV-store offload connectors.

A KV-store backend (``mooncake_store``, ``umbp_store``) moves 256-token KV
chunks between the worker's KV cache and an external key-value store; backends
share that machinery and differ only in the store client and what its answers
mean. This package is that machinery, parameterized by a small client protocol:

* ``client`` -- the protocol a backend's client implements, and ``CallRefused``;
* ``keys`` -- the layout namespace, the prompt's chunk hash chain, object keys;
* ``layouts`` -- the layouts a store moves, and the codec for each;
* ``pool`` -- the registered transfer slots, in HBM;
* ``scheduler`` -- the lookup helpers and the scheduler every backend extends;
* ``worker`` -- the worker every backend extends: windowed put/get through the
  pool and the completion protocol.

A backend package adds its client, its settings, how it marks a store it no
longer trusts, and its startup probe.
"""
