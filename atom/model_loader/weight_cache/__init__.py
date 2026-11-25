# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Keep a model's processed weights on the GPU across server restarts.

A daemon (daemon.py) loads each GPU's shard once and serves it over CUDA IPC
(transport.py); a client (client.py) builds its model on meta and maps them.
The cache key (key.py) is the checkpoint, the shard and the settings listed
there as deciding the weights; what loading produced is found by diffing
before and after it (export.py).
ATOM_PRELOAD (preload.py) starts and restarts daemons with a server's launch.
"""
