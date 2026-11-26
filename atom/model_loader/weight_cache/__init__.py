# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Keep a model's processed weights on the GPU across server restarts.

A daemon (daemon.py) loads each GPU's shard once in a process of its own; its
main process serves them all on one socket (transport.py). A server launch's
main process finds the daemon that serves it, or starts or replaces one, once
(session.py); its rank processes build their model on meta and map their
GPU's weights over CUDA IPC (client.py). The cache key (key.py) is the
checkpoint, the shard and the settings listed there as deciding the weights;
what loading produced is found by diffing before and after it (export.py).
preload.py holds what a launch and a daemon share: the launch's record,
starting and stopping a daemon. ``stop`` and ``load`` (a server asks them of
its daemon: ``session.change_weights``) act on the weights a daemon holds:
the servers that have them stop using them, and go on once another
checkpoint of the model is written into them.

What the rest of ATOM calls: ``daemon.run_daemon``, ``session.change_weights``
and ``weights_status`` (the API server), ``session.open_session`` (it records
the launch: ``preload.record_launch``) and ``serve_notices`` (the engine),
and ``client.build_and_load_model``, ``build_and_load_drafter`` and
``daemon_load_extra`` (the model runner): a rank's model and drafter with
their weights, the daemon's or, with none to use, the checkpoint's.
"""
