# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Keep a model's processed weights on the GPU across server restarts.

A *daemon* is started like a server, with ``ATOM_WEIGHT_CACHE=daemon``: in
place of an engine it starts one process per GPU that builds, loads and
post-processes its shard as a serving runner would, then stops (daemon.py).
Each exports what loading produced as CUDA IPC handles on a Unix socket named
after its GPU (transport.py). A *client* (``ATOM_WEIGHT_CACHE=client``) builds
its model on the meta device and maps those tensors instead of reading the
checkpoint (client.py).

What is served is found by diffing the model and the ATOM/AITER objects
hanging off it before and after ``load_model`` (export.py). A client takes the
cache only if all of these match the daemon's:

* identity: the checkpoint's files (name, size, mtime, inode; safetensors
  headers), hf_config, rank, GPU arch, library versions;
* what the daemon's load read (recorder.py): env vars, config fields,
  construction-time attributes, globals and files. Serving flags that loading
  does not read, such as --max-model-len, do not count;
* code: an AST hash of every function that ran during load and the key of
  every Triton kernel it launched; editing code that does not run during load
  keeps the cache (key.py);
* structure: module tree, types and parameter/exported buffer shapes.

Once a daemon holds a launch's GPUs there is no silent fallback to disk: all
ranks agree, and a mismatch fails with the reason. ``ATOM_PRELOAD=1`` instead
starts, reuses or (once) restarts the daemon with the server's own launch
(preload.py); if the daemon it started cannot be used either, it stops it and
loads from disk. ``python -m atom.model_loader.weight_cache status|stop`` shows
or stops this user's daemons.
"""
