# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

"""Terminal-checkpoint offload: the ``hybrid`` layout family of
``lmcache_offload`` (one opaque N-component offload unit per aligned boundary).

Model-agnostic machinery (unit container/codec/gpu_connector/store/admission/
policy + the profile-driven ``HybridOffloadConnector``/``HybridOffloadScheduler``);
each model is one profile under ``hybrid/profiles/`` (DSV4, Qwen3-Next). Adding a
hybrid-KV target means adding a profile, not code here. Some class names still
carry a ``DSV4`` prefix for historical reasons — they are container/metadata
types, not model-specific. See the parent ``_offload_common`` for machinery
shared with the ``dense`` family."""
