# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""The worker half of capability discovery.

Reached as ``collective_rpc("get_worker_capabilities")``, so the answer arrives
by the same path the caller will later use to drive whatever it negotiates. A
rank that cannot answer is therefore a rank that cannot be driven, which is the
useful failure.
"""

import logging

from atom.model_engine.capabilities import COLLECTIVE_RPC_PROTOCOL_VERSION

logger = logging.getLogger("atom")

# Control-plane methods a consumer may reasonably drive. Deliberately a fixed
# list rather than a dir() sweep: dir() would advertise every private helper and
# turn any refactor into a capability change, and it would leak methods that are
# not safe to call from outside.
_ADVERTISED_METHODS = (
    # weight sync
    "update_weights",
    "update_weights_from_shm",
    "update_weights_from_ipc",
    # memory lifecycle
    "release_memory",
    "resume_memory",
    "clear_kv_cache",
    # hidden-state extraction
    "configure_hidden_states",
    # capability discovery itself
    "get_worker_capabilities",
)


class CapabilityProviderMixin:
    """Lets a ModelRunner describe itself to a negotiating caller."""

    def get_worker_capabilities(self) -> dict:
        """Report this rank's protocol version, position, and what it supports.

        Returns a plain dict rather than a dataclass so the reply stays
        picklable across the worker boundary without the engine and the worker
        having to agree on a class definition.
        """
        methods = sorted(
            name for name in _ADVERTISED_METHODS if callable(getattr(self, name, None))
        )

        features = set()
        # Reported as features rather than inferred by the caller from a method
        # name, because "the method exists" and "the feature works here" differ:
        # an FP8 runner has the same methods as a BF16 one.
        if getattr(self, "_true_vocab_size", 0):
            features.add("vocab_masking")
        if callable(getattr(self, "_is_fp8_param", None)):
            features.add("fp8_weight_update")
        if callable(getattr(self, "receive_weights_rdma", None)):
            features.add("rdma_weight_receive")

        config = getattr(self, "config", None)
        parallel = getattr(config, "parallel_config", None)
        return {
            "protocol_version": COLLECTIVE_RPC_PROTOCOL_VERSION,
            "tp_rank": int(getattr(self, "rank", -1)),
            "dp_rank_local": int(getattr(parallel, "data_parallel_rank_local", 0) or 0),
            "methods": methods,
            "features": sorted(features),
        }
