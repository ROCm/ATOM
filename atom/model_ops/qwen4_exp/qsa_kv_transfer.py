# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Which KV transfer may move a Qwen3.8 engine's caches.

Kept out of `qsa_cache_rules`, which `offload.config` imports: this module
imports `KVConnectorFactory`, whose module imports `offload.config`. Imports
neither Triton nor AITER, so CPU CI runs its tests.
"""

from atom.kv_transfer.disaggregation.factory import KVConnectorFactory


def kv_transfer_leaves(config) -> list[str]:
    return [name for name, _ in KVConnectorFactory.leaf_connectors(config) or ()]


def check_kv_transfer(config) -> None:
    """Qwen3.8's GDN and PLE state ride the PAGE checkpoint image, so only an
    offload tier registered with `reads_page_images` may run."""
    for name in kv_transfer_leaves(config):
        if not KVConnectorFactory.is_offload_backend(name):
            # A P/D connector moves the QSA pages alone, and decode would
            # continue from a zero recurrent state: wrong output, no error.
            raise ValueError(
                "Qwen3.8 disaggregated serving is not supported: "
                f"kv_connector={name!r} would transfer the QSA pages without "
                "the GDN and PLE state. Use an offload connector "
                "(lmcache_offload or lmcache_mp) or run without "
                "kv_transfer_config."
            )
        if not KVConnectorFactory.reads_page_images(name):
            raise ValueError(
                "Qwen3.8 offload supports lmcache_offload and lmcache_mp; "
                f"kv_connector={name!r} does not read the QSA/GDN/PLE PAGE "
                "checkpoint image."
            )
