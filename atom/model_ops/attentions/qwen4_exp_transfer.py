# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Which KV transfer connectors Qwen3.8-Flash-Next (`qwen4_exp`) may run.

Qwen3.8's GDN and PLE state ride the PAGE checkpoint image behind its QSA
pages, and only the LMCache offload tiers read that image. Kept apart from
`qwen4_exp_attn` so the connector rules import without AITER and are tested
on a plain runner.
"""

from __future__ import annotations

# The offload tiers that read Qwen3.8's PAGE checkpoint image. Other offload
# backends (mooncake_store) check a Qwen3-Next/3.5 layout and would refuse this
# one only at startup, with a hint for the wrong model.
QWEN38_OFFLOAD_BACKENDS = frozenset({"lmcache_offload", "lmcache_mp"})


def qwen38_transfer_leaves(config) -> list[str]:
    """Canonical names of the transports ``kv_transfer_config`` runs.

    ``multi`` is expanded and every name goes through the factory's aliases
    (strip + casefold), so ``"MoRIIO"`` is ``moriio`` here as it is when the
    connector is built. Empty without a transfer config, and for a malformed
    ``multi``, which the factory refuses when it builds the subs.
    """

    if not getattr(config, "kv_transfer_config", None):
        return []
    from atom.kv_transfer.disaggregation.factory import KVConnectorFactory

    return [name for name, _ in KVConnectorFactory.leaf_connectors(config) or ()]


def qwen38_offload_configured(config) -> bool:
    """Whether a configured offload tier reads the PAGE checkpoint image.

    A server without one keeps fork-slot checkpoints, as on main: PAGE images
    change the KV budget, the hit behaviour and the copy path, and have been
    validated only as the offload tier's image so far.
    """

    return any(
        name in QWEN38_OFFLOAD_BACKENDS for name in qwen38_transfer_leaves(config)
    )


def check_qwen38_kv_transfer(config) -> None:
    """Refuse every configured transport but the two LMCache offload tiers."""

    from atom.kv_transfer.disaggregation.factory import KVConnectorFactory

    for name in qwen38_transfer_leaves(config):
        if not KVConnectorFactory.is_offload_backend(name):
            # A P/D connector moves the QSA pages alone, and decode would
            # continue from a zero recurrent state: fluent, wrong output and
            # no error.
            raise ValueError(
                "Qwen3.8 disaggregated serving is not supported: "
                f"kv_connector={name!r} would transfer the QSA pages without "
                "the GDN and PLE state. Use an offload connector "
                "(lmcache_offload or lmcache_mp) or run without "
                "kv_transfer_config."
            )
        if name not in QWEN38_OFFLOAD_BACKENDS:
            raise ValueError(
                "Qwen3.8 offload supports lmcache_offload and lmcache_mp; "
                f"kv_connector={name!r} does not read the QSA/GDN/PLE PAGE "
                "checkpoint image."
            )
