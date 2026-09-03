# SPDX-License-Identifier: MIT
"""Build the small config object ATOM's offload connector expects from vLLM's.

ATOM's offload code was written against ``atom.config.Config``, but it only ever
reads three things from it. Rather than construct a whole ATOM Config inside the
plugin -- which would mean keeping two sources of truth for block size and role
in sync -- this hands it a shim carrying exactly those fields, read off the
vLLM config that is already authoritative in plugin mode.

Kept apart from ``connector.py`` so it is importable (and testable) without
vLLM installed.
"""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class OffloadConfigShim:
    """The ATOM-offload-shaped view of a vLLM config.

    Field names match what ``_init_worker_common`` / ``DenseOffloadConnector``
    read; nothing else in ATOM's Config is touched by the offload path.
    """

    kv_cache_block_size: int
    kv_transfer_config: dict[str, Any] = field(default_factory=dict)
    decode_context_parallel_size: int = 1


def _extra_config(kv_transfer_config: Any) -> dict[str, Any]:
    extra = getattr(kv_transfer_config, "kv_connector_extra_config", None)
    return dict(extra) if isinstance(extra, dict) else {}


def build_offload_config(vllm_config: Any) -> OffloadConfigShim:
    """Derive ATOM's offload config from ``VllmConfig``.

    The role is normalized to ATOM's vocabulary: vLLM connectors are configured
    with ``kv_role`` values like ``kv_both``, while ATOM's native path spells
    the same intent ``offload``. ``_init_worker_common`` accepts both, so the
    value is passed through unchanged rather than translated -- translating
    would silently change save/load enablement if the two vocabularies drift.
    """
    cache_config = getattr(vllm_config, "cache_config", None)
    block_size = getattr(cache_config, "block_size", None)
    if not block_size:
        raise ValueError(
            "ATOM offload connector: vLLM reported no cache_config.block_size; "
            "the byte codec addresses KV by block and cannot proceed without it"
        )

    kv_transfer_config = getattr(vllm_config, "kv_transfer_config", None)
    role = getattr(kv_transfer_config, "kv_role", None) or "kv_both"

    parallel_config = getattr(vllm_config, "parallel_config", None)
    dcp = getattr(parallel_config, "decode_context_parallel_size", 1) or 1

    return OffloadConfigShim(
        kv_cache_block_size=int(block_size),
        kv_transfer_config={"kv_role": role, **_extra_config(kv_transfer_config)},
        decode_context_parallel_size=int(dcp),
    )
