"""ATOM vLLM GDN attention backend overrides.

vLLM still owns construction of CommonAttentionMetadata.  ATOM only replaces the
GDN backend-specific metadata builder so GDN fixes can live in the plugin
without monkeypatching vLLM classes in place.
"""

from __future__ import annotations

import logging

from vllm.v1.attention.backends.gdn_attn import (
    GDNAttentionBackend,
    GDNAttentionMetadataBuilder,
)
from vllm.v1.attention.backends.registry import (
    MambaAttentionBackendEnum,
    register_backend,
)
logger = logging.getLogger("atom")

_GDN_BACKEND_REGISTERED = False


class AtomGDNAttentionMetadataBuilder(GDNAttentionMetadataBuilder):
    """Use vLLM 0.26.1's native FULL-replay dummy-row handling.

    vLLM now rebuilds FULL-replay metadata from zeroed dummy block-table rows.
    The older ATOM post-build compaction overwrote those PAD slots with live
    state indices and could make high-concurrency GDN decode access invalid
    state-cache rows.
    """


class AtomGDNAttentionBackend(GDNAttentionBackend):
    @staticmethod
    def get_builder_cls() -> type[AtomGDNAttentionMetadataBuilder]:
        return AtomGDNAttentionMetadataBuilder


def register_gdn_attention_backend() -> None:
    global _GDN_BACKEND_REGISTERED
    if _GDN_BACKEND_REGISTERED:
        return

    register_backend(
        MambaAttentionBackendEnum.GDN_ATTN,
        f"{AtomGDNAttentionBackend.__module__}.{AtomGDNAttentionBackend.__qualname__}",
        is_mamba=True,
    )
    _GDN_BACKEND_REGISTERED = True
    logger.info(
        "ATOM plugin: registered GDN attention backend override with ATOM "
        "metadata builder."
    )
