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
    """ATOM GDN metadata builder.

    Inherits vLLM's GDN builder so runner-side ``isinstance`` checks for spec
    decode continue to work.  vLLM 0.27+ already pads FULL-cudagraph decode
    metadata by ``num_reqs`` (NULL_BLOCK_ID for padded request slots).  A prior
    post-build compaction pass rewrote that metadata using ``num_actual_tokens``
    (token-padded graph size) and shrunk ``num_decodes``, which corrupted GDN
    ssm_state on Qwen3.5 block-FP8 under cudagraph replay.
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
