"""Backward-compatible facade for GLM-5.2 MTP SGLang bridge modules.

Implementation lives under ``atom.plugin.sglang.glm52_mtp.*``:

- ``common``          — shared helpers
- ``target_verify``   — target verify metadata
- ``draft_extend``    — draft extend metadata
- ``draft_decode``    — draft forward decode metadata
- ``multi_token``     — bs×K metadata shared by verify/extend
- ``prefill``         — prefill metadata
- ``decode_graph``    — target decode CUDA graph
- ``dispatcher``      — forward-mode router
- ``cache_bind``      — KV pool binding
- ``worker_*``        — EAGLE worker patches
"""

from atom.plugin.sglang.glm52_mtp.cache_bind import bind_glm52_dsa_cache_views
from atom.plugin.sglang.glm52_mtp.common import (
    ATTENTION_PAGE_SIZE_ATTR,
    DECODE_GRAPH_BUFFERS_ATTR,
    DRAFT_SUB_STEP_ATTR,
    EMPTY_VALUE_CACHE_ATTR,
    INDEXER_PAGE_SIZE_ATTR,
    SHARED_SPARSE_INDICES_ATTR,
    is_glm52_dsa_arch,
    maybe_get_glm52_dsa_pools_from_sglang_backend,
)
from atom.plugin.sglang.glm52_mtp.decode_graph import (
    build_atom_glm52_decode_graph_metadata_from_sglang,
)
from atom.plugin.sglang.glm52_mtp.dispatcher import (
    build_atom_glm52_attention_metadata_from_sglang,
)
from atom.plugin.sglang.glm52_mtp.draft_decode import (
    build_decode_metadata as _build_decode_metadata,
    clear_draft_decode_sub_step,
    get_draft_decode_sub_step,
    set_draft_decode_sub_step,
)
from atom.plugin.sglang.glm52_mtp.draft_extend import (
    build_mtp_draft_extend_decode_metadata as _build_mtp_draft_extend_decode_metadata,
    draft_extend_token_num,
    draft_extend_token_num as _draft_extend_token_num,
    should_use_mtp_draft_extend_decode_path as _should_use_mtp_draft_extend_decode_path,
)
from atom.plugin.sglang.glm52_mtp.target_verify import (
    build_mtp_verify_decode_metadata as _build_mtp_verify_decode_metadata,
    should_use_mtp_verify_prefill_path as _should_use_mtp_verify_prefill_path,
)

__all__ = [
    "ATTENTION_PAGE_SIZE_ATTR",
    "DECODE_GRAPH_BUFFERS_ATTR",
    "DRAFT_SUB_STEP_ATTR",
    "EMPTY_VALUE_CACHE_ATTR",
    "INDEXER_PAGE_SIZE_ATTR",
    "SHARED_SPARSE_INDICES_ATTR",
    "_build_decode_metadata",
    "_build_mtp_draft_extend_decode_metadata",
    "_build_mtp_verify_decode_metadata",
    "_draft_extend_token_num",
    "_should_use_mtp_draft_extend_decode_path",
    "_should_use_mtp_verify_prefill_path",
    "bind_glm52_dsa_cache_views",
    "build_atom_glm52_attention_metadata_from_sglang",
    "build_atom_glm52_decode_graph_metadata_from_sglang",
    "clear_draft_decode_sub_step",
    "draft_extend_token_num",
    "get_draft_decode_sub_step",
    "is_glm52_dsa_arch",
    "maybe_get_glm52_dsa_pools_from_sglang_backend",
    "set_draft_decode_sub_step",
]
