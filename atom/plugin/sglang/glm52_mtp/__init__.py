"""GLM-5.2 MTP SGLang plugin — split by speculative phase.

Modules
-------
common
    Shared ForwardBatch → AttentionMetaData helpers (token tables, work buffers).
target_verify
    Target verify metadata (``ForwardMode.TARGET_VERIFY``, bs×K queries).
draft_extend
    Draft extend metadata (``DRAFT_EXTEND_V2``, verify→draft KV fill).
draft_decode
    Draft forward decode metadata (``draft_forward`` sub-steps, 1 tok/query).
decode_graph
    Target decode CUDA-graph metadata buffers.
prefill
    Prefill / legacy draft_extend prefill-path metadata.
dispatcher
    ``build_atom_glm52_attention_metadata_from_sglang`` mode router.
cache_bind
    ``bind_glm52_dsa_cache_views`` — KV/indexer pool wiring.
"""

from atom.plugin.sglang.glm52_mtp.cache_bind import bind_glm52_dsa_cache_views
from atom.plugin.sglang.glm52_mtp.common import (
    maybe_get_glm52_dsa_pools_from_sglang_backend,
)
from atom.plugin.sglang.glm52_mtp.dispatcher import (
    build_atom_glm52_attention_metadata_from_sglang,
)
from atom.plugin.sglang.glm52_mtp.decode_graph import (
    build_atom_glm52_decode_graph_metadata_from_sglang,
)
from atom.plugin.sglang.glm52_mtp.draft_decode import (
    clear_draft_decode_sub_step,
    get_draft_decode_sub_step,
    set_draft_decode_sub_step,
)
from atom.plugin.sglang.glm52_mtp.draft_extend import draft_extend_token_num

__all__ = [
    "bind_glm52_dsa_cache_views",
    "build_atom_glm52_attention_metadata_from_sglang",
    "build_atom_glm52_decode_graph_metadata_from_sglang",
    "clear_draft_decode_sub_step",
    "draft_extend_token_num",
    "get_draft_decode_sub_step",
    "maybe_get_glm52_dsa_pools_from_sglang_backend",
    "set_draft_decode_sub_step",
]
