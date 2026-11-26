# SPDX-License-Identifier: MIT
"""The CSA2 pool geometry, built from a V4.1 text config."""

from atom.model_ops.attentions.pool_layout.v41_pool_geometry import V41PoolGeometry
from atom.models.deepseek_v41.config import AttentionMode, build_attention_topology


def build_v41_pool_geometry(
    hf_config,
    block_size: int,
    *,
    kv_dtype: str,
    speculative_tokens: int,
    index_fp4: bool,
    index_layout: str | None,
) -> V41PoolGeometry:
    """The pool geometry a CSA2 configuration runs, from its text config alone.

    The single authority for it. The pool is declared twice -- once by the
    scheduler that sizes it and once by whoever hands the runtime its backing
    store -- and the two only describe the same bytes if they derive the shape
    the same way. A frontend that sizes its own pool sizes it from here and
    native's metadata builder builds its cache from here, so a change to the
    topology reaches both or neither.

    ``kv_dtype`` is the main pool's format (native's ``kv_cache_dtype``: bf16,
    fp4 or fp8), ``speculative_tokens`` the draft width a verify step retains,
    ``index_fp4`` the index plane's format and ``index_layout`` its physical
    ordering (native's ``index_cache_layout``; None is the format's original
    one); each moves the geometry, so none has a default that guesses. Beside
    the model's config rather than the attention backend, so a frontend that
    only sizes the pool imports neither the kernels nor AITER; and outside
    `pool_layout`, which reaches nothing of `atom`
    (`tests/test_layout_packages.py`).
    """
    topology = build_attention_topology(hf_config)[: hf_config.num_hidden_layers]
    return V41PoolGeometry(
        len(topology)
        + (hf_config.num_nextn_predict_layers if speculative_tokens else 0),
        tuple(
            (spec.layer_id, spec.ratio)
            for spec in topology
            if spec.mode == AttentionMode.FULL
        ),
        block_size,
        hf_config.sliding_window,
        hf_config.head_dim,
        hf_config.index_head_dim,
        hf_config.engram_max_ngram_size - 1,
        kv_dtype=kv_dtype,
        speculative_tokens=speculative_tokens,
        # Only the ratios the built layers run: a configuration with no
        # window-only layer gets no buffer for one.
        layer_ratios=tuple(sorted({spec.ratio for spec in topology})),
        index_topk=hf_config.index_topk,
        # Paged at the length candidates are picked in, which is what lets
        # a candidate list be a block table. A GPU that cannot page that
        # short refuses when asked, so there is nothing to pre-empt here.
        index_block_rows=hf_config.candidate_block_size,
        index_fp4=index_fp4,
        index_layout=index_layout,
    )
