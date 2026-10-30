"""Use the validated target QSA preparation arithmetic for draft q1/q4."""

from atom.models.qwen4_exp_mono import native_qsa
from atom.utils.forward_context import get_forward_context


def attention(module, positions, mixed):
    fc = get_forward_context()
    meta = fc.attn_metadata.qsa_metadata
    if meta is None or fc.context.is_prefill:
        return module(positions, mixed)
    requests = meta.block_tables.shape[0]
    rows = mixed.shape[0]
    if rows not in (requests, 4 * requests):
        return module(positions, mixed)
    # The first q4 pass publishes every raw row before group compression.
    # RoPE positions come from the actual draft position cache, including +1.
    # Quantized compressed index storage needs a separate prepared endpoint.
    assert fc.context.is_draft
    assert module.compressed_key_scale is None
    assert module.rope_position_cache is not None
    assert module.indexer.compress_ratio == 4 and module.indexer.index_head_dim == 128
    prepared = native_qsa.prepare(
        module,
        positions,
        mixed,
        fuse_decode_index=True,
        index_query_length=4 if rows == 4 * requests else 1,
    )
    return native_qsa.finish(module, prepared)
