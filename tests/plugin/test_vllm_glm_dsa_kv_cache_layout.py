"""GLM DSA (``GlmMoeDsaForCausalLM``) registration -> ATOM ``KVCacheTensor``.

Covers every GLM model that registers a DSA indexer: **GLM-5.2 and GLM-5.3**,
which are the same architecture and produce a byte-for-byte identical KV
registration. That is measured, not assumed -- every checkpoint in ``_MODELS``
below was served at TP=4 and logged

    ATOM LMCache offload: registered 78 layers, num_blocks=8192
    ATOM LMCache offload:   78 x kv tail_shape=(64, 576) dtype=torch.uint8
    ATOM LMCache offload:   21 x kv tail_shape=(64, 132) dtype=torch.uint8

so the mapping is shared rather than duplicated, and ``_MODELS`` below pins the
config fields it is derived from. Nothing in ``atom/plugin/vllm/kv_transfer``
or ``atom/kv_transfer/offload`` reads a model name or version; if a future GLM
changes a KV field, add a row to ``_MODELS`` and this file will report the new
byte geometry rather than silently offloading the wrong stride.

GLM DSA differs from MiniMax-M3 in three ways that each had to be handled
rather than assumed:

* its indexer entries are spelled ``<p>.indexer.k_cache``, not ``<p>.index_cache``
* IndexShare means only some layers own an indexer at all (21 of 78 measured)
* its fp8 indexer packs the scale INTO the row, so no scale hook is needed

The shapes below are the ones measured at TP=4 (see
``recipes/atom_vllm/GLM-5.2-LMCache-Byte-Offload.md`` and
``recipes/atom_vllm/GLM-5.3-LMCache-Byte-Offload.md``), with the block count
scaled down -- what the mapping depends on is the axis order, not how many
blocks there are.
"""

from __future__ import annotations

import inspect

import pytest
import torch

from atom.plugin.vllm.kv_transfer.kv_cache_layout import (
    build_kv_cache_tensors,
    index_cache_owner,
    split_kv_tensor,
)

NB, BS = 8, 64
MLA_DIM = 576  # kv_lora_rank 512 + qk_rope_head_dim 64, K and V fused
IDX_DIM = 132  # 128 fp8 key bytes + 4 bytes of scale packed into the row

TOTAL_LAYERS, INDEXER_LAYERS = 78, 21

_IDX_SCALE_BYTES = 4  # the fp8 indexer packs its scale into its own row

# The KV-relevant fields of every shipped GLM DSA checkpoint, read off the model
# directories. ``indexers`` is the one number the config does not carry --
# IndexShare is a property of the checkpoint -- so it is the measured count.
# ``moe_router_dtype`` is the only field GLM-5.3 adds over GLM-5.2, and no file
# under ``atom/kv_transfer/offload`` or ``atom/plugin/vllm/kv_transfer`` reads
# it; it selects MoE routing precision and has no bearing on the KV layout.
_MODELS = {
    "GLM-5.2-MXFP4": {
        "num_hidden_layers": 78,
        "kv_lora_rank": 512,
        "qk_rope_head_dim": 64,
        "index_head_dim": 128,
        "indexers": 21,
    },
    "GLM-5.3-MXFP4": {
        "num_hidden_layers": 78,
        "kv_lora_rank": 512,
        "qk_rope_head_dim": 64,
        "index_head_dim": 128,
        "indexers": 21,
    },
    "GLM-5.3-FP8": {
        "num_hidden_layers": 78,
        "kv_lora_rank": 512,
        "qk_rope_head_dim": 64,
        "index_head_dim": 128,
        "indexers": 21,
    },
}


def _mla(nb: int = NB) -> torch.Tensor:
    return torch.zeros((nb, BS, MLA_DIM), dtype=torch.uint8)


def _indexer(nb: int = NB) -> torch.Tensor:
    return torch.zeros((nb, BS, IDX_DIM), dtype=torch.uint8)


def _glm_dsa_registration(
    layers: int = TOTAL_LAYERS, indexers: int = INDEXER_LAYERS
) -> dict[str, torch.Tensor]:
    """Every ``layers`` gets an MLA cache; the first ``indexers`` also own a DSA
    indexer, which is how IndexShare presents itself to the connector."""
    kv: dict[str, torch.Tensor] = {}
    for i in range(layers):
        kv[f"model.layers.{i}.self_attn.attn"] = _mla()
        if i < indexers:
            kv[f"model.layers.{i}.self_attn.indexer.k_cache"] = _indexer()
    return kv


def test_glm_indexer_suffix_folds_onto_the_attention_layer():
    """``<p>.indexer.k_cache`` belongs to ``<p>.attn`` -- the pairing
    ``AiterMlaSparseIndexerMetadataBuilder`` itself uses."""
    assert (
        index_cache_owner("model.layers.3.self_attn.indexer.k_cache")
        == "model.layers.3.self_attn.attn"
    )


def test_m3_spelling_still_folds():
    assert (
        index_cache_owner("model.layers.3.self_attn.index_cache")
        == "model.layers.3.self_attn"
    )


def test_a_plain_layer_is_not_folded():
    assert index_cache_owner("model.layers.3.self_attn.attn") is None


def test_indexers_do_not_become_layers_of_their_own():
    """The measured count: 78 registered layers out of 99 registered entries --
    identical on GLM-5.2 and GLM-5.3."""
    tensors = build_kv_cache_tensors(_glm_dsa_registration())

    assert len(tensors) == TOTAL_LAYERS
    with_index = [t for t in tensors if t.index_cache is not None]
    assert len(with_index) == INDEXER_LAYERS


def test_index_cache_lands_on_its_own_layer_not_a_neighbour():
    """Layers are emitted in numeric order, so an off-by-one fold would restore
    one layer's indexer bytes under another layer's key -- silently."""
    kv = _glm_dsa_registration(layers=4, indexers=4)
    for i in range(4):
        kv[f"model.layers.{i}.self_attn.indexer.k_cache"].fill_(i + 1)

    tensors = build_kv_cache_tensors(kv)

    assert [int(t.index_cache.flatten()[0]) for t in tensors] == [1, 2, 3, 4]


def test_shared_layers_carry_no_index_cache():
    """IndexShare layers reuse the previous full layer's indexer, so vLLM never
    registers one for them -- and the mapping must not invent an empty slot,
    which would change the codec's per-block byte stride."""
    tensors = build_kv_cache_tensors(_glm_dsa_registration(layers=4, indexers=2))

    assert [t.index_cache is not None for t in tensors] == [True, True, False, False]


def test_both_glm_tensors_travel_whole():
    """Neither GLM DSA tensor has M3's separated K/V region, so neither is
    split."""
    for tensor in (_mla(), _indexer()):
        k_cache, v_cache = split_kv_tensor(tensor)
        assert v_cache is None
        assert k_cache.data_ptr() == tensor.data_ptr()
        assert k_cache.is_contiguous()


def test_mla_scalar_scale_needs_no_hook():
    """vLLM's per-tensor ``_k_scale`` is a constant and correctly ignored; only
    a multi-element scale is unmovable state."""

    class _MlaLayer:
        def __init__(self) -> None:
            self._k_scale = torch.tensor(0.5)

    kv = {"model.layers.0.self_attn.attn": _mla()}
    tensors = build_kv_cache_tensors(kv, {"model.layers.0.self_attn.attn": _MlaLayer()})

    assert tensors[0].k_scale is None and tensors[0].v_scale is None


def test_orphan_indexer_is_rejected():
    """An indexer whose owner is missing means the fold rule guessed the wrong
    owner name -- louder here than as corrupt bytes later."""
    kv = {"model.layers.0.self_attn.indexer.k_cache": _indexer()}

    with pytest.raises(ValueError, match="without their owning layer"):
        build_kv_cache_tensors(kv)


def test_mismatched_block_counts_are_rejected():
    """Two KV cache groups would still divide evenly and slice the smaller
    tensor at the wrong granularity, with nothing logged."""
    kv = _glm_dsa_registration(layers=2, indexers=2)
    kv["model.layers.1.self_attn.indexer.k_cache"] = _indexer(nb=NB // 2)

    with pytest.raises(ValueError, match="do not share a block count"):
        build_kv_cache_tensors(kv)


def test_block_count_check_names_both_sides():
    """The error has to say which tensors disagree; a bare count would leave
    the reader to guess which group vLLM split off."""
    kv = _glm_dsa_registration(layers=2, indexers=2)
    kv["model.layers.1.self_attn.indexer.k_cache"] = _indexer(nb=NB // 2)

    with pytest.raises(ValueError) as excinfo:
        build_kv_cache_tensors(kv)

    message = str(excinfo.value)
    assert f"{NB}: " in message and f"{NB // 2}: " in message
    assert "index_cache" in message


def test_codec_accepts_the_mapped_tensors():
    """The mapping's whole purpose: make GLM DSA pass ATOM's byte codec, with
    the per-block stride that the measured 47,700 B/token comes from."""
    codec_mod = pytest.importorskip(
        "atom.kv_transfer.offload.dense.kv_byte_codec",
        reason="offload codec pulls aiter",
    )
    tensors = build_kv_cache_tensors(_glm_dsa_registration())

    codec = codec_mod.DenseKVByteCodec(
        {str(t.layer_num): t for t in tensors}, num_blocks=NB
    )

    per_token = TOTAL_LAYERS * MLA_DIM + INDEXER_LAYERS * IDX_DIM
    assert per_token == 47700
    assert codec.bytes_per_block == BS * per_token


@pytest.mark.parametrize("model", sorted(_MODELS))
def test_every_glm_dsa_checkpoint_has_the_same_byte_geometry(model):
    """GLM-5.3 needed no new mapping code because its KV fields are identical to
    GLM-5.2's -- not because the connector recognises it. Derive the geometry
    from each config rather than restating 47,700, so a future GLM that moves
    ``kv_lora_rank`` or ``index_head_dim`` reports the new stride here instead
    of offloading at the wrong one.
    """
    fields = _MODELS[model]
    mla_dim = fields["kv_lora_rank"] + fields["qk_rope_head_dim"]
    idx_dim = fields["index_head_dim"] + _IDX_SCALE_BYTES
    layers, indexers = fields["num_hidden_layers"], fields["indexers"]

    assert (mla_dim, idx_dim) == (MLA_DIM, IDX_DIM)
    assert (layers, indexers) == (TOTAL_LAYERS, INDEXER_LAYERS)
    assert layers * mla_dim + indexers * idx_dim == 47700

    tensors = build_kv_cache_tensors(_glm_dsa_registration(layers, indexers))

    assert len(tensors) == layers
    assert sum(t.index_cache is not None for t in tensors) == indexers


def test_the_mapping_is_never_told_which_model_it_has():
    """Why GLM-5.3 was a zero-line change, stated as an invariant: the layout is
    decided by the registration vLLM hands over, never by the model's identity.
    Every other test here would still pass if someone added a version branch, so
    guard the one thing that makes such a branch impossible.
    """
    taken = set(inspect.signature(build_kv_cache_tensors).parameters)

    assert taken == {"kv_caches", "layers"}
