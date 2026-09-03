"""vLLM registration -> ATOM KVCacheTensor mapping, on MiniMax-M3's real layouts.

M3 is the model that forced this mapping: it registers three different physical
layouts at once, and one of them (the sparse layers) is not contiguous as a
whole, so it cannot be handed to ``DenseKVByteCodec`` unsplit. The strides here
are the ones measured on M3-MXFP4 (see FINDINGS in the M3 offload notes), scaled
down in block count -- the contiguity properties depend on the axis order, not
on how many blocks there are.
"""

from __future__ import annotations

import pytest
import torch

from atom.plugin.vllm.kv_transfer.kv_cache_layout import (
    build_kv_cache_tensors,
    split_kv_tensor,
)

NB, BS, HD = 8, 128, 128
DENSE_LAYERS, SPARSE_LAYERS = 3, 5


def _sparse_kv(nb: int = NB) -> torch.Tensor:
    """K and V in two separate regions, exactly as M3 allocates them.

    ``stride(1)`` jumps the whole K region, so the tensor is not contiguous --
    which is the entire point of the split this test covers.
    """
    k_block = BS * HD
    k_total = nb * k_block
    buf = torch.zeros(2 * k_total, dtype=torch.uint8)
    return buf.as_strided((nb, 2, BS, 1, HD), (k_block, k_total, HD, HD, 1))


def _m3_registration() -> dict[str, torch.Tensor]:
    kv: dict[str, torch.Tensor] = {}
    for i in range(DENSE_LAYERS):  # K/V interleaved per token
        kv[f"model.layers.{i}.self_attn.attn"] = torch.zeros(
            (NB, 1, BS, 2 * HD), dtype=torch.uint8
        )
    for i in range(DENSE_LAYERS, DENSE_LAYERS + SPARSE_LAYERS):
        name = f"model.layers.{i}.self_attn.attn"
        kv[name] = _sparse_kv()
        kv[f"{name}.index_cache"] = torch.zeros((NB, BS, HD), dtype=torch.float8_e4m3fn)
    return kv


def test_sparse_layer_is_only_movable_once_split():
    sparse = _sparse_kv()
    assert not sparse.is_contiguous(), "fixture no longer reproduces M3's layout"

    k, v = split_kv_tensor(sparse)
    assert k.is_contiguous() and v.is_contiguous()
    assert k.numel() == v.numel() == NB * BS * HD


def test_dense_layer_travels_whole():
    dense = torch.zeros((NB, 1, BS, 2 * HD), dtype=torch.uint8)
    k, v = split_kv_tensor(dense)
    assert v is None, "dense K/V interleave inside a block; splitting is meaningless"
    assert k is dense


def test_index_caches_fold_into_their_owning_layer():
    tensors = build_kv_cache_tensors(_m3_registration())

    assert (
        len(tensors) == DENSE_LAYERS + SPARSE_LAYERS
    ), "index caches must not become layers of their own"
    with_index = [t for t in tensors if t.index_cache is not None]
    assert len(with_index) == SPARSE_LAYERS


def test_layer_order_is_numeric_not_dict_order():
    kv = _m3_registration()
    shuffled = dict(reversed(list(kv.items())))

    got = build_kv_cache_tensors(shuffled)

    # Segment order must not depend on registration order: a save written in
    # one order and restored in another would scatter bytes to the wrong layers.
    assert [t.layer_num for t in got] == list(range(DENSE_LAYERS + SPARSE_LAYERS))
    assert got[0].v_cache.numel() == 0, "layer 0 is dense -> no separate V"
    assert got[-1].index_cache is not None, "last layer is sparse -> has an index cache"


def test_orphan_index_cache_is_rejected():
    with pytest.raises(ValueError, match="without their owning layer"):
        build_kv_cache_tensors(
            {"model.layers.0.self_attn.attn.index_cache": torch.zeros((NB, BS, HD))}
        )


def test_codec_accepts_the_mapped_tensors():
    """The mapping's whole purpose: make M3 pass ATOM's byte codec."""
    codec_mod = pytest.importorskip(
        "atom.kv_transfer.offload.dense.kv_byte_codec",
        reason="offload codec pulls aiter",
    )
    tensors = build_kv_cache_tensors(_m3_registration())

    codec = codec_mod.DenseKVByteCodec(
        {str(t.layer_num): t for t in tensors}, num_blocks=NB
    )

    dense_bytes = BS * 2 * HD  # one opaque K/V run
    sparse_bytes = 2 * (BS * HD) + BS * HD  # K + V + index
    assert codec.bytes_per_block == (
        DENSE_LAYERS * dense_bytes + SPARSE_LAYERS * sparse_bytes
    )
