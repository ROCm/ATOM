# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""GLM-5.3-Flash on the vllm-atom backend: MLA width, geometry, and guards.

CPU-only. The config fixture mirrors the real checkpoint's shape rather than
reading the 328 GB model, so this runs anywhere.
"""

import pytest

pytest.importorskip("vllm")

from types import SimpleNamespace

import torch
from vllm.transformers_utils.configs.glm5_next import Glm5NextConfig

from atom.plugin.vllm.glm5_next_mla_head_size import patch_vllm_mla_head_size
from atom.plugin.vllm.kimi_k3_kda_autotune_patch import (
    apply_vllm_kimi_k3_kda_autotune_patch,
)


@pytest.fixture(autouse=True, scope="module")
def _kimi_k3_kda_autotune_patch():
    """The plugin model reuses Kimi-K3's KDA adapter, which imports vLLM's
    Kimi-K3 kernels. A server installs this Triton fix in register_model before
    that import; these tests import the model directly, so install it here too
    -- and put `triton.autotune` back afterwards, since the patch's own tests
    assert that they are the ones installing it."""
    import triton

    original = triton.autotune
    apply_vllm_kimi_k3_kda_autotune_patch()
    yield
    triton.autotune = original


def _text_config_dict(index_kpool: int = 4) -> dict:
    # 45 layers, KDA everywhere except the 11 "full" MLA layers, as shipped.
    full = [3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43]
    return {
        "model_type": "glm5_next_text",
        "num_hidden_layers": 45,
        "hidden_size": 4096,
        "num_attention_heads": 64,
        "kv_lora_rank": 512,
        "qk_nope_head_dim": 256,
        "qk_rope_head_dim": 0,
        "v_head_dim": 256,
        "index_topk": 2048,
        "index_kpool": index_kpool,
        "index_head_dim": 128,
        "first_k_dense_replace": 1,
        "n_routed_experts": 288,
        "num_experts_per_tok": 8,
        "layer_types": [
            "deepseek_sparse_attention" if i in full else "linear_attention"
            for i in range(45)
        ],
        "linear_attn_config": {
            "num_heads": 64,
            "head_dim": 128,
            "short_conv_kernel_size": 4,
            "kda_layers": [i for i in range(45) if i not in full],
            "full_attn_layers": full,
        },
    }


def _config(index_kpool: int = 4) -> Glm5NextConfig:
    return Glm5NextConfig(text_config=_text_config_dict(index_kpool))


def _vllm_config(
    cfg,
    *,
    pcp=1,
    dcp=1,
    spec=None,
    block_size=1024,
    prefix_caching=False,
    kv_transfer=None,
) -> SimpleNamespace:
    # _check_unsupported / the shape helpers only read attributes.
    return SimpleNamespace(
        model_config=SimpleNamespace(hf_config=cfg, dtype=torch.bfloat16),
        parallel_config=SimpleNamespace(
            tensor_parallel_size=8,
            prefill_context_parallel_size=pcp,
            decode_context_parallel_size=dcp,
        ),
        cache_config=SimpleNamespace(
            mamba_cache_dtype="auto",
            block_size=block_size,
            enable_prefix_caching=prefix_caching,
        ),
        speculative_config=spec,
        kv_transfer_config=kv_transfer,
    )


def test_mla_head_size_is_the_nope_padded_entry():
    """vLLM's own answer is kv_lora_rank + qk_rope_head_dim = 512, because the
    checkpoint keeps the rope dim at 0; ATOM's MLA entry carries the 64-wide
    zero rope pad, so the cache must be sized at 576 or the first reshape fails.
    """
    # Patch first: it imports vllm.config, which the convertor module needs
    # loaded before it can be imported (the two import each other).
    patch_vllm_mla_head_size()
    from vllm.transformers_utils.model_arch_config_convertor import (
        ModelArchConfigConvertorBase,
    )

    cfg = _config()
    convertor = ModelArchConfigConvertorBase(cfg, cfg.get_text_config())
    assert convertor.is_deepseek_mla()
    assert convertor.get_head_size() == 576


def test_mla_head_size_patch_leaves_other_models_alone():
    from transformers import PretrainedConfig

    patch_vllm_mla_head_size()
    from vllm.transformers_utils.model_arch_config_convertor import (
        ModelArchConfigConvertorBase,
    )

    other = PretrainedConfig(
        model_type="glm_moe_dsa", kv_lora_rank=512, qk_rope_head_dim=64
    )
    convertor = ModelArchConfigConvertorBase(other, other)
    assert convertor.get_head_size() == 576  # vLLM's sum, not the patch


def test_text_config_fields_survive_verbatim():
    # ATOM reads these straight off vLLM's config object; a field vLLM's class
    # renamed or defaulted would silently disagree with the checkpoint.
    t = _config().get_text_config()
    assert t.num_hidden_layers == 45
    assert t.index_topk == 2048
    assert t.index_kpool == 4
    assert isinstance(t.linear_attn_config, dict)


def test_state_shape_matches_glms_kda_geometry():
    """The recurrent state vLLM allocates has to match GLM's KDA dims.

    Wrong dims here do not fail loudly -- vLLM would allocate a state of the
    wrong width and the KDA kernels would read it as if it were right.
    """
    from atom.plugin.vllm.models.glm5_next import _get_glm5_state_shape

    tp = 8
    heads_per_rank = 64 // tp
    conv_shape, state_shape = _get_glm5_state_shape(_vllm_config(_config()))

    # conv: (kernel - 1, q|k|v packed across the rank's heads)
    assert conv_shape == (4 - 1, 3 * heads_per_rank * 128)
    # recurrent state: one (head_k x head_v) matrix per head on this rank
    assert state_shape == (heads_per_rank, 128, 128)


def test_state_shape_scales_with_tensor_parallel():
    from atom.plugin.vllm.models.glm5_next import _get_glm5_state_shape

    vc = _vllm_config(_config())
    vc.parallel_config.tensor_parallel_size = 4
    conv_shape, state_shape = _get_glm5_state_shape(vc)
    assert state_shape == (64 // 4, 128, 128)
    assert conv_shape == (3, 3 * (64 // 4) * 128)


def test_kda_layer_count_matches_the_shipped_split():
    from atom.models.glm5_next import _normalize_glm5_next_config

    t = _config().get_text_config()
    _normalize_glm5_next_config(t)
    assert len(t.glm5_kda_layers) == 34
    assert len(t.glm5_full_attn_layers) == 11
    # 0-based, unlike Kimi's 1-based list -- layer 0 really is KDA.
    assert 0 in t.glm5_kda_layers


@pytest.mark.parametrize(
    "kwargs, expected",
    [
        ({"pcp": 2}, "PCP"),
        ({"dcp": 2}, "DCP"),
        ({"spec": object()}, "speculative decoding"),
    ],
)
def test_unsupported_features_are_refused(kwargs, expected):
    """Plugin mode never runs Config.__post_init__, so these guards would
    otherwise be silently absent and the run would produce wrong output."""
    from atom.plugin.vllm.models.glm5_next import _check_unsupported

    with pytest.raises(ValueError, match=expected):
        _check_unsupported(_vllm_config(_config(index_kpool=1), **kwargs))


def test_kpool_runs_when_the_pool_size_divides_the_block_size(monkeypatch):
    """The pooled cache holds one row per `index_kpool` tokens, so the pool size
    has to divide the block size vLLM settled on. 1024 (TP4) and 512 (TP8) both
    do, which is the only reason the pooled path can be allowed at all here."""
    from atom.plugin.vllm.models import glm5_next as mod

    monkeypatch.setattr(mod, "pooled_path_enabled", lambda _k: True)
    mod._check_unsupported(_vllm_config(_config(index_kpool=4), block_size=1024))
    mod._check_unsupported(_vllm_config(_config(index_kpool=4), block_size=512))


def test_kpool_is_refused_when_the_pool_does_not_divide_the_block(monkeypatch):
    """Refuse loudly rather than mis-address the pooled cache.

    1022 is deliberately not a multiple of the pool size; the block sizes vLLM
    actually picks for this model are, but nothing in vLLM promises that, and a
    mismatch would put pooled rows in the wrong block instead of raising.
    """
    from atom.plugin.vllm.models import glm5_next as mod

    monkeypatch.setattr(mod, "pooled_path_enabled", lambda _k: True)
    with pytest.raises(ValueError, match="divide the KV block size"):
        mod._check_unsupported(_vllm_config(_config(index_kpool=4), block_size=1022))


def test_prefix_caching_is_accepted_on_the_pooled_path(monkeypatch):
    """Not refused: the tail ring is per request and never prefix-cached, and
    vLLM's align mode carries the KDA state across cached blocks."""
    from atom.plugin.vllm.models import glm5_next as mod

    monkeypatch.setattr(mod, "pooled_path_enabled", lambda _k: True)
    mod._check_unsupported(_vllm_config(_config(index_kpool=4), prefix_caching=True))


def test_kv_transfer_is_refused():
    """KV offloading/transfer assumes one cached row per token, which the
    pooled index cache is not."""
    from atom.plugin.vllm.models import glm5_next as mod

    with pytest.raises(ValueError, match="KV transfer"):
        mod._check_unsupported(
            _vllm_config(_config(index_kpool=1), kv_transfer=SimpleNamespace())
        )


def test_kpool_disabled_is_allowed():
    from atom.plugin.vllm.models import glm5_next as mod

    # index_kpool == 1 makes pooled_path_enabled False regardless of the env.
    mod._check_unsupported(_vllm_config(_config(index_kpool=1)))


def test_tail_ring_is_one_pool_without_speculation_and_native_depth_with_it():
    """Without speculation a ring of one pool holds the open pool. With it the
    ring must be as deep as native ATOM's bound, because the plugin runs
    native's verification kernels, which are only correct at that depth."""
    from atom.plugin.vllm.attention.glm5_kpool_indexer import tail_ring_rows

    assert tail_ring_rows(4, 0) == 4
    assert tail_ring_rows(4, 1) == 8  # 4 + 2 * 2
    assert tail_ring_rows(4, 3) == 16  # 4 + 2 * 4 = 12, rounded up
    assert tail_ring_rows(4, 4) == 16  # 4 + 2 * 5 = 14, rounded up


def _tail_ring_layer(block_size=1024, index_kpool=4):
    from atom.plugin.vllm.attention.glm5_kpool_indexer import Glm5NextTailRingVllm

    layer = Glm5NextTailRingVllm.__new__(Glm5NextTailRingVllm)
    torch.nn.Module.__init__(layer)
    layer.head_dim = 128
    layer._index_kpool = index_kpool
    layer.cache_config = SimpleNamespace(block_size=block_size)
    return layer


def test_tail_ring_is_a_per_request_block_vllm_never_prefix_caches():
    """The ring is vLLM's KpoolTailSpec: one block per request for its
    lifetime, never prefix-cached, so a request's tail follows the request
    rather than its KDA state slot, and an MTP draft layer gets one too."""
    from vllm.v1.kv_cache_interface import KpoolTailSpec

    spec = _tail_ring_layer().get_kv_cache_spec(
        SimpleNamespace(num_speculative_tokens=3)
    )
    assert isinstance(spec, KpoolTailSpec)
    assert (spec.block_size, spec.sliding_window) == (16, 16)
    assert (spec.num_kv_heads, spec.head_size, spec.dtype) == (2, 128, torch.bfloat16)
    assert not spec.prefix_cacheable
    assert spec.max_num_blocks_per_req(None, 8192) == 1


def test_tail_ring_must_divide_the_block_size():
    with pytest.raises(ValueError, match="must divide the KV block size"):
        _tail_ring_layer(block_size=1032).get_kv_cache_spec(
            SimpleNamespace(num_speculative_tokens=3)
        )


def _padded_tail_view(num_blocks=8, ring=4, head_dim=8, page_elems=96):
    """Rings inside wider pages, the way vLLM aliases them into the indexer."""
    pages = torch.zeros(num_blocks, page_elems, dtype=torch.bfloat16)
    view = pages[:, : 2 * ring * head_dim].view(num_blocks, 2, ring, head_dim)
    view.copy_(torch.arange(view.numel(), dtype=torch.bfloat16).view_as(view))
    return pages, view


def test_tail_ring_view_accepts_padded_pages_and_rejects_other_rings():
    from atom.plugin.vllm.attention import glm5_kpool_indexer as op

    _, view = _padded_tail_view()
    assert op._tail_ring_view(view, 4, 8) is view
    with pytest.raises(RuntimeError, match="expected"):
        op._tail_ring_view(view, 8, 8)


def test_tail_rings_round_trip_and_skip_requests_without_a_ring():
    """Requests 0 and 2 own blocks 5 and 2; request 1 has no tokens and request
    3 is CUDA-graph padding (slot -1). Only the two owners get a kernel slot,
    and only their blocks change on the way back."""
    from atom.plugin.vllm.attention import glm5_kpool_indexer as op

    pages, view = _padded_tail_view()
    before = pages.clone()
    ring = 4
    qsl = torch.tensor([0, 3, 3, 5, 6])
    positions = torch.tensor([6, 7, 8, 1, 2, 0])
    blocks_of_tokens = torch.tensor([5, 5, 5, 2, 2, -1])
    slot_mapping = torch.where(
        blocks_of_tokens >= 0,
        blocks_of_tokens * ring + positions % ring,
        torch.full_like(positions, -1),
    )

    rings, slots, blocks = op._gather_tail_rings(view, slot_mapping, qsl, 4, ring)
    assert slots.tolist() == [0, -1, 2, -1]
    assert torch.equal(rings[0], view[5]) and torch.equal(rings[2], view[2])
    assert rings.is_contiguous()

    rings[0].fill_(-1)
    rings[2].fill_(-2)
    op._scatter_tail_rings(view, rings, blocks)
    assert (view[5] == -1).all() and (view[2] == -2).all()
    untouched = [b for b in range(8) if b not in (2, 5)]
    assert torch.equal(pages[untouched], before[untouched])
    # The padding past each ring (the rest of the indexer page) is never written.
    assert torch.equal(pages[:, 64:], before[:, 64:])


def test_plugin_classes_replace_native_ones_only_during_construction():
    """The plugin subclasses the native KDA layer and indexer and swaps them in
    while the model is built; importing it must leave the native module alone,
    so a native ATOM engine in the same process keeps its own classes."""
    from atom.models import glm5_next as native
    from atom.plugin.vllm.models import glm5_next as mod

    assert issubclass(mod.Glm5NextIndexerVllm, native.Glm5NextIndexer)
    assert issubclass(mod.Glm5NextKDAAttentionVllm, native.Glm5NextKDAAttention)
    assert native.Glm5NextIndexer is not mod.Glm5NextIndexerVllm
    assert native.Glm5NextKDAAttention is not mod.Glm5NextKDAAttentionVllm


def _prefill_chunk(seq_lens, q_lens, reqs_start=0, reqs_end=None, tokens_per_state=4):
    from atom.plugin.vllm.attention.metadata import (
        AiterMlaSparseIndexerMetadataBuilder as Builder,
    )

    reqs_end = len(seq_lens) if reqs_end is None else reqs_end
    qsl = torch.tensor([0, *torch.tensor(q_lens).cumsum(0).tolist()], dtype=torch.int32)
    builder = SimpleNamespace(
        device=torch.device("cpu"), max_prefill_buffer_size=1 << 20
    )
    return Builder._build_indexer_one_prefill_chunk(
        builder,
        reqs_start,
        reqs_end,
        qsl,
        torch.tensor(seq_lens, dtype=torch.int32),
        torch.zeros(len(seq_lens), 8, dtype=torch.int32),
        tokens_per_state=tokens_per_state,
    )


def test_prefill_chunks_shorter_than_a_pool_publish_only_complete_pools():
    """Whole prompts shorter than index_kpool, one exactly a pool, and chunks
    that resume an earlier chunk and end mid-pool. Only COMPLETE pools are
    cached and scored, counted over each request's whole context so far."""
    # seq_lens: tokens known after this chunk; q_lens: rows this chunk adds.
    m = _prefill_chunk(seq_lens=[1, 3, 4, 9, 7, 6], q_lens=[1, 3, 4, 2, 3, 1])
    assert m.pool_cu_seq_lens.tolist() == [0, 0, 0, 1, 3, 4, 5]
    assert m.total_pools == 5
    assert (m.token_start, m.token_end, m.num_reqs) == (0, 14, 6)


def test_prefill_chunk_of_a_later_request_range_offsets_its_rows():
    """A chunk covering requests 2.. of the batch: its row range starts after
    the earlier requests' rows and its pooled offsets restart at zero."""
    m = _prefill_chunk(
        seq_lens=[5, 2, 3, 8], q_lens=[5, 2, 3, 1], reqs_start=2, reqs_end=4
    )
    assert (m.reqs_start, m.reqs_end) == (2, 4)
    assert (m.token_start, m.token_end) == (7, 11)
    assert m.pool_cu_seq_lens.tolist() == [0, 0, 2]
    assert m.total_pools == 2


def test_unpooled_prefill_chunk_has_no_pool_metadata():
    m = _prefill_chunk(seq_lens=[3, 9], q_lens=[3, 9], tokens_per_state=1)
    assert m.pool_cu_seq_lens is None
    assert m.total_pools == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="runs the Triton kernels")
def test_prefill_rows_shorter_than_a_pool_attend_their_own_tokens():
    """A prefill chunk in which no request has closed a pool (prompts of 1, 2
    and 3 tokens at index_kpool=4) has nothing to score, but every row must
    still select its own open pool. A row left at -1 would be mapped to slot 0
    by the global-index conversion and attend the wrong token."""
    from atom.model_ops.glm5_next.geometry import topk_output_width
    from atom.plugin.vllm.attention.glm5_kpool_indexer import _prefill_chunk_topk

    dev = torch.device("cuda")
    kpool, topk, head_dim, n_head = 4, 2048, 128, 2
    rows_pos = torch.tensor([0, 0, 1, 0, 1, 2], dtype=torch.int64, device=dev)
    req_idx = torch.tensor([0, 1, 1, 2, 2, 2], dtype=torch.int64, device=dev)
    out = torch.full(
        (6, topk_output_width(topk, kpool)), -7, dtype=torch.int32, device=dev
    )
    _prefill_chunk_topk(
        kv_cache=torch.empty(0, 16, 144, dtype=torch.uint8, device=dev),
        q_fp8=torch.zeros(6, n_head, head_dim, device=dev),
        weights=torch.zeros(6, n_head, device=dev),
        rows_pos=rows_pos,
        req_idx=req_idx,
        pool_cu=torch.zeros(4, dtype=torch.int32, device=dev),
        total_pools=0,
        block_table=torch.zeros(3, 1, dtype=torch.int32, device=dev),
        index_kpool=kpool,
        head_dim=head_dim,
        select_k=topk // kpool,
        stable_topk=True,
        out=out,
    )
    for row, pos in enumerate(rows_pos.tolist()):
        picked = sorted(t for t in out[row].tolist() if t >= 0)
        assert picked == list(range(pos + 1)), (row, picked)


def _mtp(n=3, method="mtp"):
    return SimpleNamespace(method=method, num_speculative_tokens=n)


def test_mtp_is_allowed_and_other_speculation_is_refused(monkeypatch):
    from atom.plugin.vllm.models import glm5_next as mod

    monkeypatch.setattr(mod, "pooled_path_enabled", lambda _k: True)
    mod._check_unsupported(_vllm_config(_config(), spec=_mtp()))
    with pytest.raises(ValueError, match="'eagle3'"):
        mod._check_unsupported(_vllm_config(_config(), spec=_mtp(method="eagle3")))
    monkeypatch.setattr(mod, "pooled_path_enabled", lambda _k: False)
    mod._check_unsupported(_vllm_config(_config(), spec=_mtp()))


def test_expanded_block_table_drops_columns_past_max_model_len():
    """MTP verify rows on the dense indexer: vLLM rounds the indexer group's
    block table to whole manager blocks (128 kernel blocks), the expansion
    buffer holds the 32 that max_model_len 2048 can address."""
    from atom.plugin.vllm.attention.metadata import expand_block_table_rows

    table = torch.arange(2 * 128, dtype=torch.int32).reshape(2, 128)
    decode_lens = torch.tensor([3, 1])
    out = torch.full((8, 32), -1, dtype=torch.int32)
    expand_block_table_rows(table, decode_lens, out, 4)
    assert torch.equal(out[:3], table[0, :32].expand(3, 32))
    assert torch.equal(out[3], table[1, :32])
    assert bool((out[4:] == -1).all())

    # A table exactly as wide as the buffer is copied whole.
    same = torch.arange(2 * 32, dtype=torch.int32).reshape(2, 32)
    expand_block_table_rows(same, decode_lens, out, 4)
    assert torch.equal(out[3], same[1])


def test_state_shape_widens_the_conv_window_for_draft_tokens():
    """The model-level shape must match each KDA layer's own, which keeps the
    conv inputs of the tokens a verification may roll back."""
    from atom.plugin.vllm.models.glm5_next import _get_glm5_state_shape

    conv0, state0 = _get_glm5_state_shape(_vllm_config(_config()))
    conv3, state3 = _get_glm5_state_shape(_vllm_config(_config(), spec=_mtp(3)))
    assert conv3 == (conv0[0] + 3, conv0[1])
    assert state3 == state0


def test_kda_metadata_keeps_verify_rows_on_the_write_slot():
    """ATOM's shared KDA forward reads a state read-side index for plain and
    verify rows alike; vLLM's metadata has neither, and None keeps both on
    the slot vLLM writes."""
    from vllm.models.kimi_k3.nvidia.kda_metadata import KimiK3KDAMetadata

    import atom.plugin.vllm.kda_backend  # noqa: F401

    assert KimiK3KDAMetadata.non_spec_state_indices_in_tensor is None
    assert KimiK3KDAMetadata.spec_state_indices_in_tensor is None


def test_mtp_draft_is_registered_with_the_deepseek_mtp_contract():
    from atom.plugin.vllm import model_wrapper as mw
    from atom.plugin.vllm import register

    assert mw._ATOM_MODEL_CLASSES["Glm5NextMTPModel"].endswith(
        "glm5_next:Glm5NextMTPVllm"
    )
    for arches in (
        mw._DEEPSEEK_MTP_ARCHES,
        mw._MTP_MASK_INPUT_ARCH,
        mw._MTP_DRAFT_MODEL_ARCHES,
    ):
        assert "Glm5NextMTPModel" in arches
    assert "Glm5NextMTPModel" in register._VLLM_MODEL_REGISTRY_OVERRIDES


def test_mtp_draft_builds_with_the_plugin_indexer_only(monkeypatch):
    from atom.models import glm5_next as native
    from atom.plugin.vllm.models import glm5_next as mod

    seen = {}

    def record_indexer_cls(self, *args, **kwargs):
        seen["indexer_cls"] = native.Glm5NextIndexer

    monkeypatch.setattr(mod.Glm5NextMTP, "__init__", record_indexer_cls)
    original = native.Glm5NextIndexer
    mod.Glm5NextMTPVllm(None)
    assert seen["indexer_cls"] is mod.Glm5NextIndexerVllm
    assert native.Glm5NextIndexer is original


@pytest.mark.skipif(not torch.cuda.is_available(), reason="runs the Triton kernels")
def test_a_verification_matches_the_same_rows_decoded_one_at_a_time():
    """The plugin decodes one row per request with the stash kernel and
    verifies drafts with native's verification kernels, on the same ring. A
    four-row verification must leave the pooled keys and the ring exactly as
    four single-row decodes would. Request 0 closes pool 1 (positions 4-7) from
    one ring row and three fresh rows; request 1 closes pool 2 (8-11) from two
    of each."""
    from atom.model_ops.glm5_next import kpool
    from atom.model_ops.glm5_next.speculative import (
        build_speculative_pool_candidates,
        update_speculative_kpool_history,
    )
    from atom.plugin.vllm.attention.glm5_kpool_indexer import tail_ring_rows

    dev, head_dim, pool, num_spec = "cuda", 128, 4, 3
    ring = tail_ring_rows(pool, num_spec)
    width = num_spec + 1
    gen = torch.Generator(device=dev).manual_seed(0)

    def rand(*shape, dtype=torch.bfloat16):
        return torch.randn(*shape, device=dev, generator=gen).to(dtype)

    verified = rand(2, 2, ring, head_dim)
    stepped = verified.clone()
    keys, gates = rand(2 * width, head_dim), rand(2 * width, head_dim)
    ape = rand(pool, head_dim, dtype=torch.float32)
    positions = torch.tensor([5, 6, 7, 8, 10, 11, 12, 13], device=dev)
    cu_q = torch.tensor([0, width, 2 * width], device=dev, dtype=torch.int32)
    slots = torch.tensor([0, 1], device=dev)

    pooled, _ = build_speculative_pool_candidates(
        verified, keys, gates, positions, cu_q, slots, ape, pool
    )
    update_speculative_kpool_history(
        verified, keys, gates, positions, cu_q, slots, slots
    )
    for step in range(width):
        rows = torch.tensor([step, width + step], device=dev)
        one_row = kpool.kpool_decode_stash_and_pool(
            stepped, keys[rows], gates[rows], positions[rows], slots, ape, pool
        )
        closes = positions[rows] % pool == pool - 1
        torch.testing.assert_close(
            pooled[rows][closes].float(), one_row[closes].float(), atol=2e-2, rtol=2e-2
        )
    assert torch.equal(verified, stepped)
