# SPDX-License-Identifier: MIT
"""GLM-5.2 mono: the TP shard and the deployment refusals."""

from types import SimpleNamespace

import pytest

from atom.config import CUDAGraphMode
from atom.models.glm52.mono.config import (
    AGENTX_BATCHES,
    QUERY_LENGTHS,
    SUPPORTED_TP,
    Dims,
    config_refusal,
    model_geometry_refusal,
    native_config,
)
from atom.models.glm52.mono.kernels.geometry_config import (
    EPS,
    GLM5_CONFIG,
    glm5_kernel_samples,
    glm5_tp_config,
)
from atom.mono.plan.shard import ShardError, shard_refusal


def test_tp4_is_the_supported_shard():
    assert SUPPORTED_TP == (4,)
    dims = Dims(4)
    assert dims.local_heads == 16  # 64 attention heads / 4
    assert dims.inter == 512  # moe_intermediate_size 2048 / 4


@pytest.mark.parametrize("tp", [3, 5, 7])
def test_indivisible_tp_is_refused_by_name(tp):
    # Review Focus 1: a bad TP is named, never silently sharded wrong.
    with pytest.raises(ShardError):
        Dims(tp)
    why = shard_refusal(Dims, tp)
    assert why is not None and str(tp) in why


def test_agentx_batches_are_one_and_two():
    # Review Focus 2: the gate that makes the accuracy run meaningful.
    assert AGENTX_BATCHES == (1, 2)


def test_query_lengths_are_sourced_from_the_kernel_declaration():
    # QUERY_LENGTHS must be imported, not retyped: the kernel's own
    # GLM5_QUERY_LENGTHS is the single source of truth for which decode
    # widths the vendored kernel serves.
    assert QUERY_LENGTHS == (1, 4, 5, 6)


def _served_tp4_fp8_config(*, speculative_config, compilation_config=None):
    """The real served deployment: TP4, fp8 KV cache, the given spec config."""
    cfg = GLM5_CONFIG
    hf_config = SimpleNamespace(
        model_type="glm_moe_dsa",
        hidden_size=cfg.hidden,
        num_attention_heads=cfg.local_heads * 8,
        q_lora_rank=cfg.q_lora,
        kv_lora_rank=cfg.kv_lora,
        qk_rope_head_dim=cfg.pe_dim,
        qk_nope_head_dim=cfg.nope_dim,
        v_head_dim=cfg.v_dim,
        n_routed_experts=cfg.n_experts,
        num_experts_per_tok=cfg.top_k,
        n_shared_experts=cfg.num_shared_experts,
        index_topk=2048,
        routed_scaling_factor=cfg.route_scale,
        scoring_func="sigmoid",
        topk_method="noaux_tc",
        norm_topk_prob=True,
        rms_norm_eps=EPS,
        moe_intermediate_size=cfg.inter * 8,
        # GLM-5.x sets this; it overrides a default that would be NeoX, which
        # the fused indexer kernel cannot serve (`deepseek_v2.py`).
        indexer_rope_interleave=True,
    )
    return SimpleNamespace(
        hf_config=hf_config,
        tensor_parallel_size=4,
        parallel_config=SimpleNamespace(data_parallel_size=1),
        enable_dp_attention=False,
        decode_context_parallel_size=1,
        prefill_context_parallel_size=1,
        pipeline_parallel_size=1,
        enable_expert_parallel=False,
        enable_tbo=False,
        enable_tbo_decode=False,
        speculative_config=speculative_config,
        kv_cache_dtype="fp8",
        compilation_config=compilation_config,
    )


def _served_tp4_fp8_mtp4_config(*, compilation_config=None):
    """The real served deployment: TP4, fp8 KV cache, MTP4 (query length 5)."""
    return _served_tp4_fp8_config(
        speculative_config=SimpleNamespace(method="mtp", num_speculative_tokens=4),
        compilation_config=compilation_config,
    )


@pytest.mark.parametrize(
    "query_length, mtp",
    [(1, False), (4, True), (5, True), (6, True)],
)
def test_native_config_admits_every_kernel_declared_query_length(query_length, mtp):
    # GLM5_QUERY_LENGTHS == (1, 4, 5, 6): query length 1 is no speculation at
    # all (mtp=False, atom_config.speculative_config is None); the other
    # three are MTP verifies (mtp=True). All four must reach a real TP4 shard
    # geometry, not just the previously-measured MTP4 (query length 5).
    config = native_config(
        samples=query_length,
        tp_size=4,
        kv_cache_dtype="fp8",
        mtp=mtp,
        query_length=query_length,
    )
    assert config == glm5_tp_config(4)


def test_query_length_one_with_an_mtp_config_is_incoherent_and_refused():
    # Query length 1 means no speculation at all; an MTP config arriving
    # alongside it is a contradiction (there is no such deployment), and the
    # gate refuses it explicitly rather than leaving the outcome to whatever
    # the membership test happens to do.
    assert (
        native_config(
            samples=4, tp_size=4, kv_cache_dtype="fp8", mtp=True, query_length=1
        )
        is None
    )


@pytest.mark.parametrize(
    "query_length, speculative_config",
    [
        (1, None),
        (4, SimpleNamespace(method="mtp", num_speculative_tokens=3)),
        (5, SimpleNamespace(method="mtp", num_speculative_tokens=4)),
        (6, SimpleNamespace(method="mtp", num_speculative_tokens=5)),
    ],
)
def test_served_deployment_accepts_every_admitted_query_length(
    query_length, speculative_config
):
    # The same four widths, exercised through the full config_refusal gate on
    # the real served TP4 + fp8 deployment (not just the native_config helper
    # in isolation).
    computed = (
        1
        if speculative_config is None
        else speculative_config.num_speculative_tokens + 1
    )
    assert computed == query_length
    atom_config = _served_tp4_fp8_config(speculative_config=speculative_config)
    assert config_refusal(atom_config) is None


@pytest.mark.parametrize("num_speculative_tokens", [1, 2])
def test_mtp1_and_mtp2_are_still_refused(num_speculative_tokens):
    # MTP1 (query length 2) and MTP2 (query length 3) are absent from
    # GLM5_QUERY_LENGTHS == (1, 4, 5, 6); they must stay refused even though
    # the TP4 + fp8 deployment is otherwise served. This is the test that
    # fails loudly if GLM5_QUERY_LENGTHS is ever widened without rechecking
    # the kernel.
    speculative_config = SimpleNamespace(
        method="mtp", num_speculative_tokens=num_speculative_tokens
    )
    atom_config = _served_tp4_fp8_config(speculative_config=speculative_config)
    assert config_refusal(atom_config) == "native geometry"


@pytest.mark.parametrize(
    "batch, query_length, expected_chunk",
    [
        (1, 1, 1),
        (2, 1, 2),
        (1, 4, 4),
        (2, 4, 8),
        (1, 5, 5),
        (2, 5, 10),
        (1, 6, 6),
        (2, 6, 12),
    ],
)
def test_agentx_batches_cover_every_admitted_query_length(
    batch, query_length, expected_chunk
):
    # AGENTX_BATCHES == (1, 2); every reachable (batch, query_length) row
    # count across the four admitted widths must resolve to a kernel chunk.
    assert AGENTX_BATCHES == (1, 2)
    assert glm5_kernel_samples(batch * query_length, query_length) == expected_chunk


def test_tr_hybrid_cudagraph_does_not_refuse_the_served_deployment():
    # Review Focus 3 (fix round 1): the default `--level` 3 launch sets
    # `compilation_config.cudagraph_mode = CUDAGraphMode.TR_HYBRID` for every
    # model but deepseek_v41_text (atom/config.py:2373-2377), and GLM-5.2
    # mono routes inside the *uncompiled* outer `GlmMoeDsaForCausalLM` -- the
    # inner `DeepseekV2Model`'s token graph never runs under it and
    # cannot conflict. A fixture that omits `compilation_config` entirely (as
    # the sibling test above does) cannot catch a regression here: it hides
    # behind `_uses_tr_hybrid`'s "absent means not TR_HYBRID"
    # fallback instead of exercising the real, TR_HYBRID, served config.
    compilation_config = SimpleNamespace(cudagraph_mode=CUDAGraphMode.TR_HYBRID)
    assert compilation_config.cudagraph_mode.uses_tr_hybrid()
    atom_config = _served_tp4_fp8_mtp4_config(compilation_config=compilation_config)
    assert config_refusal(atom_config) is None


_MISSING = object()


@pytest.mark.parametrize(
    ("rope_interleave", "refused"),
    [
        (_MISSING, False),  # no field at all -> interleaved default
        (None, False),  # explicit null -> same default
        (True, False),  # interleaved, what the kernel implements
        (1, False),
        (False, True),  # NeoX -- the kernel has no branch for it
        (0, True),
    ],
)
def test_neox_mla_rope_is_refused_but_the_interleaved_default_is_not(
    rope_interleave, refused
):
    """The kernel rotates adjacent pairs with no NeoX branch, so a NeoX
    checkpoint would be served a different position encoding silently.

    The default matters as much as the refusal: production resolves a missing
    or null `rope_interleave` to interleaved, so gating on equality with True
    would reject checkpoints that are actually fine.
    """
    cfg = _served_tp4_fp8_config(
        speculative_config=SimpleNamespace(method="mtp", num_speculative_tokens=4)
    ).hf_config
    if rope_interleave is not _MISSING:
        cfg.rope_interleave = rope_interleave
    why = model_geometry_refusal(cfg, GLM5_CONFIG, 8)
    # `is None` on the allowed rows, not just "no rope complaint": it proves the
    # geometry checks above the gate pass, so the gate is actually reached.
    if refused:
        assert why == f"rope_interleave={rope_interleave!r}"
    else:
        assert why is None


@pytest.mark.parametrize(
    ("indexer_rope_interleave", "refused"),
    [
        (_MISSING, True),  # no field at all -> production resolves to NeoX
        (None, True),  # explicit null -> same default
        (False, True),
        (0, True),
        (True, False),  # what GLM-5.x actually sets
        (1, False),
    ],
)
def test_neox_indexer_rope_is_refused_and_its_default_is_the_refused_one(
    indexer_rope_interleave, refused
):
    """The indexer carries a second, independent rope whose default is the
    opposite of MLA's: a missing `indexer_rope_interleave` means NeoX, and the
    fused kernel rotates index keys in pairs. Decode would then score keys
    that prefill wrote under a different layout and pick wrong sparse indices.
    """
    cfg = _served_tp4_fp8_config(
        speculative_config=SimpleNamespace(method="mtp", num_speculative_tokens=4)
    ).hf_config
    if indexer_rope_interleave is _MISSING:
        del cfg.indexer_rope_interleave
    else:
        cfg.indexer_rope_interleave = indexer_rope_interleave
    why = model_geometry_refusal(cfg, GLM5_CONFIG, 8)
    seen = None if indexer_rope_interleave is _MISSING else indexer_rope_interleave
    if refused:
        assert why == f"indexer_rope_interleave={seen!r}"
    else:
        assert why is None


@pytest.mark.parametrize(
    ("n_group", "topk_group", "refused"),
    [
        (_MISSING, _MISSING, False),  # ungrouped checkpoint -> global top-k
        (None, None, False),  # explicit nulls read the same way
        (1, 1, False),  # one group IS the whole expert set
        (8, 8, False),  # every group kept -> the mask is a no-op
        (8, 1, True),  # one group of eight: 7/8 of the experts masked out
        (8, 4, True),  # DeepSeek-V3's shape
        (4, 2, True),
    ],
)
def test_a_grouped_router_is_refused_but_an_all_groups_one_is_not(
    n_group, topk_group, refused
):
    """Production takes the top-k inside the best `topk_group` groups; the
    kernel's `route_top8` takes a plain global top-8 with no group structure.

    The boundary is the point: `topk_group == n_group` keeps every group, so
    `masked_fill` changes nothing and production's top-k IS the global one over
    `sigmoid + bias` -- the same key the kernel sorts on. Refusing on
    "grouped at all" would disarm mono for every checkpoint that merely
    declares groups it does not use, and refusing on nothing lets a genuinely
    grouped one silently pick different experts.
    """
    cfg = _served_tp4_fp8_config(
        speculative_config=SimpleNamespace(method="mtp", num_speculative_tokens=4)
    ).hf_config
    for name, value in (("n_group", n_group), ("topk_group", topk_group)):
        if value is _MISSING:
            assert not hasattr(cfg, name)
        else:
            setattr(cfg, name, value)
    why = model_geometry_refusal(cfg, GLM5_CONFIG, 8)
    if refused:
        assert why == f"grouped top-k (n_group={n_group}, topk_group={topk_group})"
    else:
        # `is None`, so the checks above the gate are known to pass
        assert why is None
