# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Unit tests for the SpecForge DFlash draft (RadixArk/Kimi-K3-DSpark).

GPU-free except the block-read check, which skips without one. Covers the
things that decide whether the draft is even reachable, and that no amount of
GPU bring-up would localize:

- config normalization: the arch rename and the nested ``dflash_config`` lift
- attention-flavor resolution: the draft must resolve to MHA and own a KV pool
- KV-pool geometry: the scheduler block size, and the DCP head-sharding
  invariant that makes a DCP8 step additive rather than a rewrite
- the block read: the fold that makes aiter's causal `unified_attention` read
  the block bidirectionally, and that read against an fp32 reference

The checkpoint these mirror is ``RadixArk/Kimi-K3-DSpark``: a 5-layer Qwen3-style
GQA draft (64 q / 16 kv heads, head_dim 64) for the Kimi-K3 MLA+KDA target.
"""

import types

import pytest


def _real_hf_config_override():
    """Load the real SpeculativeConfig despite conftest's ``atom.config`` stub.

    Same approach as tests/test_dspark.py: exec the real source by file path
    under a throwaway module name, then restore the stub.
    """
    import importlib.util
    import os
    import sys

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    path = os.path.join(root, "atom", "config.py")
    spec = importlib.util.spec_from_file_location("_atom_config_real_dflash", path)
    mod = importlib.util.module_from_spec(spec)
    saved = sys.modules.get("atom.config")
    try:
        spec.loader.exec_module(mod)
    except Exception:  # noqa: BLE001 - a partial checkout has no SpeculativeConfig
        return None
    finally:
        if saved is not None:
            sys.modules["atom.config"] = saved
    return getattr(mod, "SpeculativeConfig", None)


def _radixark_cfg(**overrides):
    """The RadixArk checkpoint's config.json, as transformers hands it over."""
    cfg = types.SimpleNamespace(
        architectures=["DSparkDraftModel"],
        model_type="qwen3",
        block_size=7,
        dflash_config={
            "mask_token_id": 163824,
            "target_layer_ids": [7, 23, 51, 67, 83],
        },
        head_dim=64,
        hidden_size=7168,
        intermediate_size=14336,
        num_attention_heads=64,
        num_hidden_layers=5,
        num_key_value_heads=16,
        num_target_layers=93,
        markov_rank=256,
        markov_head_type="vanilla",
        enable_confidence_head=True,
        vocab_size=163840,
        max_position_embeddings=1048576,
        rms_norm_eps=1e-05,
        rope_parameters={
            "rope_type": "yarn",
            "factor": 16.0,
            "original_max_position_embeddings": 65536,
            "rope_theta": 10000.0,
        },
    )
    for k, v in overrides.items():
        setattr(cfg, k, v)
    cfg.update = lambda d: [setattr(cfg, k, v) for k, v in d.items()]
    return cfg


# --------------------------------------------------------------------------
# Config normalization
# --------------------------------------------------------------------------


def test_dflash_config_normalize():
    """DSparkDraftModel + dflash_config -> K3DFlashModel with lifted fields."""
    SpeculativeConfig = _real_hf_config_override()
    if SpeculativeConfig is None:
        pytest.skip("atom.config not importable in this sandbox")

    hf = _radixark_cfg()
    SpeculativeConfig.hf_config_override(hf, model_path=None)

    # Renamed away from the SpecForge string, which collides with ATOM's own
    # draft base class and is reused across every DFlash draft.
    assert hf.architectures == ["K3DFlashModel"]

    # Lifted out of the nested dflash_config onto the canonical names.
    assert hf.dspark_target_layer_ids == [7, 23, 51, 67, 83]
    assert hf.dspark_noise_token_id == 163824
    assert hf.dspark_markov_rank == 256
    assert hf.dspark_with_draft is True

    # The trained block width, recorded so the horizon can be warned against.
    assert hf.dspark_block_size == 7

    # num_target_layers is the TARGET's depth, not the aux-tap count. The fusion
    # width is len(target_layer_ids); sizing off 93 would be a 93/5 overshoot.
    assert hf.num_target_layers == 93
    assert len(hf.dspark_target_layer_ids) == 5


def test_dflash_rope_is_the_drafts_own():
    """The draft keeps theta 10000 / YaRN x16; it must not inherit the target's
    theta 50000 / x32. A shared rope loads clean and only moves acceptance."""
    SpeculativeConfig = _real_hf_config_override()
    if SpeculativeConfig is None:
        pytest.skip("atom.config not importable in this sandbox")

    hf = _radixark_cfg()
    SpeculativeConfig.hf_config_override(hf, model_path=None)

    assert hf.rope_parameters["rope_theta"] == 10000.0
    assert hf.rope_parameters["factor"] == 16.0
    assert hf.rope_parameters["original_max_position_embeddings"] == 65536


def test_dflash_does_not_capture_the_k3_dspark_checkpoint():
    """The MLA K3-DSpark draft keeps its own route and gains no block size."""
    SpeculativeConfig = _real_hf_config_override()
    if SpeculativeConfig is None:
        pytest.skip("atom.config not importable in this sandbox")

    hf = types.SimpleNamespace(
        architectures=["K3DSparkModel"],
        model_type="k3_dspark",
        target_layer_ids=[2, 23, 47, 71, 89],
        mask_token_id=163837,
        markov_rank=256,
        kv_lora_rank=512,
    )
    hf.update = lambda d: [setattr(hf, k, v) for k, v in d.items()]
    SpeculativeConfig.hf_config_override(hf, model_path=None)

    assert hf.architectures == ["K3DSparkModel"]
    assert hf.dspark_target_layer_ids == [2, 23, 47, 71, 89]
    assert hf.dspark_noise_token_id == 163837
    assert hf.dspark_with_draft is True
    # This checkpoint records no width; the block is --num-speculative-tokens.
    assert getattr(hf, "dspark_block_size", None) is None


def test_dflash_branch_needs_dflash_config():
    """A bare DSparkDraftModel with no dflash_config is NOT claimed as DFlash.

    SpecForge stamps that arch on every DFlash draft, so the arch string alone
    is not a model identity; claiming it would silently bind the wrong class.
    """
    SpeculativeConfig = _real_hf_config_override()
    if SpeculativeConfig is None:
        pytest.skip("atom.config not importable in this sandbox")

    hf = _radixark_cfg(dflash_config=None)
    SpeculativeConfig.hf_config_override(hf, model_path=None)

    # Unclaimed: the arch is left alone and no dspark_* marker is synthesized,
    # so the drafter registry raises a KeyError on an unknown arch rather than
    # binding this checkpoint to the Kimi-K3 draft class.
    assert hf.architectures == ["DSparkDraftModel"]
    assert getattr(hf, "dspark_with_draft", False) is False
    assert getattr(hf, "dspark_target_layer_ids", None) is None


def test_dflash_plain_qwen3_target_is_untouched():
    """model_type qwen3 is a legitimate TARGET type and must not be routed."""
    SpeculativeConfig = _real_hf_config_override()
    if SpeculativeConfig is None:
        pytest.skip("atom.config not importable in this sandbox")

    hf = types.SimpleNamespace(
        architectures=["Qwen3ForCausalLM"],
        model_type="qwen3",
        num_hidden_layers=64,
    )
    hf.update = lambda d: [setattr(hf, k, v) for k, v in d.items()]
    SpeculativeConfig.hf_config_override(hf, model_path=None)

    assert hf.architectures == ["Qwen3ForCausalLM"]
    assert getattr(hf, "dspark_with_draft", False) is False


# --------------------------------------------------------------------------
# Attention flavor
# --------------------------------------------------------------------------


def test_dflash_resolves_to_mha_not_mla():
    """No kv_lora_rank -> MHA family.

    This is the fork that hands the draft a sibling KV pool instead of binding
    it into the target's latent rows, and it is decided by SHAPE (the absence of
    a latent rank), not by a model-type list -- so it must hold for a draft
    whose model_type is the same `qwen3` a target could carry.

    Kept off `get_attn_backend`, which reads the GPU arch through rocminfo; the
    DRAFT_OWNS_KV_POOL flag it would return is asserted separately below.
    """
    selector = pytest.importorskip(
        "atom.utils.selector",
        reason="attn backend selection imports aiter at module load",
        exc_type=ImportError,
    )

    assert selector.attn_family(_radixark_cfg()) is selector.Family.MHA
    # The MLA draft it sits beside must still resolve the other way.
    assert (
        selector.attn_family(
            types.SimpleNamespace(model_type="k3_dspark", kv_lora_rank=512)
        )
        is not selector.Family.MHA
    )


def test_dflash_backend_owns_a_kv_pool():
    """The MHA backend answers "yes" to owning a draft pool.

    A GQA draft caches rows the target's MLA pool cannot express, so it has to
    own a sibling pool; `draft_kv_builder` returns None for any backend that
    says otherwise, and the draft would then silently have no KV at all.
    """
    # RuntimeError as well as ImportError: importing the backend runs aiter's
    # chip detection, which reads the GPU arch out of rocminfo and raises when
    # there is no GPU. The flag itself is a ClassVar and needs neither.
    mod = pytest.importorskip(
        "atom.model_ops.attentions.aiter_attention",
        reason="backend imports aiter, which probes the GPU arch at load",
        exc_type=(ImportError, RuntimeError),
    )
    assert mod.AiterBackend.DRAFT_OWNS_KV_POOL is True


# --------------------------------------------------------------------------
# KV pool geometry
# --------------------------------------------------------------------------


def _pool(block_size, world_size, kv_dtype=None, blocks_per_entry=1):
    torch = pytest.importorskip("torch")
    from atom.model_ops.attentions.mha_kv_pool import MhaKvPool

    hf = _radixark_cfg()
    return MhaKvPool.from_hf_config(
        hf,
        world_size=world_size,
        kv_dtype=kv_dtype or torch.float8_e4m3fn,
        layers=5,
        block_size=block_size,
        blocks_per_entry=blocks_per_entry,
        per_tensor_scales=True,
    )


def test_dflash_pool_needs_the_scheduler_block_size():
    """The pool must be built at the scheduler's block, not an MLA kernel page.

    An MLA target's builder pages at ATOM_MLA_PAGE_SIZE (1 by default). Sizing
    the draft pool from it cannot even construct -- SHUFFLE packs 16 elements
    per tile and 16 does not divide 1 -- which is why DraftKvBuilder reads
    model_runner.block_size.
    """
    pytest.importorskip(
        "atom.model_ops.attentions.mha_kv_pool",
        reason="pool imports aiter at module load",
        exc_type=ImportError,
    )
    assert _pool(block_size=128, world_size=8).block_size == 128
    with pytest.raises(ValueError, match="SHUFFLE"):
        _pool(block_size=1, world_size=8)


@pytest.mark.parametrize("tp,dcp", [(8, 1), (8, 2), (8, 8)])
def test_dflash_pool_footprint_is_dcp_invariant(tp, dcp):
    """A wider entry over fewer blocks is the same per-rank footprint.

    The draft is NOT in the dcp group: its KV heads are sharded by the full TP
    world size whatever `-dcp` says, and every rank holds the whole token axis.
    What moves with dcp is the ENTRY: one target block id addresses
    `block_size * dcp` global positions, so the pool is built with
    `blocks_per_entry = dcp` -- `entry_bytes` grows by dcp while the number of
    entries a request needs falls by dcp.

    This is the invariant that lets capacity planning done at DCP1 carry to
    DCP8. It held under the OLD geometry too (heads replicated at tp // dcp,
    tokens sharded), by the opposite cancellation -- which is exactly why it
    cannot be the only test of the decision; `test_dflash_pool_is_built_at_full_tp`
    pins which of the two we build.
    """
    pytest.importorskip(
        "atom.model_ops.attentions.mha_kv_pool",
        reason="pool imports aiter at module load",
        exc_type=ImportError,
    )
    pool = _pool(block_size=128, world_size=tp, blocks_per_entry=dcp)

    # Full TP, every dcp: 16 KV heads over 8 ranks is 2 each.
    assert pool.num_kv_heads == 16 // tp
    assert pool.entry_tokens == 128 * dcp

    # A sequence of SEQ global tokens needs SEQ/(128*dcp) entries, each
    # `dcp` times as wide. 163840 B is the one-block cost at 2 KV heads
    # (5 layers x {k,v} x 2 heads x 64 head_dim x 128 tokens x 1 B, and not a
    # byte of field alignment); the product below is what must not move.
    # (163840 / 128 = 1280 B per token per rank.) It was 174080 while the pool
    # declared per-token scale planes -- 5 x 2 x 2 x 128 x 4 B = 10240 of fp32
    # the per-tensor scale does not need.
    seq = 128 * 8 * 4
    entries = seq // (128 * dcp)
    assert pool.entry_bytes * entries == 163840 * (seq // 128), (
        f"per-rank draft KV moved with dcp: {pool.entry_bytes} B/entry x "
        f"{entries} entries at dcp={dcp}"
    )


# --------------------------------------------------------------------------
# Non-causal KV: the predicate and the dtype it forces
# --------------------------------------------------------------------------


def _normalized_radixark():
    SpeculativeConfig = _real_hf_config_override()
    if SpeculativeConfig is None:
        pytest.skip("atom.config not importable in this sandbox")
    hf = _radixark_cfg()
    SpeculativeConfig.hf_config_override(hf, model_path=None)
    return hf


def test_dflash_uses_dflash_pool():
    """Only the SpecForge DFlash draft owns the DCP1 FLASH-layout pool."""
    dk = pytest.importorskip(
        "atom.spec_decode.draft_kv",
        reason="draft_kv imports aiter lazily but config does not",
        exc_type=(ImportError, RuntimeError),
    )
    assert dk.uses_dflash_pool(_normalized_radixark()) is True

    # MiniMax-M3.1's DSpark draft is ALSO a non-MLA block drafter with a pool of
    # its own, but it binds a SHUFFLE pool with per-token scales; keying on
    # "non-MLA block drafter" handed it this one.
    minimax = types.SimpleNamespace(
        architectures=["DSparkMiniMaxDraftModel"], dspark_with_draft=True
    )
    assert dk.uses_dflash_pool(minimax) is False

    # The MLA DSpark draft binds into the target's latent rows: no pool.
    mla = types.SimpleNamespace(
        architectures=["K3DSparkModel"], dspark_with_draft=True, kv_lora_rank=512
    )
    assert dk.uses_dflash_pool(mla) is False

    # A serial MTP/EAGLE draft, and a config with no architectures at all.
    eagle = types.SimpleNamespace(architectures=["Eagle3LlamaModel"])
    assert dk.uses_dflash_pool(eagle) is False
    assert dk.uses_dflash_pool(types.SimpleNamespace(num_hidden_layers=1)) is False


def test_dflash_pool_inherits_the_targets_fp8_and_declares_no_scale_planes():
    """The draft caches fp8 beside an fp8 target, at ONE per-tensor scale.

    SGLang serves this checkpoint with `--speculative-draft-kv-cache-dtype
    fp8_e4m3`, and both ends of ATOM's pool take exactly the scale it stores:
    `aiter.reshape_and_cache_flash` dereferences `*k_scale` once per launch and
    `unified_attention` loads one `k_descale` per program. Neither can read a
    per-(kv head, token) plane, so the pool must declare none -- `per_tensor_
    scales=True`. Getting that wrong is not a crash: both read the plane's
    first element and apply it to every token.

    `float8_e4m3fn` specifically, never `fnuz`: triton upcasts fnuz to fp16 on
    gfx950, which measures slower than the bf16 it was meant to beat.
    """
    import torch

    dk = pytest.importorskip(
        "atom.spec_decode.draft_kv",
        reason="draft_kv imports aiter at call time",
        exc_type=(ImportError, RuntimeError),
    )
    pytest.importorskip(
        "atom.model_ops.attentions.aiter_attention",
        reason="backend probes the GPU arch at import",
        exc_type=(ImportError, RuntimeError),
    )

    seen = {}

    class _Backend:
        DRAFT_OWNS_KV_POOL = True

        @staticmethod
        def make_kv_pool(hf, *, world_size, kv_dtype, **kw):
            seen["kv_dtype"] = kv_dtype
            seen.update(kw)
            return object()

    runner = _stub_runner()
    hf = _normalized_radixark()

    # Patch the SELECTOR module, not draft_kv: `draft_kv_builder` imports
    # attn_family/get_attn_backend function-locally, so a module-attribute
    # patch on draft_kv would never be consulted.
    from unittest import mock

    from atom.utils import selector

    with (
        mock.patch.object(selector, "get_attn_backend", lambda _f: _Backend),
        mock.patch.object(selector, "attn_family", lambda _c: None),
    ):
        builder = dk.draft_kv_builder(runner, hf)

    assert builder is not None
    builder._make_pool(layers=5, target_block_size=128)
    assert (
        seen["kv_dtype"] is torch.float8_e4m3fn
    ), f"draft pool must be fp8_e4m3fn under an fp8 target, got {seen['kv_dtype']}"
    assert seen["per_tensor_scales"] is True

    # ... and the pool it describes really has no scale region.
    from atom.model_ops.attentions.mha_kv_pool import MhaKvPool

    pool = MhaKvPool.from_hf_config(
        hf,
        world_size=8,
        block_size=128,
        layers=5,
        kv_dtype=torch.float8_e4m3fn,
        per_tensor_scales=True,
    )
    assert pool.scale_fields == []
    with pytest.raises(KeyError):
        pool.allocate(4, "cpu")
        pool.scale_views(0)


# --------------------------------------------------------------------------
# D1: the draft's effective TP, and the plumbing that carries it
# --------------------------------------------------------------------------


def _stub_runner(tp=8, dcp=1, kv_cache_dtype="fp8"):
    """A ModelRunner stand-in carrying only what `draft_kv_builder` reads.

    `world_size` IS `config.tensor_parallel_size` on the real runner
    (model_runner.py); both are present so a regression that goes back to
    reading `world_size` is visible as a wrong number rather than an
    AttributeError.
    """
    return types.SimpleNamespace(
        world_size=tp,
        config=types.SimpleNamespace(
            kv_cache_dtype=kv_cache_dtype,
            tensor_parallel_size=tp,
            decode_context_parallel_size=dcp,
        ),
    )


@pytest.mark.parametrize("tp,dcp", [(8, 1), (8, 2), (8, 8)])
def test_dflash_pool_is_built_at_full_tp(tp, dcp):
    """`draft_kv_builder` must size the pool at FULL TP, `dcp` blocks an entry.

    The draft does not join the dcp group -- SGLang's does not either
    (`models/qwen3_5.py:1112`) -- so its KV heads shard by the whole TP world
    and every rank holds the whole token axis. What DCP costs it is the ENTRY:
    one target block id spans `block_size * dcp` global positions.

    The DCP-invariance test above builds the pool by calling `_pool` directly,
    so it pins the ARITHMETIC but not the PLUMBING -- and that arithmetic is
    satisfied by the OLD geometry too (heads replicated at tp // dcp, tokens
    sharded), which a read that returns no LSE cannot merge across ranks.
    This is the test that pins which of the two is built.
    """
    dk = pytest.importorskip(
        "atom.spec_decode.draft_kv",
        reason="draft_kv imports aiter at call time",
        exc_type=(ImportError, RuntimeError),
    )
    pytest.importorskip(
        "atom.model_ops.attentions.aiter_attention",
        reason="backend probes the GPU arch at import",
        exc_type=(ImportError, RuntimeError),
    )
    from unittest import mock

    from atom.utils import selector

    seen = {}

    class _Backend:
        DRAFT_OWNS_KV_POOL = True

        @staticmethod
        def make_kv_pool(hf, *, world_size, kv_dtype, **kw):
            seen["world_size"] = world_size
            seen.update(kw)
            return object()

    hf = _normalized_radixark()
    with (
        mock.patch.object(selector, "get_attn_backend", lambda _f: _Backend),
        mock.patch.object(selector, "attn_family", lambda _c: None),
    ):
        builder = dk.draft_kv_builder(_stub_runner(tp=tp, dcp=dcp), hf)

    builder._make_pool(layers=5, target_block_size=128)
    assert seen["world_size"] == tp
    assert seen["blocks_per_entry"] == dcp


# --------------------------------------------------------------------------
# The target feature: Kimi-K3's aggregate stream
# --------------------------------------------------------------------------
#
# Kimi-K3 carries a BANK of per-`attn_res_block_size` snapshots plus a running
# PREFIX, recombined by a softmax mixture. The generic aux extractor hands a
# draft the prefix; SpecForge trained THIS draft against a live SGLang target,
# whose dspark tap returns the mixture. Feeding it the prefix instead measured
# -6.43 pp acceptance on GSM8K.
#
# The routing is by draft CLASS, not by a flag: `KimiK3DFlash` implements
# `target_aux_capture_spec`, the MLA draft beside it deliberately does not. The
# tests below pin both halves of that, because the failure mode in either
# direction is silent -- a draft fed the wrong feature still proposes tokens,
# they are just accepted less often.


def _reference_mixture(prefix, bank, score_weight, eps):
    """The mix `apply_attn_res` computes, transcribed into plain torch.

    From `_attn_res_fused_kernel`: over the B bank rows plus the prefix,
    score = <v, score_weight> * rsqrt(mean(v^2) + eps), softmax over the B+1,
    then a weighted sum of the RAW candidates (the rmsnorm scales the SCORE,
    it does not normalize what is summed).
    """
    import torch

    rows = torch.cat([bank, prefix.unsqueeze(1)], dim=1).float()
    rstd = torch.rsqrt(rows.pow(2).mean(-1) + eps)
    probs = torch.softmax((rows * score_weight).sum(-1) * rstd, dim=-1)
    return (probs.unsqueeze(-1) * rows).sum(1).to(prefix.dtype)


def _fake_attn_res_kernel(monkeypatch, calls):
    """Stand in for `atom.model_ops.kimi_k3`, which needs aiter and a GPU.

    The draft imports `apply_attn_res` inside the extract, so swapping the
    module in `sys.modules` is enough -- and it keeps these tests CPU-only,
    which is the only way they run in CI at all. The stand-in computes the real
    mixture, so the assertions below pin the ARGUMENTS the draft marshals *and*
    that what comes back is the mixture rather than the prefix.
    """
    import sys

    module = types.ModuleType("atom.model_ops.kimi_k3")

    def apply_attn_res(
        prefix_sum,
        block_residual,
        score_weight,
        eps,
        add_hidden=None,
        out_norm_weight=None,
        out_eps=1e-6,
        add_hidden2=None,
        quant_dtype=None,
    ):
        calls.append(
            {
                "prefix": prefix_sum,
                "bank": block_residual,
                "score_weight": score_weight,
                "eps": eps,
                "out_norm_weight": out_norm_weight,
                "quant_dtype": quant_dtype,
            }
        )
        return _reference_mixture(prefix_sum, block_residual, score_weight, eps), None

    module.apply_attn_res = apply_attn_res
    monkeypatch.setitem(sys.modules, "atom.model_ops.kimi_k3", module)
    return calls


def _k3_aux_hidden_state(output):
    """`KimiDecoderLayer.aux_hidden_state`, transcribed.

    The real one lives on a `@support_torch_compile` model whose import pulls in
    aiter; its contract -- (prefix_sum, *pendings, block_residual), pendings
    summed back into the prefix -- is all the extract under test depends on.
    """
    prefix_sum, *pendings, _bank = output
    if prefix_sum is None:
        return None
    for pending in pendings:
        if pending is not None:
            prefix_sum = prefix_sum + pending
    return prefix_sum


def _k3_target_stub(hidden=16, n_layers=16, block_size=4, enabled=True):
    """A Kimi-K3 target, down to what aux capture reads off it.

    `_resolve_decoder_layers` wants `.layers`; the extract wants each layer's
    `layer_idx` and `aux_hidden_state`, and the NEXT layer's
    `self_attention_attn_res` (that is the whole reason the hook needs the
    target model and not just the tapped block).
    """
    import torch

    torch.manual_seed(0)
    layers = [
        types.SimpleNamespace(
            layer_idx=i,
            aux_hidden_state=_k3_aux_hidden_state,
            self_attention_attn_res=types.SimpleNamespace(
                enabled=enabled,
                block_size=block_size,
                eps=1e-6,
                out_eps=1e-5,
                score_weight=torch.randn(hidden) * 0.1,
            ),
        )
        for i in range(n_layers)
    ]
    return types.SimpleNamespace(layers=layers)


def _dflash_spec(target, layer_ids, hidden=16):
    """Resolve `KimiK3DFlash`'s capture spec without building the draft.

    Called unbound on purpose: the method reads nothing off `self` (its inputs
    are the tap ids, the target's hidden size, and the target model), and
    constructing the real draft would want a loaded checkpoint, a TP group and
    a GPU.
    """
    dflash = pytest.importorskip(
        "atom.models.kimi_k3_dflash",
        reason="the draft model imports aiter at module load",
        exc_type=(ImportError, RuntimeError),
    )
    return dflash.KimiK3DFlash.target_aux_capture_spec(None, layer_ids, hidden, target)


def _generic_extract():
    proposer = pytest.importorskip(
        "atom.spec_decode.dspark_proposer",
        reason="the proposer imports aiter/triton at module load",
        exc_type=(ImportError, RuntimeError),
    )
    return proposer.DSparkProposer._extract_layer_hidden


def test_dflash_owns_a_capture_spec_that_feeds_the_aggregate(monkeypatch):
    """(a) The draft resolves to its OWN extract, and that extract mixes.

    Both halves matter: resolving to its own spec is what makes the routing
    draft-owned, and the extract returning the mixture rather than the prefix is
    the feature itself. The second is asserted against an independent
    computation of the mixture, not merely "something different".
    """
    torch = pytest.importorskip("torch")
    generic = _generic_extract()

    hidden, block_size = 16, 4
    target = _k3_target_stub(hidden=hidden, n_layers=16, block_size=block_size)
    # Tap 6, not 7: with block_size 4 the ceiling and the floor of
    # (lid + 1) / block_size differ there, so the bank slice below actually
    # pins the rounding.
    spec = _dflash_spec(target, (6, 11), hidden=hidden)

    assert spec.layer_ids == (6, 11)
    assert spec.hidden_size == hidden
    assert spec.capture == "output"  # an OUTPUT tap; V4.1's is an input tap
    assert spec.extract is not generic

    calls = _fake_attn_res_kernel(monkeypatch, [])
    rows = 5
    prefix = torch.randn(rows, hidden)
    pending = torch.randn(rows, hidden)
    bank = torch.randn(rows, 6, hidden)
    block = target.layers[6]

    out = spec.extract((prefix, pending, bank), block)

    # The prefix half is still the generic reconstruction: the layer's deferred
    # addends folded back in. The aggregate is a mixture OF that, not a
    # replacement for it.
    reconstructed = generic((prefix, pending, bank), block)
    assert torch.allclose(reconstructed, prefix + pending)

    # nvb is a CEILING, matching SGLang's `prev_valid_blocks`: layer 6 sits
    # above ceil(7 / 4) = 2 closed blocks (a floor would say 1), so only 2 of
    # the bank's 6 rows are candidates.
    assert len(calls) == 1
    call = calls[0]
    assert call["bank"].shape == (rows, 2, hidden)
    assert torch.equal(call["bank"], bank[:, :2, :])
    assert torch.equal(call["prefix"], reconstructed)
    # Scored by the NEXT layer's attn-residual weights -- the one thing only the
    # target model can supply.
    res = target.layers[7].self_attention_attn_res
    assert call["score_weight"] is res.score_weight
    assert call["eps"] == res.eps
    # Un-normed and unquantized: SGLang's aggregate is the bare mixture.
    assert call["out_norm_weight"] is None
    assert call["quant_dtype"] is None

    expected = _reference_mixture(
        reconstructed, bank[:, :2, :], res.score_weight, res.eps
    )
    assert torch.allclose(out, expected)
    # ...and it is NOT the plain post-layer hidden state, which is the bug.
    assert not torch.allclose(out, reconstructed)


def test_dflash_aggregate_leaves_untapped_layers_on_the_prefix(monkeypatch):
    """A layer the draft does not tap keeps the plain reconstruction.

    The hook is installed per tapped layer, but ONE extract serves all of them,
    so it dispatches on the block it is handed. A miss must degrade to the
    prefix rather than mix against another layer's bank.
    """
    torch = pytest.importorskip("torch")
    hidden = 16
    target = _k3_target_stub(hidden=hidden)
    spec = _dflash_spec(target, (6, 11), hidden=hidden)

    calls = _fake_attn_res_kernel(monkeypatch, [])
    prefix = torch.randn(3, hidden)
    bank = torch.randn(3, 6, hidden)

    out = spec.extract((prefix, None, bank), target.layers[5])
    assert torch.equal(out, prefix)
    assert calls == []

    # Fewer closed blocks than the tap implies: nothing to mix against.
    short = spec.extract((prefix, None, bank[:, :1, :]), target.layers[6])
    assert torch.equal(short, prefix)
    assert calls == []


def test_dflash_falls_back_when_the_target_has_no_attn_residuals():
    """No attn-residual sites -> the prefix IS the whole stream.

    Guards the shape of the degradation: the draft must not try to mix against
    a bank that does not exist, and must not raise either -- the generic
    extractor is a correct answer for a target with no mixture.
    """
    generic = _generic_extract()
    target = _k3_target_stub(enabled=False)
    spec = _dflash_spec(target, (7, 11))
    assert spec.extract is generic


def test_k3_dspark_mla_draft_resolves_to_the_generic_extractor():
    """(b) REGRESSION GUARD. The Inferact MLA draft must NOT get the aggregate.

    `Inferact/Kimi-K3-DSpark` was trained through vLLM/HF against the plain
    residual stream. Measured, the aggregate costs it ~2.6 pp acceptance -- and
    it is the draft `.github/benchmark/models_accuracy.json` tracks, so an
    earlier, globally-defaulted-on version of this feature regressed CI. It
    keeps the prefix by not implementing the hook at all.
    """
    dspark = pytest.importorskip(
        "atom.models.kimi_k3_dspark",
        reason="the draft model imports aiter at module load",
        exc_type=(ImportError, RuntimeError),
    )
    proposer_mod = pytest.importorskip(
        "atom.spec_decode.dspark_proposer",
        reason="the proposer imports aiter/triton at module load",
        exc_type=(ImportError, RuntimeError),
    )
    DSparkProposer = proposer_mod.DSparkProposer

    assert not hasattr(dspark.KimiK3DSpark, "target_aux_capture_spec")

    # ...and that absence must actually route it to the generic extractor.
    proposer = DSparkProposer.__new__(DSparkProposer)
    proposer.model = dspark.KimiK3DSpark.__new__(dspark.KimiK3DSpark)
    proposer.speculative_config = types.SimpleNamespace(
        draft_model_hf_config=types.SimpleNamespace(dspark_target_layer_ids=[7, 11])
    )
    proposer.config = types.SimpleNamespace(
        hf_config=types.SimpleNamespace(hidden_size=16)
    )
    spec = proposer._aux_capture_spec(_k3_target_stub())
    assert spec.extract is DSparkProposer._extract_layer_hidden
    assert spec.capture == "output"
    assert spec.layer_ids == (7, 11)


def test_dsv41_dspark_still_owns_its_input_tap():
    """(c) The signature widening must not disturb the other implementor.

    V4.1 reads its feature off the tapped block's own input and needs nothing
    from the target model; it takes the new argument and ignores it. Pinned
    because the call site passes it POSITIONALLY to every implementor -- there
    is no `inspect.signature` sniffing that could quietly skip it.
    """
    v41 = pytest.importorskip(
        "atom.models.deepseek_v41.dspark",
        reason="the draft model imports aiter at module load",
        exc_type=(ImportError, RuntimeError),
    )
    model = v41.DeepseekV41DSpark.__new__(v41.DeepseekV41DSpark)
    model.config = types.SimpleNamespace(engram_layer_ids=[1, 2, 3])

    spec = model.target_aux_capture_spec((37, 38, 39), 7168, _k3_target_stub())
    assert spec.layer_ids == (37, 38, 39)
    assert spec.hidden_size == 7168
    assert spec.capture == "input"
    assert spec.extract is v41.DeepseekV41DSpark._target_layer_input

    # The Engram guard still fires, i.e. the extra argument did not shift the
    # ones it reads.
    with pytest.raises(ValueError, match="Engram"):
        model.target_aux_capture_spec((1, 37), 7168, _k3_target_stub())


def test_dspark_proposer_hands_the_target_model_to_the_draft():
    """The call site passes the target model through, positionally.

    This is the plumbing the K3 aggregate needs: without it the draft cannot
    reach layer `lid + 1`'s scoring weights, and the feature silently degrades
    to the prefix.
    """
    proposer_mod = pytest.importorskip(
        "atom.spec_decode.dspark_proposer",
        reason="the proposer imports aiter/triton at module load",
        exc_type=(ImportError, RuntimeError),
    )
    from atom.spec_decode.drafter import AuxCaptureSpec

    seen = {}

    def own(layer_ids, hidden_size, target_model):
        seen.update(
            layer_ids=layer_ids, hidden_size=hidden_size, target_model=target_model
        )
        return AuxCaptureSpec(layer_ids, hidden_size, lambda output, block: None)

    proposer = proposer_mod.DSparkProposer.__new__(proposer_mod.DSparkProposer)
    proposer.model = types.SimpleNamespace(target_aux_capture_spec=own)
    proposer.speculative_config = types.SimpleNamespace(
        draft_model_hf_config=types.SimpleNamespace(dspark_target_layer_ids=[7, 11])
    )
    proposer.config = types.SimpleNamespace(
        hf_config=types.SimpleNamespace(hidden_size=16)
    )
    target = _k3_target_stub()

    proposer._aux_capture_spec(target)
    assert seen == {"layer_ids": (7, 11), "hidden_size": 16, "target_model": target}


# --------------------------------------------------------------------------- #
# the DCP admission guard (config.py __post_init__)
#
# The guard refuses DCP for a draft that OWNS a sibling KV pool and does NOT
# read it non-causally. These exercise the PREDICATE exactly as `config.py`
# computes it. They do NOT exercise the `raise` itself: reaching it needs
# `aiter.jit.utils.chip_info.get_gfx` to answer "gfx950", so the end-to-end
# refusal belongs in the GPU batch. What is covered here is the part that
# decides WHICH configs are refused -- the half that silently regressed once.
# --------------------------------------------------------------------------- #


def _guard_refuses(draft_hf):
    """Mirror of `config.py` 2352-2358, the predicate under the DCP guard."""
    # get_attn_backend resolves to aiter_attention, which imports aiter at module
    # load. Skip per test, not per module, so the source-only guard test below
    # still runs on the CPU-only CI runner.
    pytest.importorskip("aiter")
    from atom.spec_decode.draft_kv import uses_dflash_pool
    from atom.utils.selector import attn_family, get_attn_backend

    owns_sibling_pool = draft_hf is not None and bool(
        get_attn_backend(attn_family(draft_hf)).DRAFT_OWNS_KV_POOL
    )
    return owns_sibling_pool and not uses_dflash_pool(draft_hf)


def test_dcp_guard_admits_the_dflash_draft():
    """DFlash owns a pool but stays out of the dcp group, so DCP is fine."""
    draft_hf = _normalized_radixark()
    assert _guard_refuses(draft_hf) is False


def test_dcp_guard_refuses_a_pool_owning_causal_draft():
    """An Eagle3-llama style MHA draft double-shards: head axis AND token axis.

    This is the case the guard exists for. It owns a sibling pool like DFlash
    does, but reads it causally, so it has no DCP1 escape -- and the failure is
    silent (lost acceptance length, no fault), which is why it must be refused
    at launch rather than diagnosed later.
    """
    draft_hf = types.SimpleNamespace(
        architectures=["LlamaForCausalLMEagle3"],
        model_type="llama",
        num_attention_heads=32,
        num_key_value_heads=8,
        head_dim=128,
        num_hidden_layers=1,
    )
    assert _guard_refuses(draft_hf) is True


def test_dcp_guard_admits_an_mtp_draft():
    """`--method mtp` must still pass -- an earlier guard version broke it.

    For mtp the draft hf config IS the target's, so `attn_family` returns the
    TARGET's family (KIMI_MLA here), whose backend leaves `DRAFT_OWNS_KV_POOL`
    at the default False: a latent-KV draft shares the target's cache and owns
    no sibling pool. Keying on ownership rather than on `Family.MLA` is what
    keeps `recipes/Agentic-GLM-5.2.md`'s documented
    `--method mtp --decode-context-parallel-size 4` working.
    """
    target_hf = types.SimpleNamespace(
        architectures=["KimiK3ForCausalLM"],
        model_type="kimi_k3",
        num_attention_heads=64,
        num_hidden_layers=2,
        kv_lora_rank=512,
        qk_rope_head_dim=64,
        qk_nope_head_dim=128,
        v_head_dim=128,
    )
    assert _guard_refuses(target_hf) is False


def test_dcp_guard_message_names_the_real_reason():
    """The refusal text must describe DCP1, not a replicate-and-merge scheme.

    The draft's read returns only `o` and never an LSE, so the draft cannot
    merge per-rank partial softmaxes -- it is exempt because it never joins the
    dcp group (`draft_kv_builder`). A message claiming
    otherwise sends a reader looking for machinery that does not exist.
    """
    import os
    import re

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    with open(os.path.join(root, "atom", "config.py"), encoding="utf-8") as fh:
        src = fh.read()
    guard = src[src.index("is only supported for the") :][:600]
    assert "stays out of the DCP group" in guard
    assert "DCP1" in guard
    assert not re.search(r"merges the per-rank partial softmaxes", guard)


# --------------------------------------------------------------------------- #
# draft pool address arithmetic (dspark_proposer)
#
# `_draft_entry_tokens` is called on the REAL property. The slot and
# prefix-length cases MIRROR the production expressions rather than driving
# `_publish_draft_ctx_slots` end to end (that needs a forward_context, a
# published block_table_state and attn_metadata). They are regression tests on
# the ARITHMETIC -- they prove the aliasing the production comment names is
# real and that the chosen formula avoids it. Driving the real method belongs
# in the GPU batch alongside the rest of the proposer.
# --------------------------------------------------------------------------- #


def _entry_tokens(block_size, dcp, non_causal_gqa):
    """Call the real property against a minimal stub."""
    # dspark_proposer imports triton at module load; the CPU-only CI runner has
    # none. The pure-arithmetic mirrors below do not need it and keep running.
    pytest.importorskip("triton")
    from atom.spec_decode.dspark_proposer import DSparkProposer

    stub = types.SimpleNamespace(
        runner=types.SimpleNamespace(block_size=block_size),
        dcp_world_size=dcp,
        _draft_uses_dflash_pool=non_causal_gqa,
    )
    return DSparkProposer._draft_entry_tokens.fget(stub)


@pytest.mark.parametrize("dcp", [1, 2, 8])
def test_draft_entry_spans_the_global_block_for_the_gqa_draft(dcp):
    """A draft outside the dcp group stores all `block_size * dcp` positions.

    The target's block id addresses `block_size` LOCAL tokens per rank. The
    DFlash draft is not in the dcp group, so one id has to cover the whole
    virtual block `dcp_prefill_slot_mapping` writes.
    """
    assert _entry_tokens(128, dcp, non_causal_gqa=True) == 128 * dcp


@pytest.mark.parametrize("dcp", [1, 2, 8])
def test_draft_entry_is_the_plain_block_for_the_mla_draft(dcp):
    """A DCP-sharded draft addresses in LOCAL terms -- one id, one block."""
    assert _entry_tokens(128, dcp, non_causal_gqa=False) == 128


def test_block_size_instead_of_entry_tokens_aliases_requests():
    """The regression `_publish_draft_ctx_slots` guards against.

    Mirrors the slot expression at `dspark_proposer.py:844-846`:
        block_tables[req, page] * vblock + pos % vblock
    Using `block_size` in place of `vblock` addresses one `dcp`-th of each
    entry, so `dcp` distinct block ids collapse onto overlapping slot ranges.
    """
    import torch

    block_size, dcp = 128, 8
    vblock = block_size * dcp
    # Two requests holding ADJACENT entries, each reading its own first token.
    block_ids = torch.tensor([3, 4], dtype=torch.int64)
    pos = torch.tensor([0, 0], dtype=torch.int64)

    correct = block_ids * vblock + torch.remainder(pos, vblock)
    assert correct.tolist() == [3 * 1024, 4 * 1024]
    # Distinct entries stay a full entry apart.
    assert int(correct[1] - correct[0]) == vblock

    wrong = block_ids * block_size + torch.remainder(pos, block_size)
    # The bug: ids land `block_size` apart, so entry 3's 1024-token span runs
    # straight through entries 4..10 -- eight requests on one range.
    assert int(wrong[1] - wrong[0]) == block_size
    assert int(wrong[0]) + vblock > int(wrong[1]), "must overlap to be the bug"


def test_pages_past_the_table_are_dropped_not_clamped():
    """Mirrors `dspark_proposer.py:838-848`.

    A column the table does not have becomes -1. Clamping it into the last
    column instead would hand the row ANOTHER request's KV, which reads as a
    plausible answer rather than a fault.
    """
    import torch

    vblock, width = 1024, 4
    block_tables = torch.tensor([[10, 11, 12, 13]], dtype=torch.int64)
    pos = torch.tensor([0, 3 * vblock, 4 * vblock], dtype=torch.int64)
    req = torch.zeros(3, dtype=torch.int64)

    page = torch.div(pos, vblock, rounding_mode="floor")
    in_table = page < width
    page = page.clamp_(max=width - 1)
    out = torch.where(
        in_table,
        block_tables[req, page] * vblock + torch.remainder(pos, vblock),
        torch.tensor(-1, dtype=torch.int64),
    )
    assert out.tolist() == [10 * vblock, 13 * vblock, -1]


# --------------------------------------------------------------------------- #
# the block read (model_ops/attentions/dspark_draft_attn.py)
#
# `unified_attention` asserts causal; the read gets a non-causal block out of it
# by folding each request's T rows into its head axis and decoding ONE row per
# request. The fold is pure tensor reshaping and runs anywhere; the kernel check
# below needs a GPU.
# --------------------------------------------------------------------------- #


def _fold_mod():
    pytest.importorskip("torch")
    from atom.model_ops.attentions import dspark_draft_attn

    return dspark_draft_attn


@pytest.mark.parametrize("bs,T,hkv,g", [(1, 7, 2, 4), (3, 7, 2, 4), (2, 4, 1, 8)])
def test_block_fold_puts_every_row_on_its_own_kv_head(bs, T, hkv, g):
    """`unified_attention` reads head `j` against KV head `j // (T * G)`.

    So in the folded layout, every one of a request's T rows times its G query
    heads per KV head must sit in that KV head's contiguous run -- and nothing
    of another request's. Tag each element with (request, row, query head) and
    read the tags back.
    """
    import torch

    dda = _fold_mod()
    hq = hkv * g
    b = torch.arange(bs).view(bs, 1, 1).expand(bs, T, hq)
    t = torch.arange(T).view(1, T, 1).expand(bs, T, hq)
    h = torch.arange(hq).view(1, 1, hq).expand(bs, T, hq)
    # One scalar per (b, t, h), carried in the D axis so the fold moves it.
    tag = torch.stack([b, t, h], dim=-1).reshape(bs * T, hq, 3)

    qf = dda.fold_block_queries(tag, bs, hkv)
    assert qf.shape == (bs, hkv * T * g, 3)
    for req in range(bs):
        for j in range(hkv * T * g):
            fb, ft, fh = qf[req, j].tolist()
            assert fb == req
            # The head the kernel assigns, and the head the row's query head
            # belongs to in the unfolded GQA layer.
            assert j // (T * g) == fh // g
            # Inside that run: row-major over (t, g), i.e. `(kv_head, t, g)`.
            assert (j % (T * g)) // g == ft
            assert j % g == fh % g

    out = torch.empty_like(tag)
    dda.unfold_block_output(qf, out, hkv)
    assert torch.equal(out, tag)


def _reference_block_attn(q, k_pool, v_pool, block_tables, kv_lens, scale, hkv):
    """Bidirectional GQA attention of each request's T rows over its pool rows,
    in fp32, gathered slot by slot from the block table."""
    import torch

    bs = kv_lens.shape[0]
    rows, hq, d = q.shape
    T, g, E = rows // bs, hq // hkv, k_pool.shape[1]
    out = torch.empty(rows, hq, d, dtype=torch.float32, device=q.device)
    for b in range(bs):
        n = int(kv_lens[b])
        pos = torch.arange(n, device=q.device)
        ent = block_tables[b, pos // E].long()
        k = k_pool[ent, pos % E].float()  # [n, hkv, d]
        v = v_pool[ent, pos % E].float()
        qb = q[b * T : (b + 1) * T].float().view(T, hkv, g, d)
        s = torch.einsum("thgd,nhd->thgn", qb, k) * scale
        p = torch.softmax(s, dim=-1)
        out[b * T : (b + 1) * T] = torch.einsum("thgn,nhd->thgd", p, v).reshape(
            T, hq, d
        )
    return out


@pytest.mark.parametrize("kv_dtype", ["bf16", "fp8"])
@pytest.mark.parametrize("far_entries", [False, True])
def test_unified_block_read_matches_the_bidirectional_reference(kv_dtype, far_entries):
    """The folded `unified_attention` read IS the block's non-causal attention.

    Draft shape at TP8: 8 q / 2 kv heads, head_dim 64, T = 7, an entry of
    `block_size 128 * dcp 8 = 1024` tokens. Requests of different lengths, one
    ending mid-entry, so a causal mask -- which a non-folded read would apply
    -- would change every row but the last.

    `far_entries` puts a request's pages past entry 16384, where
    `entry * page_stride` (1024 * 2 * 64 per entry) passes 2**31: the boundary
    a 32-bit address computation wraps at.
    """
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("needs a GPU")
    pytest.importorskip("aiter")
    dda = _fold_mod()

    torch.manual_seed(0)
    dev = torch.device("cuda")
    bs, T, hkv, g, d, E = 3, 7, 2, 4, 64, 1024
    hq = hkv * g
    kv_lens = torch.tensor([3 * E + 5, E - 1, 2 * E], dtype=torch.int32, device=dev)
    W = 4
    base = 16383 if far_entries else 0
    n_entries = base + bs * W + 1
    # Distinct, shuffled entry ids per request, so a wrong row of the table
    # reads another request's keys rather than the right ones by accident.
    ids = (base + torch.randperm(bs * W)).view(bs, W).to(torch.int32)
    block_tables = ids.to(dev)

    k_pool = torch.randn(n_entries, E, hkv, d, device=dev, dtype=torch.bfloat16)
    v_pool = torch.randn_like(k_pool)
    if kv_dtype == "fp8":
        from aiter import dtypes

        k_pool, v_pool = k_pool.to(dtypes.fp8), v_pool.to(dtypes.fp8)
    q = torch.randn(bs * T, hq, d, device=dev, dtype=torch.bfloat16)
    scale = d**-0.5

    read = dda.DSparkDraftUnifiedAttention(dev, max_bs=8)
    o = torch.empty_like(q)
    one = torch.ones(1, dtype=torch.float32, device=dev)
    read(q, k_pool, v_pool, block_tables, kv_lens, o, scale, 1.0, 1.0, one, one)

    ref = _reference_block_attn(q, k_pool, v_pool, block_tables, kv_lens, scale, hkv)
    torch.testing.assert_close(o.float(), ref, atol=2e-2, rtol=2e-2)


def test_the_read_needs_no_csr_from_the_proposer():
    """The DFlash path returns before the generic CSR build.

    Source-level, like the guard-message test above: the proposer's metadata
    build is a method on a fully wired runner. What it pins is that the early
    return for the DFlash pool sits AHEAD of the `kv_indices_generate_triton`
    call, so the read's block table is the only addressing published for it.
    """
    import os

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    path = os.path.join(root, "atom", "spec_decode", "dspark_proposer.py")
    with open(path, encoding="utf-8") as fh:
        src = fh.read()
    body = src[src.index("    def _build_paged_block_metadata(") :]
    early = body.index("        if self._draft_uses_dflash_pool:\n")
    assert early < body.index("kv_indices_generate_triton(")
    assert "return" in body[early : body.index("kv_indices_generate_triton(")]
