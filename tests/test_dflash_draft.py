# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Unit tests for the SpecForge DFlash draft (RadixArk/Kimi-K3-DSpark).

GPU-free. Covers the three things that decide whether the draft is even
reachable, and that no amount of GPU bring-up would localize:

- config normalization: the arch rename and the nested ``dflash_config`` lift
- attention-flavor resolution: the draft must resolve to MHA and own a KV pool
- KV-pool geometry: the scheduler block size, and the DCP head-sharding
  invariant that makes a DCP8 step additive rather than a rewrite

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


def test_dflash_needs_non_causal_kv():
    """Only a non-MLA BLOCK drafter reads its own pool bidirectionally."""
    dk = pytest.importorskip(
        "atom.spec_decode.draft_kv",
        reason="draft_kv imports aiter lazily but config does not",
        exc_type=(ImportError, RuntimeError),
    )
    assert dk.needs_non_causal_kv(_normalized_radixark()) is True

    # The MLA DSpark draft is also a block drafter, but it binds into the
    # target's latent rows and gets non-causal from the MLA path instead.
    mla = types.SimpleNamespace(dspark_with_draft=True, kv_lora_rank=512)
    assert dk.needs_non_causal_kv(mla) is False

    # A serial MTP/EAGLE draft emits one token per pass: plainly causal.
    serial = types.SimpleNamespace(num_hidden_layers=1)
    assert dk.needs_non_causal_kv(serial) is False


def test_dflash_pool_inherits_the_targets_fp8_and_declares_no_scale_planes():
    """The draft caches fp8 beside an fp8 target, at ONE per-tensor scale.

    SGLang serves this checkpoint with `--speculative-draft-kv-cache-dtype
    fp8_e4m3`, and both ends of ATOM's pool take exactly the scale it stores:
    `aiter.reshape_and_cache_flash` dereferences `*k_scale` once per launch and
    `verify_shared_kv_fwd` folds `float(k_scale)` into its qk. Neither can read
    a per-(kv head, token) plane, so the pool must declare none -- `per_tensor_
    scales=True`. Getting that wrong is not a crash: `verify_shared_kv_fwd`
    catches the `float()` ValueError and silently substitutes 1.0.

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
    sharded), which is the one that cannot be read by `verify_shared_kv_fwd`.
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
