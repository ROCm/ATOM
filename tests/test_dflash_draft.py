# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Unit tests for the SpecForge DFlash draft (RadixArk/Kimi-K3-DSpark).

The config-normalization and fold tests run on CPU; the rest need the
aiter/triton stack and skip without it, and the kernel checks need a GPU.
Covers the things that decide whether the draft is even reachable, and that no
amount of GPU bring-up would localize:

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


def _gpu_stack_import(name):
    """Import an atom module that may need the aiter/triton GPU stack.

    Tries the import first, so a module that imports fine without the stack
    still runs on the CPU-only CI runner. A failure is turned into a skip only
    when the stack is genuinely absent -- no aiter or triton, or no GPU for
    aiter to probe the arch of at import. On a GPU host any import error FAILS.
    """
    import importlib

    try:
        return importlib.import_module(name)
    except Exception:
        try:
            importlib.import_module("aiter")
            importlib.import_module("triton")
            import torch

            gpu = torch.cuda.is_available()
        except Exception as e:  # noqa: BLE001 - any failure here means no stack
            pytest.skip(f"{name} needs the aiter/triton stack, absent here: {e}")
        if not gpu:
            pytest.skip(f"{name} needs a GPU: aiter probes the arch at import")
        raise


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
    from atom.config import SpeculativeConfig

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


def test_dflash_config_normalize_is_idempotent():
    """A second override keeps the DFlash draft rather than routing it, by its
    `dspark_block_size`, to the V4 DSpark draft."""
    from atom.config import SpeculativeConfig

    hf = _radixark_cfg()
    SpeculativeConfig.hf_config_override(hf, model_path=None)
    model_type = hf.model_type
    SpeculativeConfig.hf_config_override(hf, model_path=None)

    assert hf.architectures == ["K3DFlashModel"]
    assert hf.dspark_block_size == 7
    assert hf.model_type == model_type != "deepseek_v4_dspark"


def test_dflash_rope_is_the_drafts_own():
    """The rope the draft attention builds comes from the draft's config:
    theta 10000 / YaRN x16 from 65536, neox-style -- not the target's theta
    50000 / x32. A wrong rope loads clean and only moves acceptance."""
    from atom.config import SpeculativeConfig

    kwargs_fn = _draft_rope_kwargs_fn("kimi_k3_dflash.py")
    hf = _radixark_cfg()
    SpeculativeConfig.hf_config_override(hf, model_path=None)
    kw = kwargs_fn(hf)
    assert kw["base"] == 10000.0
    assert kw["is_neox_style"] is True
    assert kw["rope_scaling"]["rope_type"] == "yarn"
    assert kw["rope_scaling"]["factor"] == 16.0
    assert kw["rope_scaling"]["original_max_position_embeddings"] == 65536

    # No scaling in the config: a plain rope, not YaRN.
    plain = _radixark_cfg(rope_parameters={"rope_theta": 10000.0})
    assert kwargs_fn(plain)["rope_scaling"] is None


def _inferact_rope_cfg(**overrides):
    """The rope fields of the Inferact MLA K3-DSpark checkpoint's config.json."""
    cfg = types.SimpleNamespace(
        max_position_embeddings=1048576,
        rope_theta=50000.0,
        rope_parameters={
            "rope_type": "yarn",
            "factor": 32.0,
            "original_max_position_embeddings": 32768,
            "rope_theta": 50000.0,
            "beta_fast": 32,
            "beta_slow": 1,
            "mscale": 1.0,
            "mscale_all_dim": 1.0,
        },
    )
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg


def _lift_fn(module_file, name, ns):
    """A draft module's top-level function `name`, runnable without the aiter
    stack.

    The draft modules import aiter at load, so the one function is lifted from
    the source and bound to the CPU-importable names in `ns` it calls.
    """
    import ast
    import pathlib

    from atom.models import dspark_draft

    path = pathlib.Path(dspark_draft.__file__).with_name(module_file)
    tree = ast.parse(path.read_text())
    (fn,) = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name]
    exec(  # noqa: S102 - execute only AST nodes from this repository's source.
        compile(ast.Module(body=[fn], type_ignores=[]), str(path), "exec"), ns
    )
    return ns[name]


def _draft_rope_kwargs_fn(module_file):
    """A draft module's `_draft_rope_kwargs`, bound to the shared parser."""
    from atom.models import dspark_draft

    ns = {"draft_rope_scaling": dspark_draft.draft_rope_scaling}
    return _lift_fn(module_file, "_draft_rope_kwargs", ns)


def test_dflash_refuses_a_max_model_len_past_the_rope_cache():
    """The context write ropes unclamped positions, all below `max_model_len`.
    A length equal to the cache rows -- the shipped pair's default -- fits;
    one more row is refused at startup."""
    check = _lift_fn("kimi_k3_dflash.py", "_check_rope_covers", {})
    rows = 65536 * 16
    check(rows, rows)
    check(4096, rows)
    with pytest.raises(ValueError, match="lower --max-model-len"):
        check(rows + 1, rows)


def test_k3_dspark_mla_rope_kwargs_are_unchanged():
    """The MLA draft's `get_rope` arguments, key order included (it is part of
    aiter's rope cache key): DeepSeek YaRN, interleaved, the draft's own theta.
    """
    kwargs_fn = _draft_rope_kwargs_fn("kimi_k3_dspark.py")
    kw = kwargs_fn(_inferact_rope_cfg())
    assert kw == {
        "max_position": 1048576,
        "base": 50000.0,
        "rope_scaling": {
            "rope_type": "deepseek_yarn",
            "factor": 32.0,
            "original_max_position_embeddings": 32768,
            "rope_theta": 50000.0,
            "beta_fast": 32,
            "beta_slow": 1,
            "mscale": 1.0,
            "mscale_all_dim": 1.0,
        },
        "is_neox_style": False,
    }
    assert list(kw["rope_scaling"]) == [
        "rope_type",
        "factor",
        "original_max_position_embeddings",
        "rope_theta",
        "beta_fast",
        "beta_slow",
        "mscale",
        "mscale_all_dim",
    ]

    # `rope_interleave` is the config's neox switch; a factor alone is YaRN and
    # a missing original length falls back to the max position.
    cfg = _inferact_rope_cfg(
        rope_interleave=True,
        rope_parameters={"factor": 4.0, "rope_theta": 20000.0},
    )
    kw = kwargs_fn(cfg)
    assert kw["is_neox_style"] is True
    assert kw["base"] == 20000.0
    assert kw["rope_scaling"] == {
        "factor": 4.0,
        "rope_theta": 20000.0,
        "rope_type": "deepseek_yarn",
        "original_max_position_embeddings": 1048576,
    }

    # No rope_parameters: the top-level theta, plain rope.
    kw = kwargs_fn(_inferact_rope_cfg(rope_parameters=None))
    assert kw["base"] == 50000.0
    assert kw["rope_scaling"] is None


def test_draft_rope_scaling_leaves_the_type_to_the_caller():
    """The shared parser copies the config's scaling dict; a caller's override
    of the rope type never reaches the config."""
    from atom.models.dspark_draft import draft_rope_scaling

    cfg = _radixark_cfg()
    theta, scaling = draft_rope_scaling(cfg)
    assert theta == 10000.0
    assert scaling == cfg.rope_parameters
    scaling["rope_type"] = "deepseek_yarn"
    assert cfg.rope_parameters["rope_type"] == "yarn"

    assert draft_rope_scaling(_radixark_cfg(rope_parameters={}))[1] is None


def _stub_draft(vocab_size=None):
    """A bare `DSparkDraftModel`, to run its default methods on."""
    from atom.models.dspark_draft import DSparkDraftModel

    draft = type("StubDraft", (DSparkDraftModel,), {})()
    draft.vocab_size = vocab_size
    draft.embed_tokens = None
    draft.lm_head = None
    return draft


def _target_base(vocab):
    import torch

    model = types.SimpleNamespace(embed_tokens=torch.nn.Embedding(vocab, 4))
    return types.SimpleNamespace(model=model, lm_head=object())


def test_default_share_with_target_binds_the_targets_embedding_and_head():
    draft = _stub_draft(163840)
    target = _target_base(163840)
    draft.share_with_target(target, set())
    assert draft.embed_tokens is target.model.embed_tokens
    assert draft.lm_head is target.lm_head


def test_default_share_with_target_refuses_a_vocab_mismatch():
    draft = _stub_draft(163840)
    with pytest.raises(ValueError, match="StubDraft draft vocab 163840"):
        draft.share_with_target(_target_base(200064), set())
    assert draft.embed_tokens is None and draft.lm_head is None


def test_default_sample_block_chains_each_token_into_the_next():
    """Each position samples off the previous id, starting from the anchor."""
    import torch

    class _Head:
        def sample_next(self, prev, logits, out):
            out.copy_(prev + logits.argmax(-1))

    draft = _stub_draft()
    draft.markov_head = _Head()
    base_logits = torch.zeros(2, 3, 5)
    base_logits[:, :, 1] = 1.0
    ids = draft._sample_block(base_logits, torch.tensor([10, 20]))
    assert ids.tolist() == [[11, 12, 13], [21, 22, 23]]


def test_dflash_does_not_capture_the_k3_dspark_checkpoint():
    """The MLA K3-DSpark draft keeps its own route and gains no block size."""
    from atom.config import SpeculativeConfig

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
    from atom.config import SpeculativeConfig

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
    from atom.config import SpeculativeConfig

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

    Kept off `get_attn_backend`, which reads the GPU arch through rocminfo.
    """
    selector = _gpu_stack_import("atom.utils.selector")

    assert selector.attn_family(_radixark_cfg()) is selector.Family.MHA
    # The MLA draft it sits beside must still resolve the other way.
    assert (
        selector.attn_family(
            types.SimpleNamespace(model_type="k3_dspark", kv_lora_rank=512)
        )
        is not selector.Family.MHA
    )


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
        flash_layout=True,
    )


@pytest.mark.parametrize("tp,dcp", [(8, 1), (8, 2), (8, 8)])
def test_dflash_pool_footprint_is_dcp_invariant(tp, dcp):
    """A wider entry over fewer blocks is the same per-rank footprint.

    The draft is NOT in the dcp group: its KV heads are sharded by the full TP
    world size whatever `-dcp` says, and every rank holds the whole token axis.
    What moves with dcp is the ENTRY: one target block id addresses
    `block_size * dcp` global positions, so the pool is built with
    `blocks_per_entry = dcp` -- `entry_bytes` grows by dcp while the number of
    entries a request needs falls by dcp.

    This pins the arithmetic; `test_dflash_pool_entry_is_dcp_blocks` pins that
    `draft_kv_builder` builds this geometry.
    """
    _gpu_stack_import("atom.model_ops.attentions.mha_kv_pool")
    pool = _pool(block_size=128, world_size=tp, blocks_per_entry=dcp)

    # Full TP, every dcp: 16 KV heads over 8 ranks is 2 each.
    assert pool.num_kv_heads == 16 // tp
    assert pool.entry_tokens == 128 * dcp

    # A sequence of SEQ global tokens needs SEQ/(128*dcp) entries, each
    # `dcp` times as wide. 163840 B is the one-block cost at 2 KV heads
    # (5 layers x {k,v} x 2 heads x 64 head_dim x 128 tokens x 1 B, and not a
    # byte of field alignment); the product below is what must not move.
    # (163840 / 128 = 1280 B per token per rank.)
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
    from atom.config import SpeculativeConfig

    hf = _radixark_cfg()
    SpeculativeConfig.hf_config_override(hf, model_path=None)
    return hf


def _stub_runner(tp=8, dcp=1, kv_cache_dtype="fp8"):
    """A ModelRunner stand-in carrying only what `draft_kv_builder` reads."""
    return types.SimpleNamespace(
        world_size=tp,
        block_size=128,
        config=types.SimpleNamespace(
            kv_cache_dtype=kv_cache_dtype,
            tensor_parallel_size=tp,
            decode_context_parallel_size=dcp,
        ),
    )


def _make_kv_pool_kwargs(runner, model=None):
    """Build a draft's pool through a stub backend, the DFlash draft's unless
    `model` names another.

    Returns what `make_kv_pool` was called with, and the builder under
    ``"builder"``. The selector module is
    patched, not draft_kv: `draft_kv_builder` imports `attn_family` and
    `get_attn_backend` function-locally.
    """
    from unittest import mock

    dk = _gpu_stack_import("atom.spec_decode.draft_kv")
    _gpu_stack_import("atom.model_ops.attentions.aiter_attention")
    from atom.utils import selector

    seen = {}

    class _Backend:
        DRAFT_OWNS_KV_POOL = True

        @staticmethod
        def make_kv_pool(hf, *, world_size, kv_dtype, **kw):
            seen.update(kw, world_size=world_size, kv_dtype=kv_dtype)
            return object()

    with (
        mock.patch.object(selector, "get_attn_backend", lambda _f: _Backend),
        mock.patch.object(selector, "attn_family", lambda _c: None),
    ):
        builder = dk.draft_kv_builder(
            runner,
            _normalized_radixark(),
            model=model or _gpu_stack_import("atom.models.kimi_k3_dflash").KimiK3DFlash,
        )

    assert builder is not None
    seen["builder"] = builder
    builder._make_pool(layers=5, target_block_size=128)
    return seen


def test_dflash_pool_inherits_the_targets_fp8_and_declares_no_scale_planes():
    """The draft caches fp8 beside an fp8 target, at ONE per-tensor scale.

    Both ends of the pool take one scale: `aiter.reshape_and_cache_flash`
    dereferences `*k_scale` once per launch and `unified_attention` loads one
    `k_descale` per program. Neither can read a per-(kv head, token) plane, so
    the pool must declare none. Getting that wrong is not a crash: both read
    the plane's first element and apply it to every token.

    The platform's fp8 -- `d_dtypes["fp8"]`, as the target's pool is.
    """
    seen = _make_kv_pool_kwargs(_stub_runner())
    from aiter import dtypes

    assert seen["kv_dtype"] is dtypes.d_dtypes["fp8"], (
        f"draft pool must take the target's fp8 ({dtypes.d_dtypes['fp8']}), "
        f"got {seen['kv_dtype']}"
    )
    assert seen["flash_layout"] is True

    # ... and the pool it describes really has no scale region.
    from atom.model_ops.attentions.mha_kv_pool import MhaKvPool

    pool = MhaKvPool.from_hf_config(
        _normalized_radixark(),
        world_size=8,
        block_size=128,
        layers=5,
        kv_dtype=dtypes.d_dtypes["fp8"],
        flash_layout=True,
    )
    assert pool.scale_fields == []
    pool.allocate(4, "cpu")
    with pytest.raises(KeyError):
        pool.scale_views(0)


@pytest.mark.parametrize("dcp", [1, 2, 8])
def test_dflash_pool_entry_is_dcp_blocks(dcp):
    """One target block id spans `block_size * dcp` global positions, and the
    draft, outside the dcp group, stores all of them: `dcp` blocks an entry.
    """
    seen = _make_kv_pool_kwargs(_stub_runner(dcp=dcp))
    assert seen["blocks_per_entry"] == dcp


@pytest.mark.parametrize("model", ["eagle3", "minimax"])
def test_a_draft_without_a_pool_spec_gets_the_generic_pool(model):
    """Eagle3 declares no `kv_pool_spec` and MiniMax-M3.1 DSpark inherits
    None: neither adds a pool kwarg, and both keep the target builder's
    block."""
    if model == "eagle3":
        draft = types.SimpleNamespace()
    else:
        draft = _gpu_stack_import("atom.models.minimax_m3.dspark").MiniMaxM3DSpark
        assert draft.kv_pool_spec is None
    seen = _make_kv_pool_kwargs(_stub_runner(dcp=8), model=draft)
    builder = seen.pop("builder")
    assert set(seen) == {"layers", "target_block_size", "world_size", "kv_dtype"}
    assert builder._block_size is None


def test_a_draft_outside_the_dcp_group_needs_the_flash_pool():
    """Out of the group, an entry is `block_size * dcp` tokens, which only the
    FLASH order reads; the spec refuses the pair rather than let the
    proposer's addressing and the pool's geometry part."""
    dk = _gpu_stack_import("atom.spec_decode.draft_kv")
    with pytest.raises(ValueError, match="FLASH"):
        dk.DraftKvPoolSpec(in_dcp_group=False)
    spec = dk.DraftKvPoolSpec(flash_layout=True, in_dcp_group=False)
    assert spec.pool_kwargs(8) == {"blocks_per_entry": 8, "flash_layout": True}
    assert dk.DraftKvPoolSpec().pool_kwargs(8) == {
        "blocks_per_entry": 1,
        "flash_layout": False,
    }


@pytest.mark.parametrize(
    "module,cls_name,sharded",
    [
        ("atom.models.kimi_k3_dspark", "KimiK3DSpark", True),
        ("atom.models.kimi_k3_dflash", "KimiK3DFlash", False),
        ("atom.models.minimax_m3.dspark", "MiniMaxM3DSpark", True),
    ],
)
def test_dcp_sharding_is_read_from_the_pool_spec(module, cls_name, sharded):
    """Whether the draft addresses in per-rank terms comes from the same
    declaration that shapes its pool."""
    proposer = _gpu_stack_import("atom.spec_decode.dspark_proposer")
    draft_cls = getattr(_gpu_stack_import(module), cls_name)
    stub = types.SimpleNamespace(dcp_world_size=8, model=draft_cls)
    assert proposer.DSparkProposer._draft_is_dcp_sharded.fget(stub) is sharded
    stub.dcp_world_size = 1
    assert proposer.DSparkProposer._draft_is_dcp_sharded.fget(stub) is False


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


def _fake_attn_res_kernel(monkeypatch, calls):
    """Stand in for `apply_attn_res`, whose kernel needs a GPU.

    Patched on `kimi_k3_dflash`, which imports it at module level; the module
    itself needs aiter, so these tests skip on the CPU-only CI job and run
    where aiter is installed. The stand-in records the ARGUMENTS the draft
    marshals and returns a sentinel, so a test can check the extract hands back
    the kernel's result rather than the prefix.
    """
    import torch

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
        result = torch.full_like(prefix_sum, 7.0)
        calls.append(
            {
                "result": result,
                "prefix": prefix_sum,
                "bank": block_residual,
                "score_weight": score_weight,
                "eps": eps,
                "out_norm_weight": out_norm_weight,
                "quant_dtype": quant_dtype,
            }
        )
        return result, None

    dflash = _gpu_stack_import("atom.models.kimi_k3_dflash")
    monkeypatch.setattr(dflash, "apply_attn_res", apply_attn_res)
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
    dflash = _gpu_stack_import("atom.models.kimi_k3_dflash")
    return dflash.KimiK3DFlash.target_aux_capture_spec(None, layer_ids, hidden, target)


def _generic_extract():
    proposer = _gpu_stack_import("atom.spec_decode.dspark_proposer")
    return proposer.DSparkProposer._extract_layer_hidden


def test_dflash_owns_a_capture_spec_that_feeds_the_aggregate(monkeypatch):
    """(a) The draft resolves to its OWN extract, and that extract mixes.

    Both halves matter: resolving to its own spec is what makes the routing
    draft-owned, and the extract returning the mixture rather than the prefix is
    the feature itself.
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

    # The kernel's mixture, not the plain post-layer hidden state.
    assert out is call["result"]


def test_dflash_aggregate_leaves_untapped_layers_on_the_prefix(monkeypatch):
    """A layer the draft does not tap keeps the plain reconstruction.

    The hook is installed per tapped layer, but ONE extract serves all of them,
    so it dispatches on the block it is handed. An untapped layer keeps the
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

    # Fewer closed blocks than the tap implies is a target layout change, not
    # a reason to drop silently back to the prefix feature.
    with pytest.raises(RuntimeError, match="expected a bank"):
        spec.extract((prefix, None, bank[:, :1, :]), target.layers[6])
    assert calls == []


def test_dflash_aggregate_folds_the_score_weight_when_armed():
    """The fold runs once at arm time, never inside the traced forward."""
    pytest.importorskip("torch")
    target = _k3_target_stub(hidden=16)
    res = target.layers[7].self_attention_attn_res
    folded = res.score_weight
    res.score_weight = None
    res.process_weights_after_loading = lambda: setattr(res, "score_weight", folded)
    _dflash_spec(target, (6, 11), hidden=16)
    assert res.score_weight is folded


def test_dflash_refuses_a_tap_that_cannot_form_the_aggregate():
    """A tap without a mixture would feed the prefix, which this checkpoint was
    not trained on, so arming refuses it rather than fall back per tap."""
    pytest.importorskip("torch")
    with pytest.raises(ValueError, match="no enabled attn-residual"):
        _dflash_spec(_k3_target_stub(enabled=False), (7, 11))
    with pytest.raises(ValueError, match="successor layer"):
        _dflash_spec(_k3_target_stub(n_layers=16), (6, 15))
    target = _k3_target_stub()
    target.layers[6].layer_idx = 5
    with pytest.raises(ValueError, match="layer_idx 5"):
        _dflash_spec(target, (6,))


def test_dflash_embedding_tap_keeps_the_prefix():
    """`-1` taps the embedding, below every mixture: the prefix is the stream."""
    pytest.importorskip("torch")
    generic = _generic_extract()
    assert _dflash_spec(_k3_target_stub(), (-1,)).extract is generic
    spec = _dflash_spec(_k3_target_stub(), (-1, 6))
    assert spec.extract is not generic


def test_k3_dspark_mla_draft_resolves_to_the_generic_extractor():
    """(b) REGRESSION GUARD. The Inferact MLA draft must NOT get the aggregate.

    `Inferact/Kimi-K3-DSpark` was trained through vLLM/HF against the plain
    residual stream. Measured, the aggregate costs it ~2.6 pp acceptance -- and
    it is the draft `.github/benchmark/models_accuracy.json` tracks. It keeps
    the prefix by not implementing the hook at all.
    """
    dspark = _gpu_stack_import("atom.models.kimi_k3_dspark")
    proposer_mod = _gpu_stack_import("atom.spec_decode.dspark_proposer")
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
    v41 = _gpu_stack_import("atom.models.deepseek_v41.dspark")
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
    proposer_mod = _gpu_stack_import("atom.spec_decode.dspark_proposer")
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
# draft context slots (kimi_mla_gdn_attn builder -> dspark_proposer -> draft)
#
# The slot math is `token_layout/slots.py`'s, on the host, and runs on CPU.
# The builder's publication, the proposer's hand-off and the DFlash layer are
# the REAL methods on stubs; their modules import aiter/triton, so those skip
# on the CPU-only CI job. The base draft's fan-out runs on CPU.
# --------------------------------------------------------------------------- #


def _host_slots(tables, seqlens, positions, block_size):
    """The host builder (`token_layout/slots.py`) for the same tokens."""
    import numpy as np

    from atom.model_ops.attentions.token_layout.slots import slot_mapping

    return slot_mapping(
        np.asarray(positions, dtype=np.int64),
        np.asarray(seqlens),
        np.ascontiguousarray(tables, dtype=np.int32),
        block_size,
    ).tolist()


# A prefill chunk crossing an entry boundary, then two decode tokens.
_TABLES = [[7, 9] + [0] * 14, [4] + [0] * 15]
_SEQLENS = [3, 2]


@pytest.mark.parametrize("dcp", [1, 8])
def test_draft_slots_store_every_token_at_entry_granularity(dcp):
    """Every token gets a slot, addressed in `block_size * dcp`-token entries.

    At DCP>1 they differ from the target's `block_size` addressing -- which
    would alias `dcp` requests' entries onto one range.
    """
    vblock = 128 * dcp
    positions = [vblock - 2, vblock - 1, vblock, 5, 6]
    got = _host_slots(_TABLES, _SEQLENS, positions, vblock)
    assert got == [
        7 * vblock + vblock - 2,
        7 * vblock + vblock - 1,
        9 * vblock,
        4 * vblock + 5,
        4 * vblock + 6,
    ]
    if dcp > 1:
        assert got != _host_slots(_TABLES, _SEQLENS, positions, 128)


def _builder_stub(entry_tokens, tables, positions, rows=64):
    """What `_publish_draft_ctx_slots` reads off the builder and its runner."""
    import numpy as np
    import torch

    from atom.utils import CpuGpuBuffer

    def buf(*size, dtype):
        return CpuGpuBuffer(*size, dtype=dtype, device="cpu", pin_memory=False)

    bt = buf(len(tables), len(tables[0]), dtype=torch.int32)
    bt.np[:] = tables
    pos = buf(rows, dtype=torch.int64)
    pos.np[: len(positions)] = positions
    out = buf(rows, dtype=torch.int64)
    out.np[:] = 7
    return types.SimpleNamespace(
        model_runner=types.SimpleNamespace(
            forward_vars={
                "draft_ctx_slots": out,
                "positions": pos,
                "block_tables": bt,
            },
            draft_kv_builder=types.SimpleNamespace(
                kv_pool=types.SimpleNamespace(entry_tokens=entry_tokens)
            ),
        ),
        token_axis_scratch=np.empty(rows, dtype=np.int64),
    )


def _publish(stub, seqlens, rows, tables, dummy=False):
    """Drive the real `_publish_draft_ctx_slots`; return the device buffer."""
    import numpy as np

    kimi = _gpu_stack_import("atom.model_ops.attentions.kimi_mla_gdn_attn")
    buf = stub.model_runner.forward_vars.get("draft_ctx_slots")
    batch = types.SimpleNamespace(is_dummy_run=dummy, block_tables=tables)
    kimi._KimiMLAGDNCommon._publish_draft_ctx_slots(
        stub, batch, np.asarray(seqlens, dtype=np.int32), rows
    )
    return None if buf is None else buf.gpu


@pytest.mark.parametrize("dcp", [1, 8])
def test_the_builder_uploads_the_draft_slots(dcp):
    """The target's builder uploads, at entry granularity and with no DCP
    filter, the host slot of every scheduled token."""
    vblock = 128 * dcp
    positions = [vblock - 2, vblock - 1, vblock, 5, 6]
    stub = _builder_stub(vblock, _TABLES, positions)
    got = _publish(stub, _SEQLENS, len(positions), _TABLES)
    assert got[:5].tolist() == _host_slots(_TABLES, _SEQLENS, positions, vblock)


def test_the_builder_drops_the_padded_decode_rows():
    """Graph-padded rows past the scheduled tokens get `-1`, written nowhere."""
    stub = _builder_stub(1024, _TABLES, [3, 4, 7, 8])
    got = _publish(stub, [2, 2], 6, _TABLES)
    assert got[:6].tolist() == [
        7 * 1024 + 3,
        7 * 1024 + 4,
        4 * 1024 + 7,
        4 * 1024 + 8,
        -1,
        -1,
    ]


def test_the_builder_skips_dummy_runs_and_drafts_without_a_pool():
    """A dummy run uploads nothing; a target with no pooled draft has no
    buffer to fill and is a no-op."""
    stub = _builder_stub(1024, _TABLES, [0, 1])
    assert _publish(stub, [2], 2, _TABLES, dummy=True)[:2].tolist() == [0, 0]
    del stub.model_runner.forward_vars["draft_ctx_slots"]
    assert _publish(stub, [2], 2, _TABLES) is None


def _compute_draft_kv(kv_pool_spec, n_tok):
    """Drive the real `compute_draft_kv`; return what reached the model."""
    import torch

    proposer = _gpu_stack_import("atom.spec_decode.dspark_proposer")
    from atom.utils import CpuGpuBuffer

    slots = CpuGpuBuffer(16, dtype=torch.int64, device="cpu", pin_memory=False)
    slots.gpu.copy_(torch.arange(16) + 100)
    seen = {}
    stub = types.SimpleNamespace(
        aux_for=lambda h: [h],
        runner=types.SimpleNamespace(forward_vars={"draft_ctx_slots": slots}),
        model=types.SimpleNamespace(
            kv_pool_spec=kv_pool_spec,
            write_context_kv=lambda aux, pos, slot_mapping: seen.update(
                rows=aux.shape[0], slots=slot_mapping
            ),
        ),
    )
    fwd = types.SimpleNamespace(context=types.SimpleNamespace(scheduled_bs=1))
    orig = proposer.get_forward_context
    proposer.get_forward_context = lambda: fwd
    try:
        proposer.DSparkProposer.compute_draft_kv(
            stub, torch.arange(n_tok), torch.zeros(n_tok, 4), None
        )
    finally:
        proposer.get_forward_context = orig
    return seen


def test_the_proposer_hands_a_pooled_draft_its_own_slots():
    """A draft with a declared pool gets the builder's slots, one per row."""
    from atom.spec_decode.draft_kv import DraftKvPoolSpec

    seen = _compute_draft_kv(DraftKvPoolSpec(flash_layout=True, in_dcp_group=False), 5)
    assert seen["rows"] == 5
    assert seen["slots"].tolist() == [100, 101, 102, 103, 104]


def test_the_proposer_hands_other_drafts_no_slots():
    """Every other draft writes at the target's slots, so gets None."""
    assert _compute_draft_kv(None, 5)["slots"] is None


def test_the_draft_fans_its_slots_out_to_every_context_layer(monkeypatch):
    """`DSparkDraftModel.write_context_kv` passes the slots to each layer."""
    import torch

    import atom.utils.forward_context as fc
    from atom.models.dspark_draft import DSparkDraftModel

    calls = []

    class _Layer:
        def write_context_kv(self, ctx_hidden, positions, slot_mapping):
            calls.append(slot_mapping)

    layers = [_Layer(), _Layer()]

    class _Draft(DSparkDraftModel):
        def project_context(self, aux_concat):
            return aux_concat

        @property
        def context_layers(self):
            return layers

    monkeypatch.setattr(
        fc,
        "get_forward_context",
        lambda: types.SimpleNamespace(
            context=types.SimpleNamespace(is_dummy_run=False)
        ),
    )
    slots = torch.arange(4)
    DSparkDraftModel.write_context_kv(_Draft(), torch.zeros(4, 2), None, slots)
    assert len(calls) == 2 and all(c is slots for c in calls)


def test_the_dflash_layer_writes_at_the_slots_it_is_handed():
    """No read of the target's DCP-filtered `slot_mapping` on the way down."""
    import torch

    m = _gpu_stack_import("atom.models.kimi_k3_dflash")
    seen = {}
    layer = types.SimpleNamespace(
        self_attn=types.SimpleNamespace(
            write_context_kv=lambda h, p, s: seen.update(slots=s)
        )
    )
    slots = torch.arange(4)
    m.KimiK3DFlashDecoderLayer.write_context_kv(
        layer, torch.zeros(4, 8), torch.arange(4), slots
    )
    assert seen["slots"] is slots


# --------------------------------------------------------------------------- #
# the block read (model_ops/attentions/dflash_draft_attn.py)
#
# `unified_attention` asserts causal; the read gets a non-causal block out of it
# by folding each request's T rows into its head axis and decoding ONE row per
# request. The fold is pure tensor reshaping, but its module imports aiter's
# kernel, so these skip where aiter is absent; the kernel check needs a GPU.
# --------------------------------------------------------------------------- #


def _fold_mod():
    pytest.importorskip("torch")
    return _gpu_stack_import("atom.model_ops.attentions.dflash_draft_attn")


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


def _reference_block_attn(
    q, k_pool, v_pool, block_tables, kv_lens, scale, hkv, causal=False
):
    """GQA attention of each request's T rows over its pool rows, in fp32,
    gathered slot by slot from the block table. Bidirectional by default;
    `causal` masks block row t to keys `< kv_len - T + t + 1`, the mask a
    non-folded read of the same block would apply."""
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
        if causal:
            last = n - T + torch.arange(T, device=q.device)  # [T]
            hide = pos.view(1, n) > last.view(T, 1)  # [T, n]
            s = s.masked_fill(hide.view(T, 1, 1, n), float("-inf"))
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
    ending mid-entry.

    Over ~1000 random keys the block's own T keys would carry under 1% of the
    softmax mass, and a causal mask -- it hides at most T - 1 of them from a row
    -- would move the output by less than fp8 noise. So the data gives them
    weight: q and every key share one direction, pushed apart for the prefix
    and together for the block, which makes a block key worth ~50 prefix keys.
    The read must then sit far closer to the bidirectional reference than to
    the causal one (measured: 38x and up at fp8, 74x at bf16, over 8 seeds).

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
    q = torch.randn(bs * T, hq, d, device=dev, dtype=torch.bfloat16)
    # See the docstring: one shared lane, prefix keys pushed off it, block keys
    # onto it.
    q[..., 0] += 4
    k_pool[..., 0] -= 4
    for b in range(bs):
        n = int(kv_lens[b])
        pos = torch.arange(n - T, n, device=dev)
        k_pool[block_tables[b, pos // E].long(), pos % E, :, 0] += 8
    if kv_dtype == "fp8":
        from aiter import dtypes

        k_pool, v_pool = k_pool.to(dtypes.fp8), v_pool.to(dtypes.fp8)
    scale = d**-0.5

    read = dda.DFlashDraftUnifiedAttention(dev, max_bs=8, max_model_len=W * E)
    one = torch.ones(1, dtype=torch.float32, device=dev)
    # Where the read casts q to an fp8 pool's dtype (gfx950), the reference
    # takes the same q. Taken BEFORE the call, which clamps q in place.
    cast = kv_dtype == "fp8" and read.q_to_pool_dtype
    q_ref = q.to(k_pool.dtype).to(q.dtype) if cast else q.clone()
    o = read(q, k_pool, v_pool, block_tables, kv_lens, scale, one, one)

    args = (q_ref, k_pool, v_pool, block_tables, kv_lens, scale, hkv)
    ref = _reference_block_attn(*args)
    # fp8 runs both dots at fp8 on gfx950 (Gluon); measured max 3.7e-2 there,
    # 1.3e-2 at bf16, on O(1) outputs.
    tol = 6e-2 if kv_dtype == "fp8" else 2.5e-2
    torch.testing.assert_close(o.float(), ref, atol=tol, rtol=5e-2)

    causal = _reference_block_attn(*args, causal=True)
    err_bidir = (o.float() - ref).abs().max().item()
    err_causal = (o.float() - causal).abs().max().item()
    assert err_causal > 10 * err_bidir, (err_bidir, err_causal)


# The two non-MLA block drafts, by their real classes: DFlash addresses a
# `block_size * dcp` entry (dcp 8 here), MiniMax-M3.1 the scheduler block.
_NON_MLA_DRAFTS = [
    pytest.param("atom.models.kimi_k3_dflash", "KimiK3DFlash", 128 * 8, id="dflash"),
    pytest.param(
        "atom.models.minimax_m3.dspark", "MiniMaxM3DSpark", 128, id="minimax_m31"
    ),
]


@pytest.mark.parametrize("module,cls_name,vblock", _NON_MLA_DRAFTS)
def test_non_mla_block_drafts_skip_the_csr_and_the_mla_plan(
    monkeypatch, module, cls_name, vblock
):
    """The real `_build_paged_block_metadata` for a non-MLA block draft.

    It hands `prepare_paged_block` every token of the block at entry
    granularity, drops the pad row and the position past the table, and
    returns before the CSR generator, which a non-MLA read never consumes.
    The CSR fields still point at the draft's own buffers.
    """
    import torch

    proposer = _gpu_stack_import("atom.spec_decode.dspark_proposer")
    draft_cls = getattr(_gpu_stack_import(module), cls_name)
    from atom.utils import CpuGpuBuffer
    from atom.utils.forward_context import AttentionMetaData

    def _no_csr(*a, **k):
        raise AssertionError("a non-MLA block draft built the CSR")

    monkeypatch.setattr(proposer, "kv_indices_generate_triton", _no_csr)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    T, running_bs = 2, 3
    bt = CpuGpuBuffer(running_bs, 2, dtype=torch.int32, device=dev)
    bt.np[:] = [[5, 6], [8, 9], [0, 0]]
    bt.copy_to_gpu()
    seen = {}
    i64 = {"dtype": torch.int64, "device": dev}
    i32 = {"dtype": torch.int32, "device": dev}
    stub = types.SimpleNamespace(
        draft_tokens_per_seq=T,
        runner=types.SimpleNamespace(block_size=128, forward_vars={"block_tables": bt}),
        _blk_positions=torch.zeros(running_bs, T, **i64),
        _blk_offsets=torch.arange(1, T + 1, **i64),
        _blk_dtype_q=torch.bfloat16,
        _bound_draft_k_cache=lambda fc: object(),
        _blk_slots=torch.zeros(running_bs * T, **i64),
        _blk_ctx_lens=torch.zeros(running_bs, **i32),
        _draft_entry_tokens=vblock,
        _draft_is_dcp_sharded=False,
        _paged_pad_ctx_len=1 + T,
        _blk_cu_seqlens_q=torch.zeros(running_bs + 1, **i32),
        _blk_kv_indptr=torch.zeros(running_bs + 1, **i32),
        _blk_kv_indices=torch.zeros(0, **i32),
        _blk_last_page_lens=torch.ones(running_bs, **i32),
        model=types.SimpleNamespace(
            block_attention_is_mla=draft_cls.block_attention_is_mla,
            prepare_paged_block=lambda md, bt_, pos, slots, ctx, *a: seen.update(
                slots=slots.cpu().tolist(), ctx=ctx.cpu().tolist()
            ),
        ),
    )
    md = AttentionMetaData()
    # Row 0 crosses into its second entry; row 1's block runs past its table.
    anchors = torch.tensor([vblock - 2, 2 * vblock - 2, 0], **i64)
    proposer.DSparkProposer._build_paged_block_metadata(
        stub, None, md, anchors, scheduled_bs=2, max_seqlen_k=4 * vblock
    )
    assert seen["slots"] == [
        [5 * vblock + vblock - 1, 6 * vblock],
        [9 * vblock + vblock - 1, -1],
        [-1, -1],
    ]
    assert seen["ctx"] == [vblock + 1, 2 * vblock, 1 + T]
    assert md.causal is False
    assert md.kv_indices is stub._blk_kv_indices
    assert md.kv_indptr.shape[0] == running_bs + 1


@pytest.mark.parametrize(
    "module,cls_name,is_mla",
    [
        ("atom.models.kimi_k3_dspark", "KimiK3DSpark", True),
        ("atom.models.kimi_k3_dflash", "KimiK3DFlash", False),
        ("atom.models.minimax_m3.dspark", "MiniMaxM3DSpark", False),
    ],
)
def test_kv_indices_buffer_is_allocated_only_for_mla_block_plans(
    module, cls_name, is_mla
):
    """Only the MLA block plan fills `_blk_kv_indices` (max_bs x max_model_len).

    A non-MLA draft returns from `_build_paged_block_metadata` before the CSR
    build, so it gets an empty placeholder instead of the full buffer.
    """
    torch = pytest.importorskip("torch")
    proposer = _gpu_stack_import("atom.spec_decode.dspark_proposer")
    draft_cls = getattr(_gpu_stack_import(module), cls_name)
    assert draft_cls.block_attention_is_mla is is_mla

    p = proposer.DSparkProposer.__new__(proposer.DSparkProposer)
    p.config = types.SimpleNamespace(
        max_num_seqs=8, max_model_len=4096, max_num_batched_tokens=1024
    )
    p.device = torch.device("cpu")
    p.mtp_k = 7
    p.model = types.SimpleNamespace(
        block_attention_is_mla=draft_cls.block_attention_is_mla,
        kv_pool_spec=draft_cls.kv_pool_spec,
    )
    p._init_draft_block_buffers()

    if is_mla:
        assert p._blk_kv_indices.numel() == 8 * (4096 + 7)
    else:
        assert p._blk_kv_indices.numel() == 0


def test_context_write_drops_padded_target_rows(monkeypatch):
    """Under TR_HYBRID the runner passes `running_tokens` hidden rows but only
    the scheduled rows' positions; only those rows may reach the projection."""
    torch = pytest.importorskip("torch")
    dflash = _gpu_stack_import("atom.models.kimi_k3_dflash")
    from atom.models import dspark_draft

    seen = {}
    monkeypatch.setattr(
        dspark_draft.DSparkDraftModel,
        "write_context_kv",
        lambda self, aux, pos, slots: seen.update(
            rows=aux.shape[0], pos=pos.shape[0], slots=slots
        ),
    )
    draft = dflash.KimiK3DFlash.__new__(dflash.KimiK3DFlash)
    slots = torch.arange(9)
    draft.write_context_kv(torch.zeros(12, 4), torch.arange(9), slots)
    assert seen == {"rows": 9, "pos": 9, "slots": slots}


def test_dflash_draft_is_refused_on_a_non_k3_target():
    """SpecForge stamps one arch on every DFlash draft, so the rename to
    `K3DFlashModel` cannot tell a Kimi-K3 draft from a Qwen3/Llama one; the
    target is checked where both configs are known."""
    import atom.config as cfg

    draft = types.SimpleNamespace(architectures=["K3DFlashModel"])
    cfg._check_dflash_draft_target(
        types.SimpleNamespace(architectures=["KimiK3ForConditionalGeneration"]), draft
    )
    with pytest.raises(ValueError, match="Kimi-K3 target"):
        cfg._check_dflash_draft_target(
            types.SimpleNamespace(architectures=["Qwen3ForCausalLM"]), draft
        )
    # Any other draft is not this check's business.
    cfg._check_dflash_draft_target(
        types.SimpleNamespace(architectures=["Qwen3ForCausalLM"]),
        types.SimpleNamespace(architectures=["Eagle3LlamaModel"]),
    )
    cfg._check_dflash_draft_target(types.SimpleNamespace(architectures=[]), None)


def test_fp8_q_cast_saturates_instead_of_producing_nan():
    """A plain cast to fp8 turns |x| > fp8 max into NaN, and one NaN q row
    poisons that row's softmax against every key; the read clamps first."""
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("needs a GPU")
    pytest.importorskip("aiter")
    from aiter import dtypes

    dda = _fold_mod()
    dev = torch.device("cuda")
    bs, T, hkv, g, d, E = 2, 7, 2, 4, 64, 1024
    read = dda.DFlashDraftUnifiedAttention(dev, max_bs=4, max_model_len=E)
    if not read.q_to_pool_dtype:
        pytest.skip("q is only cast to fp8 where aiter's Gluon path serves the read")
    k_pool = torch.randn(3, E, hkv, d, device=dev).to(dtypes.fp8)
    v_pool = torch.randn_like(k_pool, dtype=torch.float32).to(dtypes.fp8)
    q = torch.randn(bs * T, hkv * g, d, device=dev, dtype=torch.bfloat16)
    q[0, 0, 0] = 1e4  # far past fp8_e4m3's 448
    block_tables = torch.tensor([[1], [2]], dtype=torch.int32, device=dev)
    kv_lens = torch.tensor([100, 200], dtype=torch.int32, device=dev)
    one = torch.ones(1, dtype=torch.float32, device=dev)
    o = read(q, k_pool, v_pool, block_tables, kv_lens, d**-0.5, one, one)
    assert torch.isfinite(o).all()


def test_block_writes_and_reads_stay_inside_the_allocated_blocks():
    """The block runs up to T past the anchor, which can pass the request's
    allocation; that column reads as block 0, another request's entry. The real
    `KimiK3DFlash.prepare_paged_block` drops those writes and caps the read."""
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("needs a GPU: pinned staging and a CUDA event")
    m = _gpu_stack_import("atom.models.kimi_k3_dflash")
    draft = m.KimiK3DFlash.__new__(m.KimiK3DFlash)
    vblock, T = 1024, 3
    # The bound FLASH page is one entry wide.
    k_cache = torch.empty(1, vblock, 1, 1, device="cuda")
    kv = types.SimpleNamespace(k_cache=k_cache)
    draft.layers = [types.SimpleNamespace(self_attn=types.SimpleNamespace(kv=kv))]
    draft._alloc_cap = _gpu_stack_import("atom.models.dspark_draft").AllocationCap(
        4, "cuda"
    )
    # Row 0 holds one block and its block crosses into a second; row 1 holds
    # two and stays inside; row 2 is a pad row.
    pos = torch.tensor(
        [[vblock - 2, vblock - 1, vblock], [5, 6, 7], [1, 2, 3]], device="cuda"
    )
    slots = torch.tensor([[10, 11, 12], [20, 21, 22], [-1, -1, -1]], device="cuda")
    ctx = torch.tensor([vblock + 1, 8, 1 + T], dtype=torch.int32, device="cuda")
    draft.prepare_paged_block(None, None, pos, slots, ctx, [1, 2], 2)
    assert slots.tolist() == [[10, 11, -1], [20, 21, 22], [-1, -1, -1]]
    assert ctx.tolist() == [vblock, 8, 1 + T]


def test_the_draft_ignores_the_targets_quantization():
    """The draft atom config is a copy of the target's; its quant config, and
    the online re-quantization it carries, must not reach this BF16 draft."""
    m = _gpu_stack_import("atom.models.kimi_k3_dflash")
    qc = m._draft_quant_config(_radixark_cfg())
    assert qc.quant_method == "" and qc.online_quant is False
    with pytest.raises(ValueError, match="BF16 weights only"):
        m._draft_quant_config(
            _radixark_cfg(quantization_config={"quant_method": "fp8"})
        )
