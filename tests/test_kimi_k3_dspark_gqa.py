# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Unit tests for the Kimi-K3 DSpark GQA draft (RadixArk/Kimi-K3-DSpark, a
SpecForge DFlash checkpoint).

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


def test_gqa_draft_config_normalize():
    """DSparkDraftModel -> K3DSparkGQAModel, normalized like the other standalone
    DSpark drafts."""
    from atom.config import SpeculativeConfig

    hf = _radixark_cfg()
    SpeculativeConfig.hf_config_override(hf, model_path=None)

    # Renamed away from the generic SpecForge string, which every SpecForge
    # DFlash checkpoint carries.
    assert hf.architectures == ["K3DSparkGQAModel"]

    # Lifted out of the nested dflash_config onto the canonical names.
    assert hf.dspark_target_layer_ids == [7, 23, 51, 67, 83]
    assert hf.dspark_noise_token_id == 163824
    assert hf.dspark_markov_rank == 256
    assert hf.dspark_with_draft is True

    # No block width: as for the K3 MLA draft, --num-speculative-tokens sets it.
    assert getattr(hf, "dspark_block_size", None) is None

    # num_target_layers is the TARGET's depth, not the aux-tap count. The fusion
    # width is len(target_layer_ids); sizing off 93 would be a 93/5 overshoot.
    assert hf.num_target_layers == 93
    assert len(hf.dspark_target_layer_ids) == 5


def test_gqa_draft_config_normalize_is_idempotent():
    """A second override keeps the DSpark GQA draft and its normalized fields."""
    from atom.config import SpeculativeConfig

    hf = _radixark_cfg()
    SpeculativeConfig.hf_config_override(hf, model_path=None)
    model_type = hf.model_type
    SpeculativeConfig.hf_config_override(hf, model_path=None)

    assert hf.architectures == ["K3DSparkGQAModel"]
    assert hf.dspark_target_layer_ids == [7, 23, 51, 67, 83]
    assert hf.model_type == model_type


def test_gqa_draft_rope_is_the_drafts_own():
    """The rope the draft attention builds comes from the draft's config:
    theta 10000 / YaRN x16 from 65536, neox-style -- not the target's theta
    50000 / x32. A wrong rope loads clean and only moves acceptance."""
    from atom.config import SpeculativeConfig

    kwargs_fn = _draft_rope_kwargs_fn("kimi_k3_dspark_gqa.py")
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
    from atom.models import dspark_draft, utils

    ns = {
        "draft_rope_scaling": dspark_draft.draft_rope_scaling,
        "is_neox_rope_style": utils.is_neox_rope_style,
    }
    return _lift_fn(module_file, "_draft_rope_kwargs", ns)


def test_gqa_draft_refuses_a_max_model_len_past_the_rope_cache():
    """A length equal to the cache rows -- the shipped pair's default -- fits;
    one more row is refused at startup."""
    check = _lift_fn("kimi_k3_dspark_gqa.py", "_check_rope_covers", {})
    rows = 65536 * 16
    check(rows, rows)
    check(4096, rows)
    with pytest.raises(ValueError, match="lower --max-model-len"):
        check(rows + 1, rows)


def test_gqa_context_write_clamps_verify_rows_to_the_rope_cache(monkeypatch):
    """Verify rows reach `mtp_k` past the last token, past the rope cache at
    the default `max_model_len`; they are roped at its last row, and the
    target's positions are left as they were."""
    import torch

    mod = _gpu_stack_import("atom.models.kimi_k3_dspark_gqa")
    seen = {}
    monkeypatch.setattr(
        mod.DSparkDraftModel,
        "write_context_kv",
        lambda self, aux, positions, slots=None: seen.update(positions=positions),
    )
    draft = object.__new__(mod.KimiK3DSparkGQA)
    draft._rope_max_row = 99
    positions = torch.tensor([97, 98, 99, 100, 101])
    draft.write_context_kv(torch.zeros(5, 4), positions, None)
    assert seen["positions"].tolist() == [97, 98, 99, 99, 99]
    assert positions.tolist() == [97, 98, 99, 100, 101]


def test_gqa_draft_does_not_capture_the_k3_dspark_checkpoint():
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


def test_gqa_draft_without_dflash_config_is_refused():
    """A `DSparkDraftModel` without the `dflash_config` fields fails
    normalization, as any standalone DSpark draft missing `target_layer_ids`
    does."""
    from atom.config import SpeculativeConfig

    hf = _radixark_cfg(dflash_config=None)
    with pytest.raises(ValueError, match="target_layer_ids"):
        SpeculativeConfig.hf_config_override(hf, model_path=None)


def test_gqa_draft_plain_qwen3_target_is_untouched():
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


def test_gqa_draft_resolves_to_mha_not_mla():
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
def test_gqa_draft_pool_footprint_is_dcp_invariant(tp, dcp):
    """A wider entry over fewer blocks is the same per-rank footprint.

    The draft is NOT in the dcp group: its KV heads are sharded by the full TP
    world size whatever `-dcp` says, and every rank holds the whole token axis.
    What moves with dcp is the ENTRY: one target block id addresses
    `block_size * dcp` global positions, so the pool is built with
    `blocks_per_entry = dcp` -- `entry_bytes` grows by dcp while the number of
    entries a request needs falls by dcp.

    This pins the arithmetic; `test_gqa_draft_pool_entry_is_dcp_blocks` pins that
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
    """Build a draft's pool through a stub backend, the DSpark GQA draft's unless
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
            model=model
            or _gpu_stack_import("atom.models.kimi_k3_dspark_gqa").KimiK3DSparkGQA,
        )

    assert builder is not None
    seen["builder"] = builder
    builder._make_pool(layers=5, target_block_size=128)
    return seen


def test_gqa_draft_pool_inherits_the_targets_fp8_and_declares_no_scale_planes():
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
def test_gqa_draft_pool_entry_is_dcp_blocks(dcp):
    """One target block id spans `block_size * dcp` global positions, and the
    draft, outside the dcp group, stores all of them: `dcp` blocks an entry,
    at the scheduler's block, the unit those ids count in.
    """
    runner = _stub_runner(dcp=dcp)
    seen = _make_kv_pool_kwargs(runner)
    assert seen["blocks_per_entry"] == dcp
    assert seen["builder"]._block_size == runner.block_size


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
# The routing is by draft CLASS, not by a flag: `KimiK3DSparkGQA` implements
# `target_aux_capture_spec`, the MLA draft beside it deliberately does not. The
# tests below pin both halves of that, because the failure mode in either
# direction is silent -- a draft fed the wrong feature still proposes tokens,
# they are just accepted less often.


def _fake_attn_res_kernel(monkeypatch, calls):
    """Stand in for `apply_attn_res`, whose kernel needs a GPU.

    Patched on `atom.model_ops.kimi_k3`, which `KimiDecoderLayer` imports it
    from at call time; that package needs aiter, so these tests skip on the
    CPU-only CI job and run where aiter is installed. The stand-in records the
    ARGUMENTS the layer marshals and returns a sentinel, so a test can check the
    extract hands back the kernel's result rather than the prefix.
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
                "add_hidden": add_hidden,
                "out_norm_weight": out_norm_weight,
                "out_eps": out_eps,
                "add_hidden2": add_hidden2,
                "quant_dtype": quant_dtype,
            }
        )
        return result, None

    ops = _gpu_stack_import("atom.model_ops.kimi_k3")
    monkeypatch.setattr(ops, "apply_attn_res", apply_attn_res)
    return calls


def _k3_target_stub(hidden=16, n_layers=16, block_size=4, enabled=True):
    """A Kimi-K3 target's layer stack, down to what aux capture reads off it.

    Each layer carries `KimiDecoderLayer`'s own `check_aggregate_source` and
    `aggregate_hidden_state`, bound to a namespace
    holding the attributes they read: building the real layer would want a
    checkpoint, a TP group and a GPU.
    """
    import torch

    layer_cls = _gpu_stack_import("atom.models.kimi_k3.model").KimiDecoderLayer
    torch.manual_seed(0)
    layers = []
    for i in range(n_layers):
        layer = types.SimpleNamespace(
            layer_idx=i,
            self_attention_attn_res=types.SimpleNamespace(
                enabled=enabled,
                block_size=block_size,
                layer_idx=i,
                eps=1e-6,
                out_eps=1e-5,
                score_weight=torch.randn(hidden) * 0.1,
            ),
        )
        for name in ("check_aggregate_source", "aggregate_hidden_state"):
            setattr(layer, name, types.MethodType(getattr(layer_cls, name), layer))
        layers.append(layer)
    return types.SimpleNamespace(layers=layers)


def _gqa_spec(target, layer_ids, hidden=16):
    """Resolve `KimiK3DSparkGQA`'s capture spec without building the draft.

    Called unbound on purpose: the method reads nothing off `self` (its inputs
    are the tap ids, the target's hidden size, and the target's layers), and
    constructing the real draft would want a loaded checkpoint, a TP group and
    a GPU.
    """
    mod = _gpu_stack_import("atom.models.kimi_k3_dspark_gqa")
    return mod.KimiK3DSparkGQA.target_aux_capture_spec(
        None, layer_ids, hidden, target.layers
    )


def test_gqa_draft_owns_a_capture_spec_that_feeds_the_aggregate(monkeypatch):
    """(a) The draft resolves to its OWN extract, and that extract mixes.

    Both halves matter: resolving to its own spec is what makes the routing
    draft-owned, and the extract returning the mixture rather than the prefix is
    the feature itself.
    """
    torch = pytest.importorskip("torch")
    proposer = _gpu_stack_import("atom.spec_decode.dspark_proposer")

    hidden, block_size = 16, 4
    target = _k3_target_stub(hidden=hidden, n_layers=16, block_size=block_size)
    # Tap 6, not 7: with block_size 4, layer 6 is preceded by closes at layers
    # 0 and 4 only, so the bank slice below pins the block count.
    spec = _gqa_spec(target, (6, 11), hidden=hidden)

    assert spec.layer_ids == (6, 11)
    assert spec.hidden_size == hidden
    assert spec.capture == "output"  # an OUTPUT tap; V4.1's is an input tap
    assert spec.extract is not proposer.DSparkProposer._extract_layer_hidden

    calls = _fake_attn_res_kernel(monkeypatch, [])
    rows = 5
    prefix = torch.randn(rows, hidden)
    pending = torch.randn(rows, hidden)
    pending2 = torch.randn(rows, hidden)
    bank = torch.randn(rows, 6, hidden)
    block = target.layers[6]

    out = spec.extract((prefix, pending, pending2, bank), block)

    # Layer 6 is preceded by closes at layers 0 and 4 (6 // 4 + 1 = 2, as
    # SGLang's `prev_valid_blocks`), so only 2 of the bank's 6 rows are mixed.
    assert len(calls) == 1
    call = calls[0]
    assert call["bank"].shape == (rows, 2, hidden)
    assert torch.equal(call["bank"], bank[:, :2, :])
    # The layer's deferred addends go to the kernel as the target's next layer
    # passes them, folded on-load rather than summed into the prefix first.
    assert call["prefix"] is prefix
    assert call["add_hidden"] is pending
    assert call["add_hidden2"] is pending2
    # Scored by the NEXT layer's attn-residual weights -- the one thing only the
    # target model can supply.
    res = target.layers[7].self_attention_attn_res
    assert call["score_weight"] is res.score_weight
    assert call["eps"] == res.eps
    assert call["out_eps"] == res.out_eps
    # Un-normed and unquantized: SGLang's aggregate is the bare mixture.
    assert call["out_norm_weight"] is None
    assert call["quant_dtype"] is None

    # The kernel's mixture, not the plain post-layer hidden state.
    assert out is call["result"]


def test_gqa_draft_aggregate_refuses_a_short_bank(monkeypatch):
    """Fewer closed blocks than the tap implies is a target layout change, not
    a reason to drop silently back to the prefix feature."""
    torch = pytest.importorskip("torch")
    hidden = 16
    target = _k3_target_stub(hidden=hidden)
    spec = _gqa_spec(target, (6, 11), hidden=hidden)

    calls = _fake_attn_res_kernel(monkeypatch, [])
    prefix = torch.randn(3, hidden)
    pending = torch.randn(3, hidden)
    bank = torch.randn(3, 6, hidden)
    with pytest.raises(RuntimeError, match="expected a bank"):
        spec.extract((prefix, pending, None, bank[:, :1, :]), target.layers[6])
    assert calls == []


def test_gqa_draft_aggregate_folds_the_score_weight_when_armed():
    """The fold runs once at arm time, never inside the traced forward."""
    pytest.importorskip("torch")
    target = _k3_target_stub(hidden=16)
    res = target.layers[7].self_attention_attn_res
    folded = res.score_weight
    res.score_weight = None
    res.process_weights_after_loading = lambda: setattr(res, "score_weight", folded)
    _gqa_spec(target, (6, 11), hidden=16)
    assert res.score_weight is folded


def test_gqa_draft_refuses_a_tap_that_cannot_form_the_aggregate():
    """A tap without a mixture would feed the prefix, which this checkpoint was
    not trained on, so arming refuses it rather than fall back per tap."""
    pytest.importorskip("torch")
    with pytest.raises(ValueError, match="no enabled attn-residual"):
        _gqa_spec(_k3_target_stub(enabled=False), (7, 11))
    with pytest.raises(ValueError, match="successor layer"):
        _gqa_spec(_k3_target_stub(n_layers=16), (6, 15))
    target = _k3_target_stub()
    target.layers[6].layer_idx = 5
    with pytest.raises(ValueError, match="layer_idx 5"):
        _gqa_spec(target, (6,))


def test_gqa_draft_arms_the_embedding_tap_without_a_successor():
    """`-1` taps the embedding, below every mixture, and the drafter reads it
    verbatim, so arming needs no successor layer for it."""
    pytest.importorskip("torch")
    spec = _gqa_spec(_k3_target_stub(n_layers=1), (-1,))
    assert spec.layer_ids == (-1,)


# --------------------------------------------------------------------------- #
# draft context slots (kimi_mla_gdn_attn builder -> dspark_proposer -> draft)
#
# The builder's staging is the REAL method on a stub; its module imports
# aiter/triton, so those tests skip on the CPU-only CI job. The proposer's
# hand-off is in `test_dspark.py`, the base draft's fan-out in
# `test_dspark_draft.py`.
# --------------------------------------------------------------------------- #


# A prefill chunk crossing an entry boundary, then two decode tokens.
_TABLES = [[7, 9] + [0] * 14, [4] + [0] * 15]
_SEQLENS = [3, 2]


def _builder_stub(entry_tokens, tables, positions, rows=64):
    """What `_stage_draft_ctx_slots` reads off the builder and its runner."""
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


class _Group:
    """The phase's H2D group: records each member's count."""

    def __init__(self):
        self.counts = {}

    def set_count(self, buffer, count):
        self.counts[id(buffer)] = count


def _stage(stub, seqlens, rows, tables, dummy=False):
    """Drive the real `_stage_draft_ctx_slots`; return the host slots and the
    count it recorded into the group (None if it recorded none)."""
    import numpy as np

    kimi = _gpu_stack_import("atom.model_ops.attentions.kimi_mla_gdn_attn")
    buf = stub.model_runner.forward_vars.get("draft_ctx_slots")
    batch = types.SimpleNamespace(is_dummy_run=dummy, block_tables=tables)
    group = _Group()
    kimi._KimiMLAGDNCommon._stage_draft_ctx_slots(
        stub, batch, np.asarray(seqlens, dtype=np.int32), rows, group
    )
    if buf is None:
        assert not group.counts
        return None, None
    return buf.np, group.counts.get(id(buf))


@pytest.mark.parametrize("dcp", [1, 8])
def test_the_builder_stages_the_draft_slots(dcp):
    """The target's builder stages, with no DCP filter, the slot of every
    scheduled token in `block_size * dcp`-token entries, and counts them into
    the phase's group. At DCP>1 the target's `block_size` addressing would
    alias `dcp` requests' entries onto one range."""
    vblock = 128 * dcp
    positions = [vblock - 2, vblock - 1, vblock, 5, 6]
    stub = _builder_stub(vblock, _TABLES, positions)
    got, count = _stage(stub, _SEQLENS, len(positions), _TABLES)
    assert count == 5
    assert got[:5].tolist() == [
        7 * vblock + vblock - 2,
        7 * vblock + vblock - 1,
        9 * vblock,
        4 * vblock + 5,
        4 * vblock + 6,
    ]


def test_the_builder_drops_the_padded_decode_rows():
    """Graph-padded rows past the scheduled tokens get `-1`, written nowhere."""
    stub = _builder_stub(1024, _TABLES, [3, 4, 7, 8])
    got, count = _stage(stub, [2, 2], 6, _TABLES)
    assert count == 6
    assert got[:6].tolist() == [
        7 * 1024 + 3,
        7 * 1024 + 4,
        4 * 1024 + 7,
        4 * 1024 + 8,
        -1,
        -1,
    ]


def test_a_request_less_prefill_drops_every_padded_row(monkeypatch):
    """An idle DP rank's prefill (`dummy_execution`) schedules no token but
    runs `running_tokens` padding rows through the draft: `stage_prefill`
    stages them all `-1`, out to the width the target's slots go up at."""
    import numpy as np

    kimi = _gpu_stack_import("atom.model_ops.attentions.kimi_mla_gdn_attn")
    monkeypatch.setattr(
        kimi.GDNStateMixin, "stage_prefill", lambda self, *a: ("md", "pos")
    )
    stub = _builder_stub(1024, _TABLES, [5, 6])
    builder = object.__new__(kimi.KimiAiterMLAGDNMetadataBuilder)
    builder.__dict__.update(vars(stub))
    batch = types.SimpleNamespace(
        is_dummy_run=False,
        block_tables=[],
        total_seqs_num_prefill=0,
        num_scheduled_tokens=np.zeros(0, dtype=np.int32),
    )
    group = _Group()
    assert builder.stage_prefill(batch, 1, 8, group) == ("md", "pos")
    buf = stub.model_runner.forward_vars["draft_ctx_slots"]
    assert group.counts[id(buf)] == 8
    assert buf.np[:8].tolist() == [-1] * 8


def test_a_ragged_decode_stages_each_request_at_its_own_length(monkeypatch):
    """`stage_decode` lays the slots out from `decode_spans`, the lengths the
    target's positions use, not a uniform `max_seqlen_q` per request."""
    import numpy as np

    kimi = _gpu_stack_import("atom.model_ops.attentions.kimi_mla_gdn_attn")
    monkeypatch.setattr(
        kimi.GDNStateMixin, "stage_decode", lambda self, *a: ("md", "pos")
    )
    positions = [3, 4, 5, 8]
    stub = _builder_stub(1024, _TABLES, positions)
    builder = object.__new__(kimi.KimiAiterMLAGDNMetadataBuilder)
    builder.__dict__.update(vars(stub))
    builder.model_runner.forward_vars["cu_seqlens_q"] = types.SimpleNamespace(
        np=np.array([0, 3, 4], dtype=np.int32)
    )
    batch = types.SimpleNamespace(
        is_dummy_run=False,
        block_tables=_TABLES,
        total_seqs_num_decode=2,
        num_scheduled_tokens=np.array([3, 1], dtype=np.int32),
    )
    group = _Group()
    assert builder.stage_decode(batch, 2, 6, 3, group) == ("md", "pos")
    buf = stub.model_runner.forward_vars["draft_ctx_slots"]
    assert group.counts[id(buf)] == 6
    assert buf.np[:6].tolist() == [
        7 * 1024 + 3,
        7 * 1024 + 4,
        7 * 1024 + 5,
        4 * 1024 + 8,
        -1,
        -1,
    ]


def test_the_builder_skips_dummy_runs_and_drafts_without_a_pool():
    """A dummy run stages and counts nothing; a target with no pooled draft
    has no buffer to fill and is a no-op."""
    stub = _builder_stub(1024, _TABLES, [0, 1])
    got, count = _stage(stub, [2], 2, _TABLES, dummy=True)
    assert count is None
    assert got[:2].tolist() == [7, 7]
    del stub.model_runner.forward_vars["draft_ctx_slots"]
    assert _stage(stub, [2], 2, _TABLES) == (None, None)


# --------------------------------------------------------------------------- #
# the block read (model_ops/kimi_k3/dspark_block_attn.py)
#
# `unified_attention` asserts causal; the read gets a non-causal block out of it
# by folding each request's T rows into its head axis and decoding ONE row per
# request. The fold is pure tensor reshaping, but its module imports aiter's
# kernel, so these skip where aiter is absent; the kernel check needs a GPU.
# --------------------------------------------------------------------------- #


def _fold_mod():
    pytest.importorskip("torch")
    return _gpu_stack_import("atom.model_ops.kimi_k3.dspark_block_attn")


@pytest.mark.parametrize("bs,T,hkv,g", [(1, 7, 2, 4), (3, 7, 2, 4), (2, 4, 1, 8)])
def test_block_fold_puts_every_row_on_its_own_kv_head(bs, T, hkv, g):
    """`unified_attention` reads head `j` against KV head `j // (T * G)`.

    So in the folded layout, every one of a request's T rows times its G query
    heads per KV head must sit in that KV head's contiguous run -- and nothing
    of another request's. Tag each element with (request, row, query head) and
    read the tags back.
    """
    import torch

    dba = _fold_mod()
    hq = hkv * g
    b = torch.arange(bs).view(bs, 1, 1).expand(bs, T, hq)
    t = torch.arange(T).view(1, T, 1).expand(bs, T, hq)
    h = torch.arange(hq).view(1, 1, hq).expand(bs, T, hq)
    # One scalar per (b, t, h), carried in the D axis so the fold moves it.
    tag = torch.stack([b, t, h], dim=-1).reshape(bs * T, hq, 3)

    qf = dba.fold_block_queries(tag, bs, hkv)
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
    dba.unfold_block_output(qf, out, hkv)
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
    dba = _fold_mod()

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

    read = dba.DSparkGQABlockAttention(dev, max_bs=8, max_model_len=W * E)
    one = torch.ones(1, dtype=torch.float32, device=dev)
    # Where the read casts q to an fp8 pool's dtype (gfx950), the reference
    # takes the same q. Taken BEFORE the call, which clamps q in place.
    cast = kv_dtype == "fp8" and read.casts_q(k_pool, v_pool)
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


def test_gqa_draft_is_refused_on_a_non_k3_target():
    """The checkpoint arch is generic, so the draft checks its target where the
    target's layers are known: a stack without the Kimi-K3 aggregate refuses."""
    target = types.SimpleNamespace(
        layers=[types.SimpleNamespace(layer_idx=i) for i in range(16)]
    )
    with pytest.raises(ValueError, match="Kimi-K3 target only"):
        _gqa_spec(target, (6, 11))


def test_fp8_q_cast_asks_aiters_gluon_gate_with_the_reads_arguments(monkeypatch):
    """The cast follows aiter's own gate, asked about this read: q in the
    pool's dtype, the pool's head and page sizes, no softcap, all-decode."""
    torch = pytest.importorskip("torch")
    dba = _fold_mod()
    ua = dba.ua
    seen = []

    def gate(params):
        seen.append(params)
        return params.head_size == 64

    monkeypatch.setattr(ua, "_gfx950_gluon_supported", gate)
    read = dba.DSparkGQABlockAttention("cpu", max_bs=2, max_model_len=1024)
    fp8 = torch.float8_e4m3fn
    k = torch.empty(2, 1024, 2, 64, dtype=fp8)
    assert read.casts_q(k, k) and read.casts_q(k, k)
    assert len(seen) == 1, "the answer is cached per pool shape and dtype"
    p = seen[0]
    assert (p.head_size, p.head_size_v, p.block_size) == (64, 64, 1024)
    assert p.q_dtype == p.kv_cache_dtype == fp8
    assert p.softcap == 0 and p.max_seqlen_q == 1 and p.all_decode
    assert not (p.shuffled_kv_cache or p.use_alibi_slopes or p.use_qq_bias)

    k96 = torch.empty(2, 1024, 2, 96, dtype=fp8)
    assert not read.casts_q(k96, k96)

    monkeypatch.delattr(ua, "_gfx950_gluon_supported")
    assert not dba.gluon_takes_pool_dtype_q(64, 64, 1024, fp8)


def test_fp8_q_cast_saturates_instead_of_producing_nan():
    """A plain cast to fp8 turns |x| > fp8 max into NaN, and one NaN q row
    poisons that row's softmax against every key; the read clamps first."""
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("needs a GPU")
    pytest.importorskip("aiter")
    from aiter import dtypes

    dba = _fold_mod()
    dev = torch.device("cuda")
    bs, T, hkv, g, d, E = 2, 7, 2, 4, 64, 1024
    read = dba.DSparkGQABlockAttention(dev, max_bs=4, max_model_len=E)
    k_pool = torch.randn(3, E, hkv, d, device=dev).to(dtypes.fp8)
    v_pool = torch.randn_like(k_pool, dtype=torch.float32).to(dtypes.fp8)
    if not read.casts_q(k_pool, v_pool):
        pytest.skip("q is only cast to fp8 where aiter's Gluon path serves the read")
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
    `KimiK3DSparkGQA.prepare_paged_block` drops those writes and caps the read."""
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("needs a GPU: pinned staging and a CUDA event")
    m = _gpu_stack_import("atom.models.kimi_k3_dspark_gqa")
    draft = m.KimiK3DSparkGQA.__new__(m.KimiK3DSparkGQA)
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
    m = _gpu_stack_import("atom.models.kimi_k3_dspark_gqa")
    qc = m._draft_quant_config(_radixark_cfg())
    assert qc.quant_method == "" and qc.online_quant is False
    with pytest.raises(ValueError, match="BF16 weights only"):
        m._draft_quant_config(
            _radixark_cfg(quantization_config={"quant_method": "fp8"})
        )
