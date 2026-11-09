# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""The shared DSpark draft base, `atom/models/dspark_draft.py`, on CPU.

- `AllocationCap`, the allocation staging the paged block drafts share
  (Kimi-K3 DSpark GQA, MiniMax-M3.1 DSpark)
- `DSparkDraftModel`'s default target sharing, block sampling and context-KV
  fan-out, and the draft layers' slot hand-off
- `draft_rope_scaling`, the drafts' shared rope parser, and the Kimi-K3 MLA
  draft's rope arguments built on it

Kept apart from `test_dspark.py`, which skips whole without aiter, so these
run on the CPU-only CI job.
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


@pytest.fixture
def host_staging(monkeypatch):
    """`AllocationCap`'s pinned staging and CUDA event, on a host without a
    GPU: unpinned buffers and an event that has nothing to wait for."""
    import torch

    if torch.cuda.is_available():
        return
    zeros = torch.zeros

    def unpinned(*args, pin_memory=False, **kwargs):
        return zeros(*args, **kwargs)

    class _Event:
        def synchronize(self):
            pass

        def record(self):
            pass

    monkeypatch.setattr(torch, "zeros", unpinned)
    monkeypatch.setattr(torch.cuda, "Event", _Event)


@pytest.mark.parametrize("unit", [128, 128 * 8])
def test_allocation_cap_masks_writes_past_each_rows_allocation(host_staging, unit):
    """Rows past `scheduled_bs` and rows holding no page read one page; slots
    at or past the allocation (in tokens, `unit` per page) are dropped."""
    import torch

    from atom.models.dspark_draft import AllocationCap

    cap = AllocationCap(4, "cpu")
    # Row 0 holds one page and its block crosses into a second; row 1 holds
    # two and stays inside; row 2 is a pad row.
    pos = torch.tensor([[unit - 2, unit - 1, unit], [5, 6, 7], [1, 2, 3]])
    slots = torch.tensor([[10, 11, 12], [20, 21, 22], [-1, -1, -1]])
    alloc = cap.apply([1, 2, 9], 2, pos, slots, unit)
    assert alloc.tolist() == [unit, 2 * unit, unit]
    assert slots.tolist() == [[10, 11, -1], [20, 21, 22], [-1, -1, -1]]
    # The staging is reused: a later, smaller step sees its own pages.
    slots = torch.tensor([[30, 31, 32]])
    alloc = cap.apply([0], 1, pos[:1], slots, unit)
    assert alloc.tolist() == [unit]
    assert slots.tolist() == [[30, 31, -1]]


def test_allocation_cap_leaves_the_minimum_to_each_caller(host_staging):
    """MiniMax-M3.1 shape: page-128 units, rows capped into their own buffer
    while `ctx_lens` stays as the proposer built it. Kimi-K3 DSpark GQA shape: one
    pool entry (block size x DCP) per block id, `ctx_lens` capped in place."""
    import torch

    from atom.models.dspark_draft import AllocationCap

    T = 3
    pos = torch.tensor([[126, 127, 128], [5, 6, 7], [1, 2, 3]])
    ctx = torch.tensor([129, 8, 1 + T], dtype=torch.int32)

    cap = AllocationCap(4, "cpu")
    ctx_eff = torch.zeros(4, dtype=torch.int32)
    alloc = cap.apply([1, 2], 2, pos, pos.clone(), 128)
    torch.minimum(ctx, alloc, out=ctx_eff[:3])
    assert ctx.tolist() == [129, 8, 1 + T]
    assert ctx_eff[:3].tolist() == [128, 8, 1 + T]

    entry = 128 * 8
    pos = torch.tensor([[entry - 2, entry - 1, entry], [5, 6, 7], [1, 2, 3]])
    ctx = torch.tensor([entry + 1, 8, 1 + T], dtype=torch.int32)
    alloc = cap.apply([1, 2], 2, pos, pos.clone(), entry)
    torch.minimum(ctx, alloc, out=ctx)
    assert ctx.tolist() == [entry, 8, 1 + T]


def test_minimax_m31_dspark_caps_the_rows_but_not_ctx_lens(host_staging):
    """MiniMax-M3.1's block rows read at most the allocation; the proposer's
    `ctx_lens` (`attn_metadata.context_lens`) is left as it built it."""
    import torch

    m3_dspark = pytest.importorskip(
        "atom.models.minimax_m3.dspark",
        reason="the MiniMax-M3.1 draft imports aiter at module load",
        exc_type=ImportError,
    )
    MiniMaxM3DSpark = m3_dspark.MiniMaxM3DSpark

    draft = MiniMaxM3DSpark.__new__(MiniMaxM3DSpark)
    draft.max_num_seqs, draft.block_size, draft._bt16 = 4, 128, None
    draft.sliding_window = 0
    T = 3
    pos = torch.tensor([[126, 127, 128], [5, 6, 7], [1, 2, 3]])
    slots = torch.tensor([[10, 11, 12], [20, 21, 22], [-1, -1, -1]])
    ctx = torch.tensor([129, 8, 1 + T], dtype=torch.int32)
    md = types.SimpleNamespace()
    block_tables = torch.tensor([[3, 0], [4, 5], [0, 0]], dtype=torch.int32)
    draft.prepare_paged_block(md, block_tables, pos, slots, ctx, [1, 2], 2)
    assert slots.tolist() == [[10, 11, -1], [20, 21, 22], [-1, -1, -1]]
    assert ctx.tolist() == [129, 8, 1 + T]
    assert md.dspark_ctx_rows.tolist() == [128] * T + [8] * T + [1 + T] * T


# ---- the base draft's defaults --------------------------------------------


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
    return types.SimpleNamespace(model=model, lm_head=torch.nn.Embedding(vocab, 4))


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


def test_default_share_with_target_keeps_a_drafts_own_embedding():
    """A draft that loaded its own embedding (K3 DSpark past PP stage 0)
    keeps it and still scores with the target's head."""
    draft = _stub_draft(163840)
    own = object()
    draft.own_embed_tokens, draft.embed_tokens = True, own
    target = _target_base(163840)
    draft.share_with_target(target, set())
    assert draft.embed_tokens is own
    assert draft.lm_head is target.lm_head


def test_default_share_with_target_reads_the_vocab_off_the_head_under_pp():
    """Under PP the drafter's stage holds a PPMissingLayer for the target's
    embedding; the vocab check reads the head the draft scores with."""
    from atom.models.utils import PPMissingLayer

    draft = _stub_draft(163840)
    draft.own_embed_tokens = True
    target = _target_base(163840)
    target.model.embed_tokens = PPMissingLayer()
    draft.share_with_target(target, set())
    assert draft.lm_head is target.lm_head
    with pytest.raises(ValueError, match="target vocab 200064"):
        _stub_draft(163840).share_with_target(_target_base(200064), set())


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


@pytest.mark.parametrize(
    "module, layer",
    [
        ("atom.models.dspark_draft", "DSparkGQADecoderLayer"),
        ("atom.models.kimi_k3_dspark", "K3DSparkDecoderLayer"),
    ],
)
def test_the_draft_layer_writes_at_the_slots_it_is_handed(module, layer):
    """No read of the target's `slot_mapping` on the way down: the slots come
    from `compute_draft_kv` alone."""
    import torch

    layer_cls = getattr(_gpu_stack_import(module), layer)
    seen = {}
    stub = types.SimpleNamespace(
        self_attn=types.SimpleNamespace(
            write_context_kv=lambda h, p, s: seen.update(slots=s)
        )
    )
    slots = torch.arange(4)
    layer_cls.write_context_kv(stub, torch.zeros(4, 8), torch.arange(4), slots)
    assert seen["slots"] is slots


# ---- the shared rope parser -------------------------------------------------


def _rope_cfg(**overrides):
    """The rope fields of the RadixArk Kimi-K3 DSpark GQA checkpoint's config."""
    cfg = types.SimpleNamespace(
        max_position_embeddings=1048576,
        rope_parameters={
            "rope_type": "yarn",
            "factor": 16.0,
            "original_max_position_embeddings": 65536,
            "rope_theta": 10000.0,
        },
    )
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg


def test_draft_rope_scaling_leaves_the_type_to_the_caller():
    """The shared parser copies the config's scaling dict; a caller's override
    of the rope type never reaches the config."""
    from atom.models.dspark_draft import draft_rope_scaling

    cfg = _rope_cfg()
    theta, scaling = draft_rope_scaling(cfg)
    assert theta == 10000.0
    assert scaling == cfg.rope_parameters
    scaling["rope_type"] = "deepseek_yarn"
    assert cfg.rope_parameters["rope_type"] == "yarn"

    assert draft_rope_scaling(_rope_cfg(rope_parameters={}))[1] is None


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
    from atom.models import dspark_draft, utils

    ns = {
        "draft_rope_scaling": dspark_draft.draft_rope_scaling,
        "is_neox_rope_style": utils.is_neox_rope_style,
    }
    return _lift_fn(module_file, "_draft_rope_kwargs", ns)


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

    # `rope_interleave=true` is the same interleaved rope (K3.1 sets it); only
    # an explicit false selects neox. A factor alone is YaRN and a missing
    # original length falls back to the max position.
    cfg = _inferact_rope_cfg(
        rope_interleave=True,
        rope_parameters={"factor": 4.0, "rope_theta": 20000.0},
    )
    kw = kwargs_fn(cfg)
    assert kw["is_neox_style"] is False
    assert kwargs_fn(_inferact_rope_cfg(rope_interleave=False))["is_neox_style"]
    assert kwargs_fn(_inferact_rope_cfg(rope_interleave=None))["is_neox_style"] is False
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
