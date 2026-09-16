# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""How an expert weight reaches the fused buffers, and what happens if it
doesn't reach them the same way twice.

Two properties, both about the relayout. `weight_loader` writes plain
row-major bytes over buffers the kernel reads through aiter's per-expert
permutation, so every route into those buffers owes the same three things:
refuse the combinations the path does not implement, write, and register the
slices it wrote so the layout is re-established once at the end.

A route that skips the registration is silent -- the sync still reports
`updated=N` -- and so is a sync that registers a slice, shuffles it, and then
leaves the registration behind for the next sync to shuffle again. Per
`_finalize_expert_weight_sync`'s own docstring: shuffling an already-shuffled
slice does not undo the first shuffle, it produces a third layout.
"""

import pytest
import torch
from torch import nn

from atom.rollout.weight_updater import WeightUpdaterMixin

needs_aiter = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="atom.model_ops.utils imports aiter, which resolves the chip "
    "architecture through rocminfo",
)

HIDDEN = 8
INTERMEDIATE = 4
EXPERTS = 2


def _updater(model):
    class _Updater(WeightUpdaterMixin):
        device = torch.device("cpu")
        label = "test"
        rank = 0
        world_size = 1

        def clear_kv_cache(self):
            pass

    updater = _Updater()
    updater.model = model
    return updater


def _moe_model(dtype=torch.bfloat16, **module_attrs):
    """A layer holding the fused expert buffers under ATOM's own names."""
    experts = nn.Module()
    experts.w13_weight = nn.Parameter(
        torch.zeros(EXPERTS, 2 * INTERMEDIATE, HIDDEN, dtype=dtype),
        requires_grad=False,
    )
    experts.w2_weight = nn.Parameter(
        torch.zeros(EXPERTS, HIDDEN, INTERMEDIATE, dtype=dtype),
        requires_grad=False,
    )
    experts.weight_loader = lambda *a, **k: None
    experts.expert_map = None
    experts.num_redundant_experts = 0
    for name, value in module_attrs.items():
        setattr(experts, name, value)
    mlp = nn.Module()
    mlp.experts = experts
    model = nn.Module()
    model.mlp = mlp
    return model, experts


# ── the route a trainer takes when it mirrors ATOM's own state dict ────────


def test_an_atom_named_expert_buffer_is_not_a_plain_parameter(monkeypatch):
    """`w13_weight` is a real parameter of the FusedMoE, so it resolves in
    `_get_param_to_module_mapping` and never reaches `_apply_unmatched_weight`.

    Down the plain dispatch it is a row-major `copy_` into a buffer the kernel
    reads through the expert permutation, with nothing registered for relayout
    and `updated` counting it as a success.
    """
    model, experts = _moe_model()
    updater = _updater(model)
    routed = []
    monkeypatch.setattr(
        type(updater),
        "_apply_named_expert_buffer",
        lambda self, *a: routed.append(a[0]),
        raising=True,
    )

    assert "mlp.experts.w13_weight" in updater._get_param_to_module_mapping()
    assert (
        updater.update_weights(
            [("mlp.experts.w13_weight", torch.ones_like(experts.w13_weight))]
        )
        == 1
    )

    assert routed == ["mlp.experts.w13_weight"]


@pytest.mark.parametrize("param_name", ["w13_weight", "w2_weight"])
def test_a_named_expert_buffer_registers_every_slice(param_name):
    """One tensor covers every expert, so every slice of the buffer is new and
    all of them need the layout re-established."""
    model, experts = _moe_model()
    updater = _updater(model)
    param = getattr(experts, param_name)
    incoming = torch.full_like(param, 3.0)

    updater._apply_named_expert_buffer(
        f"mlp.experts.{param_name}", param_name, experts, param, incoming
    )

    assert param.eq(3.0).all(), "the write did not land"
    pending = updater._pending_expert_relayout
    assert list(pending) == [(experts, param_name)]
    arrived = pending[(experts, param_name)]
    assert sorted(arrived) == list(range(EXPERTS))
    expected = {"w1", "w3"} if param_name == "w13_weight" else {"w2"}
    assert all(shards == expected for shards in arrived.values())


def test_a_partial_named_expert_buffer_is_refused():
    """Re-establishing the layout works on whole expert slices, so a write
    that covers part of one cannot be relaid out."""
    model, experts = _moe_model()
    updater = _updater(model)

    with pytest.raises(NotImplementedError, match="fused expert buffer"):
        updater._apply_named_expert_buffer(
            "mlp.experts.w13_weight",
            "w13_weight",
            experts,
            experts.w13_weight,
            torch.zeros(EXPERTS, INTERMEDIATE, HIDDEN, dtype=torch.bfloat16),
        )

    assert not updater._pending_expert_relayout


def test_a_named_expert_buffer_is_refused_on_a_quantized_moe():
    """The plain dispatch would have written it and reported updated=1.
    `_check_expert_sync_supported` is the check this route was missing."""
    model, experts = _moe_model(dtype=torch.float8_e4m3fnuz)
    updater = _updater(model)

    with pytest.raises(NotImplementedError, match="quantized storage format"):
        updater._apply_named_expert_buffer(
            "mlp.experts.w13_weight",
            "w13_weight",
            experts,
            experts.w13_weight,
            torch.zeros_like(experts.w13_weight),
        )


def test_a_named_expert_buffer_is_refused_under_expert_parallelism():
    model, experts = _moe_model(expert_map=torch.zeros(EXPERTS, dtype=torch.int32))
    updater = _updater(model)

    with pytest.raises(NotImplementedError, match="expert-parallel"):
        updater._apply_named_expert_buffer(
            "mlp.experts.w13_weight",
            "w13_weight",
            experts,
            experts.w13_weight,
            torch.zeros_like(experts.w13_weight),
        )


# ── the wrappers ModelRunner rebinds onto self.model ──────────────────────


def _tbo_wrapped(model):
    """The real `UBatchWrapper`. It needs no device to construct, so nothing
    here has to stand in for it and get its shape wrong."""
    from atom.utils.tbo.ubatch_wrapper import UBatchWrapper

    return UBatchWrapper(model)


def _compiled(model):
    """What compilation level 1 rebinds. `torch.compile` compiles nothing until
    the module is called; this is only its `OptimizedModule` shell."""
    return torch.compile(model, backend="eager")


@pytest.mark.parametrize("wrap", [_tbo_wrapped, _compiled], ids=["tbo", "compiled"])
def test_a_wrapper_does_not_move_the_names_the_trainer_sends(wrap):
    """Both hold the model as a CHILD, so `named_modules()` on the wrapper
    prefixes every parameter with the wrapper's own attribute name -- `model.`
    or `_orig_mod.` -- and then nothing the trainer sends matches anything."""
    model, experts = _moe_model()
    updater = _updater(model)
    updater.model = wrap(model)

    mapping = updater._get_param_to_module_mapping()

    assert "mlp.experts.w13_weight" in mapping
    assert mapping["mlp.experts.w13_weight"][0] is experts


@pytest.mark.parametrize("wrap", [_tbo_wrapped, _compiled], ids=["tbo", "compiled"])
def test_the_mappings_reached_by_attribute_are_unaffected(wrap):
    """These two are plain attribute lookups, and BOTH wrappers forward those to
    what they wrap (`UBatchWrapper.__getattr__`, `OptimizedModule.__getattr__`),
    so they were never the broken half. Pinned because asking the unwrapped
    model instead must not be a change: one module answering all four lookups is
    the point, not a repair."""
    model, _ = _moe_model()
    model.get_expert_mapping = lambda: [("experts.w13_weight", "gate_proj", 0, "w1")]
    model.packed_modules_mapping = {"gate_proj": ("gate_up_proj", 0)}
    updater = _updater(model)
    updater.model = wrap(model)

    assert updater._get_expert_params_mapping() == [
        ("gate_proj", "experts.w13_weight", 0, "w1")
    ]
    assert updater._get_packed_modules_mapping() == {"gate_proj": ("gate_up_proj", 0)}
    assert updater._get_packed_shard_order() == {"gate_up_proj": [0]}


def test_a_sync_that_matched_nothing_says_so_above_debug(caplog):
    """What the wrapper bug looked like from outside: `updated=0, skipped=N` on
    an info line, indistinguishable from a bucket that legitimately held
    nothing. The next name convention nobody has thought of gets this."""
    model, _ = _moe_model()
    updater = _updater(model)

    with caplog.at_level("WARNING", logger="atom"):
        updated = updater.update_weights([("nobody.knows.this", torch.zeros(2))])

    assert updated == 0
    assert "matched NOTHING" in caplog.text


def test_a_rebind_after_the_first_lookup_rebuilds_the_mapping():
    """The caches were `hasattr`-keyed: nothing invalidates that, and there is
    no hook here to invalidate it from. Keyed on the model they were built from,
    they rebuild themselves when that changes."""
    model, experts = _moe_model()
    other_model, other_experts = _moe_model()
    updater = _updater(model)

    first = updater._get_param_to_module_mapping()
    assert first["mlp.experts.w13_weight"][0] is experts

    updater.model = other_model
    second = updater._get_param_to_module_mapping()

    assert second["mlp.experts.w13_weight"][0] is other_experts


# ── a sync that fails part way through ────────────────────────────────────


def _pending_two_buffers(updater, experts, *, second_is_half_written):
    """Buffer A complete, buffer B missing a shard so finalize raises on it."""
    pending = updater._pending_expert_relayout
    pending[(experts, "w13_weight")] = {e: {"w1", "w3"} for e in range(EXPERTS)}
    pending[(experts, "w2_weight")] = {
        e: (set() if second_is_half_written else {"w2"}) for e in range(EXPERTS)
    }
    return pending


@needs_aiter
def test_a_shuffled_buffer_is_not_carried_into_the_next_sync(monkeypatch):
    """`pending.clear()` sat after the loop, so the raise skipped it and left
    the entry for the buffer the loop had already shuffled. The next
    successful sync shuffled it a second time -- a third layout, silent, with
    `updated=N` logged as success."""
    import atom.model_ops.utils as utils_mod

    model, experts = _moe_model()
    updater = _updater(model)
    shuffled = []
    monkeypatch.setattr(
        utils_mod,
        "shuffle_expert_slices",
        lambda buffer, ids, **k: shuffled.append(tuple(ids)),
        raising=True,
    )
    pending = _pending_two_buffers(updater, experts, second_is_half_written=True)

    with pytest.raises(RuntimeError, match="half new and"):
        updater._finalize_expert_weight_sync()

    assert shuffled == [(0, 1)], "buffer A was shuffled before the raise"
    assert (experts, "w13_weight") not in pending, "A would be shuffled twice"

    # A later sync completes B and must not touch A again.
    pending[(experts, "w2_weight")] = {e: {"w2"} for e in range(EXPERTS)}
    updater._finalize_expert_weight_sync()

    assert shuffled == [(0, 1), (0, 1)], "exactly one shuffle per buffer"
    assert not pending


@needs_aiter
def test_a_half_delivered_expert_does_not_poison_every_later_sync(monkeypatch):
    """Keeping the entry was worse than useless.

    Nothing later completes one -- a sync sends every shard of an expert or the
    write is refused outright -- so the entry would fail this function again on
    every sync for the life of the process, and take every other buffer's
    relayout down with it each time. The bytes are recoverable without it: the
    next update that sends the whole expert overwrites both halves row-major.
    """
    import atom.model_ops.utils as utils_mod

    model, experts = _moe_model()
    updater = _updater(model)
    shuffled = []
    monkeypatch.setattr(
        utils_mod,
        "shuffle_expert_slices",
        lambda buffer, ids, **k: shuffled.append(tuple(ids)),
        raising=True,
    )
    pending = updater._pending_expert_relayout
    pending[(experts, "w13_weight")] = {0: {"w1", "w3"}, 1: {"w1"}}

    with pytest.raises(RuntimeError, match=r"w13_weight\[1\] arrived without"):
        updater._finalize_expert_weight_sync()

    # Per expert, not per buffer: expert 1 cannot be relaid out, and that is no
    # reason for expert 0 to keep reading row-major bytes through the
    # permutation.
    assert shuffled == [(0,)]
    assert not pending, "the half-delivered entry is dropped, not carried"

    # A later sync over another buffer is not taken down with it.
    pending[(experts, "w2_weight")] = {e: {"w2"} for e in range(EXPERTS)}
    updater._finalize_expert_weight_sync()

    assert shuffled == [(0,), (0, 1)]
    assert not pending


@needs_aiter
def test_a_whole_expert_resent_later_gets_its_layout_back(monkeypatch):
    """What repairs the half-delivered slice: an update carrying every shard
    writes row-major over both halves, and that entry relays out normally."""
    import atom.model_ops.utils as utils_mod

    model, experts = _moe_model()
    updater = _updater(model)
    shuffled = []
    monkeypatch.setattr(
        utils_mod,
        "shuffle_expert_slices",
        lambda buffer, ids, **k: shuffled.append(tuple(ids)),
        raising=True,
    )
    updater._pending_expert_relayout[(experts, "w13_weight")] = {1: {"w1"}}

    with pytest.raises(RuntimeError, match="half new and"):
        updater._finalize_expert_weight_sync()
    assert shuffled == []

    updater._apply_named_expert_buffer(
        "mlp.experts.w13_weight",
        "w13_weight",
        experts,
        experts.w13_weight,
        torch.full_like(experts.w13_weight, 7.0),
    )
    updater._finalize_expert_weight_sync()

    assert shuffled == [(0, 1)]
    assert not updater._pending_expert_relayout


@needs_aiter
def test_a_buffer_the_loop_never_reached_survives(monkeypatch):
    """The other direction: an entry the loop did not get to still describes a
    row-major buffer, so dropping it would leave the kernel reading row-major
    bytes through the permutation with nothing left to say so."""
    import atom.model_ops.utils as utils_mod

    model, experts = _moe_model()
    updater = _updater(model)
    monkeypatch.setattr(
        utils_mod,
        "shuffle_expert_slices",
        lambda buffer, ids, **k: (_ for _ in ()).throw(RuntimeError("boom")),
        raising=True,
    )
    pending = _pending_two_buffers(updater, experts, second_is_half_written=False)

    with pytest.raises(RuntimeError, match="boom"):
        updater._finalize_expert_weight_sync()

    assert list(pending) == [(experts, "w13_weight"), (experts, "w2_weight")]


@needs_aiter
def test_a_clean_sync_still_empties_the_pending_set(monkeypatch):
    import atom.model_ops.utils as utils_mod

    model, experts = _moe_model()
    updater = _updater(model)
    monkeypatch.setattr(
        utils_mod, "shuffle_expert_slices", lambda *a, **k: None, raising=True
    )
    pending = _pending_two_buffers(updater, experts, second_is_half_written=False)

    updater._finalize_expert_weight_sync()

    assert not pending


def test_finalize_is_a_no_op_with_nothing_pending():
    """And does not import aiter to find that out, which is why the rest of
    this file's expert-routing tests run on a CPU box."""
    model, _ = _moe_model()
    updater = _updater(model)

    updater._finalize_expert_weight_sync()

    assert not updater._pending_expert_relayout


def test_the_updater_needs_no_expert_mapping_for_this_route():
    """`get_expert_mapping` is how a *checkpoint*-named expert is resolved; a
    buffer sent under ATOM's own name never consults it."""
    model, experts = _moe_model()
    assert not hasattr(model, "get_expert_mapping")
    updater = _updater(model)

    updater._apply_named_expert_buffer(
        "mlp.experts.w2_weight",
        "w2_weight",
        experts,
        experts.w2_weight,
        torch.full_like(experts.w2_weight, 5.0),
    )

    assert experts.w2_weight.eq(5.0).all()
