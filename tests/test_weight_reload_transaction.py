# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Commit-time coverage and abort semantics of a bucketed, transactional reload.

The trainer sends checkpoint names. ATOM fuses several of them into one
parameter -- ``q_proj``/``k_proj``/``v_proj`` into ``qkv_proj``, every expert's
``gate_proj``/``up_proj`` into one ``w13_weight`` -- so coverage recorded under
the incoming name can never match ``named_parameters()``, and every reload of a
fused model was rejected as incomplete. And a fused parameter is only new once
all of its pieces are, so crediting it on the first piece would pass a reload
that left most of it old.
"""

import contextlib
import logging
from unittest import mock

import pytest
import torch
from torch import nn

from atom.rollout.weight_updater import WeightUpdaterMixin

HIDDEN = 4
EXPERTS = 2
INTERMEDIATE = 2


class _Runner(WeightUpdaterMixin):
    device = torch.device("cpu")
    label = "test"
    rank = 0
    world_size = 1

    def __init__(self, model):
        self.model = model
        self.kv_cleared = 0

    def clear_kv_cache(self):
        self.kv_cleared += 1

    def _sync_target_model(self):
        # No wrapper to peel. The real lookup imports the TBO package, whose
        # `atom.utils.forward_context` import plugin tests elsewhere in the
        # suite leave replaced by a bare stub.
        return self.model


def _param(*shape, dtype=torch.bfloat16):
    return nn.Parameter(torch.zeros(*shape, dtype=dtype), requires_grad=False)


def _qkv_loader(param, loaded, shard_id):
    rows = {"q": slice(0, 4), "k": slice(4, 6), "v": slice(6, 8)}[shard_id]
    param.data[rows].copy_(loaded)


def _dense_model():
    """One attention block the way ATOM builds it: q, k and v in one parameter."""
    attn = nn.Module()
    attn.qkv_proj = nn.Module()
    attn.qkv_proj.weight = _param(8, HIDDEN)
    attn.qkv_proj.weight_loader = _qkv_loader
    attn.o_proj = nn.Module()
    attn.o_proj.weight = _param(HIDDEN, 4)
    model = nn.Module()
    model.attn = attn
    model.packed_modules_mapping = {
        "q_proj": ("qkv_proj", "q"),
        "k_proj": ("qkv_proj", "k"),
        "v_proj": ("qkv_proj", "v"),
    }
    return model


def _dense_checkpoint(fill=1.0):
    return {
        "attn.q_proj.weight": torch.full((4, HIDDEN), fill, dtype=torch.bfloat16),
        "attn.k_proj.weight": torch.full((2, HIDDEN), fill, dtype=torch.bfloat16),
        "attn.v_proj.weight": torch.full((2, HIDDEN), fill, dtype=torch.bfloat16),
        "attn.o_proj.weight": torch.full((HIDDEN, 4), fill, dtype=torch.bfloat16),
    }


def _moe_model():
    """A FusedMoE-shaped layer, reached through the model's expert mapping."""

    def weight_loader(param, tensor, weight_name=None, shard_id=None, expert_id=0):
        if shard_id == "w2":
            param.data[expert_id].copy_(tensor)
            return
        lo = 0 if shard_id == "w1" else INTERMEDIATE
        param.data[expert_id, lo : lo + INTERMEDIATE].copy_(tensor)

    experts = nn.Module()
    experts.w13_weight = _param(EXPERTS, 2 * INTERMEDIATE, HIDDEN)
    experts.w2_weight = _param(EXPERTS, HIDDEN, INTERMEDIATE)
    experts.weight_loader = weight_loader
    experts.expert_map = None
    experts.num_redundant_experts = 0
    mlp = nn.Module()
    mlp.experts = experts
    model = nn.Module()
    model.mlp = mlp
    model.get_expert_mapping = lambda: [
        (f"experts.{param}", f"experts.{e}.{leaf}.", e, shard)
        for e in range(EXPERTS)
        for param, leaf, shard in (
            ("w13_", "gate_proj", "w1"),
            ("w13_", "up_proj", "w3"),
            ("w2_", "down_proj", "w2"),
        )
    ]
    return model


def _expert_checkpoint(experts=range(EXPERTS)):
    out = {}
    for e in experts:
        for leaf, shape in (
            ("gate_proj", (INTERMEDIATE, HIDDEN)),
            ("up_proj", (INTERMEDIATE, HIDDEN)),
            ("down_proj", (HIDDEN, INTERMEDIATE)),
        ):
            out[f"mlp.experts.{e}.{leaf}.weight"] = torch.ones(
                shape, dtype=torch.bfloat16
            )
    return out


@pytest.fixture(autouse=True)
def _cpu_relayout(monkeypatch):
    """The real relayout imports aiter. What matters here is that it consumes
    the bookkeeping, as the real one does on success: coverage read after it
    would find nothing, which is the ordering this pins."""
    monkeypatch.setattr(
        WeightUpdaterMixin,
        "_finalize_expert_weight_sync",
        lambda self: self._pending_expert_relayout.clear(),
    )


def _reload(runner, buckets, version=1):
    runner.begin_weight_update(version)
    for bucket in buckets:
        runner.apply_weight_bucket(list(bucket.items()))
    return runner.commit_weight_update(version)


# ── coverage is by the parameter written, not the name sent ────────────────


def test_a_fused_attention_block_passes_coverage_once_every_shard_lands():
    runner = _Runner(_dense_model())
    ckpt = _dense_checkpoint()
    # q and k in one bucket, v in the next: the shards of one fused parameter
    # may span buckets.
    first = {k: ckpt[k] for k in ("attn.q_proj.weight", "attn.k_proj.weight")}
    second = {k: ckpt[k] for k in ("attn.v_proj.weight", "attn.o_proj.weight")}

    manifest = _reload(runner, [first, second])

    assert manifest["loaded_internal"] == 2  # qkv_proj and o_proj
    assert runner.model.attn.qkv_proj.weight.eq(1.0).all()
    assert runner.get_weight_update_status()["healthy"] is True


def test_a_fused_parameter_missing_a_shard_is_refused():
    """Crediting qkv_proj on its first shard would pass a reload that left the
    v rows old."""
    runner = _Runner(_dense_model())
    ckpt = _dense_checkpoint()
    del ckpt["attn.v_proj.weight"]

    with pytest.raises(RuntimeError, match=r"missing=\['attn\.qkv_proj\.weight'\]"):
        _reload(runner, [ckpt])
    assert runner.get_weight_update_status()["healthy"] is False


def test_every_expert_counts_before_the_fused_buffer_does():
    runner = _Runner(_moe_model())
    manifest = _reload(runner, [_expert_checkpoint()])
    assert manifest["loaded_internal"] == 2  # w13_weight and w2_weight


def test_an_expert_nobody_sent_leaves_its_buffer_uncovered():
    runner = _Runner(_moe_model())
    with pytest.raises(RuntimeError, match="mlp.experts.w13_weight"):
        _reload(runner, [_expert_checkpoint(experts=[0])])


def test_a_tied_parameter_is_covered_under_either_name():
    """One parameter, two module paths: `named_parameters()` lists it once, the
    trainer may send it under the other name."""
    embed = nn.Module()
    embed.weight = _param(8, HIDDEN)
    head = nn.Module()
    head.weight = embed.weight
    model = nn.Module()
    model.embed = embed
    model.head = head
    runner = _Runner(model)

    _reload(runner, [{"head.weight": torch.ones(8, HIDDEN, dtype=torch.bfloat16)}])
    assert embed.weight.eq(1.0).all()


# ── FP8: only an actual write counts ───────────────────────────────────────


def _fp8_model():
    layer = nn.Module()
    layer.weight = _param(4, HIDDEN, dtype=torch.float8_e4m3fn)
    layer.weight_scale = nn.Parameter(torch.ones(1), requires_grad=False)
    model = nn.Module()
    model.layer = layer
    return model


def test_an_fp8_weight_requantisation_did_not_write_is_not_credited(monkeypatch):
    """The requantiser returns without writing on a shape it cannot shard or a
    quant type it does not know. Counting that as written let a reload commit
    with the old FP8 weight -- and credited its scale too."""
    monkeypatch.setattr(
        WeightUpdaterMixin, "_requantize_fp8_weight", lambda self, *a: False
    )
    runner = _Runner(_fp8_model())
    incoming = {"layer.weight": torch.ones(4, HIDDEN, dtype=torch.bfloat16)}

    with pytest.raises(RuntimeError, match=r"skipped=\['layer\.weight'\]"):
        _reload(runner, [incoming])


def test_a_requantised_fp8_weight_credits_its_scale_too(monkeypatch):
    """Scales are derived, never sent, so without this they read as missing."""
    monkeypatch.setattr(
        WeightUpdaterMixin, "_requantize_fp8_weight", lambda self, *a: True
    )
    runner = _Runner(_fp8_model())
    incoming = {"layer.weight": torch.ones(4, HIDDEN, dtype=torch.bfloat16)}

    assert _reload(runner, [incoming])["loaded_internal"] == 2


# ── a reload never inherits another's state ────────────────────────────────


def test_a_weight_sent_twice_is_refused():
    runner = _Runner(_dense_model())
    ckpt = _dense_checkpoint()
    runner.begin_weight_update(1)
    runner.apply_weight_bucket([("attn.o_proj.weight", ckpt["attn.o_proj.weight"])])

    with pytest.raises(RuntimeError, match="sent twice"):
        runner.apply_weight_bucket([("attn.o_proj.weight", ckpt["attn.o_proj.weight"])])
    assert runner.get_weight_update_status()["in_progress"] is None


def test_abort_drops_the_expert_slices_it_registered():
    """Left behind, a later reload relays them out again, or fails on the shards
    the aborted stream never sent -- fencing a reload that did nothing wrong."""
    runner = _Runner(_moe_model())
    half = {
        k: v for k, v in _expert_checkpoint(experts=[0]).items() if "gate_proj" in k
    }
    runner.begin_weight_update(1)
    runner.apply_weight_bucket(list(half.items()))
    assert runner._pending_expert_relayout, "the per-expert write registers a slice"

    runner.abort_weight_update(1, RuntimeError("stream failed"))

    assert not runner._pending_expert_relayout
    # The next full reload starts clean and commits.
    _reload(runner, [_expert_checkpoint()], version=2)
    assert runner.get_weight_update_status()["healthy"] is True


def test_begin_drops_bookkeeping_an_earlier_sync_left(caplog):
    """Commit credits expert buffers from this bookkeeping, so a leftover entry
    would count as written by a reload that never sent it."""
    model = _moe_model()
    runner = _Runner(model)
    runner._pending_expert_relayout[(model.mlp.experts, "w13_weight")] = {
        e: {"w1", "w3"} for e in range(EXPERTS)
    }

    with caplog.at_level(logging.WARNING, logger="atom"):
        runner.begin_weight_update(1)

    assert not runner._pending_expert_relayout
    assert "earlier sync left unfinished" in caplog.text
    ckpt = {k: v for k, v in _expert_checkpoint().items() if "down_proj" in k}
    runner.apply_weight_bucket(list(ckpt.items()))
    with pytest.raises(RuntimeError, match="mlp.experts.w13_weight"):
        runner.commit_weight_update(1)


def test_a_second_abort_keeps_the_first_reason():
    """The failing step aborts before its caller sees the error, and the caller
    aborts again on the way out; the first reason is the cause."""
    runner = _Runner(_dense_model())
    runner.begin_weight_update(1)
    runner.abort_weight_update(1, RuntimeError("bucket 3 rejected"))
    runner.abort_weight_update(1, RuntimeError("outer"))
    assert runner.get_weight_update_status()["failure"] == "bucket 3 rejected"


def test_an_abort_after_commit_still_fences():
    """The receiver synchronizes the device after commit, so a fault found
    there aborts a reload that commit had already declared good."""
    runner = _Runner(_dense_model())
    _reload(runner, [_dense_checkpoint()])
    runner.abort_weight_update(1, RuntimeError("device fault"))

    with pytest.raises(RuntimeError, match="fenced"):
        runner.assert_weight_update_ready()
    assert runner.get_weight_update_status()["failure"] == "device fault"


def test_a_partial_update_over_a_committed_version_is_refused():
    """What it leaves out would keep serving the last version beside the new
    one, which is the mixed model the transaction exists to rule out -- so
    commit offers no way to skip the check."""
    runner = _Runner(_dense_model())
    _reload(runner, [_dense_checkpoint()])
    update = {"attn.o_proj.weight": _dense_checkpoint(fill=2.0)["attn.o_proj.weight"]}

    with pytest.raises(RuntimeError, match=r"missing=\['attn\.qkv_proj\.weight'\]"):
        _reload(runner, [update], version=2)
    with pytest.raises(TypeError):
        runner.commit_weight_update(2, verify_full_load=False)


def test_the_non_transactional_path_keeps_no_coverage():
    """update_weights shares the application loop; outside a reload there is no
    transaction to credit, and nothing may accumulate."""
    runner = _Runner(_dense_model())
    runner.update_weights(list(_dense_checkpoint().items()))
    assert not getattr(runner, "_weight_update_written", set())
    assert runner.model.attn.qkv_proj.weight.eq(1.0).all()


# ── recovering through the non-transactional paths ────────────────────────


def _fenced(runner):
    runner.begin_weight_update(1)
    runner.abort_weight_update(1, RuntimeError("stream failed"))
    with pytest.raises(RuntimeError, match="fenced"):
        runner.assert_weight_update_ready()
    return runner


def test_a_direct_full_reload_lifts_the_fence_an_aborted_stream_left():
    """Only a commit took the fence down, so recovering through the direct,
    SHM or IPC path left every later forward refused."""
    runner = _fenced(_Runner(_dense_model()))
    runner.update_weights(list(_dense_checkpoint().items()))
    runner.assert_weight_update_ready()
    assert runner.get_weight_update_status()["healthy"] is True


def test_an_empty_reload_does_not_lift_it():
    """Those paths end on the caller's word, so finishing one was enough: a
    reload that sent nothing lifted the fence over all the failed stream left."""
    runner = _fenced(_Runner(_dense_model()))
    runner.update_weights([])
    with pytest.raises(RuntimeError, match="fenced"):
        runner.assert_weight_update_ready()


def test_a_partial_reload_lifts_it_only_once_the_rest_arrives():
    runner = _fenced(_Runner(_dense_model()))
    ckpt = _dense_checkpoint()
    runner.update_weights([("attn.o_proj.weight", ckpt.pop("attn.o_proj.weight"))])
    with pytest.raises(RuntimeError, match="fenced"):
        runner.assert_weight_update_ready()

    runner.update_weights(list(ckpt.items()))
    runner.assert_weight_update_ready()


def test_what_the_failed_stream_wrote_is_no_part_of_the_recovery():
    """It rewrote o_proj before failing; a recovery that sends everything else
    has still not rewritten the model."""
    runner = _Runner(_dense_model())
    ckpt = _dense_checkpoint()
    runner.begin_weight_update(1)
    runner.apply_weight_bucket([("attn.o_proj.weight", ckpt.pop("attn.o_proj.weight"))])
    runner.abort_weight_update(1, RuntimeError("stream failed"))

    runner.update_weights(list(ckpt.items()))

    with pytest.raises(RuntimeError, match="fenced"):
        runner.assert_weight_update_ready()


def test_experts_rewritten_over_several_reloads_add_up():
    """Each reload relays out what it sent, so a buffer is credited expert by
    expert; counted whole, one whose experts came in two reloads never was."""
    runner = _fenced(_Runner(_moe_model()))
    runner.update_weights(list(_expert_checkpoint(experts=[0]).items()))
    with pytest.raises(RuntimeError, match="fenced"):
        runner.assert_weight_update_ready()

    runner.update_weights(list(_expert_checkpoint(experts=[1]).items()))
    runner.assert_weight_update_ready()


def _staged(checkpoint):
    """Laid out as the SHM and IPC senders lay it out: one flat byte buffer,
    and where each tensor sits in it."""
    meta, blobs, offset = {}, [], 0
    for name, tensor in checkpoint.items():
        raw = tensor.contiguous().view(torch.uint8).reshape(-1)
        meta[name] = {
            "shape": tuple(tensor.shape),
            "dtype": str(tensor.dtype),
            "offset": offset,
            "nbytes": raw.numel(),
        }
        blobs.append(raw)
        offset += raw.numel()
    return torch.cat(blobs), meta


@contextlib.contextmanager
def _shm_segment(flat):
    from multiprocessing import shared_memory

    shm = shared_memory.SharedMemory(create=True, size=flat.numel())
    try:
        shm.buf[: flat.numel()] = flat.numpy().tobytes()
        yield shm.name
    finally:
        shm.close()
        shm.unlink()


def test_the_shm_path_lifts_it_on_its_last_bucket_only():
    runner = _fenced(_Runner(_dense_model()))
    flat, meta = _staged(_dense_checkpoint())
    names = list(meta)
    first = {n: meta[n] for n in names[:2]}
    rest = {n: meta[n] for n in names[2:]}
    with _shm_segment(flat) as shm_name:
        runner.update_weights_from_shm(shm_name, first, is_last=False)
        with pytest.raises(RuntimeError, match="fenced"):
            runner.assert_weight_update_ready()
        runner.update_weights_from_shm(shm_name, rest, is_last=True)
    runner.assert_weight_update_ready()


def _legacy_reload(runner, path, checkpoint):
    """One SHM or IPC reload of *checkpoint*, sent as its last bucket."""
    flat, meta = _staged(checkpoint)
    if path == "shm":
        with _shm_segment(flat) as shm_name:
            return runner.update_weights_from_shm(shm_name, meta)
    runner._ipc_buffer = flat  # as mapped from the sender's handle
    # Releasing the sender's mapping is not under test, and on a CPU-only torch
    # it raises an AssertionError the updater does not catch.
    with mock.patch.object(torch.cuda, "ipc_collect", lambda: None):
        return runner.update_weights_from_ipc(None, meta)


@pytest.mark.parametrize("path", ["shm", "ipc"])
def test_a_full_shm_or_ipc_reload_lifts_it(path):
    runner = _fenced(_Runner(_dense_model()))
    _legacy_reload(runner, path, _dense_checkpoint())
    runner.assert_weight_update_ready()


@pytest.mark.parametrize("path", ["shm", "ipc"])
def test_a_partial_shm_or_ipc_reload_keeps_it(path):
    """Its last bucket was all it took, however little the reload held."""
    runner = _fenced(_Runner(_dense_model()))
    ckpt = _dense_checkpoint()
    _legacy_reload(runner, path, {"attn.o_proj.weight": ckpt["attn.o_proj.weight"]})
    with pytest.raises(RuntimeError, match="fenced"):
        runner.assert_weight_update_ready()


@pytest.mark.parametrize("path", ["shm", "ipc"])
def test_a_requantisation_that_did_not_write_is_not_counted(monkeypatch, path):
    """Counted as updated regardless, a reload that wrote nothing reported
    success and lifted the fence over the old weights."""
    monkeypatch.setattr(
        WeightUpdaterMixin, "_requantize_fp8_weight", lambda self, *a: False
    )
    runner = _fenced(_Runner(_fp8_model()))
    incoming = {"layer.weight": torch.ones(4, HIDDEN, dtype=torch.bfloat16)}

    assert _legacy_reload(runner, path, incoming) == 0
    with pytest.raises(RuntimeError, match="fenced"):
        runner.assert_weight_update_ready()


def test_a_legacy_reload_that_did_not_complete_keeps_the_fence():
    """A fused FP8 parameter still waiting on a shard was never rewritten, so
    finishing the call is not the same as finishing the reload."""
    model = _dense_model()
    qkv = model.attn.qkv_proj
    qkv.weight = _param(8, HIDDEN, dtype=torch.float8_e4m3fn)
    qkv.weight_scale = nn.Parameter(torch.ones(1), requires_grad=False)
    runner = _fenced(_Runner(model))
    ckpt = _dense_checkpoint()
    del ckpt["attn.v_proj.weight"]

    runner.update_weights(list(ckpt.items()))

    with pytest.raises(RuntimeError, match="fenced"):
        runner.assert_weight_update_ready()
