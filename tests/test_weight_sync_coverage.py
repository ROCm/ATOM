# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""What a weight sync wrote, the same on the direct, SHM and IPC paths.

The engine fences serving from a sync's first update to its FINISH. Once a
sync has failed or been abandoned the weights are not whole, and the engine
says so with ``require_full``: then a sync completes only if it rewrote every
parameter. Coverage is by the parameter written, not the name sent, since ATOM
fuses several checkpoint tensors into one parameter.
"""

import contextlib
from types import SimpleNamespace
from unittest import mock

import pytest
import torch
from torch import nn

from atom.rollout.capabilities import CapabilityProviderMixin
from atom.rollout.weight_sync_record import WeightSyncRecord
from atom.rollout.weight_updater import WeightUpdaterMixin

HIDDEN = 4
EXPERTS = 2
INTERMEDIATE = 2
PATHS = ["direct", "shm", "ipc"]


class _Runner(WeightUpdaterMixin):
    device = torch.device("cpu")
    label = "test"
    rank = 0
    world_size = 1

    def __init__(self, model, config=None):
        from atom.utils.selector import attn_family

        self._init_weight_updater()
        self.model = model
        self.config = config
        # As ModelRunner declares it, from the config.
        self.attn_family = attn_family(
            getattr(config, "hf_text_config", None) or SimpleNamespace()
        )
        self.kv_cleared = 0

    def clear_kv_cache(self):
        self.kv_cleared += 1

    def _sync_target_model(self):
        # No wrapper to peel; the real lookup imports the TBO package.
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


def _dense_checkpoint(fill=1.0, names=None):
    full = {
        "attn.q_proj.weight": torch.full((4, HIDDEN), fill, dtype=torch.bfloat16),
        "attn.k_proj.weight": torch.full((2, HIDDEN), fill, dtype=torch.bfloat16),
        "attn.v_proj.weight": torch.full((2, HIDDEN), fill, dtype=torch.bfloat16),
        "attn.o_proj.weight": torch.full((HIDDEN, 4), fill, dtype=torch.bfloat16),
    }
    if names is None:
        return full
    return {f"attn.{n}.weight": full[f"attn.{n}.weight"] for n in names}


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


def _fp8_model(*, input_scale=False):
    layer = nn.Module()
    layer.weight = _param(4, HIDDEN, dtype=torch.float8_e4m3fn)
    layer.weight_scale = nn.Parameter(torch.ones(1), requires_grad=False)
    layer.quant_type = SimpleNamespace(name="per_Tensor")
    if input_scale:
        layer.input_scale = nn.Parameter(torch.ones(1), requires_grad=False)
    model = nn.Module()
    model.layer = layer
    return model


@pytest.fixture(autouse=True)
def _cpu_relayout(monkeypatch):
    """The real relayout imports aiter; what matters here is that it consumes
    the bookkeeping, as the real one does."""
    monkeypatch.setattr(
        WeightUpdaterMixin,
        "_finalize_expert_weight_sync",
        lambda self: self._expert_relayout_pending.clear(),
    )


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


def _sync(runner, path, buckets, *, require_full=False):
    """One sync of *buckets*, each a checkpoint dict; the direct path sends
    each bucket as its own update, so as a sync of its own."""
    for i, bucket in enumerate(buckets):
        last = i == len(buckets) - 1
        if path == "direct":
            runner.update_weights(list(bucket.items()), require_full=require_full)
            continue
        flat, meta = _staged(bucket)
        if path == "shm":
            with _shm_segment(flat) as shm_name:
                runner.update_weights_from_shm(shm_name, meta, last, require_full)
            continue
        runner._ipc_buffer = flat  # as mapped from the sender's handle
        # Releasing the sender's mapping is not under test, and on a CPU-only
        # torch it raises an AssertionError the updater does not catch.
        with mock.patch.object(torch.cuda, "ipc_collect", lambda: None):
            runner.update_weights_from_ipc(None, meta, last, None, require_full)


# ── the one lift rule: full coverage where the weights are not whole ───────


@pytest.mark.parametrize("path", PATHS)
def test_a_full_sync_completes_where_the_weights_are_not_whole(path):
    runner = _Runner(_dense_model())
    _sync(runner, path, [_dense_checkpoint()], require_full=True)
    assert runner.model.attn.qkv_proj.weight.eq(1.0).all()
    assert runner._sync is None


@pytest.mark.parametrize("path", PATHS)
def test_a_partial_sync_cannot_make_the_weights_whole_again(path):
    """An SHM sync failed halfway; the next sent one tensor, its FINISH took
    the fence down, and half the model served at each version."""
    runner = _Runner(_dense_model())
    with pytest.raises(RuntimeError, match=r"not whole.*'attn\.qkv_proj\.weight'"):
        _sync(runner, path, [_dense_checkpoint(names=["o_proj"])], require_full=True)
    assert runner._sync is None


@pytest.mark.parametrize("path", PATHS)
def test_a_parameter_no_checkpoint_holds_is_not_owed_by_a_full_sync(path):
    """ATOM derives it at init, so no trainer sends it, and after one failed
    sync every recovery reported it left as it was: the engine stayed fenced."""
    model = _dense_model()
    model.attn.derived = _param(2)
    runner = _Runner(model)
    runner._checkpoint_param_names = frozenset(
        {"attn.qkv_proj.weight", "attn.o_proj.weight"}
    )

    _sync(runner, path, [_dense_checkpoint()], require_full=True)

    assert runner._sync is None
    with pytest.raises(RuntimeError, match=r"not whole.*'attn\.qkv_proj\.weight'"):
        _sync(runner, path, [_dense_checkpoint(names=["o_proj"])], require_full=True)


def test_a_tied_parameter_is_owed_under_the_name_it_was_loaded_by():
    model = _dense_model()
    model.head = nn.Module()
    model.head.weight = model.attn.o_proj.weight
    runner = _Runner(model)
    runner._checkpoint_param_names = frozenset({"attn.qkv_proj.weight", "head.weight"})

    with pytest.raises(RuntimeError, match=r"not whole.*'attn\.o_proj\.weight'"):
        _sync(
            runner,
            "direct",
            [_dense_checkpoint(names=["q_proj", "k_proj", "v_proj"])],
            require_full=True,
        )


@pytest.mark.parametrize("path", PATHS)
def test_a_partial_sync_completes_where_the_weights_are_whole(path):
    runner = _Runner(_dense_model())
    _sync(runner, path, [_dense_checkpoint(names=["o_proj"])])
    assert runner.model.attn.o_proj.weight.eq(1.0).all()


def test_a_fused_parameter_split_over_direct_updates_is_not_refused():
    """A trainer splitting a bf16 sync by size sent q_proj in one call and
    k/v_proj in the next. That worked on main, then every call failed with
    "fused parameter(s) got only some of their shards"."""
    runner = _Runner(_dense_model())
    ckpt = _dense_checkpoint()
    runner.update_weights([("attn.q_proj.weight", ckpt["attn.q_proj.weight"])])
    runner.update_weights(
        [(n, ckpt[n]) for n in ("attn.k_proj.weight", "attn.v_proj.weight")]
    )
    assert runner.model.attn.qkv_proj.weight.eq(1.0).all()


@pytest.mark.parametrize("path", ["shm", "ipc"])
def test_a_fused_parameter_counts_once_its_shards_land_across_buckets(path):
    runner = _Runner(_dense_model())
    ckpt = _dense_checkpoint()
    names = list(ckpt)
    buckets = [{n: ckpt[n] for n in names[:2]}, {n: ckpt[n] for n in names[2:]}]
    _sync(runner, path, buckets, require_full=True)


@pytest.mark.parametrize("path", PATHS)
def test_a_fused_parameter_missing_a_shard_is_not_rewritten(path):
    runner = _Runner(_dense_model())
    with pytest.raises(RuntimeError, match=r"'attn\.qkv_proj\.weight'"):
        _sync(
            runner,
            path,
            [_dense_checkpoint(names=["q_proj", "k_proj", "o_proj"])],
            require_full=True,
        )


def test_every_expert_counts_before_the_fused_buffer_does():
    _sync(_Runner(_moe_model()), "direct", [_expert_checkpoint()], require_full=True)


def test_an_expert_nobody_sent_leaves_its_buffer_unwritten():
    runner = _Runner(_moe_model())
    with pytest.raises(RuntimeError, match=r"w13_weight.*w2_weight"):
        _sync(runner, "direct", [_expert_checkpoint(experts=[0])], require_full=True)


def test_a_tied_parameter_is_covered_under_either_name():
    """One parameter, two module paths: ``named_parameters()`` lists it once,
    and the trainer may send it under the other name."""
    embed = nn.Module()
    embed.weight = _param(8, HIDDEN)
    head = nn.Module()
    head.weight = embed.weight
    model = nn.Module()
    model.embed = embed
    model.head = head
    incoming = {"head.weight": torch.ones(8, HIDDEN, dtype=torch.bfloat16)}

    _sync(_Runner(model), "direct", [incoming], require_full=True)
    assert embed.weight.eq(1.0).all()


class _WritesItsWeight(nn.Module):
    """A module whose ``weight`` only marks where its values live: it writes a
    sync's tensors itself (``sync_weight``), whole or a shard at a time."""

    def __init__(self, dtype=torch.bfloat16):
        super().__init__()
        self.weight = _param(4, HIDDEN, dtype=dtype)

    def sync_weight(self, param_name, tensor, shard_id=None):
        if param_name != "weight":
            return False
        start = 0 if shard_id is None else 2 * shard_id
        self.weight.data[start : start + tensor.shape[0]].copy_(tensor)
        return True


def _self_writing_model(dtype=torch.bfloat16):
    model = nn.Module()
    model.emb = nn.Module()
    model.emb.table = _WritesItsWeight(dtype)
    model.packed_modules_mapping = {f".part_{i}.": (".table.", i) for i in range(2)}
    return model


@pytest.mark.parametrize("shards", [False, True])
@pytest.mark.parametrize("path", PATHS)
def test_a_weight_its_module_writes_counts_toward_a_full_sync(path, shards):
    runner = _Runner(_self_writing_model())
    new = torch.ones(4, HIDDEN, dtype=torch.bfloat16)
    if shards:
        bucket = {f"emb.part_{i}.weight": new[2 * i : 2 * i + 2] for i in range(2)}
    else:
        bucket = {"emb.table.weight": new}
    _sync(runner, path, [bucket], require_full=True)
    assert runner.model.emb.table.weight.eq(1.0).all()
    assert runner._sync is None


def test_a_module_that_writes_its_weights_knows_their_format():
    """An FP8 weight of no scheme the updater requantises is refused, unless
    its module writes it itself."""
    plain = nn.Module()
    plain.table = nn.Module()
    plain.table.weight = _param(4, HIDDEN, dtype=torch.float8_e4m3fn)
    assert "table.weight" in _Runner(plain)._reload_refusal()
    assert _Runner(_self_writing_model(torch.float8_e4m3fn))._reload_refusal() is None


@pytest.mark.parametrize(
    ("dtype", "refused"),
    [
        (torch.int64, False),
        (torch.int32, True),
        (torch.int8, True),
        (torch.uint8, True),
    ],
)
def test_an_int64_constant_does_not_stop_a_reload(dtype, refused):
    """A hash prime or an offset held as a parameter is no quantised weight;
    int32 is how GPTQ packs one, int8 and uint8 how others store theirs."""
    model = nn.Module()
    model.ple = nn.Module()
    model.ple.offsets = nn.Parameter(torch.zeros(4, dtype=dtype), requires_grad=False)
    refusal = _Runner(model)._reload_refusal()
    assert (refusal is not None and "ple.offsets" in refusal) == refused


def test_a_failed_syncs_writes_count_for_no_later_one():
    """What a failed sync wrote may be another version than what completes
    the rewrite after it."""
    runner = _Runner(_dense_model())
    ckpt = _dense_checkpoint()
    with pytest.raises(RuntimeError, match="could not be written"):
        runner.update_weights(
            [
                ("attn.q_proj.weight", ckpt["attn.q_proj.weight"]),
                ("attn.k_proj.weight", ckpt["attn.k_proj.weight"]),
                ("attn.o_proj.weight", torch.ones(3, 3, dtype=torch.bfloat16)),
            ]
        )
    with pytest.raises(RuntimeError, match=r"'attn\.qkv_proj\.weight'"):
        runner.update_weights(
            [(n, ckpt[n]) for n in ("attn.v_proj.weight", "attn.o_proj.weight")],
            require_full=True,
        )


# ── what counts as written ─────────────────────────────────────────────────


@pytest.mark.parametrize("path", PATHS)
def test_an_fp8_weight_requantisation_did_not_write_fails_the_update(monkeypatch, path):
    """Counted as updated, a sync that wrote nothing reported success over
    the old FP8 weight."""
    monkeypatch.setattr(
        WeightUpdaterMixin, "_requantize_fp8_weight", lambda self, *a: False
    )
    runner = _Runner(_fp8_model())
    incoming = {"layer.weight": torch.ones(4, HIDDEN, dtype=torch.bfloat16)}
    with pytest.raises(RuntimeError, match=r"could not be written.*'layer\.weight'"):
        _sync(runner, path, [incoming])


def _fp8_attention_model():
    model = _dense_model()
    qkv = model.attn.qkv_proj
    qkv.weight = _param(8, HIDDEN, dtype=torch.float8_e4m3fn)
    qkv.weight_scale = nn.Parameter(torch.ones(1), requires_grad=False)
    qkv.quant_type = SimpleNamespace(name="per_1x128")
    return model


@pytest.mark.parametrize("path", PATHS)
def test_a_fused_fp8_parameter_missing_a_shard_fails_the_update(path):
    """It is composed only once every shard has come, so it was not written at
    all; a warning reported the update a success over the old weight."""
    runner = _Runner(_fp8_attention_model())
    with pytest.raises(RuntimeError, match=r"none was written.*attn\.qkv_proj"):
        _sync(runner, path, [_dense_checkpoint(names=["q_proj", "k_proj"])])
    assert runner._packed_weight_accum == {}


def test_a_requantised_weight_rewrites_its_scale_too(monkeypatch):
    """No sender sends the scale requantisation derives."""
    monkeypatch.setattr(
        WeightUpdaterMixin, "_requantize_fp8_weight", lambda self, *a: True
    )
    incoming = {"layer.weight": torch.ones(4, HIDDEN, dtype=torch.bfloat16)}
    _sync(_Runner(_fp8_model()), "direct", [incoming], require_full=True)


def test_a_static_activation_scale_is_not_waited_for(monkeypatch):
    """Calibrated, not trained: no trainer sends one, so counting it left the
    fence up for good after any failed sync."""
    monkeypatch.setattr(
        WeightUpdaterMixin, "_requantize_fp8_weight", lambda self, *a: True
    )
    incoming = {"layer.weight": torch.ones(4, HIDDEN, dtype=torch.bfloat16)}
    runner = _Runner(_fp8_model(input_scale=True))
    _sync(runner, "direct", [incoming], require_full=True)


@pytest.mark.parametrize("path", ["shm", "ipc"])
def test_a_weight_sent_twice_in_one_sync_is_refused(path):
    """The second copy would silently replace the first."""
    runner = _Runner(_dense_model())
    ckpt = _dense_checkpoint(names=["o_proj"])
    with pytest.raises(RuntimeError, match="sent twice"):
        _sync(runner, path, [ckpt, ckpt])
    assert runner._sync is None


@pytest.mark.parametrize("path", PATHS)
def test_a_prequantised_scale_is_refused_not_ignored(path):
    """Counted among the ignored derived scales, it left the FP8 weight read
    with its old scale."""
    runner = _Runner(_fp8_model())
    incoming = {"layer.weight_scale_inv": torch.ones(1, dtype=torch.bfloat16)}
    with pytest.raises(ValueError, match="prequantised"):
        _sync(runner, path, [incoming])


def test_a_name_the_model_has_no_parameter_for_is_only_skipped():
    """An MTP head or a vision tower the rollout does not build."""
    runner = _Runner(_dense_model())
    ckpt = {**_dense_checkpoint(), "mtp.head.weight": torch.ones(2)}
    _sync(runner, "direct", [ckpt], require_full=True)


def test_the_writes_land_before_the_sync_completes(monkeypatch):
    """With clear_kv_cache=False nothing waited for them, and the forward
    FINISH lets run next may be on another stream."""
    calls = []
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda d=None: calls.append(d))
    runner = _Runner(_dense_model())
    runner.device = torch.device("cuda", 1)
    monkeypatch.setattr(
        runner,
        "_apply_named_tensors",
        lambda _: {"updated": 0, "skipped": 0, "ignored_scales": 0, "unwritten": []},
    )

    runner.update_weights([], clear_kv_cache=False)

    assert calls == [torch.device("cuda", 1)]
    assert runner.kv_cleared == 0


# ── a model whose weights a reload cannot rewrite is refused up front ──────


# A latent rank is what makes a config MLA.
_MLA_CONFIG = SimpleNamespace(hf_text_config=SimpleNamespace(kv_lora_rank=512))


def _mla_model():
    attn = nn.Module()
    attn.kv_b_proj = nn.Linear(HIDDEN, HIDDEN, bias=False)
    model = nn.Module()
    model.attn = attn
    return model


def _int8_model():
    layer = nn.Module()
    layer.weight = nn.Parameter(
        torch.zeros(4, HIDDEN, dtype=torch.int8), requires_grad=False
    )
    model = nn.Module()
    model.layer = layer
    return model


def _expert_parallel_moe():
    """Refused at its first expert tensor, after the dense ones ahead of it
    were written."""
    model = _dense_model()
    model.experts = nn.Module()
    model.experts.expert_map = torch.tensor([0, -1], dtype=torch.int32)
    return model


def _redundant_expert_moe():
    model = _dense_model()
    model.experts = nn.Module()
    model.experts.num_redundant_experts = 2
    return model


_unreloadable = pytest.mark.parametrize(
    ("model", "config", "reason"),
    [
        (_mla_model, _MLA_CONFIG, "MLA"),
        (_dense_model, SimpleNamespace(prefill_context_parallel_size=2), "prefill"),
        (_int8_model, None, "format a full-precision write cannot produce"),
        (_expert_parallel_moe, None, "expert-parallel"),
        (_redundant_expert_moe, None, "redundant expert"),
    ],
    ids=["mla", "pcp", "int8", "ep", "redundant-experts"],
)


@pytest.mark.parametrize("path", PATHS)
@_unreloadable
def test_a_model_a_reload_cannot_rewrite_is_refused_on_every_path(
    path, model, config, reason
):
    """Refused on the RDMA path alone, SHM, IPC and direct updates served a
    new kv_b_proj with the W_K/W_V derived from the old one at load."""
    runner = _Runner(model(), config)
    before = [p.clone() for p in runner.model.parameters()]
    with pytest.raises(RuntimeError, match=reason):
        _sync(runner, path, [{"anything.weight": torch.ones(2)}])
    assert all(p.equal(b) for p, b in zip(runner.model.parameters(), before))
    assert runner._sync is None


class _ReportingRunner(CapabilityProviderMixin, _Runner):
    """A runner that also says what it supports, as ModelRunner does."""


@_unreloadable
def test_a_model_a_reload_cannot_rewrite_reports_why(model, config, reason):
    report = _ReportingRunner(model(), config).get_worker_capabilities()
    assert "weight_reload" not in report["features"]
    assert reason in report["unavailable"]["weight_reload"]


def test_mla_is_what_the_config_declares_not_what_a_module_looks_like():
    """Inferred from module attributes, an MLA layer whose projection is
    named otherwise passed for MHA."""
    assert "MLA" in _Runner(_dense_model(), _MLA_CONFIG)._reload_refusal()


def test_a_model_a_reload_can_rewrite_reports_it():
    report = _ReportingRunner(_dense_model()).get_worker_capabilities()
    assert "weight_reload" in report["features"]
    assert report["unavailable"] == {}


# ── the engine and a runner together ───────────────────────────────────────


class _InProcessRunners:
    """AsyncIOProcManager with one real runner behind it."""

    proc_num = 1

    def __init__(self, runner):
        self.runner = runner

    def utility_rpc(self, method, payload, timeout=300.0):
        from atom.model_engine.collective_rpc import RpcResult, WeightReloadRefused

        try:
            value = getattr(self.runner, method)(*payload.args, **payload.call_kwargs())
        except Exception as exc:  # noqa: BLE001 - answered, as a worker does
            return [
                RpcResult(
                    payload.request_id,
                    0,
                    error=f"{type(exc).__name__}: {exc}",
                    refused_before_writing=isinstance(exc, WeightReloadRefused),
                )
            ]
        return [RpcResult(payload.request_id, 0, value=value)]


class _UnansweredQuery(_InProcessRunners):
    """The same, except that what the runner supports cannot be learnt: the
    query times out, or a rank answers it with an error."""

    def __init__(self, runner, how):
        super().__init__(runner)
        self.how = how

    def utility_rpc(self, method, payload, timeout=300.0):
        from atom.model_engine.collective_rpc import RpcResult

        if method != "get_worker_capabilities":
            return super().utility_rpc(method, payload, timeout)
        if self.how == "timeout":
            raise TimeoutError("no reply from TP rank 1")
        return [RpcResult(payload.request_id, 0, error="RuntimeError: rank 1 is down")]


class _Engine:
    def __init__(self):
        self._has_pending_utility = False
        self._is_rl_weights_offloaded = False
        self._weight_sync_fenced = False
        self._rl_weights_inconsistent = False

    def prepare_weight_sync(self, deadline):
        pass

    def prepare_kv_change(self, *, weights, deadline):
        pass


def _engine_with(runner, runners=None):
    """An engine over *runner*, which at startup asks a runner that can say
    what weight syncs it takes, as EngineCore does."""
    import queue

    from atom.model_engine.engine_utility import (
        EngineUtilityHandler,
        ask_what_runners_take,
    )

    out = queue.Queue()
    runners = runners or _InProcessRunners(runner)
    unavailable = None
    if callable(getattr(runner, "get_worker_capabilities", None)):
        unavailable = ask_what_runners_take(runners, "test")
    handler = EngineUtilityHandler(runners, out, unavailable=unavailable)
    engine = _Engine()

    def send(cmd, **args):
        engine._has_pending_utility = True
        pending = queue.Queue()
        pending.put_nowait((cmd, args))
        handler.process_queue(pending, engine)
        return out.get_nowait()[1]

    return engine, send


@_unreloadable
def test_the_engine_refuses_a_model_a_reload_cannot_rewrite_without_a_fence(
    model, config, reason
):
    """The runner refused before it wrote, but the engine had already put
    its fence up for the sync, and served nothing until another came."""
    runner = _ReportingRunner(model(), config)
    engine, send = _engine_with(runner)

    reply = send(
        "update_weights",
        sync_id="s1",
        request_id="a",
        named_tensors=[("anything.weight", torch.ones(2))],
    )

    assert reason in reply["error"]
    assert not (engine._weight_sync_fenced or engine._rl_weights_inconsistent)
    assert runner._sync is None


@pytest.mark.parametrize("how", ["timeout", "error"])
def test_runners_that_cannot_say_what_they_take_stop_the_engine_starting(how):
    """Asked on the way into every sync instead, a slow or failing answer cost
    each sync up to 30 s at its start, and then went unheeded without a word."""
    from atom.model_engine.engine_utility import ask_what_runners_take

    runner = _ReportingRunner(_dense_model())
    with pytest.raises((RuntimeError, TimeoutError), match="rank 1"):
        ask_what_runners_take(_UnansweredQuery(runner, how), "test")


def test_a_sync_asks_the_runners_nothing():
    """What they take was settled at startup."""
    asked = []

    class _Recording(_InProcessRunners):
        def utility_rpc(self, method, payload, timeout=300.0):
            asked.append(method)
            return super().utility_rpc(method, payload, timeout)

    runner = _ReportingRunner(_dense_model())
    _, send = _engine_with(runner, _Recording(runner))
    asked.clear()

    reply = send(
        "update_weights",
        sync_id="s1",
        request_id="a",
        named_tensors=list(_dense_checkpoint().items()),
    )

    assert "error" not in reply and "get_worker_capabilities" not in asked


@_unreloadable
def test_a_refusal_the_engine_was_not_told_of_fences_nothing(model, config, reason):
    """Not asked, the engine put its fence up and sent the update, which the
    runner refused before writing. DISCARD then found the fence up and kept
    it, and every later sync was refused the same way."""
    from atom.model_engine.collective_rpc import DISCARD_WEIGHT_SYNC_CMD

    runner = _Runner(model(), config)
    engine, send = _engine_with(runner)
    weights = [("anything.weight", torch.ones(2))]

    for request_id in ("a", "b"):
        reply = send(
            "update_weights", sync_id="s1", request_id=request_id, named_tensors=weights
        )
        assert reason in reply["error"]
        assert not (engine._weight_sync_fenced or engine._rl_weights_inconsistent)
    send(DISCARD_WEIGHT_SYNC_CMD, failed_cmd="update_weights", error=reply["error"])

    assert not (engine._weight_sync_fenced or engine._rl_weights_inconsistent)
    assert runner._sync is None


def test_a_refusal_the_engine_was_not_told_of_leaves_a_fence_that_was_up():
    """The engine was fenced for an earlier failed sync, and stays so."""
    runner = _Runner(_mla_model(), _MLA_CONFIG)
    engine, send = _engine_with(runner)
    engine._weight_sync_fenced = engine._rl_weights_inconsistent = True

    reply = send(
        "update_weights",
        sync_id="s1",
        request_id="a",
        named_tensors=[("anything.weight", torch.ones(2))],
    )

    assert "MLA" in reply["error"]
    assert engine._weight_sync_fenced and engine._rl_weights_inconsistent


def test_a_sync_after_one_refused_before_writing_finds_the_weights_whole(monkeypatch):
    """Only a sync abandoned between buckets, having written, leaves them not
    whole; one refused before writing is over, not abandoned."""
    runner = _Runner(_dense_model())
    engine, send = _engine_with(runner)
    monkeypatch.setattr(runner, "_reload_refusal", lambda: "stand-in refusal")
    reply = send(
        "update_weights",
        sync_id="s1",
        request_id="a",
        named_tensors=[("attn.o_proj.weight", torch.ones(3, 3))],
    )
    assert "stand-in refusal" in reply["error"]
    monkeypatch.undo()

    ckpt = _dense_checkpoint(fill=2.0)
    send(
        "update_weights", sync_id="s2", request_id="b", named_tensors=list(ckpt.items())
    )
    assert not engine._rl_weights_inconsistent
    send("finish_weight_sync", sync_id="s2", completed_request_id="b")
    assert not engine._weight_sync_fenced


@_unreloadable
def test_the_write_itself_refuses_what_a_reload_cannot_rewrite(model, config, reason):
    """A path that forgot to open its sync got an AttributeError from inside
    the loop, not the refusal."""
    runner = _Runner(model(), config)
    with pytest.raises(RuntimeError, match=reason):
        runner._apply_named_tensors([("anything.weight", torch.ones(2))])
    assert runner._sync is None


def test_the_write_opens_the_sync_it_belongs_to():
    runner = _Runner(_dense_model())
    runner._apply_named_tensors([])
    assert runner._sync is not None


def test_one_tensor_after_a_failed_sync_does_not_take_the_fence_down():
    """An SHM sync failed halfway, with half the model rewritten. The next sync
    sent one tensor through a direct update, its FINISH took the engine's
    fence down, and the engine served the two halves together."""
    runner = _Runner(_dense_model())
    engine, send = _engine_with(runner)
    ckpt = _dense_checkpoint(fill=2.0)
    flat, meta = _staged(
        {**ckpt, "attn.o_proj.weight": torch.ones(3, 3, dtype=torch.bfloat16)}
    )
    names = list(meta)
    with _shm_segment(flat) as shm_name:
        for request_id, bucket, last in (
            ("a", names[:2], False),
            ("b", names[2:], True),
        ):
            send(
                "update_weights_shm",
                sync_id="s1",
                request_id=request_id,
                shm_name=shm_name,
                bucket_meta={n: meta[n] for n in bucket},
                is_last=last,
            )
    assert engine._weight_sync_fenced and engine._rl_weights_inconsistent

    o_proj = [("attn.o_proj.weight", ckpt["attn.o_proj.weight"])]
    send("update_weights", sync_id="s2", request_id="c", named_tensors=o_proj)
    send("finish_weight_sync", sync_id="s2", completed_request_id="c")
    assert engine._weight_sync_fenced

    send(
        "update_weights", sync_id="s3", request_id="d", named_tensors=list(ckpt.items())
    )
    send("finish_weight_sync", sync_id="s3", completed_request_id="d")
    assert not engine._weight_sync_fenced and not engine._rl_weights_inconsistent


# ── the record itself ──────────────────────────────────────────────────────


def test_a_fused_parameter_is_covered_only_once_whole():
    fused = _param(2)
    record = WeightSyncRecord()
    record.credit_part(fused, "q", ("q", "k"))
    assert record.missing({"qkv": fused}) == ["qkv"]
    record.credit_part(fused, "k", ("q", "k"))
    assert record.missing({"qkv": fused}) == []
