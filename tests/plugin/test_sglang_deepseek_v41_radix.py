# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""CPU contract of the DeepSeek-V4.1 radix (prefix) cache on the SGLang plugin.

Replay first: a prefix hit at ``P`` recomputes ``[P - R, P)`` into its freshly
reset STATE slot inside the step's own forward (``deepseek_v41_radix``): the
hit's span starts at ``P - R``, which the bridge translates into ATOM's
``ScheduledBatch.replaying``. Covered here:

1. the seam: which rows replay, the record it plans, the stamps, and the
   worker / ForwardBatch / eager-view carriers;
2. the admission charge;
3. the refusal matrix, what ``prepare`` installs, and the launch policy
   (radix on by default, ``ATOM_SGLANG_V41_RADIX=0`` turning SGLang's off);
4. the record, the widened inputs and the output rows, the runtime hook;
5. the bridge: the widened spans, the stamp guard and stamp on consume, the
   decoder replay verdict, the staged inputs;
6. ATOM's cache and compress plans over a fused span;
7. the host-row invalidation that lets overlap run;
8. the GPU page-check tool (``v41_radix_page_check``);
9. admission: a 400 at the scheduler's queue door for positional embedding
   overrides;
10. DSpark under the replay: launch policy and refusals, verify rounds that
    never carry a replay, a DSpark prefill hit.
"""

import importlib.util
import pathlib
import sys
from array import array
from types import SimpleNamespace

import numpy as np
import pytest
import torch

pytest.importorskip("sglang", reason="the radix seam runs against SGLang 0.5.22")

from sglang.srt.model_executor.forward_batch_info import ForwardMode

from atom.plugin.sglang import deepseek_v41_bridge as bridge
from atom.plugin.sglang import deepseek_v41_radix as radix
from atom.plugin.sglang.deepseek_v41_bridge import (
    ATOM_DEEPSEEK_V41_SGLANG_BLOCK_SIZE,
    build_v41_scheduled_batch,
    refuse_unsupported_sglang_features,
    sglang_radix_cache_enabled,
)
from atom.plugin.sglang.deepseek_v41_radix import (
    FUSED_ATTR,
    REPLAY_ROWS,
    FusedReplay,
    ReplayHit,
    carry_fused_replay,
    check_state_stamps,
    expand_v41_forward_inputs,
    fused_layout,
    plan_prefix_replay,
    replay_admission_rows,
    run_v41_prefix_replay,
    run_with_fused_replay,
)

PAGE = ATOM_DEEPSEEK_V41_SGLANG_BLOCK_SIZE
R = REPLAY_ROWS
HISTORY = 3
EXTEND, DECODE, MIXED = ForwardMode.EXTEND, ForwardMode.DECODE, ForwardMode.MIXED


@pytest.fixture(autouse=True)
def _host_pinning(monkeypatch):
    # The replay stages its host layout from pinned memory; PyTorch can pin
    # only with a GPU, so a run without one allocates it pageable.
    if torch.cuda.is_available():
        return
    empty = torch.empty
    monkeypatch.setattr(
        torch, "empty", lambda *a, pin_memory=False, **k: empty(*a, **k)
    )


@pytest.fixture(autouse=True)
def _isolated_process_state(monkeypatch):
    # On by default; pinned so an operator's environment cannot change it.
    monkeypatch.setenv("ATOM_SGLANG_V41_RADIX", "1")
    # One builder and one install per process; no test may leak either.
    monkeypatch.setattr(bridge, "_V41_BUILDER", None)
    monkeypatch.setattr(radix, "_INSTALLED", False)
    monkeypatch.setattr(radix, "_get_builder", bridge.get_deepseek_v41_builder)
    monkeypatch.setattr(radix, "_LOGGED_FIRST_FUSED", False)
    monkeypatch.setattr(bridge, "_HOST_ROWS", bridge.HostPageRows())


@pytest.fixture
def radix_on(monkeypatch):
    """The replay installed in this process (radix cache on)."""
    monkeypatch.setattr(radix, "_INSTALLED", True)
    return R


def _req_to_token(rows, context_len, pages_by_row):
    table = torch.zeros(rows, context_len, dtype=torch.int32)
    for row, pages in pages_by_row.items():
        for i, page in enumerate(pages):
            start = i * PAGE
            width = min(PAGE, context_len - start)
            table[row, start : start + width] = page * PAGE + torch.arange(width)
    return table


@pytest.fixture
def req_to_token(monkeypatch):
    table = _req_to_token(
        rows=6,
        context_len=4 * PAGE,
        pages_by_row={1: [7, 3, 9], 2: [5, 6], 3: [2, 11, 4, 8], 4: [10, 12]},
    )
    monkeypatch.setattr(bridge, "_req_to_token", lambda: table)
    return table


def _extend_fb(prefix, lengths, slots, mode=EXTEND):
    """An SGLang EXTEND ForwardBatch as the bridge reads it."""
    bs = len(slots)
    ends = [p + n for p, n in zip(prefix, lengths)]
    return SimpleNamespace(
        forward_mode=mode,
        rids=[f"r{i}" for i in range(bs)],
        batch_size=bs,
        seq_lens=torch.tensor(ends, dtype=torch.int64),
        seq_lens_cpu=torch.tensor(ends, dtype=torch.int64),
        req_pool_indices=torch.tensor(slots, dtype=torch.int64),
        req_pool_indices_cpu=torch.tensor(slots, dtype=torch.int64),
        extend_seq_lens=None,
        extend_prefix_lens=None,
        extend_seq_lens_cpu=list(lengths),
        extend_prefix_lens_cpu=list(prefix),
        is_prefill_only=False,
        capture_hidden_mode=None,
        return_logprob=False,
        extend_logprob_start_lens_cpu=None,
    )


def _carry(fb, fused):
    """``fb`` carrying ``fused``, as the model's forward leaves it (widened)."""
    fused.expanded = "widened"
    setattr(fb, FUSED_ATTR, fused)
    return fb


@pytest.fixture
def stamps(monkeypatch):
    """The pools' stamps, as the bridge reads them off the forward context."""
    state_gen = np.zeros(8, dtype=np.int64)
    generation = torch.ones(8, dtype=torch.int64)
    monkeypatch.setattr(bridge, "_radix_stamps", lambda: (state_gen, generation))
    return state_gen, generation


def _decode_fb(slot=1, seq_len=PAGE + 1):
    return SimpleNamespace(
        forward_mode=DECODE,
        rids=["a"],
        batch_size=1,
        seq_lens=torch.tensor([seq_len]),
        seq_lens_cpu=torch.tensor([seq_len]),
        req_pool_indices=torch.tensor([slot]),
        req_pool_indices_cpu=None,
    )


def _spans(batch):
    from atom.model_ops.attentions.deepseek_v41.metadata import RequestSpan

    spans, offset = [], 0
    for rid, length, end, slot in zip(
        batch.req_ids,
        batch.num_scheduled_tokens,
        batch.context_lens,
        batch.state_slots_committed,
    ):
        spans.append(RequestSpan(rid, int(end - length), offset, int(length), slot))
        offset += int(length)
    return spans


def _sglang_module(name):
    """SGLang's ``name``, or a skip where it cannot import (some SGLang
    modules probe the GPU at import)."""
    import importlib

    try:
        return importlib.import_module(name)
    except RuntimeError as exc:
        pytest.skip(f"{name} does not import here: {exc}")


# ---------------------------------------------------------------------------
# 1. The seam
# ---------------------------------------------------------------------------


def _req(tokens, rid=None):
    return SimpleNamespace(rid=rid, full_untruncated_fill_ids=array("q", tokens))


class _World:
    """A worker, its runner and pools, and a builder."""

    def __init__(self, monkeypatch, *, rows=8, max_tokens=4096):
        self.generation = torch.ones(rows, dtype=torch.int64)
        self.pool = SimpleNamespace(state_gen=np.zeros(rows, dtype=np.int64))
        self.runner = SimpleNamespace(
            req_to_token_pool=SimpleNamespace(req_generation=self.generation),
            token_to_kv_pool=self.pool,
        )
        self.worker = SimpleNamespace(model_runner=self.runner)
        monkeypatch.setattr(
            bridge,
            "_V41_BUILDER",
            SimpleNamespace(
                block_size=PAGE,
                max_num_batched_tokens=max_tokens,
                geometry=SimpleNamespace(history_size=HISTORY),
            ),
        )

    def batch(self, prefix, slots, reqs=None, mode=EXTEND, lengths=None):
        reqs = reqs or [
            _req(range(1000, 1000 + p + 300), rid=f"q{i}") for i, p in enumerate(prefix)
        ]
        lengths = list(lengths or [7] * len(prefix))
        return SimpleNamespace(
            forward_mode=mode,
            reqs=reqs,
            prefix_lens=list(prefix),
            extend_num_tokens=sum(lengths),
            req_pool_indices_cpu=torch.tensor(slots, dtype=torch.int64),
        )

    def stamp(self, slots):
        self.pool.state_gen[slots] = self.generation[slots].numpy()


class TestSeam:
    def test_hit_rides_in_the_step(self, monkeypatch, radix_on):
        world = _World(monkeypatch)
        req = _req(range(1000, 1000 + 2 * PAGE + 7), rid="a")
        fused = run_v41_prefix_replay(world.worker, world.batch([2 * PAGE], [3], [req]))
        start, end = 2 * PAGE - R, 2 * PAGE
        assert fused.rows.tolist() == [0] and fused.slots.tolist() == [3]
        assert fused.starts.tolist() == [start] and fused.ends.tolist() == [end]
        assert fused.ids.tolist() == list(range(1000 + start, 1000 + end))
        assert fused.positions().tolist() == list(range(start, end))
        assert fused.histories.tolist() == [
            [1000 + start - 3, 1000 + start - 2, 1000 + start - 1]
        ]
        assert fused.expanded is None
        # Not stamped by the seam: the bridge stamps it when the forward
        # consumes the record, so a lost record fails the guard first.
        assert world.pool.state_gen[3] != world.generation[3]
        assert radix._LOGGED_FIRST_FUSED

    def test_only_new_tenants_above_zero_ride(self, monkeypatch, radix_on):
        world = _World(monkeypatch)
        world.stamp([4])
        reqs = [
            _req(range(k * 10000, k * 10000 + 3 * PAGE), rid=f"r{k}") for k in range(4)
        ]
        batch = world.batch([PAGE, 0, 2 * PAGE, PAGE], [1, 2, 3, 4], reqs)
        fused = run_v41_prefix_replay(world.worker, batch)
        # Row 1 is cold, row 3 continues its chunk.
        assert fused.rows.tolist() == [0, 2]
        assert fused.replay_rows(4).tolist() == [R, 0, R, 0]
        assert fused.ids.tolist() == (
            list(range(PAGE - R, PAGE))
            + list(range(20000 + 2 * PAGE - R, 20000 + 2 * PAGE))
        )
        # The cold row and the continuation are stamped; the hits wait for
        # their forward to consume the record.
        assert world.pool.state_gen[[1, 2, 3, 4]].tolist() == [0, 1, 0, 1]

    def test_retracted_readmission_replays(self, monkeypatch, radix_on):
        world = _World(monkeypatch)
        world.stamp([3])
        world.generation[3] += 1  # alloc_rows: a new tenant of row 3
        fused = run_v41_prefix_replay(world.worker, world.batch([PAGE], [3]))
        assert fused.slots.tolist() == [3]

    @pytest.mark.parametrize("prefix,stamped", [([0], False), ([PAGE], True)])
    def test_a_step_without_a_hit_carries_nothing(
        self, monkeypatch, radix_on, prefix, stamped
    ):
        world = _World(monkeypatch)
        if stamped:
            world.stamp([1])

        def boom(*args):
            raise AssertionError("a step without a hit built a record")

        monkeypatch.setattr(radix, "build_fused_replay", boom)
        assert run_v41_prefix_replay(world.worker, world.batch(prefix, [1])) is None
        # Cold or continuing, the row now belongs to its tenant.
        assert world.pool.state_gen[1] == world.generation[1]
        assert not radix._LOGGED_FIRST_FUSED

    def test_unaligned_hit_is_refused(self, monkeypatch, radix_on):
        world = _World(monkeypatch)
        with pytest.raises(ValueError, match="page boundary"):
            run_v41_prefix_replay(world.worker, world.batch([PAGE + 2], [1]))

    def test_mixed_batch_is_refused(self, monkeypatch, radix_on):
        world = _World(monkeypatch)
        with pytest.raises(NotImplementedError, match="mixed-chunk"):
            run_v41_prefix_replay(world.worker, world.batch([PAGE], [1], mode=MIXED))

    def test_a_decode_touches_nothing(self, monkeypatch, radix_on):
        world = _World(monkeypatch)
        batch = world.batch([PAGE], [1], mode=DECODE)
        assert run_v41_prefix_replay(world.worker, batch) is None
        assert not world.pool.state_gen.any()

    def test_plan_is_pure(self):
        slots = np.array([1, 2, 3], dtype=np.int64)
        generations = np.array([5, 5, 5], dtype=np.int64)
        state_gen = np.array([0, 0, 0, 5], dtype=np.int64)
        hits = plan_prefix_replay(
            [PAGE, 0, 2 * PAGE], slots, generations, state_gen, PAGE
        )
        assert hits == [ReplayHit(0, 1, PAGE - R, PAGE)]

    def test_plan_matches_a_per_row_reference(self):
        """The vectorized plan against the per-row loop it replaces."""
        rng = np.random.default_rng(0)

        def reference(prefix, slots, generations, state_gen):
            return [
                ReplayHit(i, int(s), int(p) - R, int(p))
                for i, (p, s, g) in enumerate(zip(prefix, slots, generations))
                if p > 0 and state_gen[s] != g
            ]

        for _ in range(500):
            bs = int(rng.integers(1, 40))
            slots = rng.permutation(64)[:bs].astype(np.int64)
            generations = rng.integers(1, 4, size=bs).astype(np.int64)
            state_gen = rng.integers(0, 4, size=64).astype(np.int64)
            prefix = (rng.integers(0, 4, size=bs) * PAGE).tolist()
            assert plan_prefix_replay(
                prefix, slots, generations, state_gen, PAGE
            ) == reference(prefix, slots, generations, state_gen)


class TestWorkerSeam:
    def _worker_cls(self, monkeypatch, calls):
        import types

        class TpModelWorker:
            def forward_batch_generation(self, batch=None, *args, **kwargs):
                fused = getattr(batch, FUSED_ATTR, None)
                calls.append((fused, args, kwargs))
                if fused is not None:
                    fused.expanded = "widened"  # the model ran it
                return "result"

        module = types.ModuleType("sglang.srt.managers.tp_worker")
        module.TpModelWorker = TpModelWorker
        monkeypatch.setitem(sys.modules, "sglang.srt.managers.tp_worker", module)
        radix._install_worker_seam()
        radix._install_worker_seam()  # idempotent
        return TpModelWorker

    def test_the_record_rides_on_the_batch_for_the_call(self, monkeypatch):
        calls, planned = [], []
        worker_cls = self._worker_cls(monkeypatch, calls)
        fused = _fused([0], [1], [0], [4])
        monkeypatch.setattr(
            radix,
            "run_v41_prefix_replay",
            lambda worker, batch: planned.append(batch) or fused,
        )
        batch = SimpleNamespace()
        assert worker_cls().forward_batch_generation(batch, 1, k=2) == "result"
        # Set for the call, gone after it.
        assert planned == [batch] and calls == [(fused, (1,), {"k": 2})]
        assert getattr(batch, FUSED_ATTR) is None
        monkeypatch.setattr(radix, "run_v41_prefix_replay", lambda *a: None)
        worker_cls().forward_batch_generation(batch)
        assert calls[-1] == (None, (), {})

    def test_no_batch_is_not_planned(self, monkeypatch):
        calls = []
        worker_cls = self._worker_cls(monkeypatch, calls)
        monkeypatch.setattr(
            radix, "run_v41_prefix_replay", lambda *a: pytest.fail("planned None")
        )
        worker_cls().forward_batch_generation(None, forward_batch="fb")
        assert calls == [(None, (), {"forward_batch": "fb"})]


class TestFusedCarrier:
    def test_init_new_carries_the_record(self):
        fused = _fused([0], [1], [0], [4])
        batch = SimpleNamespace()
        setattr(batch, FUSED_ATTR, fused)
        fb = SimpleNamespace(forward_mode=EXTEND)
        carry_fused_replay(batch, fb)
        assert getattr(fb, FUSED_ATTR) is fused
        # A batch with none (any other step) adds none.
        for none in (SimpleNamespace(), SimpleNamespace(**{FUSED_ATTR: None})):
            fb = SimpleNamespace(forward_mode=EXTEND)
            carry_fused_replay(none, fb)
            assert getattr(fb, FUSED_ATTR, None) is None

    def test_the_record_lives_for_one_call(self):
        fused = _fused([0], [1], [0], [4])
        batch = SimpleNamespace()
        seen = []

        def original(worker, batch, *args, **kwargs):
            seen.append((getattr(batch, FUSED_ATTR), args, kwargs))
            getattr(batch, FUSED_ATTR).expanded = "widened"  # the model ran it
            return "result"

        out = run_with_fused_replay(original, "w", batch, fused, (1,), {"k": 2})
        assert out == "result" and seen == [(fused, (1,), {"k": 2})]
        assert getattr(batch, FUSED_ATTR) is None

        def failing(worker, batch, *args, **kwargs):
            raise RuntimeError("forward failed")

        with pytest.raises(RuntimeError, match="forward failed"):
            run_with_fused_replay(failing, "w", batch, fused, (), {})
        assert getattr(batch, FUSED_ATTR) is None

    def test_the_eager_runners_view_carries_the_record(self, monkeypatch):
        import dataclasses
        import types

        @dataclasses.dataclass
        class ForwardBatch:
            forward_mode: object
            input_ids: object = None

        class EagerRunner:
            def load_batch(self, forward_batch, pp_proxy_tensors=None, **kwargs):
                # SGLang's view over its static buffers: fields only.
                return dataclasses.replace(forward_batch, input_ids="static")

        module = types.ModuleType("sglang.srt.model_executor.runner.eager_runner")
        module.EagerRunner = EagerRunner
        monkeypatch.setitem(
            sys.modules, "sglang.srt.model_executor.runner.eager_runner", module
        )
        radix._install_eager_view_carrier()
        radix._install_eager_view_carrier()  # idempotent
        fused = _fused([0], [1], [0], [4])
        fb = ForwardBatch(EXTEND)
        setattr(fb, FUSED_ATTR, fused)
        view = EagerRunner().load_batch(fb)
        assert view is not fb and view.input_ids == "static"
        assert getattr(view, FUSED_ATTR) is fused
        # Every other forward's view stays without one.
        other = EagerRunner().load_batch(ForwardBatch(EXTEND))
        assert getattr(other, FUSED_ATTR, None) is None


# ---------------------------------------------------------------------------
# 2. Admission accounting
# ---------------------------------------------------------------------------


def _admitted(pool_idx=None, retracted=False):
    return SimpleNamespace(
        kv=SimpleNamespace(req_pool_idx=pool_idx), is_retracted=retracted
    )


@pytest.mark.parametrize(
    ("req", "prefix", "charge"),
    [
        (_admitted(), 2 * PAGE, R),
        (_admitted(), 64, 64),
        (_admitted(), 0, 0),
        (_admitted(pool_idx=3), 2 * PAGE, 0),  # a chunk continuation's row
        (_admitted(pool_idx=3, retracted=True), 2 * PAGE, R),
    ],
)
def test_replay_admission_rows(req, prefix, charge):
    assert replay_admission_rows(req, prefix) == charge


def test_admission_charge_wraps_sglang(monkeypatch):
    PrefillAdder = _sglang_module("sglang.srt.managers.schedule_policy").PrefillAdder

    seen = []
    monkeypatch.setattr(
        PrefillAdder,
        "_account_prefill_cache_admission",
        lambda self, req, prefix_len: seen.append(prefix_len),
    )
    radix._install_admission_accounting()
    radix._install_admission_accounting()  # idempotent
    account = PrefillAdder._account_prefill_cache_admission
    adder = SimpleNamespace(
        log_replay_tokens=0, rem_input_tokens=1000, rem_chunk_tokens=500
    )
    account(adder, _admitted(), 2 * PAGE)
    account(adder, _admitted(pool_idx=2), 2 * PAGE)
    assert adder.log_replay_tokens == R
    assert (adder.rem_input_tokens, adder.rem_chunk_tokens) == (1000 - R, 500 - R)
    adder.rem_chunk_tokens = None
    account(adder, _admitted(retracted=True), PAGE)
    assert adder.rem_input_tokens == 1000 - 2 * R
    assert seen == [2 * PAGE] * 2 + [PAGE]


# ---------------------------------------------------------------------------
# 3. Refusal matrix, and what prepare installs
# ---------------------------------------------------------------------------


def _radix_bags(**overrides):
    bags = {
        "memory": SimpleNamespace(
            disable_radix_cache=False,
            enable_hierarchical_cache=False,
            enable_session_radix_cache=False,
            enable_unified_memory=False,
            radix_cache_backend=None,
            enable_lmcache=False,
            enable_flexkv=False,
            enable_unified_cache_external_linker=False,
        ),
        "exec": SimpleNamespace(
            features=SimpleNamespace(
                enable_encoder_swa_bounded_replay=False,
                enable_decoder_swa_bounded_replay=False,
                enable_memory_saver=False,
            ),
            moe=SimpleNamespace(moe_a2a_backend="none"),
            graph=SimpleNamespace(
                cuda_graph_config=SimpleNamespace(
                    decode=SimpleNamespace(backend="full"),
                    prefill=SimpleNamespace(backend="disabled"),
                )
            ),
        ),
        "spec": SimpleNamespace(
            speculative_algorithm=None,
            speculative_num_draft_tokens=None,
            speculative_draft_model_path=None,
            speculative_use_block_verification=False,
        ),
        "model": SimpleNamespace(model_path="/models/v41"),
        "disagg": SimpleNamespace(disaggregation_mode="null", language_model_only=True),
        "parallel": SimpleNamespace(
            enable_dp_attention=False, dcp_size=1, moe_dense_tp_size=None
        ),
        "schedule": SimpleNamespace(
            disable_hybrid_swa_memory=False,
            chunked_prefill_size=8192,
            max_prefill_tokens=16384,
            max_running_requests=64,
            mem_fraction_static=0.85,
            disable_overlap_schedule=False,
            enable_mixed_chunk=False,
        ),
    }
    bags.update(overrides)
    return bags


@pytest.fixture
def launch(monkeypatch):
    """SGLang's resolved config bags, as its scheduler publishes them."""
    from sglang.srt import runtime_context

    state = SimpleNamespace(bags=_radix_bags())
    for name in ("memory", "exec", "spec", "disagg", "parallel", "schedule", "model"):
        monkeypatch.setattr(
            runtime_context, f"get_{name}", lambda name=name: state.bags[name]
        )
    for name in (
        "SGLANG_ENABLE_METADATA_GLUE_GRAPH",
        "ATOM_ENGRAM_UVA",
        "ATOM_ENGRAM_OVERLAP",
    ):
        monkeypatch.delenv(name, raising=False)
    return state


class TestRefusals:
    def test_supported_radix_launch_passes(self, launch):
        # Overlap scheduling included: the seam runs inside
        # forward_batch_generation, and a checkpoint drops the slot's host
        # rows (section 7).
        assert sglang_radix_cache_enabled()
        refuse_unsupported_sglang_features()

    def test_the_kill_switch_is_the_launch_policys(self, launch, monkeypatch):
        # ATOM_SGLANG_V41_RADIX=0 turns SGLang's radix cache off before the bridge
        # runs (`TestLaunchPolicy`); the bridge has no stand-in refusal.
        monkeypatch.setenv("ATOM_SGLANG_V41_RADIX", "0")
        refuse_unsupported_sglang_features()

    @pytest.mark.parametrize(
        ("bag", "path", "value", "match"),
        [
            ("schedule", "enable_mixed_chunk", True, "mixed chunk"),
            ("memory", "enable_session_radix_cache", True, "session radix cache"),
            ("memory", "radix_cache_backend", "custom", "custom radix cache"),
            ("memory", "enable_lmcache", True, "KV storage backend"),
            ("memory", "enable_flexkv", True, "KV storage backend"),
            (
                "memory",
                "enable_unified_cache_external_linker",
                True,
                "KV storage backend",
            ),
        ],
    )
    def test_radix_refusals(self, launch, bag, path, value, match):
        setattr(launch.bags[bag], path, value)
        with pytest.raises(ValueError, match=match):
            refuse_unsupported_sglang_features()

    @pytest.mark.parametrize("chunk", [-1, 0, None])
    def test_disabled_chunking_is_refused_on_its_own(self, launch, chunk):
        # Refused for every launch (the builder holds one chunk plus R), so
        # no admitted step can widen past the builder.
        launch.bags["schedule"].chunked_prefill_size = chunk
        with pytest.raises(ValueError, match="disabled chunked prefill"):
            refuse_unsupported_sglang_features()

    def test_radix_off_ignores_radix_only_refusals(self, launch, monkeypatch):
        monkeypatch.setenv("ATOM_SGLANG_V41_RADIX", "0")
        launch.bags["memory"].disable_radix_cache = True
        launch.bags["schedule"].enable_mixed_chunk = True
        assert not sglang_radix_cache_enabled()
        refuse_unsupported_sglang_features()

    def test_encoder_replay_flag_stays_refused(self, launch):
        launch.bags["exec"].features.enable_encoder_swa_bounded_replay = True
        with pytest.raises(ValueError, match="encoder SWA bounded replay"):
            refuse_unsupported_sglang_features()


class TestLaunchPolicy:
    """Radix on by default; SGLang's radix cache turned off at server-args
    resolution, with one warning, for ``ATOM_SGLANG_V41_RADIX=0`` -- on a
    launch ATOM serves only."""

    @pytest.fixture(autouse=True)
    def _fresh_policy(self, monkeypatch):
        monkeypatch.setattr(radix, "_POLICY_WARNED", False)
        monkeypatch.setattr(radix, "_POLICY_REGISTERED", False)
        monkeypatch.delenv("ATOM_SGLANG_V41_RADIX", raising=False)
        monkeypatch.setenv("SGLANG_EXTERNAL_MODEL_PACKAGE", "atom.plugin.sglang.models")
        # A plain record: SGLang's resolving view reads it the same way.
        monkeypatch.setattr(radix, "_resolving_view", lambda server_args: server_args)

    ARGS = SimpleNamespace(disable_radix_cache=False, speculative_algorithm=None)
    V41 = SimpleNamespace(model_type="deepseek_v41")

    @staticmethod
    def _args(spec=None):
        return SimpleNamespace(disable_radix_cache=False, speculative_algorithm=spec)

    def test_default_launch_keeps_radix(self, caplog):
        with caplog.at_level("WARNING", logger="atom"):
            assert radix.v41_radix_launch_override(self.ARGS, self.V41) == {}
        assert not caplog.records

    def test_kill_switch_turns_radix_off_once(self, monkeypatch, caplog):
        monkeypatch.setenv("ATOM_SGLANG_V41_RADIX", "0")
        with caplog.at_level("WARNING", logger="atom"):
            for _ in range(3):
                assert radix.v41_radix_launch_override(self.ARGS, self.V41) == {
                    "disable_radix_cache": True
                }
        (record,) = caplog.records
        assert "ATOM_SGLANG_V41_RADIX is off" in record.getMessage()
        assert "--disable-radix-cache" in record.getMessage()

    def test_radix_already_off_declares_nothing(self, monkeypatch, caplog):
        monkeypatch.setenv("ATOM_SGLANG_V41_RADIX", "0")
        args = SimpleNamespace(disable_radix_cache=True, speculative_algorithm="DSPARK")
        with caplog.at_level("WARNING", logger="atom"):
            assert radix.v41_radix_launch_override(args, self.V41) == {}
        assert not caplog.records

    def test_dspark_keeps_radix(self, caplog):
        with caplog.at_level("WARNING", logger="atom"):
            assert radix.v41_radix_launch_override(self._args("DSPARK"), self.V41) == {}
        assert not caplog.records

    def test_radix_off_wins_over_dspark(self, monkeypatch, caplog):
        monkeypatch.setenv("ATOM_SGLANG_V41_RADIX", "0")
        with caplog.at_level("WARNING", logger="atom"):
            assert radix.v41_radix_launch_override(self._args("DSPARK"), self.V41) == {
                "disable_radix_cache": True
            }
        assert "ATOM_SGLANG_V41_RADIX is off" in caplog.records[0].getMessage()

    # ATOM's text type never reaches SGLang's registry: SGLang's V4.1 config
    # forces ``deepseek_v41``.
    @pytest.mark.parametrize(
        "model_type", ["deepseek_v4", "deepseek_v41_text", "qwen3", None]
    )
    def test_other_models_are_left_alone(self, monkeypatch, model_type):
        monkeypatch.setenv("ATOM_SGLANG_V41_RADIX", "0")
        config = SimpleNamespace(model_type=model_type)
        assert radix.v41_radix_launch_override(self.ARGS, config) == {}

    @pytest.mark.parametrize("package", [None, "", "sglang.srt.models"])
    def test_native_v41_launches_are_left_alone(self, monkeypatch, package, caplog):
        # The entry point registers the policy on every SGLang launch; a V4.1
        # launch SGLang serves natively keeps its radix cache.
        monkeypatch.setenv("ATOM_SGLANG_V41_RADIX", "0")
        if package is None:
            monkeypatch.delenv("SGLANG_EXTERNAL_MODEL_PACKAGE")
        else:
            monkeypatch.setenv("SGLANG_EXTERNAL_MODEL_PACKAGE", package)
        with caplog.at_level("WARNING", logger="atom"):
            assert radix.v41_radix_launch_override(self.ARGS, self.V41) == {}
        assert not caplog.records

    def test_sglang_resolving_view_sees_earlier_declarations(self, monkeypatch):
        base = _sglang_module("sglang.srt.arg_groups.model_override_base")
        monkeypatch.setattr(radix, "_resolving_view", base.resolving_view)
        monkeypatch.setenv("ATOM_SGLANG_V41_RADIX", "0")
        args = SimpleNamespace(disable_radix_cache=False, speculative_algorithm=None)
        args._resolved_overrides = [("earlier", {"disable_radix_cache": True})]
        assert radix.v41_radix_launch_override(args, self.V41) == {}
        args._resolved_overrides = []
        assert radix.v41_radix_launch_override(args, self.V41) == {
            "disable_radix_cache": True
        }

    def test_registers_under_sglangs_v41_architecture_once(self, monkeypatch):
        registered = []
        fake = SimpleNamespace(
            register_model_override=lambda arch: (
                lambda fn: registered.append((arch, fn)) or fn
            )
        )
        monkeypatch.setitem(
            sys.modules, "sglang.srt.arg_groups.model_override_base", fake
        )
        radix.register_v41_radix_launch_policy()
        radix.register_v41_radix_launch_policy()
        assert registered == [
            ("DeepseekV4ForCausalLM", radix.v41_radix_launch_override)
        ]

    def test_sglang_hands_v41_under_v4s_architecture(self):
        # The registry is keyed on `hf_config.architectures[0]`; SGLang's V4.1
        # config renames V4.1's and forces the model type the policy checks.
        configs = _sglang_module("sglang.srt.configs.deepseek_v41")
        values = configs.normalize_deepseek_v41_config(
            {"architectures": ["DeepseekV41ForCausalLM"], "model_type": "x"}
        )
        assert values["architectures"] == ["DeepseekV4ForCausalLM"]
        assert values["model_type"] == "deepseek_v41"

    def test_plugin_entry_point_registers_the_policy(self, monkeypatch):
        """``register_plugin`` (the ``sglang.srt.plugins`` entry point) calls
        the registration; every other patch it installs is stubbed out."""
        import importlib

        from atom.plugin.sglang import register

        calls = []
        stubs = {
            register: (
                "_ensure_aiter_gpu_archs_env",
                "_install_model_config_quant_patch",
                "_install_loader_quant_patch",
                "_register_tc_piecewise_attention_split_ops",
                "_install_decode_graph_forward_context_patch",
                "apply_prefill_compile_only_patch",
                "apply_triton_kernel_retention_patch",
                "apply_xgrammar_hip_bitmask_patch",
                "_keep_atom_full_attn_for_native_qwen4_exp",
                "register_kimi_k3_text_only_processor",
            ),
            radix: ("register_v41_radix_launch_policy",),
            "atom.plugin.sglang.attention_backend.gdn_replayssm": (
                "install_sglang_replayssm_commit",
            ),
            "atom.plugin.sglang.patches.qwen4_exp_recognition_patch": (
                "apply_qwen4_exp_recognition_patch",
            ),
            "atom.plugin.sglang.patches.qwen4_exp_rocm_patch": (
                "apply_qwen4_exp_rocm_patch",
            ),
            "atom.plugin.sglang.runtime": ("apply_load_config_patch",),
        }
        for module, names in stubs.items():
            if isinstance(module, str):
                module = importlib.import_module(module)
            for name in names:
                monkeypatch.setattr(
                    module, name, lambda *a, name=name, **k: calls.append(name)
                )
        register.register_plugin()
        assert "register_v41_radix_launch_policy" in calls


def _v41_atom_config():
    return SimpleNamespace(
        max_num_batched_tokens=16384,
        max_num_seqs=64,
        gpu_memory_utilization=0.85,
        kv_cache_block_size=128,
        enable_decoder_swa_bounded_replay=True,
        enable_encoder_swa_bounded_replay=True,
        speculative_config=None,
        online_quant_config={"layer_quant_config": {}},
        quant_config=None,
        hf_config=SimpleNamespace(sliding_window=128, fp4ptpc=False),
        kv_cache_dtype="bf16",
        index_cache_dtype="fp8",
    )


@pytest.mark.parametrize("decoder_replay", ["1", "0"])
@pytest.mark.parametrize("radix_cache", [False, True])
def test_prepare_installs_the_replay_only_with_radix(
    launch, monkeypatch, radix_cache, decoder_replay
):
    from atom.plugin import register
    from atom.plugin.sglang.runtime import model_arch

    # The quant remap reads only the model class's mappings; the class itself
    # needs a GPU build of AITER.
    monkeypatch.setattr(
        register, "_ATOM_SUPPORTED_MODELS", {"DeepseekV41ForCausalLM": object}
    )
    # Decoder bounded replay is independent of radix cache: on unless
    # ATOM_SGLANG_V41_DECODER_REPLAY=0, with radix on or off.
    monkeypatch.setenv("ATOM_SGLANG_V41_DECODER_REPLAY", decoder_replay)
    installed = []
    monkeypatch.setattr(
        model_arch,
        "install_deepseek_v41_radix",
        lambda window=None: installed.append(window) or R,
    )
    launch.bags["memory"].disable_radix_cache = not radix_cache
    config = _v41_atom_config()
    model_arch._prepare_deepseek_v41_config(config, "DeepseekV41ForCausalLM")
    assert config.enable_encoder_swa_bounded_replay is radix_cache
    assert config.enable_decoder_swa_bounded_replay is (decoder_replay == "1")
    assert installed == ([128] if radix_cache else [])
    assert config.kv_cache_block_size == PAGE
    # A step runs its hits' replay rows on top of its chunk.
    assert config.max_num_batched_tokens == 16384 + (R if radix_cache else 0)


def test_install_turns_the_seam_on_once_per_process(monkeypatch):
    installed = []
    for name in (
        "_install_worker_seam",
        "_install_forward_batch_carrier",
        "_install_eager_view_carrier",
        "_install_checkpoint_invalidation",
        "_install_admission_accounting",
        "_install_request_admission",
    ):
        monkeypatch.setattr(radix, name, lambda name=name: installed.append(name))
    monkeypatch.setattr(radix, "_get_builder", None)
    assert radix.radix_replay_rows() is None
    assert radix.install_deepseek_v41_radix(window=128) == R
    assert radix.radix_replay_rows() == R
    assert radix._get_builder is bridge.get_deepseek_v41_builder
    assert len(installed) == 6 and "_install_request_admission" in installed
    # R covers the sliding window, or the rows at a hit read unwritten rings.
    with pytest.raises(ValueError, match="sliding window 256"):
        radix.install_deepseek_v41_radix(window=256)


def test_proxy_pool_owns_a_stamp_per_request_slot(monkeypatch):
    from tests.plugin.test_sglang_deepseek_v41_bridge import (
        _SGLANG_MODULES,
        _install_builder,
        _proxy,
    )
    from tests.plugin.test_sglang_deepseek_v41_bridge import (
        _small_geometry as bridge_geometry,
    )

    for name, module in _SGLANG_MODULES.items():
        monkeypatch.setitem(sys.modules, name, module)
    _install_builder(monkeypatch, bridge_geometry())
    pool = _proxy()
    assert pool.state_gen.dtype == np.int64
    assert pool.state_gen.tolist() == [0] * pool.num_req_slots
    with pytest.raises(NotImplementedError, match="relocation"):
        pool.move_kv_cache(torch.zeros(1), torch.zeros(1))


# ---------------------------------------------------------------------------
# 4. The record, the widened inputs, the runtime hook
# ---------------------------------------------------------------------------


def _fused(rows, slots, starts, ends, history=((1, 2, 3),), ids=None):
    if ids is None:
        ids = [
            1000 * (k + 1) + t
            for k, (s, e) in enumerate(zip(starts, ends))
            for t in range(e - s)
        ]
    histories = np.asarray(history, np.int64)
    if histories.shape[0] != len(rows):
        histories = np.repeat(histories[:1], len(rows), axis=0)
    return FusedReplay(rows, slots, starts, ends, ids, histories)


class TestFusedRecord:
    def test_replay_rows_per_request(self):
        fused = _fused([0, 2], [1, 3], [0, 4], [4, 6])
        assert fused.num_rows == 6
        assert fused.replay_rows(3).tolist() == [4, 0, 2]
        assert fused.positions().tolist() == [0, 1, 2, 3, 4, 5]

    def test_layout_puts_each_replay_before_its_request(self):
        keep, replay_at = fused_layout([3, 2, 4], [0, 5, 2])
        # req0: 0..2; req1: replay 3..7, own 8..9; req2: replay 10..11, own 12..15
        assert keep.tolist() == [0, 1, 2, 8, 9, 12, 13, 14, 15]
        assert replay_at.tolist() == [3, 4, 5, 6, 7, 10, 11]
        keep, replay_at = fused_layout([4], [0])
        assert keep.tolist() == [0, 1, 2, 3] and replay_at.tolist() == []


def _fused_fb(lengths, fused, mode=EXTEND):
    fb = SimpleNamespace(
        forward_mode=mode,
        batch_size=len(lengths),
        extend_seq_lens_cpu=list(lengths),
    )
    setattr(fb, FUSED_ATTR, fused)
    return fb


class TestFusedInputs:
    def _case(self):
        fused = _fused([1], [4], [6], [10], ids=[60, 70, 80, 90])
        fb = _fused_fb([3, 2], fused)
        ids = torch.tensor([11, 12, 13, 21, 22])
        positions = torch.tensor([0, 1, 2, 10, 11])
        return fused, fb, ids, positions

    def test_rows_below_the_hit_are_prepended(self):
        fused, fb, ids, positions = self._case()
        wide_ids, wide_pos, rows = expand_v41_forward_inputs(fb, ids, positions)
        assert wide_ids.tolist() == [11, 12, 13, 60, 70, 80, 90, 21, 22]
        assert wide_pos.tolist() == [0, 1, 2, 6, 7, 8, 9, 10, 11]
        assert wide_ids.dtype == ids.dtype and wide_pos.dtype == positions.dtype
        # SGLang's rows of the widened output, in order.
        assert rows.tolist() == [0, 1, 2, 7, 8]
        hidden = torch.arange(9)[:, None].expand(9, 4)
        assert hidden.index_select(0, rows)[:, 0].tolist() == [0, 1, 2, 7, 8]
        # Kept on the record for the bridge.
        assert fused.expanded.input_ids is wide_ids

    def test_other_forwards_are_not_widened(self):
        fb = _fused_fb([3], None)
        assert expand_v41_forward_inputs(fb, torch.zeros(3), torch.zeros(3)) is None

    def test_the_v41_adapter_hook(self):
        from atom.plugin.sglang.runtime import model_arch

        spec = model_arch.MODEL_ADAPTER_SPECS[model_arch.DEEPSEEK_V41_ARCH]
        assert spec.expand_forward_inputs is expand_v41_forward_inputs
        # Every other adapter has none, so its runtime skips the hook.
        others = [
            arch
            for arch, other in model_arch.MODEL_ADAPTER_SPECS.items()
            if other.expand_forward_inputs is not None
        ]
        assert others == [model_arch.DEEPSEEK_V41_ARCH]


class TestPinnedStaging:
    def test_the_host_layout_rides_on_the_record(self):
        fused = _fused([1], [4], [6], [10], ids=[60, 70, 80, 90])
        fb = _fused_fb([3, 2], fused)
        ids = torch.tensor([11, 12, 13, 21, 22])
        positions = torch.tensor([0, 1, 2, 10, 11])
        expand_v41_forward_inputs(fb, ids, positions)
        # keep (5), replay_at (4), ids (4), positions (4), in that order.
        assert fused.host.tolist() == (
            [0, 1, 2, 7, 8] + [3, 4, 5, 6] + [60, 70, 80, 90] + [6, 7, 8, 9]
        )

    def test_a_device_forward_copies_from_pinned_memory(self, monkeypatch):
        if not torch.cuda.is_available():
            pytest.skip("needs a visible GPU")
        copies = []
        original_to = torch.Tensor.to

        def to(self, *args, **kwargs):
            if self.device.type == "cpu":
                copies.append((self.is_pinned(), kwargs.get("non_blocking")))
            return original_to(self, *args, **kwargs)

        fused = _fused([1], [4], [6], [10], ids=[60, 70, 80, 90])
        fb = _fused_fb([3, 2], fused)
        ids = torch.tensor([11, 12, 13, 21, 22], device="cuda")
        positions = torch.tensor([0, 1, 2, 10, 11], device="cuda")
        monkeypatch.setattr(torch.Tensor, "to", to)
        wide_ids, wide_pos, rows = expand_v41_forward_inputs(fb, ids, positions)
        monkeypatch.undo()
        assert copies == [(True, True)]
        assert fused.host.is_pinned()
        assert wide_ids.tolist() == [11, 12, 13, 60, 70, 80, 90, 21, 22]
        assert wide_pos.tolist() == [0, 1, 2, 6, 7, 8, 9, 10, 11]
        assert rows.tolist() == [0, 1, 2, 7, 8]


class TestRuntimeExpansion:
    def _runtime(self, expand, mode=EXTEND):
        from atom.plugin.sglang.runtime.forward_context import SGLangPluginRuntime

        return SGLangPluginRuntime(
            atom_config=SimpleNamespace(hf_config=None),
            forward_batch=SimpleNamespace(forward_mode=mode),
            positions=torch.tensor([0, 1, 2]),
            input_ids=torch.tensor([5, 6, 7]),
            set_forward_context=False,
            expand_forward_inputs=expand,
        )

    def test_widened_inputs_and_sglang_rows_out(self):
        seen = []

        def expand(fb, ids, positions, embeds):
            seen.append((ids.tolist(), positions.tolist(), embeds))
            return (
                torch.tensor([9, 9, 5, 6, 7]),
                torch.tensor([8, 9, 10, 11, 12]),
                torch.tensor([2, 3, 4]),
            )

        with self._runtime(expand) as runtime:
            assert runtime.input_ids.tolist() == [9, 9, 5, 6, 7]
            assert runtime.positions.tolist() == [8, 9, 10, 11, 12]
            hidden = torch.arange(10).view(5, 2)
            assert runtime.trim_output(hidden).tolist() == [[4, 5], [6, 7], [8, 9]]
            pair = runtime.trim_output((hidden, None))
            assert pair[0].shape == (3, 2) and pair[1] is None
            with pytest.raises(TypeError, match="widened forward"):
                runtime.trim_output([hidden])
        assert seen == [([5, 6, 7], [0, 1, 2], None)]

    def test_forwards_the_hook_declines_are_untouched(self):
        with self._runtime(lambda *args: None) as runtime:
            assert runtime.input_ids.tolist() == [5, 6, 7]
            hidden = torch.ones(3, 2)
            assert runtime.trim_output(hidden) is hidden

    def test_dummy_forwards_are_not_expanded(self):
        def expand(*args):
            raise AssertionError("an IDLE forward was expanded")

        with self._runtime(expand, mode=ForwardMode.IDLE) as runtime:
            assert runtime.trim_output(torch.ones(1, 2)).shape == (0, 2)

    def test_without_a_hook_no_adapter_is_resolved(self, monkeypatch):
        # Every other plugin model: no per-forward adapter lookup at all.
        from atom.plugin.sglang.runtime import forward_context, model_arch

        def boom(*args):
            raise AssertionError("adapter resolved per forward")

        monkeypatch.setattr(model_arch, "resolve_model_arch_spec", boom)
        monkeypatch.setattr(forward_context, "cached_model_arch_spec", boom)
        with self._runtime(None) as runtime:
            assert runtime.input_ids.tolist() == [5, 6, 7]

    def test_the_wrapper_passes_its_adapters_hook(self, monkeypatch):
        """``EntryClass.forward`` hands the runtime the adapter's hook it
        resolved at construction, so the runtime looks nothing up."""
        import contextlib

        from atom.plugin.sglang.models import base_model_wrapper

        passed = []

        class Runtime:
            def __init__(self, **kwargs):
                passed.append(kwargs["expand_forward_inputs"])
                raise RuntimeError("stop")

        monkeypatch.setattr(base_model_wrapper, "SGLangPluginRuntime", Runtime)
        monkeypatch.setattr(
            base_model_wrapper,
            "plugin_runtime_scope",
            lambda **kwargs: contextlib.nullcontext(),
        )
        for hook in (expand_v41_forward_inputs, None):
            wrapper = SimpleNamespace(
                atom_config=None,
                model_arch_spec=SimpleNamespace(
                    wrapper_binds_gdn_context=False, expand_forward_inputs=hook
                ),
            )
            with pytest.raises(RuntimeError, match="stop"):
                base_model_wrapper.EntryClass[0].forward(
                    wrapper, torch.zeros(1), torch.zeros(1), SimpleNamespace()
                )
        assert passed == [expand_v41_forward_inputs, None]


# ---------------------------------------------------------------------------
# 5. The bridge: widened spans, the stamp guard, staged inputs
# ---------------------------------------------------------------------------


@pytest.mark.usefixtures("stamps")
class TestFusedBridge:
    def _fb(self):
        fb = _extend_fb([2 * PAGE, 0, PAGE], [5, 9, 300], [1, 2, 3])
        fused = _fused(
            [0, 2],
            [1, 3],
            [2 * PAGE - R, PAGE - R],
            [2 * PAGE, PAGE],
            history=[[1, 2, 3], [4, 5, 6]],
        )
        _carry(fb, fused)
        return fb, fused

    def test_hits_are_scheduled_from_below_the_hit(self, req_to_token):
        from atom.model_engine.sequence import Replay

        fb, _ = self._fb()
        batch = build_v41_scheduled_batch(fb, PAGE)
        assert batch.num_scheduled_tokens.tolist() == [5 + R, 9, 300 + R]
        assert batch.context_lens.tolist() == [2 * PAGE + 5, 9, PAGE + 300]
        assert batch.replay_rows.tolist() == [R, 0, R]
        assert batch.total_tokens_num == 5 + 9 + 300 + 2 * R
        assert [span.position for span in _spans(batch)] == [2 * PAGE - R, 0, PAGE - R]
        assert batch.replaying == {
            0: Replay(2 * PAGE, True, (1, 2, 3), 2 * PAGE - R),
            2: Replay(PAGE, True, (4, 5, 6), PAGE - R),
        }
        # The block tables cover the request, prefix pages included.
        assert [row.tolist() for row in batch.block_tables] == [
            [7, 3, 9],
            [5],
            [2, 11, 4],
        ]

    @pytest.mark.parametrize(("start", "kept"), [(0, ()), (2, (7, 8)), (4, (6, 7, 8))])
    def test_history_keeps_only_real_tokens(self, req_to_token, start, kept):
        # SGLang left-pads a short history; native ATOM carries what exists.
        fb = _extend_fb([PAGE], [3], [1])
        _carry(fb, _fused([0], [1], [start], [PAGE], [[6, 7, 8]]))
        batch = build_v41_scheduled_batch(fb, PAGE)
        assert batch.replaying[0].history_ids == kept
        assert batch.replaying[0].start == start

    def test_the_stamp_guard_reads_sglang_prefix(
        self, monkeypatch, req_to_token, radix_on
    ):
        seen, stamped = [], []
        monkeypatch.setattr(
            radix,
            "check_state_stamps",
            lambda state_gen, generation, slots, prefix, carried: seen.append(
                (prefix.tolist(), carried.tolist())
            ),
        )
        monkeypatch.setattr(
            radix,
            "stamp_fused_hits",
            lambda state_gen, generation, fused: stamped.append(fused.slots.tolist()),
        )
        monkeypatch.setattr(bridge, "_radix_stamps", lambda: (None, None))
        fb, _ = self._fb()
        build_v41_scheduled_batch(fb, PAGE)
        # SGLang's prefix, and the slots the carried record rebuilds.
        assert seen == [([2 * PAGE, 0, PAGE], [1, 3])]
        assert stamped == [[1, 3]]

    def test_a_short_suffix_still_replays_its_decoder(self, req_to_token):
        """The fused span is at least R >= window rows, so even a hit whose
        own suffix is shorter than the window runs the decoder on its tail,
        as native ATOM's hit forward from ``P - 128`` does."""
        fb, _ = self._fb()
        batch = build_v41_scheduled_batch(fb, PAGE)
        assert bridge.v41_extend_replays(fb, batch, 128)
        # Without the fused rows, the 5-row hit could not.
        setattr(fb, FUSED_ATTR, None)
        plain = build_v41_scheduled_batch(fb, PAGE)
        assert not bridge.v41_extend_replays(fb, plain, 128)

    @pytest.mark.parametrize(
        "starts,replays", [([5, 9, 300], True), ([4, 9, 300], False)]
    )
    def test_input_logprobs_count_sglang_rows_only(self, req_to_token, starts, replays):
        fb, _ = self._fb()
        fb.return_logprob = True
        fb.extend_logprob_start_lens_cpu = starts
        batch = build_v41_scheduled_batch(fb, PAGE)
        assert bridge.v41_extend_replays(fb, batch, 128) is replays


class TestStampGuard:
    def test_radix_off_reads_no_stamps(self, monkeypatch, req_to_token):
        def boom():
            raise AssertionError("stamps read with radix off")

        monkeypatch.setattr(bridge, "_radix_stamps", boom)
        batch = build_v41_scheduled_batch(_extend_fb([PAGE], [3], [1]), PAGE)
        assert batch.replaying == {}

    def test_resuming_row_needs_its_tenant_stamp(
        self, monkeypatch, req_to_token, radix_on
    ):
        state_gen = np.array([0, 4, 7, 0, 0, 0], dtype=np.int64)
        generation = torch.tensor([0, 4, 8, 0, 0, 0], dtype=torch.int64)
        monkeypatch.setattr(bridge, "_radix_stamps", lambda: (state_gen, generation))
        # Slot 1 is stamped; slot 2 at P = 0 needs no stamp.
        build_v41_scheduled_batch(_extend_fb([PAGE, 0], [3, 5], [1, 2]), PAGE)
        # Slot 2 resuming above 0 on a stale stamp: the seam was skipped.
        with pytest.raises(RuntimeError, match=r"slots \[2\]"):
            build_v41_scheduled_batch(_extend_fb([PAGE, PAGE], [3, 5], [1, 2]), PAGE)

    def test_decode_is_not_checked(self, monkeypatch, req_to_token, radix_on):
        monkeypatch.setattr(
            bridge, "_radix_stamps", lambda: pytest.fail("decode read stamps")
        )
        build_v41_scheduled_batch(_decode_fb(), PAGE)

    def test_check_state_stamps(self, monkeypatch):
        state_gen = np.array([0, 3, 5])
        generation = torch.tensor([0, 3, 6])
        prefix = np.array([PAGE, PAGE])
        check_state_stamps(state_gen, generation, [1, 2], np.array([PAGE, 0]))
        with pytest.raises(RuntimeError, match=r"slots \[2\].*not rebuilt"):
            check_state_stamps(state_gen, generation, [1, 2], prefix)
        # Carried slots pass; others still need their stamp.
        check_state_stamps(state_gen, generation, [1, 2], prefix, np.array([2]))
        with pytest.raises(RuntimeError, match=r"slots \[2\]"):
            check_state_stamps(state_gen, generation, [1, 2], prefix, np.array([1]))
        # Nothing carried: no membership test.
        monkeypatch.setattr(radix.np, "isin", lambda *a: pytest.fail("np.isin"))
        check_state_stamps(state_gen, generation, [1], np.array([PAGE]))


class TestStampOnConsume:
    """The seam leaves a hit unstamped; the bridge's guard accepts it only on
    a forward carrying its record, and stamps it there."""

    def _seam(self, monkeypatch):
        world = _World(monkeypatch)
        req = _req(range(1000, 1000 + 2 * PAGE + 7), rid="a")
        batch = world.batch([2 * PAGE, 0], [3, 2], [req, _req(range(9), rid="c")])
        fused = run_v41_prefix_replay(world.worker, batch)
        monkeypatch.setattr(
            bridge,
            "_radix_stamps",
            lambda: (world.pool.state_gen, world.generation),
        )
        table = _req_to_token(
            rows=8, context_len=4 * PAGE, pages_by_row={3: [7, 3, 9], 2: [5]}
        )
        monkeypatch.setattr(bridge, "_req_to_token", lambda: table)
        fb = _extend_fb([2 * PAGE, 0], [7, 9], [3, 2])
        fb.rids = ["a", "c"]
        return world, fused, fb

    def test_a_forward_that_lost_the_record_fails_before_compute(
        self, monkeypatch, radix_on
    ):
        world, fused, fb = self._seam(monkeypatch)
        assert fused is not None and world.pool.state_gen[3] == 0
        # The ForwardBatch reached the bridge without the record.
        with pytest.raises(RuntimeError, match=r"slots \[3\].*not rebuilt"):
            build_v41_scheduled_batch(fb, PAGE)
        assert world.pool.state_gen[3] == 0

    def test_the_forward_carrying_the_record_stamps_its_hits(
        self, monkeypatch, radix_on
    ):
        world, fused, fb = self._seam(monkeypatch)
        setattr(fb, FUSED_ATTR, fused)
        expand_v41_forward_inputs(fb, torch.arange(16), torch.arange(16))
        batch = build_v41_scheduled_batch(fb, PAGE)
        assert batch.replay_rows.tolist() == [R, 0]
        assert world.pool.state_gen[3] == world.generation[3]
        # The next chunk of the same tenant is a plain continuation.
        setattr(fb, FUSED_ATTR, None)
        build_v41_scheduled_batch(fb, PAGE)


@pytest.mark.usefixtures("stamps")
class TestFusedMetadata:
    def _builder(self, monkeypatch, captured):
        def prepare_prefill(batch, running_bs, running_tokens):
            captured.update(batch=batch, bs=running_bs, tokens=running_tokens)
            metadata = SimpleNamespace(
                engram_embeddings=None,
                step=SimpleNamespace(),
            )
            return metadata, None

        builder = SimpleNamespace(
            cache=object(),
            block_size=PAGE,
            geometry=SimpleNamespace(
                history_size=HISTORY, window_size=128, speculative_tokens=0
            ),
            max_bs=8,
            max_num_batched_tokens=4096,
            _decoder_replays=True,
            prepare_prefill=prepare_prefill,
            prepare_model_inputs=lambda ids, md: captured.update(ids=ids),
        )
        monkeypatch.setattr(bridge, "_V41_BUILDER", builder)
        # The pool's decoder row table (outside a SGLang forward context).
        monkeypatch.setattr(bridge, "_decoder_rows_from", lambda: np.zeros(8, np.int64))
        return builder

    def _fb(self):
        fb = _extend_fb([PAGE, 0], [3, 2], [1, 2])
        fb.input_ids = torch.tensor([11, 12, 13, 21, 22])
        fused = _fused([0], [1], [PAGE - R], [PAGE])
        setattr(fb, FUSED_ATTR, fused)
        return fb, fused

    def test_atom_stages_the_widened_inputs(self, monkeypatch, req_to_token):
        captured = {}
        self._builder(monkeypatch, captured)
        fb, _ = self._fb()
        positions = torch.tensor([PAGE, PAGE + 1, PAGE + 2, 0, 1])
        wide_ids, _, _ = expand_v41_forward_inputs(fb, fb.input_ids, positions)
        metadata = bridge.build_deepseek_v41_forward_metadata(fb)
        assert metadata.staged_outside_forward
        assert captured["ids"] is wide_ids
        assert captured["tokens"] == 5 + R and captured["bs"] == 2
        assert captured["batch"].num_scheduled_tokens.tolist() == [3 + R, 2]
        assert captured["batch"].decoder_replays


# ---------------------------------------------------------------------------
# 6. ATOM's cache and compress plans over a fused span
# ---------------------------------------------------------------------------


def _small_geometry():
    from atom.model_ops.attentions.pool_layout.v41_pool_geometry import (
        V41PoolGeometry,
    )

    return V41PoolGeometry(
        4, ((0, 2), (2, 1)), PAGE, 128, 512, 32, layer_ratios=(0, 1, 2), index_topk=4
    )


@pytest.mark.parametrize("prefix,suffix", [(PAGE, 6), (3 * PAGE, 200)])
def test_fused_compress_plans_write_only_from_the_hit(prefix, suffix):
    """One span over ``[P - R, P + suffix)`` with ``write_from = P``: every
    compressed group from the hit on is written, none below it."""
    from atom.model_ops.v4_kernels import make_compress_plans
    from atom.utils import CpuGpuBuffer

    geo = _small_geometry()
    assert {ratio for ratio, _ in geo.compress_ratios} == {1, 2}

    def plans(lengths, ends, write_from):
        width = int(lengths.max()) + 1
        buffers = {
            ratio: {
                name: CpuGpuBuffer(
                    width, 4, dtype=torch.int32, device="cpu", pin_memory=False
                )
                for name in ("compress", "write")
            }
            for ratio, _ in geo.compress_ratios
        }
        return make_compress_plans(
            lengths,
            ends,
            geo.compress_ratios,
            plan_buffers=buffers,
            extra_write=0,
            write_from=write_from,
        )

    ends = np.asarray([prefix + suffix], dtype=np.int32)
    fused = plans(
        np.asarray([R + suffix], dtype=np.int32),
        ends,
        np.asarray([prefix], dtype=np.int32),
    )
    # What an extend resuming at P compresses.
    resumed = plans(
        np.asarray([suffix], dtype=np.int32), ends, np.zeros(1, dtype=np.int32)
    )
    for ratio, plan in fused.items():
        assert plan.num_compress == resumed[ratio].num_compress == suffix // ratio
    # The replay rows alone write no group, and still refill the ring.
    below = plans(
        np.asarray([R], dtype=np.int32),
        np.asarray([prefix], dtype=np.int32),
        np.asarray([prefix], dtype=np.int32),
    )
    for ratio, plan in below.items():
        assert plan.num_compress == 0 and plan.num_write > 0, ratio


def _tiny_cache(slots=3, kv_dtype="bf16"):
    pytest.importorskip("aiter")
    from atom.model_ops.attentions.deepseek_v41.cache import PagedAttentionCache
    from atom.model_ops.attentions.pool_layout.v41_pool_geometry import (
        V41PoolGeometry,
    )

    # A window of 8 rows, PAGEs of 32 (the CPU floor of an FP8 index plane).
    geo = V41PoolGeometry(
        2,
        ((0, 1),),
        32,
        8,
        512,
        32,
        kv_dtype=kv_dtype,
        layer_ratios=(0, 1),
        index_topk=4,
    )
    return PagedAttentionCache(geo, 16, slots, "cpu")


def _span(rid, position, length, slot, offset=0):
    from atom.model_ops.attentions.deepseek_v41.metadata import RequestSpan

    return RequestSpan(rid, position, offset, length, slot)


class TestCacheContract:
    P, START = 64, 32

    def _dirty(self, cache, slot, cursor):
        # The last tenant's leftovers.
        cache.state_bytes[slot].fill_(77)
        cache.cursor[slot] = torch.tensor(cursor, dtype=torch.int64)

    def _main(self, cache, slot):
        step = cache.begin_step(
            [_span(9, self.P, 5, slot)], block_tables=[np.arange(1, 4, dtype=np.int32)]
        )
        return cache.prepare_state(step)

    def test_no_replay_raises(self):
        cache = _tiny_cache()
        self._dirty(cache, 1, [1000, 1, 2, 3])
        with pytest.raises(ValueError, match=f"needs state at {self.P}"):
            self._main(cache, 1)

    def test_coincidental_cursor_is_caught_by_the_stamp_guard(self):
        # The last tenant stopped exactly at P: ATOM's cursor check passes on
        # its state. Only the per-slot stamp tells the tenants apart.
        cache = _tiny_cache()
        self._dirty(cache, 1, [self.P, 1, 2, 3])
        self._main(cache, 1)
        state_gen = np.array([0, 3, 0])
        generation = torch.tensor([0, 4, 0])
        with pytest.raises(RuntimeError, match="not rebuilt"):
            check_state_stamps(state_gen, generation, [1], np.array([self.P]))

    def test_the_fused_span_resets_its_slot_and_resumes_past_the_hit(self):
        """One request's span starts below its hit beside a continuing one."""
        cache = _tiny_cache()
        self._dirty(cache, 1, [1000, 1, 2, 3])
        cache.state_bytes[2].fill_(55)
        cache.cursor[2] = torch.tensor([40, 7, 8, 9])
        length = self.P + 5 - self.START
        step = cache.begin_step(
            [_span(9, self.START, length, 1), _span(8, 40, 6, 2, offset=length)],
            block_tables=[
                np.arange(1, 4, dtype=np.int32),
                np.arange(4, 6, dtype=np.int32),
            ],
            floors=np.asarray([self.START, 0], np.int32),
            cached_until=np.zeros(2, np.int32),
            start_histories=np.asarray([[4, 5, 6], [-1, -1, -1]], np.int64),
        )
        histories = cache.prepare_state(step)
        # The hit's slot starts over at its floor with its seeded history; the
        # continuation keeps its state and history.
        assert histories.tolist() == [[4, 5, 6], [7, 8, 9]]
        window = cache.state.view("window")
        assert window[:, 1].count_nonzero() == 0
        assert bool((window[:, 2] != 0).all())
        cache.advance_cursor(step, histories)
        assert cache.cursor[1, 0] == self.P + 5 and cache.cursor[2, 0] == 46


def test_replay_layout_over_a_fused_step(req_to_token, stamps):
    pytest.importorskip("aiter")
    if not torch.cuda.is_available():
        pytest.skip("aiter's arch probe needs a visible GPU")
    from atom.model_ops.attentions.deepseek_v41.backend import (
        DeepseekV41MetadataBuilder,
    )
    from tests.model_ops.engram.test_hash_bounds import build, tiny_config

    mapping = build(tiny_config())
    width = mapping.config.max_ngram_size - 1
    fb = _extend_fb([2 * PAGE, 0], [5, 9], [1, 2])
    _carry(
        fb,
        _fused([0], [1], [2 * PAGE - R], [2 * PAGE], [[11, 12, 13][-width:]]),
    )
    batch = build_v41_scheduled_batch(fb, PAGE)
    builder = DeepseekV41MetadataBuilder.__new__(DeepseekV41MetadataBuilder)
    builder.engram = SimpleNamespace(mapping=mapping)
    builder.geometry = SimpleNamespace(history_size=width)
    builder.config = SimpleNamespace(image_token_id=None)
    resumes, floors, cached, histories = builder._replay_layout(
        batch.replaying, _spans(batch)
    )
    # The hit: nothing global below P, its window floored at P - R, its slot
    # rebuilt; the cold row beside it untouched by the replay.
    assert resumes.tolist() == [2 * PAGE, 0]
    assert floors.tolist() == [2 * PAGE - R, 0]
    assert cached.tolist() == [0, 0]
    expected = mapping.advance_history(
        None, mapping.compress_tokens(np.asarray([11, 12, 13][-width:], np.int64))
    )[0]
    assert histories[0].tolist() == np.asarray(expected).tolist()


# ---------------------------------------------------------------------------
# 7. Host PAGE rows under radix: a checkpoint drops the slot's copy
# ---------------------------------------------------------------------------


def test_a_checkpoint_drops_its_slots_host_rows(monkeypatch):
    import types

    class UnifiedRadixCache:
        def checkpoint_into_tree(self, req, *, up_to, **kwargs):
            if up_to < 0:
                raise RuntimeError("insert failed")
            return up_to

    module = types.ModuleType("sglang.srt.mem_cache.unified_radix_cache")
    module.UnifiedRadixCache = UnifiedRadixCache
    monkeypatch.setitem(sys.modules, "sglang.srt.mem_cache.unified_radix_cache", module)
    rows = [np.asarray([7, 3], np.int32), np.asarray([5], np.int32)]
    bridge._HOST_ROWS.record([3, 4], ["a", "b"], rows)
    radix._install_checkpoint_invalidation()
    radix._install_checkpoint_invalidation()  # idempotent
    req = SimpleNamespace(kv=SimpleNamespace(req_pool_idx=3))
    assert UnifiedRadixCache().checkpoint_into_tree(req, up_to=512) == 512
    assert bridge._HOST_ROWS.lookup([3], ["a"], [1]) is None
    assert bridge._HOST_ROWS.lookup([4], ["b"], [1])[0].tolist() == [5]
    # A failed insert may have rewritten part of the row: dropped too.
    req = SimpleNamespace(kv=SimpleNamespace(req_pool_idx=4))
    with pytest.raises(RuntimeError, match="insert failed"):
        UnifiedRadixCache().checkpoint_into_tree(req, up_to=-1)
    assert bridge._HOST_ROWS.lookup([4], ["b"], [1]) is None
    # A request without a row has nothing to drop.
    req = SimpleNamespace(kv=SimpleNamespace(req_pool_idx=None))
    assert UnifiedRadixCache().checkpoint_into_tree(req, up_to=0) == 0


# ---------------------------------------------------------------------------
# 8. The GPU page-check tool (tests/plugin/v41_radix_page_check)
# ---------------------------------------------------------------------------


def _page_check_tool():
    path = pathlib.Path(__file__).parent / "v41_radix_page_check" / "sitecustomize.py"
    spec = importlib.util.spec_from_file_location("v41_radix_page_check", path)
    tool = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tool)
    return tool


@pytest.mark.parametrize("kv_dtype", ["bf16", "fp8"])
def test_page_digests_see_main_and_index_writes(kv_dtype):
    tool = _page_check_tool()
    cache = _tiny_cache(kv_dtype=kv_dtype)
    pages = torch.tensor([1, 2, 3])
    before = tool.page_digests(cache, pages)
    assert torch.equal(tool.page_digests(cache, pages), before)
    # page 2's main rows: the flat PAGE (BF16) or the main source's data
    # plane (FP8, its own buffer)
    main = next(
        view for name, view in cache.page_regions if name in ("page", "main_0.data")
    )
    main.view(int(cache.num_pages), -1)[2, 5] = 1
    plane = next(iter(cache.index_planes.values()))[0]
    plane[3, 0, 1] = 9  # page 3's index rows
    plane[5, 0, 0] = 9  # a page outside the set
    changed = (tool.page_digests(cache, pages) != before).any(dim=1)
    assert changed.tolist() == [False, True, True]


def test_the_page_check_wraps_the_seam(monkeypatch, radix_on, caplog):
    tool = _page_check_tool()
    monkeypatch.setattr(radix, "run_v41_prefix_replay", radix.run_v41_prefix_replay)
    monkeypatch.setattr(radix, "run_with_fused_replay", radix.run_with_fused_replay)
    monkeypatch.setattr(tool, "tp_agreement", lambda count: "single")
    digests = iter([torch.zeros(2, 2)] * 3 + [torch.ones(2, 2)])
    monkeypatch.setattr(tool, "page_digests", lambda cache, pages: next(digests))
    tool.install(radix)
    tool.install(radix)  # idempotent
    world = _World(monkeypatch)
    world.runner.req_to_token_pool.req_to_token = torch.arange(
        8 * 4 * PAGE, dtype=torch.int32
    ).view(8, 4 * PAGE)
    bridge._V41_BUILDER.cache = object()

    def forward(worker, batch):
        getattr(batch, FUSED_ATTR).expanded = "widened"
        return "result"

    with caplog.at_level("INFO", logger="atom"):
        for changed in (False, True):
            batch = world.batch([2 * PAGE], [3])
            world.generation[3] += 1
            fused = radix.run_v41_prefix_replay(world.worker, batch)
            assert radix.run_with_fused_replay(
                forward, world.worker, batch, fused, (), {}
            )
    records = [r for r in caplog.records if "radix page check" in r.getMessage()]
    assert [r.levelname for r in records] == ["INFO", "ERROR"]
    assert "pages_changed=0" in records[0].getMessage()
    assert "pages_changed=2" in records[1].getMessage()
    assert tool._PENDING == {}


# ---------------------------------------------------------------------------
# 9. Admission: a 400 at the scheduler's queue door
# ---------------------------------------------------------------------------


def _queued_req(rid, overrides=None):
    return SimpleNamespace(rid=rid, positional_embed_overrides=overrides)


class TestOverrideAdmission:
    def test_positional_overrides_are_refused(self):
        error = radix.radix_request_refusal(_queued_req("p", overrides="pe"))
        assert "positional_embed_overrides" in error
        assert "--disable-radix-cache" in error
        assert radix.radix_request_refusal(_queued_req("t")) is None

    def test_the_scheduler_queue_door_is_wrapped(self, monkeypatch):
        Scheduler = _sglang_module("sglang.srt.managers.scheduler").Scheduler
        queued, rejected = [], []

        def original(self, req, is_retracted=False):
            queued.append((req.rid, is_retracted))

        monkeypatch.setattr(Scheduler, "_add_request_to_queue", original)
        monkeypatch.setattr(
            Scheduler,
            "_reject_sampling_mask_request",
            lambda self, req, error: rejected.append((req.rid, error)),
            raising=False,
        )
        radix._install_request_admission()
        radix._install_request_admission()  # idempotent
        door = Scheduler._add_request_to_queue
        assert door._atom_v41_radix_patched
        scheduler = object.__new__(Scheduler)
        door(scheduler, _queued_req("text"))
        door(scheduler, _queued_req("pe", overrides="pe"))
        door(scheduler, _queued_req("back"), is_retracted=True)
        assert queued == [("text", False), ("back", True)]
        assert [rid for rid, _ in rejected] == ["pe"]
        assert "positional_embed_overrides" in rejected[0][1]

    def test_sglang_answers_the_refusal_with_a_400(self):
        """The door's answer is SGLang's own: a 400 abort streamed without
        queueing or running the model."""
        Scheduler = _sglang_module("sglang.srt.managers.scheduler").Scheduler
        streamed, aborted = [], []
        scheduler = object.__new__(Scheduler)
        scheduler.output_streamer = SimpleNamespace(
            stream_output=lambda reqs, return_logprob: streamed.append(
                [r.rid for r in reqs]
            )
        )
        req = _queued_req("pe", overrides="pe")
        req.return_logprob = False
        req.time_stats = SimpleNamespace(
            trace_ctx=SimpleNamespace(abort=lambda abort_info: aborted.append(1))
        )
        error = radix.radix_request_refusal(req)
        Scheduler._reject_sampling_mask_request(scheduler, req, error)
        reason = req.finished_reason
        assert type(reason).__name__ == "FINISH_ABORT"
        assert reason.message == error and int(reason.status_code) == 400
        assert streamed == [["pe"]] and aborted == [1]


# ---------------------------------------------------------------------------
# 10. DSpark under the replay
# ---------------------------------------------------------------------------


VERIFY = ForwardMode.TARGET_VERIFY
DSPARK_WIDTH = bridge.V41_DSPARK_VERIFY_TOKENS


def _verify_fb(prefix, slots):
    ends = [p + DSPARK_WIDTH for p in prefix]
    return SimpleNamespace(
        forward_mode=VERIFY,
        rids=[f"v{i}" for i in range(len(slots))],
        batch_size=len(slots),
        seq_lens=torch.tensor(ends, dtype=torch.int64),
        seq_lens_cpu=torch.tensor(ends, dtype=torch.int64),
        req_pool_indices=torch.tensor(slots, dtype=torch.int64),
        req_pool_indices_cpu=None,
        atom_v41_req_pool_indices_cpu=torch.tensor(slots, dtype=torch.int64),
        spec_info=SimpleNamespace(
            draft_token_num=DSPARK_WIDTH,
            live_seq_lens_cpu=torch.tensor(prefix, dtype=torch.int64),
        ),
    )


class TestDSparkUnderTheReplay:
    def test_dspark_is_not_a_radix_refusal(self, launch):
        launch.bags["spec"].speculative_algorithm = "DSPARK"
        bags = launch.bags
        refusals = bridge._radix_refusals(bags["memory"], bags["schedule"])
        assert [name for name, on in refusals if on] == []

    def test_verify_rounds_read_no_stamps(self, monkeypatch, req_to_token, radix_on):
        monkeypatch.setattr(
            bridge, "_radix_stamps", lambda: pytest.fail("verify read stamps")
        )
        batch = build_v41_scheduled_batch(_verify_fb([PAGE + 3, 40], [1, 2]), PAGE)
        assert batch.replaying == {}
        assert batch.replay_rows.tolist() == [0, 0]
        assert batch.num_scheduled_tokens.tolist() == [DSPARK_WIDTH] * 2

    def test_the_seam_leaves_verify_batches_alone(self, monkeypatch, radix_on):
        world = _World(monkeypatch)
        batch = world.batch([PAGE], [1], mode=VERIFY)
        assert run_v41_prefix_replay(world.worker, batch) is None
        assert not world.pool.state_gen.any()
        assert not radix._LOGGED_FIRST_FUSED

    def test_the_seam_leaves_a_deferred_decode_round_alone(self, monkeypatch, radix_on):
        """`dspark_deferred_seq_lens_patch` leaves a DSpark decode round's
        host ``seq_lens_cpu`` / ``seq_lens_sum`` None until its verify: the
        seam decides on the mode alone and never reads them (a prefill keeps
        upstream's resolve, so an EXTEND always has them)."""

        class _Deferred(SimpleNamespace):
            def __getattribute__(self, name):
                if name in ("seq_lens_cpu", "seq_lens_sum", "prefix_lens"):
                    pytest.fail(f"the seam read {name} of a decode round")
                return super().__getattribute__(name)

        world = _World(monkeypatch)
        for mode in (DECODE, VERIFY):
            batch = _Deferred(forward_mode=mode, spec_algorithm="DSPARK")
            assert run_v41_prefix_replay(world.worker, batch) is None
        assert not world.pool.state_gen.any()

    def test_dspark_prefill_hit_rides_in_its_step(self, monkeypatch, radix_on):
        """SGLang's ``DSparkWorkerV2._forward_prefill`` hands the target
        worker the EXTEND ``ScheduleBatch`` (FULL capture): the seam plans the
        same record as without speculation."""
        world = _World(monkeypatch)
        req = _req(range(500, 500 + 2 * PAGE + 9), rid="d")
        batch = world.batch([2 * PAGE], [3], [req])
        batch.spec_algorithm = "DSPARK"
        fused = run_v41_prefix_replay(world.worker, batch)
        assert fused.starts.tolist() == [2 * PAGE - R]
        assert fused.ends.tolist() == [2 * PAGE]


# ---------------------------------------------------------------------------
# 11. The prefill fixes under radix: rows copied at allocation, prefill
#     token graphs on a fused step
# ---------------------------------------------------------------------------


class _SyncedEvent:
    def record(self, stream=None):
        pass

    def synchronize(self):
        pass


@pytest.mark.usefixtures("stamps")
class TestExtendRowsUnderRadix:
    """`ExtendRowRead` with radix cache on: the copy holds a hit's prefix
    pages and the replay's rows below P; a checkpoint after the copy refuses
    it."""

    @pytest.fixture(autouse=True)
    def _rows(self, monkeypatch, req_to_token):
        monkeypatch.setattr(bridge, "_EXTEND_ROWS", bridge.ExtendRowRead())
        monkeypatch.setattr(
            bridge, "_device_module", lambda device: SimpleNamespace(Event=_SyncedEvent)
        )
        monkeypatch.setattr(
            bridge, "_V41_BUILDER", SimpleNamespace(cache=object(), block_size=PAGE)
        )
        self.reads = []
        original = bridge._block_rows

        def counted(*args, **kwargs):
            self.reads.append(len(args[1]))
            return original(*args, **kwargs)

        monkeypatch.setattr(bridge, "_block_rows", counted)
        return req_to_token

    def _alloc(self, table, slots, rids, ends):
        batch = SimpleNamespace(
            reqs=[SimpleNamespace(rid=rid) for rid in rids],
            seq_lens_cpu=torch.tensor(ends, dtype=torch.int64),
            req_to_token_pool=SimpleNamespace(req_to_token=table),
        )
        slots_cpu = torch.tensor(slots, dtype=torch.int64)
        bridge.read_v41_extend_rows_at_alloc(batch, slots_cpu, slots_cpu)

    def _hit_step(self):
        # r0 hits at 2 PAGEs (replaying [2 PAGE - R, 2 PAGE)); r1 cold.
        fb = _extend_fb([2 * PAGE, 0], [5, 9], [1, 2])
        fb.req_pool_indices_cpu = torch.tensor([1, 2], dtype=torch.int64)
        return _carry(fb, _fused([0], [1], [2 * PAGE - R], [2 * PAGE]))

    def test_a_hit_takes_its_copy_prefix_pages_included(self, req_to_token):
        self._alloc(req_to_token, [1, 2], ["r0", "r1"], [2 * PAGE + 5, 9])
        batch = build_v41_scheduled_batch(self._hit_step(), PAGE)
        assert self.reads == []
        # The prefix pages the tree handed back, with the extend's own: the
        # replay's rows [2 PAGE - R, 2 PAGE) sit in page 3.
        assert [row.tolist() for row in batch.block_tables] == [[7, 3, 9], [5]]
        assert batch.num_scheduled_tokens.tolist() == [5 + R, 9]

    def test_a_checkpoint_after_the_copy_refuses_it(self, req_to_token):
        self._alloc(req_to_token, [1, 2], ["r0", "r1"], [2 * PAGE + 5, 9])
        # The tree repoints slot 1's row in place (`checkpoint_into_tree`).
        req_to_token[1, :PAGE] = 13 * PAGE + torch.arange(PAGE)
        bridge.forget_host_page_rows(1)
        batch = build_v41_scheduled_batch(self._hit_step(), PAGE)
        assert self.reads == [2]
        assert [row.tolist() for row in batch.block_tables] == [[13, 3, 9], [5]]

    def test_a_checkpoint_before_the_copy_keeps_it(self, req_to_token):
        # A chunked request is checkpointed while its next batch is built,
        # before that batch's allocation queues the copy.
        req_to_token[1, :PAGE] = 13 * PAGE + torch.arange(PAGE)
        bridge.forget_host_page_rows(1)
        self._alloc(req_to_token, [1, 2], ["r0", "r1"], [2 * PAGE + 5, 9])
        batch = build_v41_scheduled_batch(self._hit_step(), PAGE)
        assert self.reads == []
        assert [row.tolist() for row in batch.block_tables] == [[13, 3, 9], [5]]


@pytest.mark.usefixtures("stamps")
class TestPrefillTokenGraphsUnderRadix:
    """A fused step pads to the bucket holding its widened rows, as native
    pads a replaying prefill; SGLang gets its own rows back."""

    @pytest.fixture(autouse=True)
    def _graphs(self, monkeypatch):
        from atom.utils import region_graph

        graphs = bridge._PrefillTokenGraphs()
        graphs.sizes = (256, 512)
        monkeypatch.setattr(bridge, "_PREFILL_GRAPHS", graphs)
        monkeypatch.setattr(
            region_graph.TokenGraph,
            "bucket",
            staticmethod(
                lambda tokens, phase, multiple_of=1: next(
                    (n for n in (256, 512) if n >= tokens), None
                )
            ),
        )
        return graphs

    def _builder(self, monkeypatch, captured):
        def prepare_prefill(batch, running_bs, running_tokens):
            captured.update(batch=batch, bs=running_bs, tokens=running_tokens)
            return (
                SimpleNamespace(
                    engram_embeddings=None,
                    step=SimpleNamespace(),
                ),
                torch.arange(1000, 1000 + running_tokens),
            )

        builder = SimpleNamespace(
            cache=object(),
            block_size=PAGE,
            geometry=SimpleNamespace(
                history_size=HISTORY, window_size=128, speculative_tokens=0
            ),
            max_bs=8,
            max_num_batched_tokens=4096,
            _decoder_replays=True,
            prepare_prefill=prepare_prefill,
            prepare_model_inputs=lambda ids, md: captured.update(ids=ids.clone()),
        )
        monkeypatch.setattr(bridge, "_V41_BUILDER", builder)
        monkeypatch.setattr(bridge, "_decoder_rows_from", lambda: np.zeros(8, np.int64))
        return builder

    def _fused_step(self):
        fb = _extend_fb([PAGE, 0], [3, 2], [1, 2])
        fb.input_ids = torch.tensor([11, 12, 13, 21, 22])
        setattr(fb, FUSED_ATTR, _fused([0], [1], [PAGE - R], [PAGE]))
        positions = torch.tensor([PAGE, PAGE + 1, PAGE + 2, 0, 1])
        wide_ids, wide_pos, keep = expand_v41_forward_inputs(
            fb, fb.input_ids, positions
        )
        return fb, wide_ids, wide_pos, keep

    def test_a_fused_step_pads_on_its_widened_rows(self, monkeypatch, req_to_token):
        captured = {}
        self._builder(monkeypatch, captured)
        fb, wide_ids, _, _ = self._fused_step()
        metadata = bridge.build_deepseek_v41_forward_metadata(fb)
        # 5 + R = 133 rows ran by ATOM: the 256 bucket, not SGLang's 5 rows'.
        assert captured["tokens"] == 256 and captured["bs"] == 2
        assert captured["batch"].num_scheduled_tokens.tolist() == [3 + R, 2]
        assert captured["batch"].replaying[0].until == PAGE
        padded = metadata.sglang_padded_inputs
        assert padded.scheduled_tokens == 5 + R
        assert padded.input_ids.shape[0] == 256
        assert torch.equal(padded.input_ids[: 5 + R], wide_ids)
        assert not padded.input_ids[5 + R :].any()
        assert torch.equal(captured["ids"], padded.input_ids)
        assert padded.positions.tolist() == list(range(1000, 1256))

    def test_a_widened_step_past_every_bucket_runs_eagerly(
        self, monkeypatch, req_to_token
    ):
        captured = {}
        self._builder(monkeypatch, captured)
        fb = _extend_fb([PAGE, 0], [3, 400], [1, 4])
        fb.input_ids = torch.arange(403)
        setattr(fb, FUSED_ATTR, _fused([0], [1], [PAGE - R], [PAGE]))
        positions = torch.cat([torch.arange(PAGE, PAGE + 3), torch.arange(400)])
        expand_v41_forward_inputs(fb, fb.input_ids, positions)
        # SGLang's 403 rows would fit 512; ATOM's 531 do not.
        metadata = bridge.build_deepseek_v41_forward_metadata(fb)
        assert captured["tokens"] == 403 + R
        assert getattr(metadata, "sglang_padded_inputs", None) is None

    def test_the_runtime_hands_sglang_its_rows_out_of_the_bucket(
        self, monkeypatch, req_to_token
    ):
        from atom.plugin.sglang.runtime import forward_context as plugin_fc
        from atom.plugin.sglang.runtime import model_arch
        from atom.utils.forward_context import get_forward_context

        captured = {}
        self._builder(monkeypatch, captured)
        fb = _extend_fb([PAGE, 0], [3, 2], [1, 2])
        fb.input_ids = torch.tensor([11, 12, 13, 21, 22])
        setattr(fb, FUSED_ATTR, _fused([0], [1], [PAGE - R], [PAGE]))
        spec = model_arch.SGLangModelAdapterSpec(
            build_forward_metadata=lambda config, batch, positions: (
                bridge.build_deepseek_v41_forward_metadata(batch)
            ),
        )
        monkeypatch.setattr(
            model_arch, "resolve_model_arch_spec", lambda config: ("X", spec)
        )
        monkeypatch.setattr(
            plugin_fc, "_resolve_running_tokens_are_unified", lambda *a: True
        )
        atom_config = SimpleNamespace(
            hf_config=None,
            enable_dp_attention=False,
            parallel_config=SimpleNamespace(data_parallel_size=1, data_parallel_rank=0),
            compilation_config=SimpleNamespace(static_forward_context={}),
        )
        with plugin_fc.SGLangPluginRuntime(
            atom_config=atom_config,
            forward_batch=fb,
            positions=torch.tensor([PAGE, PAGE + 1, PAGE + 2, 0, 1]),
            input_ids=fb.input_ids,
            expand_forward_inputs=expand_v41_forward_inputs,
        ) as runtime:
            forward = get_forward_context()
            assert forward.tr_graph == "replay"
            assert forward.context.scheduled_tokens == 5 + R
            assert runtime.input_ids.shape[0] == 256
            assert runtime.positions.shape[0] == 256
            hidden = torch.arange(256)[:, None].expand(256, 2)
            out = runtime.trim_output(hidden)
            # SGLang's five rows: r0's own three after its R replay rows,
            # then r1's two.
            assert out[:, 0].tolist() == [R, R + 1, R + 2, R + 3, R + 4]
            aux = runtime.trim_output((hidden, hidden + 1))
            assert aux[1][:, 0].tolist() == [R + 1, R + 2, R + 3, R + 4, R + 5]
