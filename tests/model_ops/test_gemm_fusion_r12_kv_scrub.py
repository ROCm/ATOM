# SPDX-License-Identifier: MIT
# After a gemm_fusion device error verdict, the KV blocks the aborted requests held are zeroed (on
# every rank) before they can be handed to another request.
#
# Why (Evidence, GPU58): while the error word is set the fused path writes NaN into K/V/index entries
# (sticky NaN mode); the stock M3 paged path later reads UNWRITTEN slots of a later sequence and masks
# them by multiplying with 0, and NaN * 0 = NaN. Zeroing the KV pool after a verdict made later outputs
# byte-identical again; finite stale bytes (plain block recycling) did not diverge.
#
# Torch-free: the real methods are extracted from source with ast and run against stand-ins. The one
# real-tensor test (a CPU MhaKvPool) is skipped without torch and runs in CI.
import ast
import logging
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[2]


def _class_node(rel, cls_name):
    tree = ast.parse((ROOT / rel).read_text(encoding="utf-8"))
    return next(
        n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == cls_name
    )


def _extract(rel, cls_name, methods, namespace):
    cls = _class_node(rel, cls_name)
    cls.body = [
        n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in methods
    ]
    assert {n.name for n in cls.body} == set(methods), "a method moved or was renamed"
    cls.bases, cls.decorator_list = [], []
    module = ast.Module(
        [ast.parse("from __future__ import annotations").body[0], cls], []
    )
    ns = dict(namespace)
    exec(  # noqa: S102 - real source, extracted by ast, run against stand-ins
        compile(ast.fix_missing_locations(module), rel, "exec"), ns
    )
    return ns[cls_name]


POOL = "atom/model_ops/attentions/mha_kv_pool.py"
BUILDER_BASE = "atom/model_ops/attentions/backends.py"
BUILDER_AITER = "atom/model_ops/attentions/aiter_attention.py"
RUNNER = "atom/model_engine/model_runner.py"
SCHED = "atom/model_engine/scheduler.py"
ENGINE = "atom/model_engine/engine_core.py"


# ------------------------------------------------------------------------------------ the pool
class _View:
    """A field view ``[layers, entries, *shape]``: only its device and indexed writes are used."""

    device = "dev"

    def __init__(self):
        self.writes = []

    def __setitem__(self, key, value):
        self.writes.append((key, value))


def _pool(entries=8, views=("k", "v", "k_scale", "v_scale", "index")):
    torch_stub = SimpleNamespace(
        tensor=lambda ids, dtype=None, device=None: ("IDX", tuple(ids), device),
        long="long",
    )
    pool = _extract(POOL, "MhaKvPool", {"zero_entries"}, {"torch": torch_stub})()
    pool._views = {name: _View() for name in views}
    pool.cache = SimpleNamespace(entries=entries)
    return pool


def test_zero_entries_writes_zeros_to_every_field_of_every_layer_at_those_entries():
    pool = _pool()
    assert pool.zero_entries([5, 2, 5]) == 2  # sorted, de-duplicated
    assert set(pool._views) == {"k", "v", "k_scale", "v_scale", "index"}
    for view in pool._views.values():
        ((key, value),) = view.writes
        assert value == 0
        assert key == (slice(None), ("IDX", (2, 5), "dev"))  # all layers, those entries


def test_zero_entries_with_nothing_to_do_touches_nothing():
    pool = _pool()
    assert pool.zero_entries([]) == 0
    assert all(not v.writes for v in pool._views.values())


@pytest.mark.parametrize("bad", [[-1], [8], [3, 9]])
def test_zero_entries_refuses_entries_outside_the_pool(bad):
    pool = _pool(entries=8)
    with pytest.raises(ValueError, match="outside the pool"):
        pool.zero_entries(bad)
    assert all(not v.writes for v in pool._views.values())  # nothing was written


def test_a_released_pool_has_nothing_to_zero():
    pool = _pool()
    pool._views = {}
    assert pool.zero_entries([1]) == 0


def test_real_pool_zeroes_exactly_the_given_entries_in_every_layer_and_field():
    torch = pytest.importorskip("torch")
    from atom.model_ops.attentions.mha_kv_pool import MhaKvPool
    from atom.model_ops.attentions.pool_layout.entry_arena import EntryField

    pool = MhaKvPool(
        layers=3,
        block_size=16,
        num_kv_heads=1,
        head_dim=128,
        kv_dtype=torch.float8_e4m3fn,
        extra_fields=[EntryField("index", 2, (64,), torch.uint8)],
    )
    pool.allocate(6, "cpu")
    for view in pool._views.values():  # a previous occupant: every byte non-zero
        view.view(torch.uint8).fill_(7) if view.dtype != torch.uint8 else view.fill_(7)
    assert pool.zero_entries([1, 4]) == 2
    for name, view in pool._views.items():
        flat = view.view(torch.uint8) if view.dtype != torch.uint8 else view
        assert not flat[:, [1, 4]].any(), name  # wiped in every layer
        assert (flat[:, [0, 2, 3, 5]] == 7).all(), name  # neighbours untouched


# ---------------------------------------------------------------------------------- the builder
def test_the_base_builder_reports_it_has_no_per_block_zeroing():
    base = _extract(BUILDER_BASE, "AttentionMetadataBuilder", {"zero_kv_entries"}, {})()
    assert base.zero_kv_entries([1, 2]) is False


def test_the_mha_builder_zeroes_every_pool_and_says_so():
    builder = _extract(
        BUILDER_AITER, "AiterAttentionMetadataBuilder", {"zero_kv_entries"}, {}
    )()
    zeroed = []
    builder.kv_pools = {
        "g1": SimpleNamespace(
            zero_entries=lambda ids: zeroed.append(("g1", list(ids)))
        ),
        "g2": SimpleNamespace(
            zero_entries=lambda ids: zeroed.append(("g2", list(ids)))
        ),
    }
    assert builder.zero_kv_entries([4, 9]) is True
    assert zeroed == [("g1", [4, 9]), ("g2", [4, 9])]
    builder.kv_pools = {}
    assert builder.zero_kv_entries([4]) is False  # no pool: nothing was zeroed


# ----------------------------------------------------------------------------------- the runner
SYNCS = []


def _runner(zero_result=True, kv_transfer=None):
    ns = {
        "torch": SimpleNamespace(
            cuda=SimpleNamespace(synchronize=lambda: SYNCS.append(1))
        ),
        "logger": logging.getLogger("atom.test_scrub_runner"),
    }
    runner = _extract(RUNNER, "ModelRunner", {"scrub_kv_blocks"}, ns)()
    SYNCS.clear()
    runner.zero_calls = []
    runner.config = SimpleNamespace(kv_transfer_config=kv_transfer)
    runner.attn_metadata_builder = SimpleNamespace(
        zero_kv_entries=lambda ids: runner.zero_calls.append(list(ids)) or zero_result
    )
    return runner


def test_the_runner_zeroes_the_blocks_then_synchronises():
    runner = _runner()
    assert runner.scrub_kv_blocks([3, 4, 9]) == 3
    assert runner.zero_calls == [[3, 4, 9]] and SYNCS == [1]


def test_the_runner_does_nothing_for_no_blocks():
    runner = _runner()
    assert runner.scrub_kv_blocks([]) == 0
    assert runner.zero_calls == [] and SYNCS == []


def test_the_runner_leaves_the_pool_alone_with_a_kv_transfer_connector():
    runner = _runner(kv_transfer={"kv_role": "kv_producer"})
    assert runner.scrub_kv_blocks([3]) == 0
    assert runner.zero_calls == [] and SYNCS == []


def test_the_runner_reports_a_builder_without_per_block_zeroing(caplog):
    runner = _runner(zero_result=False)
    with caplog.at_level(logging.WARNING, logger="atom.test_scrub_runner"):
        assert runner.scrub_kv_blocks([3, 4]) == 0
    assert "no per-block KV zeroing" in caplog.text and SYNCS == []


# --------------------------------------------------------------------------------- the scheduler
SCHED_METHODS = {
    "postprocess",
    "_abort_for_device_error",
    "_freed_blocks",
    "take_kv_scrub",
}


class _Seq:
    def __init__(self, sid, blocks):
        self.id, self.block_table = sid, list(blocks)


def _scheduler(seqs, refs):
    """``seqs`` run; ``refs`` = {block: ref_count}. The body frees aborted requests' blocks like
    BlockManager.deallocate does: one reference dropped per block, then the table cleared.
    """
    ns = {"logger": logging.getLogger("atom.test_scrub_sched")}
    sched = _extract(SCHED, "Scheduler", SCHED_METHODS, ns)()
    sched.running = list(seqs)
    sched.aborted = set()
    sched.abort_request = lambda rid: sched.aborted.add(rid) or True

    def body(_seqs, _out, _stream, _batch):
        for seq in sched.running:
            if seq.id in sched.aborted:
                for b in seq.block_table:
                    refs[b] -= 1
                seq.block_table.clear()
        return []

    sched._postprocess_impl = body
    sched.block_manager = SimpleNamespace(
        clear_cache=lambda: None,
        kv=SimpleNamespace(block=lambda b: SimpleNamespace(ref_count=refs[b])),
    )
    return sched


def _verdict(ids):
    return SimpleNamespace(aborted_req_ids=ids)


def test_the_blocks_of_the_aborted_requests_are_reported_once_they_are_freed():
    refs = {1: 1, 2: 1, 3: 1, 4: 1}
    sched = _scheduler([_Seq(10, [1, 2]), _Seq(11, [3]), _Seq(12, [4])], refs)
    sched.postprocess([], _verdict([10, 11]))
    assert sched.take_kv_scrub() == [
        1,
        2,
        3,
    ]  # request 12 was not aborted: block 4 stays
    assert sched.take_kv_scrub() == []  # handed out once


def test_a_block_a_surviving_request_still_shares_is_never_reported():
    """Zeroing it would destroy the survivor's (finite, valid) KV."""
    refs = {1: 1, 2: 2, 3: 1}  # block 2 is shared by the aborted request and a survivor
    sched = _scheduler([_Seq(10, [1, 2]), _Seq(11, [2, 3])], refs)
    sched.postprocess([], _verdict([10]))
    assert sched.take_kv_scrub() == [1]


def test_the_blocks_are_read_before_the_body_frees_them():
    """The body clears every aborted request's block table, so they must be captured first."""
    refs = {5: 1, 6: 1}
    seq = _Seq(10, [5, 6])
    sched = _scheduler([seq], refs)
    sched.postprocess([], _verdict([10]))
    assert seq.block_table == []  # the body really did clear it
    assert sched.take_kv_scrub() == [5, 6]


def test_no_verdict_means_nothing_to_scrub():
    sched = _scheduler([_Seq(10, [1])], {1: 1})
    sched._postprocess_impl = lambda *a: []
    sched.postprocess([], SimpleNamespace(aborted_req_ids=None))
    assert sched.take_kv_scrub() == []


def test_an_empty_verdict_scrubs_nothing():
    sched = _scheduler([_Seq(10, [1])], {1: 1})
    sched.postprocess([], _verdict([]))
    assert sched.take_kv_scrub() == []


def test_a_stale_request_id_in_the_verdict_holds_no_blocks():
    """A request that finished inside the window is gone: nothing to read, nothing to zero."""
    refs = {1: 1}
    sched = _scheduler([_Seq(10, [1])], refs)
    sched.postprocess([], _verdict([999]))
    assert sched.take_kv_scrub() == []


# -------------------------------------------------------------------------------- the engine core
def test_the_engine_core_has_every_rank_zero_the_blocks_before_the_next_step():
    calls = []
    core = _extract(ENGINE, "EngineCore", {"_scrub_kv_after_verdict"}, {})()
    core.scheduler = SimpleNamespace(take_kv_scrub=lambda: [4, 5])
    core.runner_mgr = SimpleNamespace(call_func=lambda *a, **k: calls.append((a, k)))
    core._scrub_kv_after_verdict()
    assert calls == [(("scrub_kv_blocks", [4, 5]), {"wait_out": True})]


@pytest.mark.parametrize(
    "scheduler",
    [SimpleNamespace(take_kv_scrub=list), SimpleNamespace()],
    ids=["nothing to scrub", "a scheduler without the hook"],
)
def test_the_engine_core_makes_no_call_when_there_is_nothing_to_scrub(scheduler):
    calls = []
    core = _extract(ENGINE, "EngineCore", {"_scrub_kv_after_verdict"}, {})()
    core.scheduler = scheduler
    core.runner_mgr = SimpleNamespace(call_func=lambda *a, **k: calls.append(a))
    core._scrub_kv_after_verdict()
    assert calls == []


def test_the_scrub_runs_right_after_postprocess_in_the_step_that_serves_requests():
    """It must come before anything can schedule: the very next statement after the postprocess."""
    cls = _class_node(ENGINE, "EngineCore")
    found = 0
    for fn in (n for n in cls.body if isinstance(n, ast.FunctionDef)):
        for i, stmt in enumerate(fn.body):
            if "self.scheduler.postprocess(" in ast.unparse(stmt):
                found += 1
                assert ast.unparse(fn.body[i + 1]) == "self._scrub_kv_after_verdict()"
    assert found == 1
