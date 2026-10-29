# SPDX-License-Identifier: MIT
# Torch-free pins for the places where main, the SP series and the gemm_fusion branch meet.
#
# CI (CPU torch) caught what a no-torch machine cannot run: the cache key, the warmup budget and the
# runner methods the SP tests extract all changed under the rebase. These tests execute the REAL source
# (extracted with ast, nothing imported from atom) with small stand-ins, so they also run without torch,
# numpy or aiter.
import ast
import hashlib
import itertools
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[2]


def _module_tree(rel):
    return ast.parse((ROOT / rel).read_text(encoding="utf-8"))


def _run(nodes, namespace, rel):
    module = ast.Module(
        [ast.parse("from __future__ import annotations").body[0], *nodes], []
    )
    ns = dict(namespace)
    exec(  # noqa: S102 - real source, extracted by ast, run against stand-ins
        compile(ast.fix_missing_locations(module), rel, "exec"), ns
    )
    return ns


def _class_with_methods(rel, cls_name, method_names, namespace):
    cls = next(
        n
        for n in _module_tree(rel).body
        if isinstance(n, ast.ClassDef) and n.name == cls_name
    )
    cls.body = [
        n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in method_names
    ]
    assert {n.name for n in cls.body} == set(
        method_names
    ), "a method moved or was renamed"
    assert not any(
        n.decorator_list for n in cls.body
    ), "a decorated method cannot be extracted"
    cls.bases, cls.decorator_list = [], []
    return _run([cls], namespace, rel)[cls_name]


# ------------------------------------------------------------------------------ Config.compute_hash
def _compute_hash():
    cls = next(
        n
        for n in _module_tree("atom/config.py").body
        if isinstance(n, ast.ClassDef) and n.name == "Config"
    )
    fn = next(
        n
        for n in cls.body
        if isinstance(n, ast.FunctionDef) and n.name == "compute_hash"
    )
    fn.decorator_list = []
    ns = _run(
        [fn],
        {
            "hashlib": hashlib,
            "Any": Any,
            "envs": SimpleNamespace(
                ATOM_REPLICATE_VOCAB_EMBED=False,
                ATOM_FUSE_MXFP8_SHARED_INTO_MXFP4=False,
            ),
        },
        "atom/config.py",
    )
    return ns["compute_hash"]


# exactly the stand-in tests/test_compile_hash_spec.py (main) builds: no SP / expert / gemm_fusion field
MAIN_STUB = {
    "quant_config": None,
    "compilation_config": None,
    "parallel_config": None,
    "tensor_parallel_size": 8,
    "prefill_context_parallel_size": 1,
    "dcp_config": None,
    "enable_dp_attention": True,
    "index_cache_dtype": "fp4",
    "enable_decoder_swa_bounded_replay": True,
    "hf_config": SimpleNamespace(),
    "speculative_config": None,
    # main's #2688-era factors, read directly (not via getattr)
    "max_model_len": 4096,
    "max_num_seqs": 8,
}
NEUTRAL = {
    "sequence_parallel_size": 1,
    "enable_expert_parallel": False,
    "moe_all2all_backend": None,
    "moe_backend": None,
    "gemm_fusion": SimpleNamespace(enabled=False, profile=""),
}


def _key(**over):
    return _compute_hash()(SimpleNamespace(**{**MAIN_STUB, **over}))


def test_compute_hash_accepts_the_minimal_config_main_s_cache_key_tests_build():
    """CI failure: main's tests build a config without the fields the SP series and this branch added to
    the key (sequence_parallel_size, the expert layout, gemm_fusion) and compute_hash raised.
    """
    key = _key()
    assert isinstance(key, str) and len(key) == 10


def test_the_added_fields_at_their_neutral_values_leave_the_key_unchanged():
    """A real Config always has them; the tolerant read must give the SAME key as an absent field, so
    OFF / default runs keep their artifacts."""
    assert _key() == _key(**NEUTRAL)


@pytest.mark.parametrize(
    "over",
    [
        {"sequence_parallel_size": 4},
        {"enable_expert_parallel": True},
        {"moe_all2all_backend": "mori"},
        {"moe_backend": "triton"},
        {"gemm_fusion": SimpleNamespace(enabled=True, profile="p1")},
    ],
)
def test_each_field_this_branch_added_still_keys_the_compiled_artifact(over):
    assert _key(**{**NEUTRAL, **over}) != _key(**NEUTRAL)


def test_gemm_fusion_profiles_key_distinct_artifacts_and_off_never_does():
    def on(p):
        return SimpleNamespace(enabled=True, profile=p)

    assert _key(gemm_fusion=on("a")) != _key(gemm_fusion=on("b"))
    assert _key(gemm_fusion=SimpleNamespace(enabled=False, profile="a")) == _key()


def test_main_s_own_draft_width_factor_survived_the_merge():
    def spec(k):
        return SimpleNamespace(method="dspark", num_speculative_tokens=k)

    assert _key(speculative_config=spec(3)) != _key(speculative_config=spec(6))
    assert _key(speculative_config=spec(3)) != _key(speculative_config=None)


# --------------------------------------------------------------------------------- GemmFusionConfig
def _gemm_fusion_config():
    from dataclasses import dataclass, fields

    cls = next(
        n
        for n in _module_tree("atom/config.py").body
        if isinstance(n, ast.ClassDef) and n.name == "GemmFusionConfig"
    )
    return _run([cls], {"dataclass": dataclass, "fields": fields}, "atom/config.py")[
        "GemmFusionConfig"
    ]


def test_a_non_string_profile_is_a_type_error_and_an_unknown_key_a_value_error():
    """CI ruff (TRY004): an isinstance check must raise TypeError."""
    cfg = _gemm_fusion_config()
    with pytest.raises(TypeError, match="must be a string"):
        cfg(profile=3)
    with pytest.raises(ValueError, match="Unknown --gemm-fusion-config key"):
        cfg.from_dict({"nope": 1})
    assert cfg().enabled is False and cfg.from_dict({"profile": "p"}).enabled is True


# ------------------------------------------------------------ the ModelRunner methods the SP tests run
class _NP:
    """Just enough numpy for warmup_model (array(...).sum(), int32)."""

    int32 = "int32"

    @staticmethod
    def array(values, dtype=None):
        return SimpleNamespace(sum=lambda: sum(values))


def _runner_namespace(sp_on, dp_world=1):
    ids = itertools.count()
    forward_context = []
    return {
        "np": _NP,
        "time": SimpleNamespace(time=lambda: 0.0),
        "torch": SimpleNamespace(
            cuda=SimpleNamespace(
                empty_cache=lambda: None, reset_peak_memory_stats=lambda: None
            )
        ),
        "logger": SimpleNamespace(
            warning=lambda *a, **k: None, info=lambda *a, **k: None
        ),
        "get_dp_group": lambda: SimpleNamespace(world_size=dp_world),
        "sp_is_enabled": lambda: sp_on,
        "Sequence": lambda tokens, block_size: SimpleNamespace(id=next(ids)),
        "ScheduledBatch": lambda **kw: SimpleNamespace(**kw),
        "Context": SimpleNamespace,
        "set_forward_context": lambda **kw: forward_context.append(kw),
        "_forward_context": forward_context,
    }


WARMUP_METHODS = {"warmup_model", "warmup_max_tokens"}


@pytest.mark.parametrize(
    "sp_on,pcp,expected",
    [
        (
            True,
            4,
            17,
        ),  # SP reuses the PCP rank dimension and shards the sequence itself
        (True, 1, 17),
        (False, 1, 17),
        (
            False,
            4,
            4,
        ),  # main's behaviour for real PCP: the budget is divided by the PCP size
    ],
)
def test_warmup_budget_is_global_under_sp_and_pcp_divided_otherwise(
    sp_on, pcp, expected
):
    """CI failure: the SP warmup test extracts warmup_model by name and main moved the budget into
    warmup_max_tokens. The SP condition lives in that method now."""
    runner_cls = _class_with_methods(
        "atom/model_engine/model_runner.py",
        "ModelRunner",
        WARMUP_METHODS,
        _runner_namespace(sp_on),
    )
    runner = runner_cls()
    runner.config = SimpleNamespace(
        max_num_batched_tokens=17,
        max_model_len=17,
        max_num_seqs=1,
        enable_dp_attention=False,
        prefill_context_parallel_size=pcp,
    )
    runner.block_size, runner.label = 16, "test"
    batches = []
    runner.forward = batches.append
    runner.tokenID_processor = SimpleNamespace(clean=lambda: None)
    runner.warmup_model()
    assert len(batches) == 1 and batches[0].total_tokens_num == expected


PREPARE_METHODS = {"prepare_inputs", "_verifies_drafts"}


def _prepare_runner(sp_on, with_drafter=False, pd_producer=False):
    runner_cls = _class_with_methods(
        "atom/model_engine/model_runner.py",
        "ModelRunner",
        PREPARE_METHODS,
        _runner_namespace(sp_on),
    )
    runner = runner_cls()
    runner.config = SimpleNamespace(prefill_context_parallel_size=4 if sp_on else 1)
    runner.attn_metadata_builder = SimpleNamespace(
        build=lambda **kw: (None, None), prepare_model_inputs=lambda *a: None
    )
    runner._maybe_create_tbo_slices = lambda *a: None
    runner.tokenID_processor = SimpleNamespace(is_pd_producer=pd_producer)
    # main's #2688 prepare_inputs reads the step's GPU scalar through forward_vars
    runner.forward_vars = {"batch_info": SimpleNamespace(gpu=[0, 0, 0])}
    runner.running_tokens_per_dp_rank = SimpleNamespace(rows=[0])
    if with_drafter:
        runner.drafter = object()
    return runner


@pytest.mark.parametrize("sp_on,expected_tokens", [(True, 32), (False, 17)])
def test_prepare_inputs_publishes_the_dp_rank_row_table(sp_on, expected_tokens):
    """Main's #2688 API publishes ``running_tokens_per_dp_rank`` (the table
    ``ForwardMode.decide`` settled); under SP the SP expansion happens inside
    ``forward_context.set_forward_context``, so the runner forwards the table."""
    runner = _prepare_runner(sp_on)
    runner.running_tokens_per_dp_rank = SimpleNamespace(tokens=expected_tokens)
    batch = SimpleNamespace(
        total_tokens_num_prefill=0,
        total_tokens_num=17,
        total_seqs_num=17,
        num_scheduled_tokens=[1] * 17,
        is_dummy_run=False,
    )
    mode = SimpleNamespace(
        sync=SimpleNamespace(ub_max_tokens_across_dp=None, ub_tokens_across_dp=None),
        tbo_collective_active=False,
        running_tokens_are_unified=True,
        running_bs=32,
        running_tokens=32,
        max_seqlen_q=1,
        scheduled_bs=17,
        ubatch_split_bs=None,
    )
    ns = runner.prepare_inputs.__globals__
    ns["_forward_context"].clear()
    runner.prepare_inputs(batch, "ids", mode)
    (published,) = ns["_forward_context"]
    assert published["running_tokens_per_dp_rank"] is runner.running_tokens_per_dp_rank
    assert published["spec_decode_metadata"] is None  # no drafter: nothing to verify


@pytest.mark.parametrize(
    "prefill,drafter,producer,dummy,expected",
    [
        (False, True, False, False, True),  # a decode step with a drafter scores drafts
        (True, True, False, False, False),
        (False, False, False, False, False),
        (
            False,
            True,
            True,
            False,
            False,
        ),  # a P/D producer keeps the drafter for context KV only
        (False, True, False, True, False),
    ],
)
def test_verifies_drafts_is_what_the_stub_runner_in_the_sp_test_relies_on(
    prefill, drafter, producer, dummy, expected
):
    runner = _prepare_runner(False, with_drafter=drafter, pd_producer=producer)
    batch = SimpleNamespace(is_dummy_run=dummy)
    assert runner._verifies_drafts(batch, prefill) is expected
