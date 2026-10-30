# SPDX-License-Identifier: MIT
# Torch-free tests for the R12 configuration: check_period and the PP x SP refusal.
# The real source is extracted with ast (nothing imported from atom), like
# test_gemm_fusion_rebase_compat.py.
import ast
from dataclasses import dataclass, fields
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[2]


def _tree(rel):
    return ast.parse((ROOT / rel).read_text(encoding="utf-8"))


def _run(nodes, namespace):
    module = ast.Module(
        [ast.parse("from __future__ import annotations").body[0], *nodes], []
    )
    ns = dict(namespace)
    exec(  # noqa: S102 - real source, extracted by ast, run against stand-ins
        compile(ast.fix_missing_locations(module), "atom/config.py", "exec"), ns
    )
    return ns


def _gemm_fusion_config():
    cls = next(
        n
        for n in _tree("atom/config.py").body
        if isinstance(n, ast.ClassDef) and n.name == "GemmFusionConfig"
    )
    return _run([cls], {"dataclass": dataclass, "fields": fields})["GemmFusionConfig"]


def _config_method(name):
    cls = next(
        n
        for n in _tree("atom/config.py").body
        if isinstance(n, ast.ClassDef) and n.name == "Config"
    )
    fn = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == name)
    fn.decorator_list = []
    return _run([fn], {})[name]


def test_the_default_period_is_14_and_the_fused_path_is_still_off_by_default():
    cfg = _gemm_fusion_config()()
    assert cfg.check_period == 14 and cfg.enabled is False


@pytest.mark.parametrize("period", [1, 3, 14, 100])
def test_valid_periods_are_accepted(period):
    assert (
        _gemm_fusion_config()(profile="p", check_period=period).check_period == period
    )


@pytest.mark.parametrize("period", [0, 2, -1, -14])
def test_periods_the_two_step_schedule_cannot_honour_are_a_value_error(period):
    with pytest.raises(ValueError, match="check_period"):
        _gemm_fusion_config()(profile="p", check_period=period)


@pytest.mark.parametrize("period", [True, False, 14.0, "14", None])
def test_a_non_integer_period_is_a_type_error(period):
    with pytest.raises(TypeError, match="check_period"):
        _gemm_fusion_config()(profile="p", check_period=period)


def test_from_dict_accepts_the_new_key_and_still_rejects_unknown_ones():
    cfg = _gemm_fusion_config()
    assert cfg.from_dict({"profile": "p", "check_period": 3}).check_period == 3
    with pytest.raises(ValueError, match="Unknown --gemm-fusion-config key"):
        cfg.from_dict({"check_perid": 3})


@pytest.mark.parametrize(
    "profile,sp,pp,refused",
    [
        ("p", 4, 2, True),  # the refused combination
        ("p", 8, 4, True),
        ("p", 4, 1, False),  # SP alone is the supported deployment
        ("p", 1, 2, False),  # PP alone: no SP exchange to fuse, nothing to refuse
        ("", 4, 2, False),  # OFF is never refused
    ],
)
def test_fusion_with_sequence_and_pipeline_parallelism_is_refused_at_config_time(
    profile, sp, pp, refused
):
    validate = _config_method("_validate_gemm_fusion")
    cfg_cls = _gemm_fusion_config()
    self = SimpleNamespace(
        gemm_fusion=cfg_cls(profile=profile),
        sequence_parallel_size=sp,
        pipeline_parallel_size=pp,
    )
    if refused:
        with pytest.raises(ValueError, match="pipeline parallelism"):
            validate(self)
    else:
        validate(self)


def test_the_refusal_runs_in_post_init_right_after_the_sp_validation():
    cls = next(
        n
        for n in _tree("atom/config.py").body
        if isinstance(n, ast.ClassDef) and n.name == "Config"
    )
    post = next(
        n
        for n in cls.body
        if isinstance(n, ast.FunctionDef) and n.name == "__post_init__"
    )
    calls = [
        n
        for n in ast.walk(post)
        if isinstance(n, ast.Call)
        and isinstance(n.func, ast.Attribute)
        and isinstance(n.func.value, ast.Name)
        and n.func.value.id == "self"
        and n.func.attr in {"_init_sequence_parallel", "_validate_gemm_fusion"}
    ]
    # ast.walk is breadth-first: order by source line
    order = [n.func.attr for n in sorted(calls, key=lambda n: n.lineno)]
    assert order == ["_init_sequence_parallel", "_validate_gemm_fusion"]


def test_the_cli_option_reaches_the_config_object():
    """Source pin (EngineArgs needs torch): the flag exists and is forwarded to GemmFusionConfig."""
    src = (ROOT / "atom/model_engine/arg_utils.py").read_text(encoding="utf-8")
    assert '"--gemm-fusion-check-period"' in src
    assert 'kwargs.pop("gemm_fusion_check_period", None)' in src
    assert "GemmFusionConfig(**gemm_fusion_kwargs)" in src
