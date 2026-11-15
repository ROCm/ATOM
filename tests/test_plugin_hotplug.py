# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""The engine side of plugin hot-plug: payload rules and its wiring.

CPU only, no AITER and no GPU: the payload/row/aggregation rules live in
``atom/model_engine/plugin_hotplug.py`` (imports recipes, not an engine), and
the parts that do need an engine -- the runner's recapture, the utility
command, the manager method and the HTTP route -- are checked as source, the
same way the plugin host's own wiring is.
"""

import ast
import json
import re
import shutil
import sys
import tempfile
import types
import uuid
from pathlib import Path

import pytest

from atom import plugin_recipe as recipes
from atom.model_engine import plugin_hotplug as hotplug

REPO_ROOT = Path(__file__).resolve().parent.parent
if not (REPO_ROOT / "atom").is_dir():
    # Running the file from outside the checkout (a CPU box running this one
    # test on its own): find the tree the imported atom package came from.
    REPO_ROOT = next(
        Path(entry)
        for entry in sys.path
        if (Path(entry) / "atom" / "config.py").is_file()
    )
OUT = "v41.attn.out"
QUANT = "v41.op.quant_fp8_g32"
INVOKE = "v41.attn.index_topk"


# -- fixtures ----------------------------------------------------------------


def _recipe(plan, *, mode="any", buckets=None, model_arch="DeepseekV41ForCausalLM"):
    owners = sorted({owner for names in plan.values() for owner in names})
    document = {
        "schema": recipes.RECIPE_SCHEMA,
        "model_arch": model_arch,
        "plugins": [
            {
                "name": owner,
                "serves": sorted(
                    seam for seam, names in plan.items() if owner in names
                ),
            }
            for owner in owners
        ],
        "plan": {"seams": {seam: list(names) for seam, names in plan.items()}},
    }
    if buckets is not None:
        document["graph"] = {"mode": "capture", "buckets": list(buckets)}
    else:
        document["graph"] = {"mode": mode}
    return json.dumps(document)


def _plan(document, *, enable=()):
    return recipes.resolve([json.loads(document)])


# -- payload parsing ---------------------------------------------------------


def test_a_recipe_text_payload_parses():
    request = hotplug.parse_hotplug_payload(
        {"recipe": [_recipe({OUT: ["bmm"]})], "rows": [16, 8, 8]}
    )
    assert request.rows == (8, 16)
    assert request.capture is True
    assert request.arm == "plugin"
    plan = request.source.resolve()
    assert plan.plan_id == request.source.plan_id
    assert plan.providers == ("bmm",)
    assert plan.tables == {"any": {OUT: ("bmm",)}}


def test_a_recipe_payload_accepts_documents_and_paths():
    document = json.loads(_recipe({OUT: ["bmm"]}))
    # A scratch directory of this test's own: default mode, so a sandbox that
    # refuses to traverse a 0o700 directory pytest would create still serves.
    directory = Path(tempfile.gettempdir()) / f"atom-hotplug-{uuid.uuid4().hex}"
    directory.mkdir()
    try:
        (directory / "one.json").write_text(json.dumps(document), encoding="utf-8")
        for recipe in (
            [document],
            [str(directory / "one.json")],
            [str(directory)],
        ):
            request = hotplug.parse_hotplug_payload({"recipe": recipe})
            assert request.source.resolve().providers == ("bmm",)
    finally:
        shutil.rmtree(directory, ignore_errors=True)


def test_rows_are_optional_and_plan_derived_by_the_runner():
    request = hotplug.parse_hotplug_payload({"recipe": [_recipe({OUT: ["bmm"]})]})
    assert request.rows == ()
    assert request.capture is True
    assert (
        hotplug.parse_hotplug_payload(
            {"recipe": [_recipe({OUT: ["bmm"]})], "capture": False}
        ).capture
        is False
    )


def test_enable_narrows_the_plan_and_the_source_checks_its_id():
    document = _recipe({OUT: ["a"], QUANT: ["b"]})
    request = hotplug.parse_hotplug_payload({"recipe": [document], "enable": ["a"]})
    plan = request.source.resolve()
    assert plan.arms["plugin"] == ("a",)
    assert set(plan.providers) == {"a", "b"}


def test_a_payload_refusal_carries_a_reason():
    cases = [
        ({}, "no-recipe"),
        ({"recipe": ["not json and not a path"]}, "recipe-unreadable"),
        ({"recipe": [_recipe({OUT: ["bmm"]})], "rows": [0]}, "rows-invalid"),
        ({"recipe": [_recipe({OUT: ["bmm"]})], "rows": [True]}, "rows-invalid"),
        ({"recipe": [_recipe({OUT: ["bmm"]})], "rows": "8"}, "payload-schema"),
        ({"recipe": [_recipe({OUT: ["bmm"]})], "capture": "yes"}, "payload-schema"),
        ({"recipe": [_recipe({OUT: ["bmm"]})], "arm": "bogus"}, "unknown-arm"),
        ({"body": 1}, "no-recipe"),
    ]
    for body, reason in cases:
        with pytest.raises(hotplug.HotplugPayloadError) as refused:
            hotplug.parse_hotplug_payload(body)
        assert refused.value.reason == reason, body


def test_a_recipe_that_cannot_serve_is_refused_not_a_crash():
    with pytest.raises(recipes.RecipeRefused):
        hotplug.parse_hotplug_payload({"recipe": ['{"schema": "nope"}']})
    # An unknown --plugin-enable name is a recipe refusal too.
    with pytest.raises(recipes.RecipeRefused):
        hotplug.parse_hotplug_payload(
            {"recipe": [_recipe({OUT: ["bmm"]})], "enable": ["not-installed"]}
        )


def test_a_recipe_for_another_model_is_refused():
    with pytest.raises(hotplug.HotplugPayloadError) as refused:
        hotplug.parse_hotplug_payload(
            {"recipe": [_recipe({OUT: ["bmm"]}, model_arch="OtherModel")]},
            model_arch="DeepseekV41ForCausalLM",
        )
    assert refused.value.reason == "recipe-model-mismatch"


# -- rows -> decode buckets --------------------------------------------------


def test_rows_map_to_decode_request_buckets():
    # rows are padded running_tokens: bs * (mtp_k + 1), here mtp_k = 5.
    assert hotplug.rows_to_bs([48, 24, 48], 6) == [4, 8]
    assert hotplug.rows_to_bs([], 6) == []


def test_rows_that_are_not_target_decode_rows_are_refused():
    with pytest.raises(hotplug.HotplugPayloadError) as refused:
        hotplug.rows_to_bs([5], 6)  # a DSpark draft row, bs * mtp_k
    assert refused.value.reason == "rows-not-decodable"
    with pytest.raises(hotplug.HotplugPayloadError) as refused:
        hotplug.rows_to_bs([8], 0)
    assert refused.value.reason == "rows-invalid"


# -- which rows a plan asks for ---------------------------------------------


def test_capture_rows_are_the_plans_padded_rows():
    plan = _plan(_recipe({OUT: ["bmm"]}, buckets=[16, 8]))
    assert hotplug.capture_rows(plan) == [8, 16]
    assert hotplug.capture_rows(plan, full_q_len=4) == [8, 16]
    assert hotplug.capture_rows(None) == []


def test_a_plan_with_capture_tables_targets_those_buckets():
    plan = _plan(_recipe({OUT: ["bmm"]}, buckets=[8, 16]))
    assert hotplug.default_capture_rows(plan, 4, [1, 2, 4, 8, 16]) == [8, 16]


def test_a_plan_without_capture_tables_recaptures_the_whole_ladder():
    plan = _plan(_recipe({OUT: ["bmm"]}))
    assert hotplug.capture_rows(plan) == []
    assert hotplug.default_capture_rows(plan, 4, [1, 2, 4, 8]) == [4, 8, 16, 32]


def test_capture_buckets_outside_the_ladder_fall_back_to_the_ladder():
    # A recipe authored for a bigger ladder: its rows cannot be replayed here,
    # but its any/eager tables still changed every recorded graph.
    plan = _plan(_recipe({OUT: ["bmm"]}, buckets=[24, 48]))
    assert hotplug.default_capture_rows(plan, 6, [4, 8]) == [24, 48]


# -- per-rank report aggregation --------------------------------------------


def _report(**overrides):
    report = {
        "rank": 0,
        "tp_world_size": 4,
        "status": "ok",
        "error": None,
        "plan_id": "sha256:abc",
        "rows": [8, 16],
        "sizes": [2, 4],
        "capture_seconds": 1.5,
    }
    report.update(overrides)
    return report


def test_reports_merge_into_one_answer():
    merged = hotplug.aggregate_reports(
        [_report(capture_seconds=2.0), _report(capture_seconds=1.0)]
    )
    assert merged["status"] == "ok"
    assert merged["plan_id"] == "sha256:abc"
    assert merged["rows"] == [8, 16]
    assert merged["sizes"] == [2, 4]
    assert merged["capture_seconds"] == 2.0
    assert len(merged["ranks"]) == 2


def test_reports_that_resolved_to_different_plans_are_refused():
    with pytest.raises(hotplug.HotplugPayloadError) as refused:
        hotplug.aggregate_reports([_report(), _report(plan_id="sha256:other")])
    assert refused.value.reason == "plan-id-mismatch"
    with pytest.raises(RuntimeError):
        hotplug.aggregate_reports([])


# -- HTTP refusal mapping ----------------------------------------------------


def test_refusals_map_to_4xx_and_their_reason_code():
    cases = [
        (hotplug.HotplugPayloadError("rows-invalid", "bad"), 400, "rows-invalid"),
        (recipes.RecipeRefused("recipe-schema", "bad"), 400, "recipe-schema"),
        (TimeoutError("no reply"), 504, "hotplug-timeout"),
        (RuntimeError("hot-plug is off"), 409, "hotplug-failed"),
        (ValueError("no engine"), 400, "hotplug-refused"),
    ]
    for error, status, reason in cases:
        assert hotplug.refusal_status(error) == status
        assert hotplug.refusal_reason(error) == reason


def test_topologies_the_first_cut_cannot_capture_in_are_refused():
    def config(**overrides):
        values = {
            "pipeline_parallel_size": 1,
            "enable_dp_attention": False,
            "enable_rapidserve": False,
            "parallel_config": types.SimpleNamespace(data_parallel_size=1),
        }
        values.update(overrides)
        return types.SimpleNamespace(**values)

    assert hotplug.topology_refusal(config()) is None
    assert "pipeline" in hotplug.topology_refusal(config(pipeline_parallel_size=2))
    assert "data parallelism" in hotplug.topology_refusal(
        config(parallel_config=types.SimpleNamespace(data_parallel_size=2))
    )
    assert "DP attention" in hotplug.topology_refusal(config(enable_dp_attention=True))
    assert "disaggregation" in hotplug.topology_refusal(config(enable_rapidserve=True))


def test_hotplug_enabled_reads_the_config_then_the_env(monkeypatch):
    assert hotplug.hotplug_enabled(types.SimpleNamespace(plugin_hotplug=True))
    # The env fallback imports atom.utils lazily; stand it in so this test does
    # not need the full engine package on a bare CPU box.
    fake_utils = types.ModuleType("atom.utils")
    fake_utils.envs = types.SimpleNamespace(ATOM_PLUGIN_HOTPLUG=True)
    monkeypatch.setitem(sys.modules, "atom.utils", fake_utils)
    assert hotplug.hotplug_enabled(types.SimpleNamespace(plugin_hotplug=False))
    fake_utils.envs = types.SimpleNamespace(ATOM_PLUGIN_HOTPLUG=False)
    assert not hotplug.hotplug_enabled(types.SimpleNamespace(plugin_hotplug=False))


# -- wiring: config, CLI, runner, utility command, manager, route ------------

TEXT = {
    path: (REPO_ROOT / path).read_text(encoding="utf-8")
    for path in (
        "atom/config.py",
        "atom/model_engine/arg_utils.py",
        "atom/model_engine/model_runner.py",
        "atom/model_engine/engine_utility.py",
        "atom/model_engine/engine_core_mgr.py",
        "atom/entrypoints/openai/api_server.py",
        "atom/utils/envs.py",
    )
}


def _flat(source):
    """Whitespace-insensitive source text, for call-shape assertions."""
    return re.sub(r"\s+", " ", source)


def _classes(path):
    tree = ast.parse(TEXT[path])
    return {
        node.name: node for node in ast.walk(tree) if isinstance(node, ast.ClassDef)
    }


def _methods(path, owner):
    return {
        node.name
        for node in _classes(path)[owner].body
        if isinstance(node, ast.FunctionDef)
    }


def test_config_and_cli_carry_the_hotplug_flag():
    assert "plugin_hotplug: bool = False" in TEXT["atom/config.py"]
    flat = _flat(TEXT["atom/model_engine/arg_utils.py"])
    assert '"--plugin-hotplug"' in flat
    assert 'kwargs["plugin_hotplug"]' in flat
    assert "envs.ATOM_PLUGIN_HOTPLUG" in flat
    assert '"ATOM_PLUGIN_HOTPLUG": lambda:' in TEXT["atom/utils/envs.py"]


def test_the_runner_configures_the_host_hotplug_aware():
    flat = _flat(TEXT["atom/model_engine/model_runner.py"])
    assert 'hotplug=getattr(config, "plugin_hotplug", False)' in flat


def test_the_runner_has_the_hotplug_and_recapture_path():
    methods = _methods("atom/model_engine/model_runner.py", "ModelRunner")
    assert {
        "hotplug_plugin",
        "recapture_cudagraph",
        "_capture_request_bucket",
        "_request_capture_builder",
        "_tp_barrier",
        "_tp_agree",
    } <= methods
    flat = _flat(TEXT["atom/model_engine/model_runner.py"])
    # The plan is installed by the host, then the requested graphs recaptured.
    assert (
        'plugin_host.hotplug_plan(plan, arm=report["arm"], config=self.config)' in flat
    )
    assert "plugin_host.hotplug_enabled()" in flat
    assert "rows_to_bs(rows, full_q_len)" in flat
    # The old plan's graphs are dropped (MVP), never replayed.
    assert "self.req_graph.release_graphs()" in flat
    # The capture runs on every TP rank at once.
    assert 'self._tp_barrier("recapture")' in flat
    assert 'self._tp_barrier("recapture end")' in flat
    assert "torch.distributed.barrier(group=group.cpu_group)" in flat
    assert "torch.distributed.all_reduce(" in flat
    # The ladder narrows to what was really recorded, so ForwardMode runs the
    # rest eagerly instead of replaying a graph that no rank holds.
    assert "self.capture_sizes = surviving" in flat
    assert "self.capture_sizes_np = np.asarray(self.capture_sizes" in flat


def test_the_utility_command_is_synchronous_and_registered():
    tree = ast.parse(TEXT["atom/model_engine/engine_utility.py"])
    forget = set()
    for node in tree.body:
        if not isinstance(node, ast.Assign) or not isinstance(node.value, ast.Call):
            continue
        target = node.targets[0]
        if not (
            isinstance(target, ast.Name) and target.id == "FIRE_AND_FORGET_UTILITY_CMDS"
        ):
            continue
        for element in node.value.args[0].elts:
            forget.add(element.value)
    assert forget and "hotplug_plugin" not in forget
    flat = _flat(TEXT["atom/model_engine/engine_utility.py"])
    assert 'HOTPLUG_CMD: "_handle_hotplug_plugin"' in flat
    assert "HOTPLUG_CMD, payload, wait_out=True" in flat
    assert "engine.pause_generation(deadline=deadline)" in flat
    assert "engine.resume_generation()" in flat
    assert 'self.output_queue.put_nowait(("UTILITY_RESPONSE", reply))' in flat
    # A recapture touches the serving caches: it may only run with no request
    # holding per-request state, so the handler refuses a busy engine.
    assert "_require_idle_for_capture" in flat
    assert "needs an idle engine" in flat


def test_the_manager_exposes_a_synchronous_hotplug_method():
    methods = _methods("atom/model_engine/engine_core_mgr.py", "CoreManager")
    assert "hotplug_plugin" in methods
    flat = _flat(TEXT["atom/model_engine/engine_core_mgr.py"])
    assert (
        "self.broadcast_utility_command_sync( HOTPLUG_CMD, timeout=timeout, payload=payload )"
        in flat
    )
    assert "aggregate_reports(reports)" in flat


def test_the_api_server_serves_the_internal_route():
    tree = ast.parse(TEXT["atom/entrypoints/openai/api_server.py"])
    routes = []
    for node in tree.body:
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        for decorator in node.decorator_list:
            if not (
                isinstance(decorator, ast.Call)
                and isinstance(decorator.func, ast.Name)
                and decorator.func.id == "_engine_route"
            ):
                continue
            keywords = {kw.arg: kw.value for kw in decorator.keywords}
            path = decorator.args[0] if decorator.args else None
            routes.append((node, path, keywords))
    match = [
        (node, keywords)
        for node, path, keywords in routes
        if isinstance(path, ast.Name) and path.id == "HOTPLUG_ENDPOINT"
    ]
    assert len(match) == 1
    node, keywords = match[0]
    methods = keywords["methods"]
    assert [m.value for m in methods.elts] == ["POST"]
    assert isinstance(keywords["include_in_schema"], ast.Constant)
    assert keywords["include_in_schema"].value is False
    flat = re.sub(
        r"\s+",
        " ",
        ast.get_source_segment(TEXT["atom/entrypoints/openai/api_server.py"], node),
    )
    assert "parse_hotplug_payload(body" in flat
    assert "core_mgr" in flat
    assert "refusal_status(exc)" in flat
    assert "JSONResponse(result)" in flat
    assert "asyncio.to_thread" in flat


def test_the_route_is_not_in_the_openai_documentation():
    # include_in_schema=False keeps /plugin/hotplug out of /openapi.json, and
    # the route lives on the internal management surface, not the OpenAI one.
    assert 'HOTPLUG_ENDPOINT, methods=["POST"], include_in_schema=False' in _flat(
        TEXT["atom/entrypoints/openai/api_server.py"]
    )
