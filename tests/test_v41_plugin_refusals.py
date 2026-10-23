# SPDX-License-Identifier: MIT
"""Configurations V4.1 refuses on the vLLM plugin, collected without vLLM.

Both of these exist because the model's own `validate_runtime_config` cannot
see the setting in plugin mode: the ATOM `Config` the plugin builds never
carries the pipeline/context parallel sizes, and the connector name it does
carry is resolved through ATOM's registry rather than the one vLLM actually
dispatches on. A check that is never asked the question passes, so the server
starts and fails later somewhere that does not name the cause.

`atom/plugin/vllm/platform.py` imports vLLM at module scope, so the refusals
are extracted and executed on their own, which is also what lets CPU CI
collect them -- `tests/plugin` is excluded from it.
"""

import ast
import logging
from pathlib import Path
from types import SimpleNamespace

import pytest

# Imported normally, unlike `platform`: this module pulls in no vLLM at module
# scope, so the reserve arithmetic can be exercised directly.
from atom.plugin.vllm import deepseek_v41_state_reserve_patch as reserve_patch

PLATFORM = Path(__file__).resolve().parents[1] / "atom/plugin/vllm/platform.py"
_REFUSALS = ("_refuse_v41_multi_axis_parallelism", "_refuse_v41_kv_transfer")


def _refusals():
    tree = ast.parse(PLATFORM.read_text())
    functions = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in _REFUSALS
    ]
    assert {node.name for node in functions} == set(_REFUSALS)
    scope = {"logger": logging.getLogger("atom.test.refusals")}
    exec(  # noqa: S102 - execute only AST nodes from this repository's source.
        compile(ast.Module(body=functions, type_ignores=[]), str(PLATFORM), "exec"),
        scope,
    )
    return SimpleNamespace(**{name: scope[name] for name in _REFUSALS})


def _parallel(**sizes):
    base = {
        "pipeline_parallel_size": 1,
        "prefill_context_parallel_size": 1,
        "decode_context_parallel_size": 1,
    }
    base.update(sizes)
    return SimpleNamespace(parallel_config=SimpleNamespace(**base))


@pytest.mark.parametrize(
    "axis",
    [
        "pipeline_parallel_size",
        "prefill_context_parallel_size",
        "decode_context_parallel_size",
    ],
)
def test_parallelism_without_a_v41_split_is_refused(axis):
    refuse = _refusals()._refuse_v41_multi_axis_parallelism
    with pytest.raises(ValueError, match="tensor parallel only") as raised:
        refuse(_parallel(**{axis: 2}))
    # The message has to name the axis; "unsupported parallelism" sends the
    # reader back to the flags to work out which one.
    assert axis in str(raised.value)


def test_tensor_parallel_alone_is_accepted():
    refuse = _refusals()._refuse_v41_multi_axis_parallelism
    refuse(_parallel())
    # Several at once are all named, not just the first one found.
    with pytest.raises(ValueError) as raised:
        refuse(_parallel(pipeline_parallel_size=2, decode_context_parallel_size=4))
    assert "pipeline_parallel_size=2" in str(raised.value)
    assert "decode_context_parallel_size=4" in str(raised.value)


def test_a_config_without_parallel_sizes_is_not_refused():
    """Absent is not a violation: the sizes default to one, as vLLM does."""
    refuse = _refusals()._refuse_v41_multi_axis_parallelism
    refuse(SimpleNamespace(parallel_config=None))
    refuse(SimpleNamespace())
    refuse(SimpleNamespace(parallel_config=SimpleNamespace()))


@pytest.mark.parametrize(
    "connector",
    [
        # The one the model's own check admits, because ATOM registers it as an
        # alias of `lmcache_mp`. On this path it is vLLM's connector, and ATOM's
        # STATE checkpoint is not what runs.
        "LMCacheMPConnector",
        "lmcache_mp",
        # vLLM's own names, which ATOM's registry does not know: left to it,
        # these raise "unknown KV connector" instead of being refused.
        "NixlConnector",
        "OffloadingConnector",
        "MooncakeConnector",
    ],
)
def test_every_kv_connector_is_refused(connector):
    refuse = _refusals()._refuse_v41_kv_transfer
    for config in (
        SimpleNamespace(kv_transfer_config=SimpleNamespace(kv_connector=connector)),
        SimpleNamespace(kv_transfer_config={"kv_connector": connector}),
    ):
        with pytest.raises(ValueError, match="does not support KV transfer") as raised:
            refuse(config)
        assert connector in str(raised.value)


@pytest.mark.parametrize(
    "transfer", [None, {}, SimpleNamespace(kv_connector=None), {"kv_connector": ""}]
)
def test_running_without_a_connector_is_not_refused(transfer):
    refuse = _refusals()._refuse_v41_kv_transfer
    refuse(SimpleNamespace(kv_transfer_config=transfer))
    refuse(SimpleNamespace())


# ---------------------------------------------------------------------------
# What the STATE tail does to a pool vLLM is still sizing
# ---------------------------------------------------------------------------


def _sizing_config(*, override=None, max_model_len=8192):
    return SimpleNamespace(
        model_config=SimpleNamespace(max_model_len=max_model_len),
        scheduler_config=SimpleNamespace(max_num_seqs=256),
        cache_config=SimpleNamespace(num_gpu_blocks_override=override),
    )


def test_auto_fit_plans_against_the_pool_minus_the_tail(monkeypatch):
    """Otherwise the fitted length fills a pool the tail then comes out of."""
    seen = {}
    module = SimpleNamespace(
        _pool_bytes_per_block=lambda groups: 1000,
        _auto_fit_max_model_len=lambda cfg, groups, memory: seen.update(memory=memory),
    )
    monkeypatch.setattr(reserve_patch, "deepseek_v41_state_reserve_blocks", lambda _: 7)

    assert reserve_patch._patch_auto_fit(module) is True
    module._auto_fit_max_model_len(_sizing_config(), [["g"], ["g"]], [50_000, 60_000])

    # 7 blocks x 1000 bytes withheld from each worker's budget.
    assert seen["memory"] == [43_000, 53_000]
    # Installing twice must not subtract the tail twice.
    assert reserve_patch._patch_auto_fit(module) is False


def test_auto_fit_is_untouched_without_a_v41_tail(monkeypatch):
    seen = {}
    module = SimpleNamespace(
        _pool_bytes_per_block=lambda groups: 1000,
        _auto_fit_max_model_len=lambda cfg, groups, memory: seen.update(memory=memory),
    )
    monkeypatch.setattr(reserve_patch, "deepseek_v41_state_reserve_blocks", lambda _: 0)

    reserve_patch._patch_auto_fit(module)
    module._auto_fit_max_model_len(_sizing_config(), [["g"]], [50_000])

    assert seen["memory"] == [50_000]


def test_a_worker_with_no_groups_keeps_its_budget(monkeypatch):
    seen = {}
    module = SimpleNamespace(
        _pool_bytes_per_block=lambda groups: 1000,
        _auto_fit_max_model_len=lambda cfg, groups, memory: seen.update(memory=memory),
    )
    monkeypatch.setattr(reserve_patch, "deepseek_v41_state_reserve_blocks", lambda _: 7)

    reserve_patch._patch_auto_fit(module)
    module._auto_fit_max_model_len(_sizing_config(), [[], ["g"]], [50_000, 60_000])

    assert seen["memory"] == [50_000, 53_000]


def test_a_version_without_the_auto_fit_symbol_is_not_an_error():
    assert reserve_patch._patch_auto_fit(SimpleNamespace()) is False
