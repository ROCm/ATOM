# SPDX-License-Identifier: MIT
"""How the in-forward staging fallback decides a batch is vLLM's own.

Two paths stage a V4.1 step: the builder one, which runs outside the forward
and is what real serving takes, and this fallback, reached when the builder
did not stage -- warmup before the proxy cache is bound. They answered the
same question differently, and the difference had a direction: the builder
asks `in_dummy_run()`, the capture state and the request ids, while the
fallback asked the request ids alone.

That inference is the one `apply_vllm_dummy_run_patch` was written to replace.
Its docstring records four attempts at reading a synthetic batch off the batch
itself, each wrong under load; V2's dummy batches carry unique request ids, so
the id test reads one as real traffic and hands it the STATE slots real
requests hold.

Executed by AST extraction, so the real function is the one under test with no
vLLM, aiter or GPU present.
"""

import ast
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

BRIDGE = Path(__file__).resolve().parents[1] / "atom/plugin/vllm/deepseek_v41_bridge.py"


def _live_step_inputs(monkeypatch, *, dummy_run, capturing, duplicate_ids):
    """The real `_v41_live_step_inputs`, with its collaborators recorded."""
    tree = ast.parse(BRIDGE.read_text())
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "_v41_live_step_inputs"
    )

    # The module-local import inside the function resolves through sys.modules.
    runtime = ModuleType("atom.plugin.vllm.vllm_runtime")
    runtime.in_dummy_run = lambda: dummy_run
    for name, module in (
        ("atom", ModuleType("atom")),
        ("atom.plugin", ModuleType("atom.plugin")),
        ("atom.plugin.vllm", ModuleType("atom.plugin.vllm")),
        ("atom.plugin.vllm.vllm_runtime", runtime),
    ):
        monkeypatch.setitem(sys.modules, name, module)
    sys.modules["atom.plugin.vllm"].vllm_runtime = runtime

    chosen = {}
    scope = {
        "torch": SimpleNamespace(
            cuda=SimpleNamespace(is_current_stream_capturing=lambda: capturing)
        ),
        "get_deepseek_v41_proxy_metadata_from_vllm_context": lambda _: SimpleNamespace(
            atom_v41_snapshot=SimpleNamespace(num_reqs=2)
        ),
        "_v41_live_running_tokens": lambda _: 2,
        "_v41_request_ids_are_synthetic": lambda _: duplicate_ids,
        "v41_capture_state_slots": lambda n, cfg: chosen.setdefault("slots", "capture"),
        "_v41_request_state_slots": lambda s, cfg: chosen.setdefault("slots", "real"),
        "_v41_scheduled_batch": lambda snapshot, slots: SimpleNamespace(
            total_tokens_num=2, total_seqs_num=2
        ),
        "_v41_dummy_batch": lambda *a, **k: SimpleNamespace(
            total_tokens_num=2, total_seqs_num=2
        ),
    }
    exec(  # noqa: S102 - execute only AST nodes from this repository's source.
        compile(ast.Module(body=[function], type_ignores=[]), str(BRIDGE), "exec"),
        scope,
    )
    builder = SimpleNamespace(
        cache=object(), block_table_cols=4, block_ratio=1, block_size=256, max_bs=4
    )
    scope["_v41_live_step_inputs"](builder, None, "proxy", SimpleNamespace())
    return chosen["slots"]


def test_a_dummy_run_takes_capture_slots_even_with_unique_request_ids(monkeypatch):
    """The case the id test alone gets wrong: V2 numbers its dummies uniquely."""
    assert (
        _live_step_inputs(
            monkeypatch, dummy_run=True, capturing=False, duplicate_ids=False
        )
        == "capture"
    )


def test_a_capturing_stream_takes_capture_slots(monkeypatch):
    assert (
        _live_step_inputs(
            monkeypatch, dummy_run=False, capturing=True, duplicate_ids=False
        )
        == "capture"
    )


def test_repeated_request_ids_still_take_capture_slots(monkeypatch):
    """The original signal, kept: V1-style warmup repeats one placeholder id."""
    assert (
        _live_step_inputs(
            monkeypatch, dummy_run=False, capturing=False, duplicate_ids=True
        )
        == "capture"
    )


def test_real_traffic_takes_real_state_slots(monkeypatch):
    """None of the three fire, so this is a scheduled batch and keeps its slots."""
    assert (
        _live_step_inputs(
            monkeypatch, dummy_run=False, capturing=False, duplicate_ids=False
        )
        == "real"
    )
