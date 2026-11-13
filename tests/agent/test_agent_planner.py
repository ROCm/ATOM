# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Flag validation, recipe ranking and plan construction (no GPU)."""

import json
from pathlib import Path

import pytest

from atom.agent import hardware, planner
from atom.agent.knowledge import KnowledgeBase

FLAG_SRC = """
import argparse
def add(parser):
    parser.add_argument("--model", type=str)
    parser.add_argument("--tensor-parallel-size", "-tp", type=int, default=1)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--enable_prefix_caching", action=argparse.BooleanOptionalAction)
    parser.add_argument("--kv_cache_dtype", choices=["bf16", "fp8"], help="KV dtype")
    parser.add_argument("--level", type=int, default=3)
"""


@pytest.fixture
def table():
    return planner.parse_flag_table([FLAG_SRC])


def test_flag_table_shapes(table):
    assert table["--trust-remote-code"]["takes_value"] is False
    assert table["--enable-prefix-caching"]["takes_value"] is False
    assert "--no-enable-prefix-caching" in table
    assert table["-tp"]["takes_value"] is True
    assert table["--kv-cache-dtype"]["choices"] == ["bf16", "fp8"]


@pytest.mark.parametrize(
    "args, expected",
    [
        (["--model", "m", "-tp", "4", "--kv_cache_dtype", "fp8"], []),
        (["--kv-cache-dtype=bf16", "--no-enable_prefix_caching"], []),
        (["--levle", "3"], ["did you mean --level", "stray value '3'"]),
        (["--kv_cache_dtype", "int4"], ["not in"]),
        (["--level"], ["needs a value"]),
    ],
)
def test_validate_args(table, args, expected):
    problems = planner.validate_args(args, table)
    assert len(problems) == len(expected)
    for p, e in zip(problems, expected):
        assert e in p


def test_real_server_flags_cover_core_options():
    flags = planner.server_flags()
    for name in ("--model", "-tp", "--server-port", "--kv-cache-dtype", "--level"):
        assert name in flags
    assert planner.validate_args(["--model", "x", "--server-port", "8000"]) == []


def test_arg_helpers():
    args = ["--model", "m", "-tp", "2", "--trust-remote-code", "--level=3"]
    assert planner.get_arg(args, "--tensor-parallel-size", "-tp") == "2"
    assert planner.get_arg(args, "--level") == "3"
    assert planner.remove_arg(args, "-tp", "--level") == [
        "--model",
        "m",
        "--trust-remote-code",
    ]
    assert planner.set_arg(args, "--level", 0)[-2:] == ["--level", "0"]


def _cmd(model, section="", recipe="recipes/x.md", args=(), env=None):
    return {
        "recipe": recipe,
        "section": section,
        "env": env or {},
        "args": ["--model", model, *args],
    }


def test_score_prefers_exact_model_and_matching_gpu():
    toks = planner._name_tokens("DeepSeek-V4.1-Flash-1017")
    exact = _cmd("deepseek-ai/DeepSeek-V4.1-Flash", "TP4 on MI355X", args=["-tp", "4"])
    sibling = _cmd("deepseek-ai/DeepSeek-V4-Flash", "TP4 on MI355X", args=["-tp", "4"])
    other_gpu = _cmd(
        "deepseek-ai/DeepSeek-V4.1-Flash", "gfx1250 (MI455X)", args=["-tp", "4"]
    )
    pd = _cmd(
        "deepseek-ai/DeepSeek-V4.1-Flash",
        "prefill node",
        args=["--kv-transfer-config", "{}"],
    )
    s = {
        name: planner.score_command(c, toks, "gfx950", None, 8)[0]
        for name, c in [
            ("exact", exact),
            ("sibling", sibling),
            ("other_gpu", other_gpu),
            ("pd", pd),
        ]
    }
    assert s["exact"] > s["sibling"]
    assert s["exact"] > s["other_gpu"]
    assert s["pd"] < 0
    assert exact["exact_model"] and not sibling["exact_model"]


def test_latency_goal_prefers_more_tp():
    toks = planner._name_tokens("Foo-7B")
    tp2 = _cmd("org/Foo-7B", "TP2", args=["-tp", "2"])
    tp4 = _cmd("org/Foo-7B", "TP4", args=["-tp", "4"])
    lat = [
        planner.score_command(c, toks, None, "low latency", 8)[0] for c in (tp2, tp4)
    ]
    thr = [
        planner.score_command(c, toks, None, "max throughput", 8)[0] for c in (tp2, tp4)
    ]
    assert lat[1] > lat[0]
    assert thr[0] > thr[1]


def test_sibling_recipe_survives_user_pinned_tp():
    """A TP8 recipe for a sibling model still supplies env/flags when the user
    pins one GPU: its TP is resized, so it must not be discarded."""
    toks = planner._name_tokens("DeepSeek-V4-Flash-4layer")
    sib = _cmd("deepseek-ai/DeepSeek-V4-Pro", "FP8 on 8xMI355X", args=["-tp", "8"])
    pinned, _ = planner.score_command(sib, toks, "gfx950", "test", 1, tp_fixed=True)
    assert pinned >= 5
    # Incidental goal words ("kv cache" in a heading) do not move the score.
    noisy, _ = planner.score_command(
        dict(sib, section="FP8 on 8xMI355X (FP8 KV Cache)"),
        toks,
        "gfx950",
        "one GPU, KV cache",
        1,
        True,
    )
    assert noisy == pinned


def test_pick_gpus_prefers_aligned_blocks():
    def gpu(i, free=True):
        return hardware.Gpu(i, "gfx950", "x", "", "", 288000, 300 if free else 90000, 0)

    gpus = [gpu(0, False), gpu(1), gpu(2), gpu(3), gpu(4), gpu(5), gpu(6), gpu(7)]
    assert planner.pick_gpus(gpus, 4, set()) == [4, 5, 6, 7]
    assert planner.pick_gpus(gpus, 2, {2}) == [4, 5]
    assert planner.pick_gpus(gpus, 8, set()) is None


@pytest.fixture
def fake_machine(tmp_path, monkeypatch):
    gpus = [
        hardware.Gpu(i, "gfx950", "MI355X", "", "", 288 * 1024, 300, 0)
        for i in range(8)
    ]
    monkeypatch.setattr(hardware, "list_gpus", lambda with_names=True: gpus)
    model = tmp_path / "models" / "Foo-V2.1-0915"
    model.mkdir(parents=True)
    (model / "config.json").write_text(
        json.dumps(
            {
                "architectures": ["Qwen3ForCausalLM"],
                "model_type": "qwen3",
                "num_hidden_layers": 4,
                "hidden_size": 64,
                "num_attention_heads": 4,
                "num_key_value_heads": 2,
            }
        )
    )
    (model / "model.safetensors").write_bytes(b"\0" * 1024)
    root = tmp_path / "repo"
    (root / "recipes").mkdir(parents=True)
    (root / "recipes" / "Foo-V2.1.md").write_text(
        "# Foo-V2.1\n## TP4 on MI355X\n```bash\n"
        "AITER_LOG_LEVEL=WARNING python3 -m atom.entrypoints.openai_server "
        "--model org/Foo-V2.1 -tp 4 --kv_cache_dtype fp8 "
        "--spec-decode-acceptance-length 3.5 --server-port 8000\n```\n"
    )
    return KnowledgeBase(root), model


def test_make_plan_from_recipe(fake_machine):
    kb, model = fake_machine
    plan = planner.make_plan(kb, str(model), goal="latency", reserved_gpus={0})
    assert plan["ok"], plan
    assert plan["tp"] == 4
    assert plan["gpus"] == [4, 5, 6, 7]  # 0 is reserved, so the next aligned block
    assert plan["source"]["recipe"] == "recipes/Foo-V2.1.md"
    assert plan["args"][:4] == ["--model", str(model), "-tp", "4"]
    assert "--spec-decode-acceptance-length" not in plan["args"]
    assert any("acceptance" in n for n in plan["notes"])
    assert plan["env"]["AITER_LOG_LEVEL"] == "WARNING"


def test_make_plan_rejects_unknown_model(fake_machine):
    kb, _ = fake_machine
    plan = planner.make_plan(kb, "/nope/model")
    assert not plan["ok"]


def test_make_plan_explicit_gpus_must_match_tp(fake_machine):
    kb, model = fake_machine
    plan = planner.make_plan(kb, str(model), gpus=[1, 2], tp=4)
    assert not plan["ok"] and "GPUs" in plan["error"]


def test_port_selection_skips_taken(fake_machine):
    kb, model = fake_machine
    plan = planner.make_plan(kb, str(model), port=18000, taken_ports={18000})
    assert plan["port"] != 18000
    assert Path(plan["model"]) == model
