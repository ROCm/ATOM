# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Recipe/doc search and launch-command extraction (no GPU, no torch)."""

from pathlib import Path

from atom.agent.knowledge import (
    KnowledgeBase,
    _split_sections,
    extract_launch_commands,
    parse_launch_block,
    tokenize,
)
from atom.agent.planner import recipe_env

RECIPE = """\
# Foo-V2.1 on ATOM

## Required env

```bash
export ATOM_USE_TRITON_GEMM=1
export AITER_LOG_LEVEL=WARNING
```

## TP4 server on MI355X

```bash
export MODEL_PATH="/models/org/Foo-V2.1"
export HIP_VISIBLE_DEVICES=0,1,2,3
SIZES="[1,2,4]"
if [ "$CONC" -eq 32 ]; then
  SIZES="[1,2,3,4,5,6,7,8]"
fi

ATOM_X=1 python3 -m atom.entrypoints.openai_server \\
  --model "$MODEL_PATH" -tp 4 \\
  --cudagraph-capture-sizes "$SIZES" \\
  --kv_cache_dtype fp8 2>&1 | tee server.log
```

## Client

```bash
curl localhost:8000/v1/models
```
"""


def test_tokenize_keeps_versions_and_their_prefix():
    toks = tokenize("DeepSeek-V4.1-Flash on MI355X")
    assert {"deepseek", "v4.1", "v4", "flash", "mi355x"} <= set(toks)


def test_split_sections_markdown_and_rst():
    md = _split_sections("a.md", "# T\nx\n## A\ny\n```\n# not a heading\n```\n")
    assert [h for h, _ in md] == ["T", "T > A"]
    rst = _split_sections("a.rst", "Title\n=====\nbody\nSub\n---\nmore\n")
    assert [h for h, _ in rst] == ["Title", "Title > Sub"]


def test_parse_launch_block_env_vars_branches_and_pipes():
    block = RECIPE.split("```bash\n")[2].split("```")[0]
    cmd = parse_launch_block(block)
    assert cmd["args"] == [
        "--model",
        "/models/org/Foo-V2.1",
        "-tp",
        "4",
        "--cudagraph-capture-sizes",
        "[1,2,4]",  # the default, not the value set inside `if`
        "--kv_cache_dtype",
        "fp8",
    ]
    assert cmd["env"]["ATOM_X"] == "1"
    assert cmd["env"]["HIP_VISIBLE_DEVICES"] == "0,1,2,3"
    assert not cmd["uses_docker"]


def test_shell_default_values_expand():
    cmd = parse_launch_block(
        "TP=${TP:-4}\npython -m atom.entrypoints.openai_server -tp ${TP} "
        "--max-model-len ${LEN:-8192} --x ${UNSET}\n"
    )
    assert cmd["args"] == ["-tp", "4", "--max-model-len", "8192", "--x", "${UNSET}"]


def test_extract_launch_commands_records_section():
    cmds = extract_launch_commands(RECIPE, "recipes/Foo.md")
    assert len(cmds) == 1
    assert cmds[0]["section"] == "Foo-V2.1 on ATOM > TP4 server on MI355X"
    assert recipe_env(RECIPE) == {
        "ATOM_USE_TRITON_GEMM": "1",
        "AITER_LOG_LEVEL": "WARNING",
    }


def test_knowledge_base_search_and_read(tmp_path: Path):
    (tmp_path / "recipes").mkdir()
    (tmp_path / "recipes" / "Foo-V2.1.md").write_text(RECIPE)
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "guide.md").write_text(
        "# Guide\n## KV cache\nblocks and pages\n"
    )
    kb = KnowledgeBase(tmp_path)
    hits = kb.search("foo v2.1 mi355x server")
    assert hits[0]["path"] == "recipes/Foo-V2.1.md"
    assert "TP4" in hits[0]["section"]
    assert kb.search("kv cache", path_prefix="docs/")[0]["path"] == "docs/guide.md"
    sec = kb.read("guide.md", "kv cache")
    assert "blocks and pages" in sec["text"]
    assert "error" in kb.read("guide.md", "nope")
    assert kb.list_recipes()[0]["launch_commands"] == 1


def test_real_recipes_parse():
    """Every recipe in the checkout parses; most carry a launch command."""
    root = Path(__file__).resolve().parents[2]
    kb = KnowledgeBase(root)
    recipes = kb.list_recipes()
    assert recipes, "no recipes found"
    assert sum(r["launch_commands"] for r in recipes) >= len(recipes) // 2
