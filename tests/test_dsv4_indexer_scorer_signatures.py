# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""The two MX indexer scorers must accept exactly what `indexer_score_topk` passes.

`indexer_score_topk` calls the prefill and decode scorers with one shared
argument list, and nothing else in the test suite instantiates an Indexer -- so a
signature drift between the dispatch and either scorer is invisible until a
server boots and a request lands on that phase. That is how a decode-path
`TypeError` once survived a green unit run.

This is a source-level check (ast, no import) so it stays runnable on a CPU-only
box without aiter.
"""

import ast
from pathlib import Path

MODEL = (
    Path(__file__).resolve().parents[1] / "atom" / "models" / "deepseek_v4.py"
)
DISPATCH = "indexer_score_topk"
SCORERS = ("_score_topk_prefill_fp4", "_score_topk_decode_fp4")


def _tree():
    return ast.parse(MODEL.read_text(encoding="utf-8"), filename=str(MODEL))


def _funcs(tree):
    return {
        node.name: node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef)
    }


def _calls_in(node, names):
    out = []
    for sub in ast.walk(node):
        if (
            isinstance(sub, ast.Call)
            and isinstance(sub.func, ast.Attribute)
            and sub.func.attr in names
        ):
            out.append(sub)
    return out


def test_dispatch_matches_both_scorer_signatures():
    funcs = _funcs(_tree())
    for name in (DISPATCH, *SCORERS):
        assert name in funcs, f"{name} not found in {MODEL.name}"

    calls = _calls_in(funcs[DISPATCH], set(SCORERS))
    assert {c.func.attr for c in calls} == set(SCORERS), (
        "indexer_score_topk must dispatch to both MX scorers; found "
        f"{sorted(c.func.attr for c in calls)}"
    )

    for call in calls:
        target = funcs[call.func.attr]
        # `self` is bound, so the declared arity excludes it.
        declared = [a.arg for a in target.args.args if a.arg != "self"]
        n_required = len(declared) - len(target.args.defaults)
        n_passed = len(call.args) + len(call.keywords)
        assert n_passed == len(declared), (
            f"{call.func.attr}: dispatch passes {n_passed} args, "
            f"signature declares {len(declared)} ({declared})"
        )
        assert n_passed >= n_required, (
            f"{call.func.attr}: dispatch passes {n_passed} args, "
            f"{n_required} are required"
        )


def test_both_scorers_take_the_same_arguments():
    """They are called from one site with one arg list; keep them interchangeable."""
    funcs = _funcs(_tree())
    sigs = {
        name: [a.arg for a in funcs[name].args.args if a.arg != "self"]
        for name in SCORERS
    }
    prefill, decode = (sigs[n] for n in SCORERS)
    assert prefill == decode, (
        "prefill and decode scorers must share an argument list so the single "
        f"dispatch site stays correct for both: {sigs}"
    )
