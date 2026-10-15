# SPDX-License-Identifier: MIT
"""Exercise the device gather's address selection without importing ROCm.

The oracle is the sequence's committed span, not another copy of the device
expression. New cache rows have concurrent producers and may contain Inf.
They must gather a safe placeholder while retaining their key for mailbox fill.
"""

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest


def gather_addresses(keys, positions, window_rows, compressed_rows, ratio, tok, sample):
    source = (
        Path(__file__).resolve().parents[1]
        / "atom/models/deepseek_v4/mono/kernels/attention.py"
    )
    tree = ast.parse(source.read_text())
    gather = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "gather_old_kv"
    )
    # Execute the real pre-gather expressions with scalar expression stand-ins.
    # Stop before buffer loads; these tests require no GPU or FlyDSL installation.
    cutoff = next(
        i
        for i, node in enumerate(gather.body)
        if isinstance(node, ast.For)
        and any(isinstance(child, ast.If) for child in node.body)
    )
    gather.body = gather.body[:cutoff] + [
        ast.Return(
            ast.Tuple(
                [ast.Name("krows", ast.Load()), ast.Name("pending", ast.Load())],
                ast.Load(),
            )
        )
    ]
    namespace = {
        "fx": SimpleNamespace(Int32=lambda n: Scalar(n), min=min, max=max),
        "range_constexpr": range,
        "const_expr": bool,
        "KPW": len(keys),
        "TOK": tok,
        "CR": ratio,
        "wave": 0,
        "keys": keys,
        "lds_ld": lambda rows, i: Scalar(rows[i]),
        "ld_pos": lambda s: positions[s],
        "ld_dest": lambda kind, s: window_rows[s],
        "comp_row": lambda s, e: compressed_rows[s],
    }
    module = ast.fix_missing_locations(ast.Module([gather], type_ignores=[]))
    exec(  # noqa: S102 - trusted repo AST
        compile(module, str(source), "exec"), namespace
    )
    return namespace["gather_old_kv"](sample)


class Scalar(int):
    def select(self, yes, no):
        return yes if self else no

    def __eq__(self, other):
        return Scalar(int(self) == other)

    def __or__(self, other):
        return Scalar(bool(self) or bool(other))

    def __and__(self, other):
        return Scalar(bool(self) and bool(other))


@pytest.mark.parametrize("tok", [1, 4])
@pytest.mark.parametrize("ratio", [4, 128])
@pytest.mark.parametrize("start", [0, 892, 895, 1023])
def test_current_span_uses_mailboxes_without_reading_unwritten_cache(tok, ratio, start):
    positions = list(range(start, start + tok))
    windows = [10000 + i for i in range(tok)]
    compressed = [20000 + position // ratio for position in positions]
    keys = list(dict.fromkeys([31, -1, *windows, *compressed]))
    for sample in range(tok):
        addresses, pending = gather_addresses(
            keys, positions, windows, compressed, ratio, tok, sample
        )
        expected = set(windows[: sample + 1])
        expected.update(
            compressed[i]
            for i, position in enumerate(positions[: sample + 1])
            if (position + 1) % ratio == 0
        )
        assert [bool(value) for value in pending] == [key in expected for key in keys]
        assert addresses == [0 if key in expected else max(key, 0) for key in keys]
        assert keys[0] == 31  # old committed rows and key identities stay intact


@pytest.mark.parametrize("tok", [1, 4])
@pytest.mark.parametrize("ratio", [0, 4, 128])
def test_pending_rows_do_not_cross_request_or_future_token_boundaries(tok, ratio):
    positions = list(range(895, 895 + tok)) + list(range(1023, 1023 + tok))
    windows = [10000 + i for i in range(2 * tok)]
    compressed = [
        20000 + (i // tok) * 1000 + position // (ratio or 1)
        for i, position in enumerate(positions)
    ]
    keys = list(dict.fromkeys([31, -1, *windows, *compressed]))
    for sample in range(2 * tok):
        addresses, pending = gather_addresses(
            keys, positions, windows, compressed, ratio, tok, sample
        )
        own_span = range(sample // tok * tok, sample + 1)
        expected = {windows[i] for i in own_span}
        if ratio:
            expected.update(
                compressed[i] for i in own_span if (positions[i] + 1) % ratio == 0
            )
        assert [bool(value) for value in pending] == [key in expected for key in keys]
        assert addresses == [0 if key in expected else max(key, 0) for key in keys]
