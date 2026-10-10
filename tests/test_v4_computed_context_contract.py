# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""`num_computed` must come from the first token's position, not a subtraction.

`_batch_has_computed_context` decides whether a V4 batch goes down
`PREFILL_PREFIX` (some sequence already has KV committed) or `PREFILL_NATIVE`
(none has). The quantity it needs is `num_computed`, and there are two ways to
get it:

* the global position of each sequence's FIRST token this forward, which *is*
  `num_computed` by definition, and is exact; or
* `seq_len - query_len`, which is only as good as `seq_len`.

Only the first is usable here. vLLM 0.29 deprecated and 0.31 removed the exact
CPU mirrors (`_seq_lens_cpu`, `_num_computed_tokens_cpu`), leaving
`seq_lens_cpu_upper_bound` -- and `_build_dsv4_metadata` already documents that
on a speculative-decode (MTP) mixed prefill+verify batch that bound
OVERESTIMATES `seq_len`, so the subtraction exceeds a verify token's true
position. Overestimating here routes a batch with no committed KV onto
`PREFILL_PREFIX`.

That misclassification is invisible to the gsm8k cells: the V4-Flash cell never
builds a batch that mixes a short prefill with verify rows, so the only run that
would distinguish the two implementations is one CI does not have. Hence this
test, which is also why it lives in `tests/` rather than `tests/plugin/`:
importing the bridge pulls in vLLM, and CI's unit-test job both lacks vLLM and
passes `--ignore=tests/plugin`. Following
`tests/test_vllm_mla_bind_guard_contract.py`, the function is lifted out of the
source and evaluated, so what is asserted is the decision it makes rather than
how it is spelled.
"""

from __future__ import annotations

import ast
import pathlib
import types

import numpy as np
import pytest

torch = pytest.importorskip("torch")

_SOURCE = (
    pathlib.Path(__file__).resolve().parents[1]
    / "atom"
    / "plugin"
    / "vllm"
    / "deepseek_v4_bridge.py"
)
_FUNC = "_batch_has_computed_context"


def _load_function():
    """Compile just `_batch_has_computed_context`, with no vLLM import."""
    tree = ast.parse(_SOURCE.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == _FUNC:
            module = ast.Module(body=[node], type_ignores=[])
            namespace: dict = {"np": np, "torch": torch}
            exec(compile(module, str(_SOURCE), "exec"), namespace)  # noqa: S102
            return namespace[_FUNC]
    raise AssertionError(f"{_FUNC} not found in {_SOURCE}")


def _metadata(query_start_loc, positions, *, seq_lens_upper_bound=None):
    q = torch.tensor(query_start_loc, dtype=torch.int32)
    return types.SimpleNamespace(
        num_reqs=len(query_start_loc) - 1,
        query_start_loc_cpu=q,
        positions=(
            None if positions is None else torch.tensor(positions, dtype=torch.int64)
        ),
        seq_lens_cpu_upper_bound=(
            None
            if seq_lens_upper_bound is None
            else torch.tensor(seq_lens_upper_bound, dtype=torch.int32)
        ),
    )


def test_fresh_batch_has_no_computed_context():
    """Every sequence starts at token 0 -> nothing committed."""
    fn = _load_function()
    # two sequences of 4 tokens, both starting at position 0
    md = _metadata([0, 4, 8], [0, 1, 2, 3, 0, 1, 2, 3])
    assert fn(md) is False


def test_resumed_sequence_has_computed_context():
    """A sequence whose first token is past 0 has KV committed."""
    fn = _load_function()
    md = _metadata([0, 4], [10, 11, 12, 13])
    assert fn(md) is True


def test_overstating_upper_bound_does_not_invent_computed_context():
    """The regression this file exists for.

    A mixed prefill+verify batch where `seq_lens_cpu_upper_bound` overstates
    `seq_len` -- the MTP case `_build_dsv4_metadata` documents -- while the true
    `num_computed` is 0 for every sequence. `seq_len - query_len` would be
    positive here and classify the batch `PREFILL_PREFIX`; the first-token
    position is 0, so it must stay `PREFILL_NATIVE`.
    """
    fn = _load_function()
    query_start_loc = [0, 4, 8]
    positions = [0, 1, 2, 3, 0, 1, 2, 3]  # truth: both sequences start at 0
    # optimistic bound: assumes every draft was accepted, so seq_len > query_len
    overstated = [7, 7]
    md = _metadata(query_start_loc, positions, seq_lens_upper_bound=overstated)

    q = np.asarray(query_start_loc, dtype=np.int64)
    query_lens = np.diff(q)
    # negative control: the rejected implementation really would say True here,
    # so this test fails for the right reason rather than vacuously.
    assert bool((np.asarray(overstated) - query_lens > 0).any())

    assert fn(md) is False


def test_cudagraph_padded_rows_are_not_indexed():
    """Padded requests contribute no tokens and must not be read as rows."""
    fn = _load_function()
    # second request is padding: query_start_loc does not advance
    md = _metadata([0, 3, 3], [0, 1, 2])
    assert fn(md) is False


def test_missing_positions_reports_no_computed_context():
    """Without positions there is no exact source; match the builder's arange."""
    fn = _load_function()
    md = _metadata([0, 4], None)
    assert fn(md) is False
