# SPDX-License-Identifier: MIT
"""GLM-5.2 mono: the two layout refusals Task 6 rescued from being silently
dropped. Each is a negative-path check: a fake model whose layer list
violates exactly one condition, cheap ``SimpleNamespace`` layers, no real
model build.
"""

from types import SimpleNamespace

import pytest

# `mono.dispatch` reaches `mono.weights`, which takes its dtypes from AITER at
# module load.  Skip on the non-GPU unit gate; runs wherever AITER is present.
pytest.importorskip("aiter", reason="needs the AITER GPU kernel library")

from atom.models.glm52.mono.dispatch import _layout_refusal


def _layer(has_experts: bool, indexer, skip_topk: bool = False):
    mlp = SimpleNamespace(experts=object()) if has_experts else SimpleNamespace()
    return SimpleNamespace(
        mlp=mlp, self_attn=SimpleNamespace(indexer=indexer, skip_topk=skip_topk)
    )


def _model(layers):
    return SimpleNamespace(layers=layers, start_layer=0, end_layer=len(layers))


def test_moe_layers_not_a_suffix_is_refused():
    # Review Focus: a dense layer after the first MoE one would be left
    # reading a residual mono already folded in.
    layers = [
        _layer(has_experts=False, indexer=object()),  # dense, full indexer
        _layer(has_experts=True, indexer=None, skip_topk=True),  # MoE, IndexShare
        _layer(has_experts=False, indexer=None, skip_topk=True),  # dense again: break
    ]
    assert _layout_refusal(_model(layers)) == "MoE layers are not a suffix"


def test_indexshare_before_full_layer_is_refused():
    # Review Focus: an IndexShare layer reuses the sparse indices a preceding
    # full layer wrote; one that comes first selects over an unfilled buffer.
    layers = [
        _layer(has_experts=True, indexer=None, skip_topk=True),  # IndexShare first
        _layer(has_experts=True, indexer=object(), skip_topk=False),  # full, too late
    ]
    assert (
        _layout_refusal(_model(layers))
        == "shared IndexShare layer precedes a full layer"
    )


def test_suffix_and_indexer_order_both_valid_is_not_refused():
    layers = [
        _layer(has_experts=False, indexer=object(), skip_topk=False),  # dense, full
        _layer(has_experts=True, indexer=None, skip_topk=True),  # MoE, IndexShare
        _layer(has_experts=True, indexer=None, skip_topk=True),  # MoE, IndexShare
    ]
    assert _layout_refusal(_model(layers)) is None
