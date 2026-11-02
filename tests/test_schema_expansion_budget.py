# SPDX-License-Identifier: MIT
"""Budget exhaustion must stop expansion before allocating large results."""

import json

import pytest
from jsonschema import Draft202012Validator

from atom.model_engine import schema_normalization as normalization
from atom.model_engine.schema_normalization import (
    SchemaExpansionLimit,
    normalize_schema,
)


@pytest.mark.parametrize("key", ["enum", "required", "properties", "allOf", "anyOf"])
def test_oversized_negation_rejected_before_iterating_entries(key):
    class NoIteration(list):
        def __iter__(self):
            raise AssertionError("budget must be checked before expansion")

    class NoItems(dict):
        def items(self):
            raise AssertionError("budget must be checked before expansion")

    value = (
        NoItems({str(i): True for i in range(300)})
        if key == "properties"
        else NoIteration([True] * 300)
    )
    with pytest.raises(SchemaExpansionLimit):
        normalize_schema({"not": {key: value}})


def test_branch_budget_is_shared_across_sibling_schemas():
    child = {"not": {"required": [f"p{i}" for i in range(130)]}}
    normalize_schema(child)
    with pytest.raises(SchemaExpansionLimit):
        normalize_schema({"properties": {"a": child, "b": child}})


def test_nested_negations_share_budget():
    child = {"properties": {f"p{i}": {"const": i} for i in range(40)}}
    normalize_schema({"not": child})
    with pytest.raises(SchemaExpansionLimit):
        normalize_schema({"not": {"anyOf": [child, child]}})


def test_regex_budget_is_shared_across_short_enough_literals():
    child = {"not": {"const": "a" * 256}}
    normalize_schema(child)
    with pytest.raises(SchemaExpansionLimit):
        normalize_schema({"properties": {str(i): child for i in range(9)}})


def test_exhaustion_cannot_enter_conditional_fallback(monkeypatch):
    def forbidden(*_args):
        raise AssertionError("budget failure must not enter finite fallback")

    monkeypatch.setattr(normalization, "_finite", forbidden)
    with pytest.raises(SchemaExpansionLimit):
        normalize_schema(
            {
                "type": "object",
                "if": {"required": [str(i) for i in range(300)]},
                "then": False,
            }
        )


def test_unique_expansions_share_the_same_budget():
    child = {"type": "array", "uniqueItems": True, "items": {"enum": [0, 1, 2, 3]}}
    normalize_schema(child)
    with pytest.raises(SchemaExpansionLimit):
        normalize_schema({"properties": {str(i): child for i in range(4)}})


def test_finite_candidate_budget_precedes_validation(monkeypatch):
    def forbidden(*_args):
        raise AssertionError("candidate validation must not start")

    monkeypatch.setattr(normalization, "Draft202012Validator", forbidden)
    with pytest.raises(SchemaExpansionLimit):
        normalize_schema({"enum": list(range(5000)), "not": {"const": 1}})


def test_budget_boundary_preserves_semantics_and_resets_per_request():
    names = [f"p{i}" for i in range(256)]
    schema = {"not": {"required": names}}
    normalized = normalize_schema(schema)
    assert len(normalized["allOf"][0]["anyOf"]) == 256
    samples = [{}, dict.fromkeys(names, 1), dict.fromkeys(names[:-1], 1), 7]
    assert [Draft202012Validator(schema).is_valid(v) for v in samples] == [
        Draft202012Validator(normalized).is_valid(v) for v in samples
    ]
    with pytest.raises(SchemaExpansionLimit):
        normalize_schema({"not": {"required": names + ["overflow"]}})
    assert normalize_schema(schema) == normalized


def test_unexpanded_properties_are_not_charged_as_generated_branches():
    schema = {"properties": {str(i): {"type": "string"} for i in range(300)}}
    assert normalize_schema(schema) == schema


def test_budget_failure_precedes_backend_compilation(monkeypatch):
    llguidance = pytest.importorskip("llguidance")
    from atom.model_engine.structured_grammar import GrammarCompiler

    class NoBackend:
        @staticmethod
        def grammar_from_json_schema(*_args, **_kwargs):
            raise AssertionError("oversized expansion reached backend compilation")

    monkeypatch.setattr(llguidance, "LLMatcher", NoBackend)
    with pytest.raises(SchemaExpansionLimit):
        GrammarCompiler(object()).compile_json_schema(
            json.dumps({"not": {"enum": list(range(300))}})
        )
