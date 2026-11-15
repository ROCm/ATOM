# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

import itertools
import json
import subprocess
import sys

import pytest

from atom.structured_output import schema as normalization
from atom.structured_output.schema import (
    SchemaExpansionLimit,
    check_schema_safety,
    normalize_schema,
    schema_validator,
)
from atom.structured_output.validation import schema_from_format, validate_output

Draft202012Validator = pytest.importorskip("jsonschema").Draft202012Validator


VALUES = [
    None,
    True,
    False,
    -2,
    -1,
    0,
    1,
    1.0,
    1.5,
    2,
    3,
    "a",
    "b",
    [],
    {},
    [1],
    {"x": 1},
]


@pytest.mark.parametrize("bound", [0.0, 1.0, 2.0])
@pytest.mark.parametrize(
    "key,kind,values",
    [
        (key, kind, values)
        for low, high, kind, values in [
            ("minLength", "maxLength", "string", ["", "x", "xx", "xxx"]),
            ("minItems", "maxItems", "array", [[], [1], [1, 2], [1, 2, 3]]),
            (
                "minProperties",
                "maxProperties",
                "object",
                [{}, {"a": 1}, {"a": 1, "b": 2}, {"a": 1, "b": 2, "c": 3}],
            ),
        ]
        for key in (low, high)
    ],
)
def test_integer_valued_float_bounds_have_exact_complements(bound, key, kind, values):
    schema = {"type": kind, "not": {key: bound}}
    Draft202012Validator.check_schema(schema)
    lowered = normalize_schema(schema)
    Draft202012Validator.check_schema(lowered)
    assert [Draft202012Validator(schema).is_valid(v) for v in values] == [
        Draft202012Validator(lowered).is_valid(v) for v in values
    ]


@pytest.mark.parametrize(
    "predicate",
    [
        {},
        True,
        False,
        {"const": 0},
        {"const": True},
        {"const": None},
        {"const": "a"},
        {"const": ""},
        {"const": "[中]\\\n"},
        {"minLength": 0},
        {"maxLength": 2},
        {"required": ["x", "y"]},
        {"properties": {"x": {"const": 1}}},
        {"minimum": 0},
        {"exclusiveMaximum": 2},
        {"enum": [0, True, None]},
        {"type": ["number", "null"]},
        {"anyOf": [{"minimum": 0}, {"const": False}]},
        {"allOf": [{"type": "number"}, {"minimum": 0}]},
    ],
)
def test_negation_preserves_validation(predicate):
    original = {"not": predicate}
    lowered = normalize_schema(original)
    Draft202012Validator.check_schema(lowered)
    assert [Draft202012Validator(original).is_valid(v) for v in VALUES] == [
        Draft202012Validator(lowered).is_valid(v) for v in VALUES
    ]


@pytest.mark.parametrize("required", [[], ["kind"]])
def test_conditional_optional_discriminator_and_missing_value(required):
    original = {
        "type": "object",
        "properties": {"kind": {"enum": ["a", "b"]}, "value": {"type": "integer"}},
        "required": required,
        "additionalProperties": False,
        "if": {"properties": {"kind": {"const": "a"}}},
        "then": {"properties": {"value": {"const": 1}}},
        "else": {"properties": {"value": {"const": 2}}},
    }
    lowered = normalize_schema(original)
    for kind, value in itertools.product([None, "a", "b", "c"], [None, 0, 1, 2, "1"]):
        instance = {}
        if kind is not None:
            instance["kind"] = kind
        if value is not None:
            instance["value"] = value
        assert Draft202012Validator(original).is_valid(
            instance
        ) == Draft202012Validator(lowered).is_valid(instance), instance


def test_unique_items_boolean_numeric_equality_and_all_orders():
    original = {
        "items": {"enum": [0, False, 1, 1.0]},
        "uniqueItems": True,
        "maxItems": 3,
    }
    lowered = normalize_schema(original)
    for size in range(4):
        for values in itertools.product([0, False, 1, 1.0, 2], repeat=size):
            value = list(values)
            assert Draft202012Validator(original).is_valid(
                value
            ) == Draft202012Validator(lowered).is_valid(value), value
    for value in [None, True, 1, "text", {}]:
        assert Draft202012Validator(lowered).is_valid(value)


def test_literal_schema_like_keys_are_not_rewritten():
    original = {"const": {"not": {"const": 0}, "uniqueItems": True}}
    assert normalize_schema(original) == original


def test_unique_prefix_items_must_not_drop_values_outside_items_domain():
    schema = {
        "type": "array",
        "prefixItems": [{"const": "head"}],
        "items": {"enum": [1, 2]},
        "uniqueItems": True,
        "minItems": 1,
    }
    assert Draft202012Validator(schema).is_valid(["head", 1])
    lowered = normalize_schema(schema)
    for size in range(4):
        for value in itertools.product(["head", 1, 2, 3], repeat=size):
            assert Draft202012Validator(schema).is_valid(
                list(value)
            ) == Draft202012Validator(lowered).is_valid(list(value))


@pytest.mark.parametrize(
    "schema",
    [
        {"type": "array", "items": {"type": "integer"}, "uniqueItems": True},
        {"type": "array", "items": {"enum": list(range(20))}, "uniqueItems": True},
        {"not": {"pattern": "x"}},
        {
            "type": "object",
            "if": {"properties": {"x": {"pattern": "x"}}},
            "then": False,
        },
    ],
)
def test_unsupported_or_excessive_expansions_are_explicit(schema):
    with pytest.raises(ValueError):
        normalize_schema(schema)


@pytest.mark.parametrize("size", [257, 100_000])
def test_literal_negation_budget_checked_before_regex_allocation(monkeypatch, size):
    from atom.structured_output import schema as module

    def no_escape(_value):
        raise AssertionError("oversized literal must be rejected before expansion")

    monkeypatch.setattr(module.re, "escape", no_escape)
    with pytest.raises(ValueError, match="exceeds 256 characters"):
        normalize_schema({"not": {"const": "a" * size}})


def test_literal_negation_at_budget_remains_exact():
    literal = "a" * 256
    lowered = Draft202012Validator(normalize_schema({"not": {"const": literal}}))
    assert not lowered.is_valid(literal)
    assert all(
        lowered.is_valid(value)
        for value in [literal[:-1], literal + "a", literal[:-1] + "b", 1]
    )


@pytest.mark.parametrize(
    "guard",
    [
        {"properties": {"kind": {"const": "special"}}},
        {"properties": {"kind": {"minimum": 5}}, "required": ["kind"]},
        {"required": ["kind", "v"]},
        {"type": "string", "minLength": 2},
    ],
)
def test_unbounded_conditionals_preserve_absence_and_type_semantics(guard):
    schema = {
        "if": guard,
        "then": {"properties": {"v": {"const": 1}}},
        "else": {"properties": {"v": {"const": 2}}},
    }
    lowered = normalize_schema(schema)
    examples = VALUES + [
        {"kind": k, "v": v}
        for k, v in itertools.product(["special", "other", 4, 6, None], [1, 2, None])
    ]
    examples += [{"v": v} for v in [1, 2, 3]]
    for value in examples:
        assert Draft202012Validator(schema).is_valid(value) == Draft202012Validator(
            lowered
        ).is_valid(value), value


def test_reference_into_literal_cannot_reach_network(monkeypatch):
    import urllib.request

    def forbidden(*args, **kwargs):
        pytest.fail("schema validation attempted network retrieval")

    monkeypatch.setattr(urllib.request, "urlopen", forbidden)
    schema = {
        "const": {"$ref": "http://127.0.0.1:8000/private"},
        "$ref": "#/const",
        "not": False,
    }
    fmt = {"type": "json_schema", "name": "unsafe", "schema": schema}
    with pytest.raises(ValueError, match="Only local"):
        schema_from_format(fmt, responses=True)
    with pytest.raises(ValueError, match="Only local"):
        normalize_schema(schema)
    with pytest.raises(ValueError, match="Only local"):
        schema_validator(schema).validate(schema["const"])
    with pytest.raises(RuntimeError, match="did not satisfy"):
        validate_output(json.dumps(schema), "{}", "stop")


def test_local_reference_destinations_and_cycles():
    schema = {"const": {"type": "integer"}, "$ref": "#/const"}
    check_schema_safety(schema)
    assert schema_validator(schema).is_valid(3) is False  # const still applies
    check_schema_safety(
        {"$defs": {"node": {"$ref": "#/$defs/node"}}, "$ref": "#/$defs/node"}
    )
    safe = {"$defs": {"answer": {"type": "integer"}}, "$ref": "#/$defs/answer"}
    check_schema_safety(safe)
    assert schema_validator(safe).is_valid(3)
    assert not schema_validator(safe).is_valid("3")


@pytest.mark.parametrize("ref", ["#/$defs/a~1b", "#/%24defs/a~1b"])
def test_escaped_local_pointer_is_checked(ref):
    schema = {"$defs": {"a/b": {"$ref": "https://example.invalid/schema"}}, "$ref": ref}
    with pytest.raises(ValueError, match="Only local"):
        check_schema_safety(schema)


def test_literal_reference_text_without_schema_reference_is_data():
    value = {"$ref": "https://example.invalid/literal"}
    schema = {"const": value}
    check_schema_safety(schema)
    assert schema_validator(schema).is_valid(value)


@pytest.mark.parametrize(
    "keyword",
    ["pattern", "patternProperties", "additionalProperties", "unevaluatedProperties"],
)
def test_regex_paths_finish_under_enforced_test_deadline(keyword):
    # The subprocess timeout would stop a regression using Python's backtracking re.
    code = f"""
from atom.structured_output.schema import schema_validator
from atom.structured_output.schema import normalize_schema
from atom.structured_output.validation import validate_output
import json
bad = "a" * 40 + "!"
pattern = "^(a+)+$"
keyword = {keyword!r}
if keyword == "pattern":
    schema = {{"pattern": pattern, "enum": [bad], "not": {{"const": "x"}}}}
    normalize_schema(schema)
    assert not schema_validator(schema).is_valid(bad)
    try: validate_output(json.dumps(schema), json.dumps(bad), "stop")
    except RuntimeError: pass
    else: raise AssertionError("invalid output accepted")
else:
    schema = {{"type": "object", "patternProperties": {{pattern: {{"type": "integer"}}}}}}
    if keyword != "patternProperties": schema[keyword] = False
    assert schema_validator(schema).is_valid({{bad: 1}}) == (keyword == "patternProperties")
"""
    subprocess.run(
        [sys.executable, "-c", code], check=True, timeout=10, capture_output=True
    )


def test_nested_dialect_cannot_restore_backtracking_validator():
    code = """
from atom.structured_output.schema import schema_validator
schema = {
        "properties": {
            "x": {
                "$schema": "https://json-schema.org/draft/2020-12/schema",
                "pattern": "^(a+)+$",
            }
        }
    }
assert not schema_validator(schema).is_valid({"x": "a" * 40 + "!"})
"""
    subprocess.run(
        [sys.executable, "-c", code], check=True, timeout=10, capture_output=True
    )


def test_registry_blocks_retrieval_even_without_admission_guard(monkeypatch):
    import urllib.request

    from referencing.exceptions import Unresolvable

    from atom.structured_output import schema as schema_validation

    monkeypatch.setattr(schema_validation, "check_schema_safety", lambda schema: None)
    monkeypatch.setattr(
        urllib.request,
        "urlopen",
        lambda *args, **kwargs: pytest.fail("network retrieval attempted"),
    )
    with pytest.raises(Unresolvable):
        schema_validator({"$ref": "http://127.0.0.1:8000/private"}).validate({})


def test_unevaluated_properties_respects_valid_branches():
    schema = {
        "allOf": [{"patternProperties": {"^x": {"type": "integer"}}}],
        "unevaluatedProperties": False,
    }
    assert schema_validator(schema).is_valid({"x1": 1})
    assert not schema_validator(schema).is_valid({"y": 1})
    assert not schema_validator(schema).is_valid({"x1": "bad"})


def test_unsupported_regex_rejected_at_admission():
    with pytest.raises(ValueError, match="safe RE2"):
        check_schema_safety({"pattern": r"(a)\1"})


"""Budget exhaustion must stop expansion before allocating large results."""


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

    monkeypatch.setattr(normalization, "schema_validator", forbidden)
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
    from atom.structured_output.backend_guidance import GrammarCompiler

    class NoBackend:
        @staticmethod
        def grammar_from_json_schema(*_args, **_kwargs):
            raise AssertionError("oversized expansion reached backend compilation")

    monkeypatch.setattr(llguidance, "LLMatcher", NoBackend)
    with pytest.raises(SchemaExpansionLimit):
        GrammarCompiler(object()).compile_json_schema(
            json.dumps({"not": {"enum": list(range(300))}})
        )
