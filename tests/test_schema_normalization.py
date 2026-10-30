import itertools

import pytest

Draft202012Validator = pytest.importorskip("jsonschema").Draft202012Validator

from atom.model_engine.schema_normalization import normalize_schema

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


@pytest.mark.parametrize(
    "predicate",
    [
        {},
        True,
        False,
        {"const": 0},
        {"const": True},
        {"const": None},
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
    with pytest.raises(ValueError, match="prefixItems"):
        normalize_schema(schema)


@pytest.mark.parametrize(
    "schema",
    [
        {"type": "array", "items": {"type": "integer"}, "uniqueItems": True},
        {"type": "array", "items": {"enum": list(range(20))}, "uniqueItems": True},
        {"not": {"pattern": "x"}},
        {"type": "object", "if": {"properties": {"x": {"const": 1}}}, "then": False},
    ],
)
def test_unsupported_or_excessive_expansions_are_explicit(schema):
    with pytest.raises(ValueError):
        normalize_schema(schema)
