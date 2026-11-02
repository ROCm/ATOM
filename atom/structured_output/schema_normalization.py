# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Exact lowering for bounded schemas; never approximate unsupported predicates."""

import itertools
import math
import re
from dataclasses import dataclass

from jsonschema import Draft202012Validator

_TYPES = ["null", "boolean", "number", "string", "array", "object"]
_LIMIT = 256
_ANNOTATIONS = {"title", "description", "default", "examples", "$comment"}
_MISSING = object()


class SchemaExpansionLimit(ValueError):
    """An exhausted budget is terminal, not an unsupported-predicate fallback."""


@dataclass
class _ExpansionBudget:
    branches: int = _LIMIT
    candidates: int = 4096
    regex_chars: int = 256 * 1024

    def consume(self, *, branches=0, candidates=0, regex_chars=0):
        # Reserve capacity before comprehensions, permutations or regex joins.
        if (
            branches > self.branches
            or candidates > self.candidates
            or regex_chars > self.regex_chars
        ):
            raise SchemaExpansionLimit("JSON Schema shared expansion budget exceeded")
        self.branches -= branches
        self.candidates -= candidates
        self.regex_chars -= regex_chars


def _finite(schema, budget):
    if schema is False:
        return []
    if not isinstance(schema, dict):
        return None
    if "const" in schema:
        values = [schema["const"]]
    elif "enum" in schema:
        values = schema["enum"]
    elif schema.get("type") == "boolean":
        values = [False, True]
    elif schema.get("type") == "null":
        values = [None]
    elif (
        schema.get("type") == "integer" and "minimum" in schema and "maximum" in schema
    ):
        lo, hi = math.ceil(schema["minimum"]), math.floor(schema["maximum"])
        if hi - lo >= _LIMIT:
            return None
        values = range(lo, hi + 1)
    else:
        return None
    budget.consume(candidates=len(values))
    validator = Draft202012Validator(schema)
    return [value for value in values if validator.is_valid(value)]


def _negate(predicate, budget):
    if isinstance(predicate, bool):
        return not predicate
    p = {k: v for k, v in predicate.items() if k not in _ANNOTATIONS}
    if not p:
        return False
    # A schema is the intersection of its sibling predicates. Complement each
    # independently, preserving JSON Schema's non-applicable-type semantics.
    if len(p) > 1:
        budget.consume(branches=len(p))
        return {"anyOf": [_negate({k: v}, budget) for k, v in p.items()]}
    key, value = next(iter(p.items()))
    if key in {"allOf", "anyOf"}:
        budget.consume(branches=len(value))
        return {
            "anyOf" if key == "allOf" else "allOf": [_negate(v, budget) for v in value]
        }
    if key == "not":
        return value
    if key == "type":
        types = [value] if isinstance(value, str) else value
        if "integer" in types and "number" not in types:
            raise ValueError("Negating the integer type is not supported")
        allowed = [t for t in _TYPES if t not in types]
        return {"type": allowed} if allowed else False
    inverse = {
        "minimum": "exclusiveMaximum",
        "maximum": "exclusiveMinimum",
        "exclusiveMinimum": "maximum",
        "exclusiveMaximum": "minimum",
    }
    if key in inverse:
        return {"type": "number", inverse[key]: value}
    size_keys = {
        "minLength": ("string", "maxLength", -1),
        "maxLength": ("string", "minLength", 1),
        "minItems": ("array", "maxItems", -1),
        "maxItems": ("array", "minItems", 1),
        "minProperties": ("object", "maxProperties", -1),
        "maxProperties": ("object", "minProperties", 1),
    }
    if key in size_keys:
        type_name, bound, delta = size_keys[key]
        # JSON Schema integers include integer-valued JSON numbers such as 2.0.
        size = int(value) + delta
        return {"type": type_name, bound: size} if size >= 0 else False
    if key == "required":
        budget.consume(branches=len(value))
        branches = [{"type": "object", "properties": {name: False}} for name in value]
        return {"anyOf": branches} if branches else False
    if key == "properties":
        budget.consume(branches=len(value))
        branches = [
            {
                "type": "object",
                "required": [name],
                "properties": {name: _negate(child, budget)},
            }
            for name, child in value.items()
        ]
        return {"anyOf": branches} if branches else False
    if key == "enum":
        budget.consume(branches=len(value))
        return {"allOf": [_negate({"const": v}, budget) for v in value]}
    if key == "const":
        if isinstance(value, str):
            # The regex below emits one escaped prefix per character. Check
            # before slicing/escaping so a small request cannot allocate a
            # quadratic expansion while worker scheduling is paused.
            if len(value) > _LIMIT:
                raise SchemaExpansionLimit(
                    "Literal-string negation exceeds 256 characters"
                )
            # Count escaped characters in O(n), without constructing any prefix.
            regex_chars = (
                (
                    sum(
                        (len(value) - i) * len(re.escape(c))
                        for i, c in enumerate(value)
                    )
                    + 4 * len(value)
                    + 4
                )
                if value
                else 0
            )
            budget.consume(branches=4 if value else 2, regex_chars=regex_chars)
            # All strings except one literal: shorter/longer strings or the
            # same length with a differing character. No regex lookaround.
            branches = [
                {"type": [t for t in _TYPES if t != "string"]},
                {"type": "string", "minLength": len(value) + 1},
            ]
            if value:
                branches.extend(
                    [
                        {"type": "string", "maxLength": len(value) - 1},
                        {
                            "type": "string",
                            "pattern": "^(?:"
                            + "|".join(
                                re.escape(value[:i]) + "[^" + re.escape(c) + "]"
                                for i, c in enumerate(value)
                            )
                            + ")",
                        },
                    ]
                )
            return {"anyOf": branches}
        if value is None:
            return {"type": [t for t in _TYPES if t != "null"]}
        if isinstance(value, bool):
            budget.consume(branches=2)
            return {
                "anyOf": [
                    {"type": [t for t in _TYPES if t != "boolean"]},
                    {"const": not value},
                ]
            }
        if isinstance(value, (int, float)):
            budget.consume(branches=3)
            return {
                "anyOf": [
                    {"type": [t for t in _TYPES if t != "number"]},
                    {"type": "number", "exclusiveMaximum": value},
                    {"type": "number", "exclusiveMinimum": value},
                ]
            }
    raise ValueError(f"Unsupported negated JSON Schema predicate: {key}")


def _conditional(schema, budget):
    guard = schema["if"]
    if isinstance(guard, bool):
        return schema.get("then" if guard else "else", True)
    # Logical implication does not require a finite discriminator domain.
    # Complements below preserve absent-property and non-object semantics.
    try:
        complement = _negate(guard, budget)
    except SchemaExpansionLimit:
        raise
    except ValueError:
        pass  # Retain exact finite enumeration for more complex predicates.
    else:
        budget.consume(branches=6)
        return {
            "anyOf": [
                {"allOf": [guard, schema.get("then", True)]},
                {"allOf": [complement, schema.get("else", True)]},
            ]
        }
    keys = set(guard) - _ANNOTATIONS
    if schema.get("type") != "object" or not keys <= {"properties", "required", "type"}:
        raise ValueError("Conditional schemas require finite object discriminators")
    if guard.get("type", "object") != "object":
        raise ValueError("Conditional schemas require an object discriminator")
    names = list(
        dict.fromkeys([*guard.get("properties", {}), *guard.get("required", [])])
    )
    domains = []
    for name in names:
        values = _finite(schema.get("properties", {}).get(name), budget)
        if values is None:
            raise ValueError(
                f"Conditional discriminator {name!r} must have a finite domain"
            )
        if name not in schema.get("required", []):
            values = [*values, _MISSING]
        domains.append(values)
    if math.prod(map(len, domains)) > _LIMIT:
        raise ValueError("Conditional schema expansion exceeds 256 branches")
    budget.consume(branches=math.prod(map(len, domains)) * (3 + len(names)))
    branches = []
    validator = Draft202012Validator(guard)
    for values in itertools.product(*domains):
        instance = {n: v for n, v in zip(names, values) if v is not _MISSING}
        properties = {
            n: False if v is _MISSING else {"const": v} for n, v in zip(names, values)
        }
        restriction = {
            "type": "object",
            "properties": properties,
            "required": list(instance),
        }
        selected = schema.get("then" if validator.is_valid(instance) else "else", True)
        branches.append({"allOf": [restriction, selected]})
    return {"anyOf": branches} if branches else False


def _unique(schema, budget):
    prefixes = schema.get("prefixItems", [])
    domains = [_finite(p, budget) for p in prefixes]
    # items is irrelevant when maxItems ends within the prefix positions.
    if schema.get("maxItems", math.inf) > len(prefixes):
        domains.append(_finite(schema.get("items"), budget))
    if any(domain is None for domain in domains):
        raise ValueError("uniqueItems currently requires finite item/prefix domains")
    if sum(map(len, domains)) > _LIMIT:
        raise SchemaExpansionLimit("uniqueItems domain exceeds 256 values")
    values = [v for domain in domains for v in domain]
    # Deduplicate with JSON Schema equality (1 == 1.0, but true != 1).
    distinct = []
    for value in values:
        if not any(
            Draft202012Validator({"const": old}).is_valid(value) for old in distinct
        ):
            distinct.append(value)
    lo = schema.get("minItems", 0)
    hi = min(schema.get("maxItems", len(distinct)), len(distinct))
    count = 0
    for n in range(lo, hi + 1):
        count += math.perm(len(distinct), n)
        if count > _LIMIT:
            raise ValueError("uniqueItems expansion exceeds 256 arrays")
    budget.consume(branches=count + 2)
    validator = Draft202012Validator(schema)
    arrays = [
        list(p) for n in range(lo, hi + 1) for p in itertools.permutations(distinct, n)
    ]
    arrays = [a for a in arrays if validator.is_valid(a)]
    array_schema = {"enum": arrays} if arrays else False
    # Array keywords do not constrain non-array values.
    return {"anyOf": [{"type": [t for t in _TYPES if t != "array"]}, array_schema]}


def normalize_schema(schema):
    # One budget for the entire request, including sibling schemas and fallback
    # paths. Recursive normalization must never allocate a fresh budget.
    return _normalize_schema(schema, _ExpansionBudget())


def _normalize_schema(schema, budget):
    if not isinstance(schema, dict):
        return schema
    result = dict(schema)
    for key in (
        "minLength",
        "maxLength",
        "minItems",
        "maxItems",
        "minProperties",
        "maxProperties",
    ):
        if key in result:
            result[key] = int(result[key])
    additions = []
    if "not" in result:
        predicate = result.pop("not")
        finite = _finite(result, budget)
        if finite is not None:
            budget.consume(branches=len(finite))
            values = [
                v for v in finite if not Draft202012Validator(predicate).is_valid(v)
            ]
            additions.append({"enum": values} if values else False)
        else:
            additions.append(_negate(predicate, budget))
    if "if" in result:
        additions.append(_conditional(result, budget))
        for key in ["if", "then", "else"]:
            result.pop(key, None)
    else:
        # Without if these annotations have no assertion semantics.
        result.pop("then", None)
        result.pop("else", None)
    if result.pop("uniqueItems", False):
        original = dict(result, uniqueItems=True)
        additions.append(_unique(original, budget))
    if additions:
        result.setdefault("allOf", [])
        result["allOf"] = [*result["allOf"], *additions]
    # Traverse schema positions only. const/enum/default/examples may contain
    # literal objects whose keys must not be interpreted as schema keywords.
    for key in [
        "properties",
        "patternProperties",
        "$defs",
        "definitions",
        "dependentSchemas",
    ]:
        if key in result:
            result[key] = {
                name: _normalize_schema(v, budget) for name, v in result[key].items()
            }
    for key in [
        "items",
        "additionalProperties",
        "contains",
        "propertyNames",
        "unevaluatedProperties",
        "unevaluatedItems",
    ]:
        if key in result:
            result[key] = _normalize_schema(result[key], budget)
    for key in ["allOf", "anyOf", "oneOf", "prefixItems"]:
        if key in result:
            result[key] = [_normalize_schema(v, budget) for v in result[key]]
    return result
