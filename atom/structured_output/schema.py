# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Offline schema safety checks and exact, bounded constraint normalization."""

import itertools
import math
import re
from dataclasses import dataclass
from functools import lru_cache

from jsonschema import Draft202012Validator, ValidationError, validators
from referencing import Registry
from referencing.exceptions import NoSuchResource
from referencing.jsonschema import DRAFT202012

_MAPS = ("properties", "patternProperties", "$defs", "definitions", "dependentSchemas")


_SINGLES = (
    "items",
    "additionalProperties",
    "contains",
    "propertyNames",
    "unevaluatedProperties",
    "unevaluatedItems",
    "not",
    "if",
    "then",
    "else",
)


_ARRAYS = ("allOf", "anyOf", "oneOf", "prefixItems")


def _no_retrieval(uri):
    raise NoSuchResource(ref=uri)


OFFLINE_REGISTRY = Registry(retrieve=_no_retrieval)


@lru_cache(maxsize=32)
def _regex(pattern):
    import re2

    if len(pattern) > 262144:
        raise ValueError("JSON Schema regex exceeds 262144 characters")
    options = re2.Options()
    options.max_mem = 2 * 1024 * 1024
    options.log_errors = False
    try:
        return re2.compile(pattern, options=options)
    except re2.error as exc:
        raise ValueError(
            "JSON Schema regex is unsupported by the safe RE2 evaluator"
        ) from exc


def check_schema_safety(schema):
    """Visit schema positions AND local reference destinations, including literals."""
    resolver = OFFLINE_REGISTRY.resolver_with_root(DRAFT202012.create_resource(schema))
    seen = set()

    def visit(value, current):
        if not isinstance(value, dict) or id(value) in seen:
            return
        seen.add(id(value))
        dialect = value.get("$schema")
        if (
            dialect
            and dialect.rstrip("#") != "https://json-schema.org/draft/2020-12/schema"
        ):
            raise ValueError("Structured output requires JSON Schema draft 2020-12")
        for keyword in ("$ref", "$dynamicRef"):
            if keyword not in value:
                continue
            ref = value[keyword]
            if not isinstance(ref, str) or not ref.startswith("#"):
                raise ValueError("Only local JSON Schema references are supported")
            try:
                resolved = current.lookup(ref)
            except Exception as exc:
                raise ValueError("Cannot resolve local JSON Schema reference") from exc
            visit(resolved.contents, resolved.resolver)
        if "pattern" in value:
            _regex(value["pattern"])
        for pattern in value.get("patternProperties", {}):
            _regex(pattern)
        children = []
        for key in _MAPS:
            children.extend(value.get(key, {}).values())
        for key in _SINGLES:
            if key in value:
                children.append(value[key])
        for key in _ARRAYS:
            children.extend(value.get(key, []))
        for child in children:
            if isinstance(child, (dict, bool)):
                visit(child, current.in_subresource(DRAFT202012.create_resource(child)))

    visit(schema, resolver)


def _pattern(validator, pattern, instance, schema):
    if validator.is_type(instance, "string") and not _regex(pattern).search(instance):
        yield ValidationError("String does not match the required pattern")


def _pattern_properties(validator, patterns, instance, schema):
    if validator.is_type(instance, "object"):
        for pattern, child in patterns.items():
            compiled = _regex(pattern)
            for key, value in instance.items():
                if compiled.search(key):
                    yield from validator.descend(
                        value, child, path=key, schema_path=pattern
                    )


def _additional_properties(validator, additional, instance, schema):
    if not validator.is_type(instance, "object"):
        return
    patterns = [_regex(p) for p in schema.get("patternProperties", {})]
    for key, value in instance.items():
        if key not in schema.get("properties", {}) and not any(
            p.search(key) for p in patterns
        ):
            yield from validator.descend(value, additional, path=key)


def _evaluated_keys(validator, instance, schema):
    if isinstance(schema, bool):
        return set()
    keys = set(schema.get("properties", {})) & instance.keys()
    for keyword in ("$ref", "$dynamicRef"):
        if keyword in schema:
            resolved = validator._resolver.lookup(schema[keyword])
            keys.update(
                _evaluated_keys(
                    validator.evolve(
                        schema=resolved.contents, _resolver=resolved.resolver
                    ),
                    instance,
                    resolved.contents,
                )
            )
    for keyword in ("additionalProperties", "unevaluatedProperties"):
        if keyword in schema:
            keys.update(
                k
                for k, v in instance.items()
                if not any(validator.descend(v, schema[keyword]))
            )
    for pattern in schema.get("patternProperties", {}):
        keys.update(k for k in instance if _regex(pattern).search(k))
    for key, child in schema.get("dependentSchemas", {}).items():
        if key in instance:
            keys.update(_evaluated_keys(validator, instance, child))
    for keyword in ("allOf", "anyOf", "oneOf"):
        for child in schema.get(keyword, []):
            if not any(validator.descend(instance, child)):
                keys.update(_evaluated_keys(validator, instance, child))
    if "if" in schema:
        matched = validator.evolve(schema=schema["if"]).is_valid(instance)
        if matched:
            keys.update(_evaluated_keys(validator, instance, schema["if"]))
        branch = "then" if matched else "else"
        if branch in schema:
            keys.update(_evaluated_keys(validator, instance, schema[branch]))
    return keys


def _unevaluated_properties(validator, remaining, instance, schema):
    if validator.is_type(instance, "object"):
        evaluated = _evaluated_keys(validator, instance, schema)
        for key in instance.keys() - evaluated:
            yield from validator.descend(instance[key], remaining, path=key)


_SafeValidator = validators.extend(
    Draft202012Validator,
    {
        "pattern": _pattern,
        "patternProperties": _pattern_properties,
        "additionalProperties": _additional_properties,
        "unevaluatedProperties": _unevaluated_properties,
    },
)


_evolve = _SafeValidator.evolve


def _safe_evolve(self, **changes):
    # A nested $schema must not switch back to jsonschema's backtracking regexes.
    schema = changes.get("schema", self.schema)
    if isinstance(schema, dict) and "$schema" in schema:
        schema = {k: v for k, v in schema.items() if k != "$schema"}
    return _evolve(self, **dict(changes, schema=schema))


_SafeValidator.evolve = _safe_evolve


def schema_validator(schema):
    check_schema_safety(schema)
    return _SafeValidator(schema, registry=OFFLINE_REGISTRY)


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
    validator = schema_validator(schema)
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
    validator = schema_validator(guard)
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
            schema_validator({"const": old}).is_valid(value) for old in distinct
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
    validator = schema_validator(schema)
    arrays = [
        list(p) for n in range(lo, hi + 1) for p in itertools.permutations(distinct, n)
    ]
    arrays = [a for a in arrays if validator.is_valid(a)]
    array_schema = {"enum": arrays} if arrays else False
    # Array keywords do not constrain non-array values.
    return {"anyOf": [{"type": [t for t in _TYPES if t != "array"]}, array_schema]}


def expansion_budget():
    """A fresh budget, for a request that normalizes several schemas."""
    return _ExpansionBudget()


def normalize_schema(schema, budget=None):
    # One budget for the entire request, including sibling schemas and fallback
    # paths. Recursive normalization must never allocate a fresh budget, and a
    # request with several schemas (strict tool parameters) passes its own.
    return _normalize_schema(schema, _ExpansionBudget() if budget is None else budget)


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
            values = [v for v in finite if not schema_validator(predicate).is_valid(v)]
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


# Schema-position maps: their keys are names, never keywords.
_NAME_MAPS = (
    "properties",
    "patternProperties",
    "$defs",
    "$def",
    "definitions",
    "dependentSchemas",
)
# Literal values: a "$ref" key inside one is data, not a reference.
_LITERALS = ("const", "enum", "default", "examples")


def canonical_defs(schema):
    """``schema`` with DeepSeek's ``$def`` definitions moved into ``$defs``.

    DeepSeek's strict mode spells the definitions container ``$def``. The
    safety check, normalization and the grammar backend all walk ``$defs``,
    so what sat under ``$def`` went unchecked and unnormalized; references
    into it are rewritten to match. A name defined under both is refused.
    """
    if not isinstance(schema, dict) or "$def" not in schema:
        return schema
    defs, existing = schema["$def"], schema.get("$defs", {})
    if not isinstance(defs, dict) or not isinstance(existing, dict):
        raise ValueError("$def and $defs must be objects")  # noqa: TRY004
    clash = sorted(set(defs) & set(existing))
    if clash:
        raise ValueError(f"{clash[0]!r} is defined in both $def and $defs")
    result = _rewrite_def_refs(schema)
    result["$defs"] = {**result.get("$defs", {}), **result.pop("$def")}
    return result


def _rewrite_def_refs(node):
    if isinstance(node, list):
        return [_rewrite_def_refs(child) for child in node]
    if not isinstance(node, dict):
        return node
    result = {}
    for key, value in node.items():
        if key in _LITERALS:
            result[key] = value
        elif key == "$ref" and isinstance(value, str) and value.startswith("#/$def/"):
            result[key] = "#/$defs/" + value[len("#/$def/") :]
        elif key in _NAME_MAPS and isinstance(value, dict):
            result[key] = {name: _rewrite_def_refs(v) for name, v in value.items()}
        else:
            result[key] = _rewrite_def_refs(value)
    return result
