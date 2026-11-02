# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Offline JSON Schema validation with bounded, non-backtracking regexes."""

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
