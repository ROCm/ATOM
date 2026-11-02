# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

import json
import subprocess
import sys

import pytest

from atom.structured_output.request_validation import schema_from_format
from atom.structured_output.response_validation import validate_output
from atom.structured_output.schema_normalization import normalize_schema
from atom.structured_output.schema_validation import (
    check_schema_safety,
    schema_validator,
)


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
from atom.structured_output.schema_validation import schema_validator
from atom.structured_output.schema_normalization import normalize_schema
from atom.structured_output.response_validation import validate_output
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
from atom.structured_output.schema_validation import schema_validator
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

    from atom.structured_output import schema_validation

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
