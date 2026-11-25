# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Whether ATOM's structured output can run here, and what a client sees.

The backend is written against one llguidance release. The vLLM image pins an
older one for vLLM's own backend, so an installed llguidance is not enough.
"""

import asyncio
import json
import pathlib
import re
import sys
from importlib.metadata import PackageNotFoundError
from types import SimpleNamespace as NS

import pytest

from atom import structured_output
from atom.entrypoints.openai import api_server
from atom.structured_output import validation


@pytest.fixture
def probe(monkeypatch):
    """`available()` against a stated set of installed packages."""

    def run(installed, llguidance_version=structured_output.LLGUIDANCE_VERSION):
        monkeypatch.setattr(
            structured_output,
            "find_spec",
            lambda name: object() if name in installed else None,
        )

        def version(name):
            if llguidance_version is None:
                raise PackageNotFoundError(name)
            return llguidance_version

        monkeypatch.setattr(structured_output, "version", version)
        structured_output.available.cache_clear()
        try:
            return structured_output.available()
        finally:
            structured_output.available.cache_clear()

    return run


ALL = ("llguidance", "jsonschema", "re2")


def test_the_pinned_release_with_every_dependency_is_available(probe):
    assert probe(ALL) is True


@pytest.mark.parametrize("missing", ALL)
def test_a_missing_dependency_is_unavailable(probe, missing):
    assert probe(tuple(n for n in ALL if n != missing)) is False


def test_another_llguidance_release_is_unavailable(probe, caplog):
    with caplog.at_level("WARNING", logger="atom"):
        assert probe(ALL, llguidance_version="1.7.6") is False
    assert "llguidance 1.7.6 is installed" in caplog.text


def test_llguidance_without_package_metadata_is_unavailable(probe):
    # Not an error: every request asks for the deployment metadata.
    assert probe(ALL, llguidance_version=None) is False


def _pyproject_list(key):
    # tomllib is 3.11+; the project supports 3.10.
    pyproject = pathlib.Path(__file__).parents[2] / "pyproject.toml"
    match = re.search(rf"^{key} = (\[.*\])$", pyproject.read_text(), re.MULTILINE)
    return None if match is None else json.loads(match[1])


def test_the_base_install_carries_the_pinned_backend():
    base = _pyproject_list("dependencies")
    assert base is not None, "dependencies is not a one-line list; update the parser"
    assert f"llguidance=={structured_output.LLGUIDANCE_VERSION}" in base
    assert any(dep.startswith("google-re2") for dep in base)


def test_the_refusal_names_the_pin_without_an_install_command():
    assert (
        f"llguidance=={structured_output.LLGUIDANCE_VERSION}"
        in structured_output.UNAVAILABLE
    )
    # An install hint would upgrade llguidance past the vLLM image's <1.8 pin.
    assert "pip install" not in structured_output.UNAVAILABLE


@pytest.mark.parametrize("installed", [True, False])
def test_models_lists_it_only_when_available(monkeypatch, installed):
    monkeypatch.setattr(structured_output, "available", lambda: installed)
    monkeypatch.setattr(api_server, "model_name", "m")
    monkeypatch.setattr(api_server, "custom_message_encoder", None)
    monkeypatch.setattr(api_server, "_get_engine_config", lambda: None)
    card = asyncio.run(api_server.list_models())["data"][0]
    assert ("structured_output" in card) is installed


@pytest.fixture
def without_backend(monkeypatch):
    """An environment without llguidance and RE2 (ATOM installed --no-deps)."""
    monkeypatch.setattr(structured_output, "available", lambda: False)
    monkeypatch.setitem(sys.modules, "llguidance", None)
    monkeypatch.setitem(sys.modules, "re2", None)


# The safety checks compile `pattern` with RE2.
PATTERN_SCHEMA = {
    "type": "object",
    "properties": {"id": {"type": "string", "pattern": "^[a-z]+$"}},
    "required": ["id"],
    "additionalProperties": False,
}


@pytest.mark.parametrize(
    "fmt, responses",
    [
        (
            {
                "type": "json_schema",
                "json_schema": {"name": "r", "schema": PATTERN_SCHEMA},
            },
            False,
        ),
        ({"type": "json_schema", "name": "r", "schema": PATTERN_SCHEMA}, True),
        ({"type": "json_object"}, False),
    ],
)
def test_a_format_is_refused_before_its_checks_need_the_backend(
    without_backend, fmt, responses
):
    with pytest.raises(ValueError, match=re.escape(structured_output.UNAVAILABLE)):
        validation.schema_from_format(fmt, responses=responses)


def test_a_text_format_does_not_need_it(without_backend):
    assert validation.schema_from_format({"type": "text"}) is None


def test_a_strict_tool_is_refused_before_its_checks_need_the_backend(without_backend):
    tool = {"function": {"name": "f", "strict": True, "parameters": PATTERN_SCHEMA}}
    with pytest.raises(ValueError, match=re.escape(structured_output.UNAVAILABLE)):
        validation.strict_tool_schemas([tool])


def test_tools_that_are_not_strict_do_not_need_it(without_backend):
    tool = {"function": {"name": "f", "strict": False, "parameters": PATTERN_SCHEMA}}
    assert validation.strict_tool_schemas([tool]) == {}


def test_the_offline_engine_refuses_too(without_backend):
    from atom.model_engine.llm_engine import LLMEngine

    params = NS(json_schema='{"type": "object"}', structured_tool_format=None)
    with pytest.raises(ValueError, match=re.escape(structured_output.UNAVAILABLE)):
        LLMEngine.prepare_structured_request(NS(config=None), params)
