# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors
"""The gate's GitHub client (ci_gate/github_api.py) classifies every call:
a fact (absence included), a transient failure (retried), or an error."""

import http.client
import io
import json
import sys
import urllib.error
from email.message import Message
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / ".github" / "scripts"))

from ci_gate.github_api import GitHub, GitHubError


class _Response(io.BytesIO):
    def __init__(self, payload: bytes, headers: Message):
        super().__init__(payload)
        self.headers = headers

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _headers(**headers) -> Message:
    message = Message()
    for key, value in headers.items():
        message[key.replace("_", "-")] = value
    return message


def _http_error(code: int, body: str = "", **headers) -> urllib.error.HTTPError:
    return urllib.error.HTTPError(
        "u", code, "x", _headers(**headers), io.BytesIO(body.encode())
    )


class _Ok:
    """A 200 with headers."""

    def __init__(self, value, **headers):
        self.value = value
        self.headers = _headers(**headers)


def _client(*outcomes):
    """A client whose calls take `outcomes` in order: an exception to raise,
    an _Ok, else a JSON value to return. Returns the client and the waits it
    slept."""
    queue = list(outcomes)
    waits: list[float] = []

    def opener(request, timeout):
        outcome = queue.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        ok = outcome if isinstance(outcome, _Ok) else _Ok(outcome)
        return _Response(json.dumps(ok.value).encode(), ok.headers)

    return GitHub("t", "ROCm/ATOM", sleep=waits.append, opener=opener), waits


@pytest.mark.parametrize(
    "failure",
    [
        TimeoutError("read timed out"),
        http.client.RemoteDisconnected("closed"),
        http.client.IncompleteRead(b""),
        urllib.error.URLError("no route"),
        _http_error(502),
    ],
    ids=["timeout", "disconnected", "incomplete-read", "url-error", "5xx"],
)
def test_transient_failures_are_retried(failure):
    gh, waits = _client(failure, {"ok": 1})
    assert gh.rest("GET", "x") == {"ok": 1}
    assert len(waits) == 1


def test_rate_limits_wait_as_github_says():
    gh, waits = _client(
        _http_error(429, Retry_After="7"),
        _http_error(403, "You have exceeded a secondary rate limit"),
        {"ok": 1},
    )
    assert gh.rest("GET", "x") == {"ok": 1}
    assert waits == [7.0, 60.0]


def test_a_graphql_rate_limit_comes_in_a_200_and_is_retried():
    # GitHub's documented signals: a primary limit, an error with
    # x-ratelimit-remaining at 0; a secondary one, an error that says so.
    primary = {"errors": [{"message": "limit"}]}
    secondary = {"errors": [{"message": "You have exceeded a secondary rate limit"}]}
    typed = {"errors": [{"type": "RATE_LIMITED", "message": "x"}]}
    gh, waits = _client(
        _Ok(primary, X_RateLimit_Remaining="0", Retry_After="9"),
        _Ok(secondary),
        _Ok(typed),
        {"data": {"ok": 1}},
    )
    assert gh.graphql("{ok}", {}) == {"ok": 1}
    assert waits == [9.0, 60.0, 60.0]


def test_another_graphql_error_is_an_error_not_retried():
    gh, waits = _client(_Ok({"errors": [{"type": "FORBIDDEN", "message": "no"}]}))
    with pytest.raises(GitHubError):
        gh.graphql("{ok}", {})
    assert waits == []


def test_a_failure_that_does_not_clear_becomes_an_error():
    gh, waits = _client(*[TimeoutError()] * 5)
    with pytest.raises(GitHubError):
        gh.rest("GET", "x")
    assert len(waits) == 4


def test_a_plain_403_is_an_error_not_a_retry():
    gh, waits = _client(_http_error(403, "Resource not accessible by integration"))
    with pytest.raises(GitHubError) as raised:
        gh.rest("GET", "x")
    assert raised.value.status == 403 and not waits


def test_absence_is_a_fact():
    gh, _ = _client(_http_error(404))
    assert gh.rest_optional("repos/x/pulls/1") is None


def test_graphql_not_found_leaves_its_field_null():
    response = {
        "data": {"repository": {"d1": None, "d2": {"__typename": "PullRequest"}}},
        "errors": [{"type": "NOT_FOUND", "path": ["repository", "d1"]}],
    }
    gh, _ = _client(response)
    assert gh.graphql("q", {})["repository"]["d1"] is None


def test_any_other_graphql_error_is_raised():
    gh, _ = _client({"data": None, "errors": [{"type": "FORBIDDEN"}]})
    with pytest.raises(GitHubError):
        gh.graphql("q", {})
