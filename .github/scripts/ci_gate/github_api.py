# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors
"""A GitHub REST / GraphQL client on the standard library, and the one place
that classifies what a call returns:

- a fact, including absence: a REST 404 is None from `rest_optional`, a GraphQL
  NOT_FOUND is the null the response already holds in place of the field;
- transient: a network failure, a 5xx, a primary or secondary rate limit (a
  403 / 429, or for GraphQL also a 200 whose error is the limit);
  retried with backoff, the wait taken from Retry-After or the limit's reset;
- an error: anything else, raised as GitHubError.
"""

from __future__ import annotations

import http.client
import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Any

_API = os.environ.get("GITHUB_API_URL", "https://api.github.com")
_ATTEMPTS = 5
_MAX_WAIT_S = 60.0
_LIMIT_WAIT_S = 60.0  # a rate limit whose response does not say how long
_TRANSIENT_EXCEPTIONS = (
    urllib.error.URLError,
    TimeoutError,
    ConnectionError,
    http.client.HTTPException,
)


class GitHubError(RuntimeError):
    def __init__(self, status: int, url: str, body: str):
        super().__init__(f"GitHub API {status} for {url}: {body[:300]}")
        self.status = status


class _Transient(Exception):
    def __init__(self, reason: str, wait_s: float | None):
        super().__init__(reason)
        self.wait_s = wait_s


class GitHub:
    def __init__(
        self, token: str, repo: str, sleep=time.sleep, opener=urllib.request.urlopen
    ):
        self.token = token
        self.repo = repo
        self.owner, self.name = repo.split("/", 1)
        self._sleep = sleep
        self._open = opener

    def rest(
        self,
        method: str,
        path: str,
        params: dict[str, Any] | None = None,
        body: dict[str, Any] | None = None,
    ) -> Any:
        return self._call(method, self._url(path, params), body)

    def rest_optional(self, path: str, params: dict[str, Any] | None = None) -> Any:
        """GET a resource that may not exist: None when it does not."""
        try:
            return self.rest("GET", path, params)
        except GitHubError as exc:
            if exc.status == 404:
                return None
            raise

    def rest_pages(
        self, path: str, key: str | None, params: dict[str, Any], limit: int
    ) -> list[Any]:
        """Items of a paginated list endpoint, at most `limit`."""
        items: list[Any] = []
        page = 1
        while len(items) < limit:
            data = self.rest("GET", path, {**params, "per_page": 100, "page": page})
            batch = data[key] if key else data
            items.extend(batch)
            if len(batch) < 100:
                break
            page += 1
        return items[:limit]

    def graphql(self, query: str, variables: dict[str, Any]) -> dict[str, Any]:
        """The response data. A NOT_FOUND error leaves its field null, which
        is the answer; any other error is raised."""
        response = self._call(
            "POST", f"{_API}/graphql", {"query": query, "variables": variables}
        )
        errors = [
            e for e in response.get("errors") or [] if e.get("type") != "NOT_FOUND"
        ]
        if errors or response.get("data") is None:
            raise GitHubError(200, "graphql", json.dumps(errors or response))
        return response["data"]

    def _url(self, path: str, params: dict[str, Any] | None) -> str:
        url = f"{_API}/{path.lstrip('/')}"
        return url + "?" + urllib.parse.urlencode(params) if params else url

    def _call(self, method: str, url: str, body: dict[str, Any] | None) -> Any:
        data = json.dumps(body).encode() if body is not None else None
        for attempt in range(1, _ATTEMPTS + 1):
            try:
                return self._send(method, url, data)
            except _Transient as exc:
                if attempt == _ATTEMPTS:
                    raise GitHubError(
                        0, url, f"still failing after {attempt} attempts: {exc}"
                    ) from exc
                wait = exc.wait_s if exc.wait_s is not None else 2.0**attempt
                self._sleep(min(max(wait, 1.0), _MAX_WAIT_S))
        raise AssertionError("unreachable")

    def _send(self, method: str, url: str, data: bytes | None) -> Any:
        request = urllib.request.Request(url, data=data, method=method)
        request.add_header("Authorization", f"Bearer {self.token}")
        request.add_header("Accept", "application/vnd.github+json")
        request.add_header("X-GitHub-Api-Version", "2022-11-28")
        if data is not None:
            request.add_header("Content-Type", "application/json")
        try:
            with self._open(request, timeout=30) as response:
                payload = response.read()
                headers = response.headers
        except urllib.error.HTTPError as exc:
            body = exc.read().decode(errors="replace")
            wait = _rate_limit_wait(exc.code, exc.headers, body)
            if exc.code >= 500 or wait is not None:
                raise _Transient(f"HTTP {exc.code}", wait) from exc
            raise GitHubError(exc.code, url, body) from exc
        except _TRANSIENT_EXCEPTIONS as exc:
            raise _Transient(type(exc).__name__, None) from exc
        result = json.loads(payload) if payload else None
        if _graphql_rate_limited(result, headers):
            raise _Transient(
                "GraphQL rate limit", _limit_wait(headers) or _LIMIT_WAIT_S
            )
        return result


def _rate_limit_wait(status: int, headers, body: str) -> float | None:
    """Seconds to wait when a 403 / 429 is a rate limit, else None."""
    if status not in (403, 429):
        return None
    wait = _limit_wait(headers)
    if wait is not None:
        return wait
    if status == 429 or "rate limit" in body.lower():
        return _LIMIT_WAIT_S
    return None


def _limit_wait(headers) -> float | None:
    """The wait a rate-limited response's headers ask for, if they say."""
    retry_after = headers.get("Retry-After")
    if retry_after and retry_after.isdigit():
        return float(retry_after)
    if (
        headers.get("X-RateLimit-Remaining") == "0"
        and headers.get("X-RateLimit-Reset", "").isdigit()
    ):
        return float(headers["X-RateLimit-Reset"]) - time.time()
    return None


def _graphql_rate_limited(result: Any, headers) -> bool:
    """Whether a 200 is a GraphQL rate limit. GitHub's documented signals: a
    primary limit is an error with x-ratelimit-remaining at 0; a secondary
    one an error whose message says so (type RATE_LIMITED when given)."""
    errors = (result.get("errors") or []) if isinstance(result, dict) else []
    if not errors:
        return False
    return headers.get("X-RateLimit-Remaining") == "0" or any(
        error.get("type") == "RATE_LIMITED"
        or "rate limit" in str(error.get("message", "")).lower()
        for error in errors
    )
