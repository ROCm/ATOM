#!/usr/bin/env python3
"""Resolve a stable published OOT image reference for benchmarking.

Given a floating tag like ``vllm-latest``, this script looks up its manifest
digest and then searches the same repository for the newest
``vllm-v*-nightly_*`` tag with that exact digest. If no nightly tag matches, or
if the reverse lookup cannot complete after the floating tag digest is known, it
returns a digest-pinned ``latest`` reference instead.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from typing import Iterable
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlencode, urljoin
from urllib.request import Request, urlopen

AUTH_URL = "https://auth.docker.io/token"
REGISTRY_URL = "https://registry-1.docker.io"
MANIFEST_ACCEPT = ", ".join(
    (
        "application/vnd.oci.image.index.v1+json",
        "application/vnd.docker.distribution.manifest.list.v2+json",
        "application/vnd.oci.image.manifest.v1+json",
        "application/vnd.docker.distribution.manifest.v2+json",
    )
)
NIGHTLY_RE = re.compile(r"^vllm-v(?P<version>.+)-nightly_(?P<date>\d{8})$")
USER_AGENT = "ATOM OOT Image Resolver/1.0"


@dataclass(frozen=True)
class CandidateTag:
    tag: str
    version: str
    date: str


def build_resolution(
    repository: str,
    reference_tag: str,
    reference_digest: str,
    preferred_version: str | None,
    *,
    match_type: str,
    matched_tag: str = "",
    candidate_count: int = 0,
    resolution_error: str = "",
) -> dict[str, object]:
    reference_image = f"{repository}:{reference_tag}"
    result: dict[str, object] = {
        "repository": repository,
        "reference_tag": reference_tag,
        "reference_image": reference_image,
        "reference_digest": reference_digest,
        "preferred_version": preferred_version or "",
        "match_type": match_type,
        "matched_tag": matched_tag,
        "candidate_count": candidate_count,
    }
    if matched_tag:
        result["resolved_image"] = f"{repository}:{matched_tag}"
    else:
        result["resolved_image"] = f"{reference_image}@{reference_digest}"
    if resolution_error:
        result["resolution_error"] = resolution_error
    return result


def http_request(
    url: str, *, headers: dict[str, str] | None = None
) -> tuple[bytes, object]:
    request = Request(url, headers=headers or {})
    try:
        with urlopen(request, timeout=30) as response:
            return response.read(), response.headers
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace").strip()
        message = f"HTTP {exc.code} for {url}"
        if detail:
            message = f"{message}: {detail}"
        raise RuntimeError(message) from exc
    except URLError as exc:
        raise RuntimeError(f"Failed to reach {url}: {exc}") from exc


def get_registry_token(repository: str) -> str:
    query = urlencode(
        {
            "service": "registry.docker.io",
            "scope": f"repository:{repository}:pull",
        }
    )
    body, _ = http_request(
        f"{AUTH_URL}?{query}",
        headers={"User-Agent": USER_AGENT},
    )
    payload = json.loads(body.decode("utf-8"))
    token = payload.get("token")
    if not token:
        raise RuntimeError(
            f"Registry token response did not include a token for {repository}"
        )
    return str(token)


def registry_headers(token: str, *, accept: str | None = None) -> dict[str, str]:
    headers = {
        "Authorization": f"Bearer {token}",
        "User-Agent": USER_AGENT,
    }
    if accept:
        headers["Accept"] = accept
    return headers


def get_manifest_digest(repository: str, reference: str, token: str) -> str:
    manifest_url = (
        f"{REGISTRY_URL}/v2/{repository}/manifests/{quote(reference, safe='')}"
    )
    _, headers = http_request(
        manifest_url,
        headers=registry_headers(token, accept=MANIFEST_ACCEPT),
    )
    digest = headers.get("Docker-Content-Digest") or headers.get(
        "docker-content-digest"
    )
    if not digest:
        raise RuntimeError(
            f"Registry did not return Docker-Content-Digest for {repository}:{reference}"
        )
    return str(digest).strip()


def next_page_url(base_url: str, link_header: str | None) -> str | None:
    if not link_header:
        return None

    for part in link_header.split(","):
        match = re.match(r'\s*<([^>]+)>\s*;\s*rel="next"\s*', part)
        if match:
            return urljoin(base_url, match.group(1))
    return None


def list_tags(repository: str, token: str) -> list[str]:
    tags: list[str] = []
    url = f"{REGISTRY_URL}/v2/{repository}/tags/list?n=1000"

    while url:
        body, headers = http_request(url, headers=registry_headers(token))
        payload = json.loads(body.decode("utf-8"))
        tags.extend(payload.get("tags") or [])
        url = next_page_url(url, headers.get("Link"))

    return tags


def nightly_candidates(
    tags: Iterable[str], preferred_version: str | None
) -> list[CandidateTag]:
    candidates: list[CandidateTag] = []
    for tag in tags:
        match = NIGHTLY_RE.match(tag)
        if not match:
            continue
        candidates.append(
            CandidateTag(
                tag=tag,
                version=match.group("version"),
                date=match.group("date"),
            )
        )

    def sort_key(candidate: CandidateTag) -> tuple[int, str, str]:
        preferred = (
            1 if preferred_version and candidate.version == preferred_version else 0
        )
        return (preferred, candidate.date, candidate.tag)

    return sorted(candidates, key=sort_key, reverse=True)


def resolve_image(
    repository: str, reference_tag: str, preferred_version: str | None
) -> dict[str, object]:
    token = get_registry_token(repository)
    reference_digest = get_manifest_digest(repository, reference_tag, token)
    candidates: list[CandidateTag] = []
    try:
        candidates = nightly_candidates(list_tags(repository, token), preferred_version)
        for candidate in candidates:
            if (
                get_manifest_digest(repository, candidate.tag, token)
                == reference_digest
            ):
                return build_resolution(
                    repository,
                    reference_tag,
                    reference_digest,
                    preferred_version,
                    match_type="matched-nightly-tag",
                    matched_tag=candidate.tag,
                    candidate_count=len(candidates),
                )
    except RuntimeError as exc:
        return build_resolution(
            repository,
            reference_tag,
            reference_digest,
            preferred_version,
            match_type="reverse-lookup-failed-digest-pinned-latest",
            candidate_count=len(candidates),
            resolution_error=str(exc),
        )

    return build_resolution(
        repository,
        reference_tag,
        reference_digest,
        preferred_version,
        match_type="digest-pinned-latest",
        candidate_count=len(candidates),
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Resolve a floating OOT image tag to a same-digest nightly tag when possible."
    )
    parser.add_argument(
        "--repository",
        required=True,
        help="Docker repository, for example rocm/atom-dev",
    )
    parser.add_argument(
        "--reference-tag",
        required=True,
        help="Floating tag to resolve, for example vllm-latest",
    )
    parser.add_argument(
        "--preferred-version",
        default="",
        help="Optional preferred vLLM version to prioritize when scanning nightly tags",
    )
    args = parser.parse_args()

    resolution = resolve_image(
        repository=args.repository,
        reference_tag=args.reference_tag,
        preferred_version=args.preferred_version or None,
    )
    json.dump(resolution, sys.stdout, sort_keys=True)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
