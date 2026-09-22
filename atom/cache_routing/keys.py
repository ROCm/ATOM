# SPDX-License-Identifier: MIT
"""The sha256-prefix-u32le-v1 content identity, independent of native keys."""

from __future__ import annotations

import hashlib
import json
import struct
from collections.abc import Sequence

HASH_PROTOCOL = "sha256-prefix-u32le-v1"
NAMESPACE_FIELDS = (
    "model_revision",
    "tokenizer_revision",
    "template_revision",
    "kv_semantics",
    "adapter_revision",
    "cache_salt",
    "multimodal_identity",
)


def namespace_digest(manifest: dict) -> str:
    """Hash a frozen string/null schema whose JSON is also RFC 8785 canonical.

    All identity fields are explicit: model aliases and missing revisions must
    not silently share a namespace. Non-text request identities are reserved
    but unsupported by this first provider.
    """
    if set(manifest) != set(NAMESPACE_FIELDS):
        raise ValueError("cache namespace requires all v1 identity fields")
    for field in NAMESPACE_FIELDS:
        value = manifest[field]
        if field in NAMESPACE_FIELDS[:4]:
            if not isinstance(value, str) or not value:
                raise ValueError(f"cache namespace {field} must be nonempty")
        elif value is not None:
            raise ValueError(f"cache routing does not yet support {field}")
    canonical = json.dumps(
        manifest, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def root_key(namespace: str, block_size: int) -> bytes:
    digest = bytes.fromhex(namespace)
    if len(digest) != 32 or not 0 < block_size <= 0xFFFFFFFF:
        raise ValueError("invalid namespace or canonical block size")
    return hashlib.sha256(
        b"atom-kv-content-v1\0" + digest + struct.pack("<I", block_size)
    ).digest()


def extend_keys(parent: bytes, tokens: Sequence[int], block_size: int) -> list[str]:
    """Extend a prefix chain, ignoring the final incomplete canonical block."""
    if len(parent) != 32 or block_size <= 0:
        raise ValueError("invalid parent key or block size")
    result = []
    for start in range(0, len(tokens) - block_size + 1, block_size):
        block = tokens[start : start + block_size]
        if any(
            isinstance(t, bool) or not isinstance(t, int) or not 0 <= t <= 0xFFFFFFFF
            for t in block
        ):
            raise ValueError("token IDs must be uint32 integers")
        parent = hashlib.sha256(
            b"block\0" + parent + struct.pack(f"<{block_size}I", *block)
        ).digest()
        result.append(parent.hex())
    return result


def content_keys(namespace: str, tokens: Sequence[int], block_size: int) -> list[str]:
    return extend_keys(root_key(namespace, block_size), tokens, block_size)
