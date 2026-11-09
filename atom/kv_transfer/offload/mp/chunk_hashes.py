# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Prompts named by their LMCache chunk hashes instead of their tokens.

With `lmcache.mp.client_chunk_hashes`, LMCache's ATOM adapters send a
prompt's blake3 chunk hashes to the server instead of its tokens, and an
adapter that accepts caller hashes (`ACCEPTS_CALLER_CHUNK_HASHES`) takes them
from ATOM instead of hashing the tokens itself. The scheduler then sends a
prompt's hashes of `[0, end)` with every lookup and lock release, and every
save and load carries the hashes of `[0, end)` its worker hands to the adapter
-- no token list is copied, pickled into the step metadata, or hashed again on
each TP rank.

The chain of a request is hashed once: by the frontend's tokenizer cache, which
keeps it per conversation and hashes only a turn's new chunks
(`prefix_block_hashes.extend_lmcache_chunk_hashes`), or here, once per request,
when nothing was shipped (token-ID input, multimodal prompts).

On only when `lmcache.mp.client_chunk_hashes` is set and the installed adapter
accepts caller hashes (`chunk_hashes_enabled`); otherwise keys carry the token
prefix as before.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Any

from atom.model_engine.prefix_block_hashes import LMCACHE_CHUNK_HASH_BYTES
from atom.model_engine.prefix_hash import prompt_fingerprint
from atom.utils import envs

logger = logging.getLogger("atom")

CHUNK_HASH_BYTES = LMCACHE_CHUNK_HASH_BYTES

# kv_connector names that select the LMCache MP connector.
_MP_CONNECTORS = ("lmcache_mp", "LMCacheMPConnector")
# The deployment option (`deployment._adapter_options`) that has LMCache's
# adapters send chunk hashes instead of tokens.
_OPTION = "lmcache.mp.client_chunk_hashes"
# LMCache's default chunk size, which its MP server also starts with.
_DEFAULT_CHUNK_SIZE = 256

# Shipped chains rejected, warned about once.
_rejected = 0


@dataclass(frozen=True, slots=True)
class HashedPrompt:
    """A prompt prefix as an LMCache MP lookup names it: the hashes of the
    complete chunks of its first `num_tokens` tokens. `len()` is the prefix
    length, as for the token list it stands in for."""

    num_tokens: int
    hashes: bytes

    def __len__(self) -> int:
        return self.num_tokens


def frontend_chunk_size(config: Any) -> int:
    """The chunk size the frontend hashes prompts for, or 0 for no chain.

    Only for the LMCache MP connector with `lmcache.mp.client_chunk_hashes`.
    The size is the one LMCache is configured with on this side
    (`lmcache.chunk_size`, else `LMCACHE_CHUNK_SIZE`, else LMCache's default);
    the server's own is known only to the engine, which rehashes a chain of
    another size.
    """
    kvc = getattr(config, "kv_transfer_config", None) or {}
    if not isinstance(kvc, dict) or kvc.get("kv_connector") not in _MP_CONNECTORS:
        return 0
    extra = kvc.get("kv_connector_extra_config") or {}
    if not isinstance(extra, dict) or extra.get(_OPTION) is not True:
        return 0
    if not envs.ATOM_FRONTEND_PREFIX_HASH:
        return 0
    try:
        import blake3  # noqa: F401
    except ImportError:
        return 0
    configured = extra.get("lmcache.chunk_size")
    if configured is None:
        configured = os.environ.get("LMCACHE_CHUNK_SIZE", _DEFAULT_CHUNK_SIZE)
    try:
        return max(0, int(configured))
    except (TypeError, ValueError):
        return 0


def chunk_hashes_enabled(extra: dict[str, Any], adapter: Any) -> bool:
    """Whether ATOM names prompts by their chunk hashes: the deployment opted
    into `lmcache.mp.client_chunk_hashes` (passed to the adapters by
    `deployment._adapter_options`) and the scheduler adapter accepts caller
    hashes (`ACCEPTS_CALLER_CHUNK_HASHES`).

    Warns when the option is set but the adapter cannot take ATOM's hashes:
    the transfers then carry the whole token prefix, which defeats what the
    option was set for."""
    if extra.get(_OPTION) is not True:
        return False
    if getattr(adapter, "ACCEPTS_CALLER_CHUNK_HASHES", False):
        return True
    logger.warning(
        "%s is set but the installed LMCache ATOM adapter does not take "
        "caller chunk hashes; LMCache MP lookups, saves and loads carry the "
        "token prefix",
        _OPTION,
    )
    return False


def prompt_chunk_hashes(seq: Any, chunk_size: int) -> bytes:
    """LMCache chunk hashes of every complete chunk of `seq`'s prompt.

    Kept on the sequence (`_lmcache_chunk_hashes`); lookups, saves and loads
    all take slices of it. Starts from the frontend's chain when it checks out
    (`_adopt_shipped_chain`), and otherwise hashes the prompt once here.
    """
    n = int(seq.num_prompt_tokens) // chunk_size
    want = n * CHUNK_HASH_BYTES
    cached = getattr(seq, "_lmcache_chunk_hashes", None)
    have = cached[1] if cached is not None and cached[0] == chunk_size else b""
    if len(have) >= want:
        return have if len(have) == want else have[:want]
    hashes = None if have else _adopt_shipped_chain(seq, chunk_size, n)
    if hashes is None:
        from lmcache.integration.atom import lmcache_chunk_hashes

        k = len(have) // CHUNK_HASH_BYTES
        hashes = have + lmcache_chunk_hashes(
            seq.token_ids[k * chunk_size : n * chunk_size],
            chunk_size,
            have[-CHUNK_HASH_BYTES:] if k else None,
        )
    seq._lmcache_chunk_hashes = (chunk_size, hashes)
    return hashes


def _adopt_shipped_chain(seq: Any, chunk_size: int, n: int) -> bytes | None:
    """The frontend's chain for `seq` (consumed), or None to hash here.

    Adopted only if it was minted for this chunk size from these prompt tokens
    (`prompt_fingerprint`), and its first, middle and last links rehash equal
    with LMCache's own hasher -- the one the server resolves keys with.
    """
    shipped = getattr(seq, "lmcache_chunk_chain", None)
    if shipped is None:
        return None
    seq.lmcache_chunk_chain = None
    hashes = shipped.hashes
    if (shipped.chunk_size, len(hashes)) != (chunk_size, n * CHUNK_HASH_BYTES):
        _reject(
            f"minted for chunk size {shipped.chunk_size} with "
            f"{len(hashes) // CHUNK_HASH_BYTES} chunks; this server wants "
            f"{chunk_size}, {n}"
        )
        return None
    if (shipped.num_prompt_tokens, shipped.fingerprint) != (
        seq.num_prompt_tokens,
        prompt_fingerprint(seq),
    ):
        _reject("it was minted from other tokens")
        return None
    from lmcache.integration.atom import lmcache_chunk_hashes

    def link_holds(i: int) -> bool:
        at = i * CHUNK_HASH_BYTES
        prev = hashes[at - CHUNK_HASH_BYTES : at] if i else None
        tokens = seq.token_ids[i * chunk_size : (i + 1) * chunk_size]
        return lmcache_chunk_hashes(tokens, chunk_size, prev) == (
            hashes[at : at + CHUNK_HASH_BYTES]
        )

    if n and not all(map(link_holds, {0, n // 2, n - 1})):
        _reject("its hashes differ from LMCache's")
        return None
    return bytes(hashes)


def _reject(why: str) -> None:
    global _rejected
    # Warn once: a frontend/engine mismatch repeats on every request.
    if not _rejected:
        logger.warning(
            "LMCache MP: rehashing a prompt chunk chain the frontend shipped, "
            "%s. Keys are unaffected; the scheduler pays the hashing.",
            why,
        )
    _rejected += 1


def prompt_key(prompt: Any, end: int, adapter: Any) -> tuple[Any, dict[str, bytes]]:
    """`(token_ids, extra keyword arguments)` naming `[0, end)` of `prompt` to
    the scheduler `adapter`: a `HashedPrompt` goes as its hashes of the
    complete chunks there, which the adapter takes in place of the tokens."""
    if isinstance(prompt, HashedPrompt):
        chunk_size = int(adapter.lmcache_tokens_per_chunk)
        return (), {"chunk_hashes": _hashes_to(prompt.hashes, end, chunk_size)}
    return prompt, {}


def prefix_fields(seq: Any, end: int, chunk_size: int, hashed: bool) -> dict:
    """`LMCacheReqMeta` fields naming the first `end` tokens of `seq`'s prompt:
    their chunk hashes and `end` when `hashed`, else the tokens themselves."""
    if not hashed:
        return {"token_ids": list(seq.token_ids[:end])}
    return {
        "token_ids": [],
        "token_end": end,
        "chunk_hashes": _hashes_to(
            prompt_chunk_hashes(seq, chunk_size), end, chunk_size
        ),
    }


def request_prefix_tokens(req: Any) -> int:
    """How many prompt tokens `req` names: `len(req.token_ids)`, or with chunk
    hashes its `token_end`."""
    if req.chunk_hashes is None:
        return len(req.token_ids)
    return int(req.token_end)


def transfer_key_fields(req: Any, end: int, chunk_size: int) -> dict:
    """`AtomMPTransferSpec` fields naming a transfer ending at `end` of
    `req`'s prefix: its tokens, or the hashes of the complete chunks of
    `[0, end)` (the adapter keys `[start, end)` from them)."""
    if req.chunk_hashes is None:
        return {"token_ids": list(req.token_ids)}
    return {
        "token_ids": [],
        "chunk_hashes": _hashes_to(req.chunk_hashes, end, chunk_size),
    }


def _hashes_to(hashes: bytes, end: int, chunk_size: int) -> bytes:
    """The hashes of the complete chunks of `[0, end)`, from a chain that
    covers at least that much."""
    want = end // chunk_size * CHUNK_HASH_BYTES
    if len(hashes) < want:
        raise ValueError(
            f"{len(hashes) // CHUNK_HASH_BYTES} LMCache chunk hashes cannot "
            f"name [0, {end}) at chunk size {chunk_size}"
        )
    return hashes[:want]


__all__ = [
    "CHUNK_HASH_BYTES",
    "HashedPrompt",
    "chunk_hashes_enabled",
    "frontend_chunk_size",
    "prefix_fields",
    "prompt_chunk_hashes",
    "prompt_key",
    "request_prefix_tokens",
    "transfer_key_fields",
]
