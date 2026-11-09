"""Shared test helpers for LMCache MP incremental token shipping."""

import copy
from types import SimpleNamespace

from atom.kv_transfer.offload.mp.token_shipping import WorkerTokens, token_list


def shipped_tokens(req):
    """The token ids (a list) a worker hands the transfer for `req`: a
    request's first shipment carries its whole prefix (`WorkerTokens`)."""
    meta = SimpleNamespace(requests=[copy.copy(req)])
    WorkerTokens().resolve(meta)
    return token_list(meta.requests[0].token_ids)
