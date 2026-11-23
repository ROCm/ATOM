"""Default-off dump of MiniMax-M3 prefill top-k selections (offline analysis).

``ATOM_M3_DUMP_PREFILL_TOPK=<dir>`` saves, on rank 0, every sparse layer's
freshly selected prefill top-k of the first ``ATOM_M3_DUMP_PREFILL_TOPK_CHUNKS``
(default 4) prefill forwards whose longest cached prefix is at least
``ATOM_M3_DUMP_PREFILL_TOPK_MIN_PREFIX`` tokens (default 8192: skips warm-up and
short prompts) to ``<dir>/chunk<NNN>_layer<LL>.pt``:

- ``topk_idx`` ``[kv_heads, tokens, topk]`` int32 logical 128-blocks in
  selection order (-1 padded),
- ``sparse_bt`` / ``sparse_ctx``: the compacted page table and lengths the
  NVFP4 attention reads (None when the selector did not emit them),
- ``cu_seqlens_q``, ``prefix_lens``, ``seq_lens`` of the batch, ``layer``.

Copies to the host synchronize, so it is a debugging aid only. Prefill forwards
replayed from a token-bucket graph (or captured) are skipped: nothing can be
copied out while a graph is captured.
"""

import os

import torch

from atom.utils import envs

_state = {"chunk": -1, "last_layer": None, "skip": False}


def _rank() -> int:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_rank()
    return 0


def maybe_dump_prefill_topk(
    layer: int,
    topk_idx: torch.Tensor,
    sparse_bt: torch.Tensor | None,
    sparse_ctx: torch.Tensor | None,
    cu_seqlens_q: torch.Tensor,
    prefix_lens: torch.Tensor,
    seq_lens: torch.Tensor,
) -> None:
    """Save this layer's selection when ``ATOM_M3_DUMP_PREFILL_TOPK`` asks for it
    (a no-op otherwise)."""
    out_dir = envs.ATOM_M3_DUMP_PREFILL_TOPK
    if not out_dir or _rank() != 0:
        return
    if torch.cuda.is_available() and torch.cuda.is_current_stream_capturing():
        return
    last = _state["last_layer"]
    if last is None or layer <= last:
        # A layer at or below the last one seen starts a new prefill forward.
        if _state["chunk"] + 1 >= envs.ATOM_M3_DUMP_PREFILL_TOPK_CHUNKS:
            _state["last_layer"] = layer
            _state["skip"] = True
            return
        long_enough = (
            int(prefix_lens.max().item()) >= envs.ATOM_M3_DUMP_PREFILL_TOPK_MIN_PREFIX
            if prefix_lens.numel()
            else False
        )
        _state["skip"] = not long_enough
        if long_enough:
            _state["chunk"] += 1
    _state["last_layer"] = layer
    if _state["skip"]:
        return
    os.makedirs(out_dir, exist_ok=True)

    def host(t):
        return None if t is None else t.detach().to("cpu", copy=True)

    torch.save(
        {
            "layer": layer,
            "topk_idx": host(topk_idx),
            "sparse_bt": host(sparse_bt),
            "sparse_ctx": host(sparse_ctx),
            "cu_seqlens_q": host(cu_seqlens_q),
            "prefix_lens": host(prefix_lens),
            "seq_lens": host(seq_lens),
        },
        os.path.join(out_dir, f"chunk{_state['chunk']:03d}_layer{layer:02d}.pt"),
    )
