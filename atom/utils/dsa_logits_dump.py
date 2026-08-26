"""Env-gated mirror of the DSA indexer's decode logits plane.

Decode runs under a whole-forward CUDA graph, so the Python body of
``sparse_attn_indexer`` executes only while the graph is captured. ``record``
therefore captures a ``copy_`` into a buffer that was allocated *outside* the
graph, during the eager warmup forward that precedes each capture. The copy is
replayed with every decode step, so the buffer always holds the plane the top-k
kernel consumed on the step that just ran, and ``flush`` -- called from the
eager ``run_model`` tail -- reads it back.

Only the first ``ATOM_DSA_LOGITS_DUMP_ROWS`` rows are mirrored: a row is up to
``max_model_len`` floats wide, and the distribution of one row is what the
one-block radix kernel's cost is a function of.

    ATOM_DSA_LOGITS_DUMP        directory to write snapshots to (unset = off)
    ATOM_DSA_LOGITS_DUMP_ROWS   rows mirrored per plane        (default 8)
    ATOM_DSA_LOGITS_DUMP_EVERY  decode steps between snapshots (default 400)
    ATOM_DSA_LOGITS_DUMP_MAX    snapshots before it stops      (default 24)
    ATOM_DSA_LOGITS_DUMP_LAYERS comma-separated layer indices to mirror, each
                                getting its own buffer and its own file series
                                (default: whichever layer calls first)
"""

import logging
import os
import re

import numpy as np
import torch
import torch.distributed as dist

from atom.utils import envs

logger = logging.getLogger(__name__)

# context_lens is one entry per sequence, so max_num_seqs bounds it.
_CTX_CAP = 4096

_DIR = envs.ATOM_DSA_LOGITS_DUMP
_ROWS = envs.ATOM_DSA_LOGITS_DUMP_ROWS
_EVERY = envs.ATOM_DSA_LOGITS_DUMP_EVERY
_MAX = envs.ATOM_DSA_LOGITS_DUMP_MAX
_LAYERS = {int(x) for x in envs.ATOM_DSA_LOGITS_DUMP_LAYERS.split(",") if x.strip()}

_rank: int | None = None


def _dumping_rank() -> bool:
    """Only one rank mirrors. Under a replicated index cache every DCP rank
    computes the same plane, so the other three would write the same bytes to
    the same path at the same moment."""
    global _rank
    if _rank is None:
        if dist.is_available() and dist.is_initialized():
            _rank = dist.get_rank()
        else:
            _rank = int(os.environ.get("RANK", "0"))
    return _rank == 0


# key -> {"plane": mirror, "ctx": mirror, "width": int}
_state: dict[str, dict] = {}
# key -> the layer whose plane that key mirrors.
_owner: dict[str, str] = {}
_step = 0
_saved = 0
_late: set[str] = set()


def enabled() -> bool:
    return bool(_DIR)


def _layer_index(layer: str) -> int | None:
    m = re.search(r"layers\.(\d+)\.", layer)
    return int(m.group(1)) if m else None


def _key(tag: str, layer: str) -> str | None:
    """The buffer a layer mirrors into, or None if this layer is not mirrored.

    Every sparse layer calls record() with its own plane and a plane is up to
    `max_model_len` floats per row, so only the layers named by
    ATOM_DSA_LOGITS_DUMP_LAYERS get a buffer. With none named, the first layer
    to arrive claims one and the rest are skipped."""
    if not _LAYERS:
        return tag if _owner.setdefault(tag, layer) == layer else None
    idx = _layer_index(layer)
    if idx is None or idx not in _LAYERS:
        return None
    key = f"{tag}_L{idx:02d}"
    _owner[key] = layer
    return key


def record(tag: str, layer: str, plane: torch.Tensor, ctx_lens: torch.Tensor) -> None:
    """Mirror `plane`'s leading rows and `ctx_lens` for later readback."""
    if not _DIR or _saved >= _MAX or not _dumping_rank():
        return
    key = _key(tag, layer)
    if key is None:
        return
    st = _state.get(key)
    if st is None:
        if torch.cuda.is_current_stream_capturing():
            # No eager call came first, so there is nowhere outside the graph to
            # allocate. Report it rather than corrupting the capture.
            if key not in _late:
                _late.add(key)
                logger.warning(
                    "dsa_logits_dump: '%s' first seen during capture; not mirrored",
                    key,
                )
            return
        st = {
            "plane": torch.empty(
                (min(_ROWS, plane.shape[0]), plane.shape[1]),
                dtype=plane.dtype,
                device=plane.device,
            ),
            "ctx": torch.zeros(_CTX_CAP, dtype=torch.int32, device=plane.device),
            "width": int(plane.shape[1]),
        }
        _state[key] = st
    if int(plane.shape[1]) != st["width"]:
        return
    rows = min(st["plane"].shape[0], plane.shape[0])
    st["plane"][:rows].copy_(plane[:rows])
    n = min(ctx_lens.numel(), _CTX_CAP)
    st["ctx"][:n].copy_(ctx_lens[:n].to(torch.int32))


def flush(graph_bs: int, max_q_len: int) -> None:
    """Write a snapshot every `_EVERY` decode steps, up to `_MAX` of them."""
    global _step, _saved
    if not _DIR or not _state or _saved >= _MAX or not _dumping_rank():
        return
    _step += 1
    if _step % _EVERY:
        return
    os.makedirs(_DIR, exist_ok=True)
    for key, st in _state.items():
        ctx = st["ctx"].cpu().numpy()
        rows = int(st["plane"].shape[0])
        # The kernel reads [0, rowEnd) only; rowEnd comes from ctx_lens.
        live = ctx[: max(1, (graph_bs * max_q_len + max_q_len - 1) // max_q_len)]
        width = int(min(st["width"], max(1, int(live.max()) if live.size else 1)))
        path = os.path.join(_DIR, f"dsa_{key}_r{_rank}_s{_step:07d}.npz")
        np.savez(
            path,
            plane=st["plane"][:, :width].cpu().numpy(),
            ctx=ctx,
            rows=rows,
            width=width,
            stride0=st["width"],
            graph_bs=graph_bs,
            max_q_len=max_q_len,
            step=_step,
            layer=_owner.get(key, ""),
        )
        logger.info("dsa_logits_dump: wrote %s (%d x %d)", path, rows, width)
    _saved += 1
