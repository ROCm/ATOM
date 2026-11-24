# SPDX-License-Identifier: MIT
"""Static expert placement for expert parallelism (off by default).

``ATOM_EXPERT_PLACEMENT=<file.json>`` renumbers the routed experts of the MoE
layers named in the file: ``{"layers.5.ffn.experts": [new_id for old_id in
range(n_experts)], ...}`` (the key is the FusedMoE prefix, the value a
permutation of the checkpoint expert ids). Under EP a rank owns the contiguous
id range ``[rank * n / ep_size, (rank + 1) * n / ep_size)``, so the permutation
decides which rank serves which checkpoint expert.

The model is unchanged: expert ``e``'s weights load into slot ``perm[e]`` and
the gate's row ``e`` (weight and routing biases) moves to row ``perm[e]``, so
the router emits the new ids directly. Nothing runs on the request path.
"""

from __future__ import annotations

import json
import logging
import os
import threading

import torch

logger = logging.getLogger("atom")

_ENV = "ATOM_EXPERT_PLACEMENT"
_TABLE: dict[str, list[int]] | None = None
# The loader's worker threads all ask for the table. It is built in a local,
# validated, and only then published, under a lock: a caller that saw a
# half-built (empty) table would map its expert by identity, i.e. drop it or
# write it into another expert's slot.
_TABLE_LOCK = threading.Lock()
_GATE_ROWS = ("weight", "e_score_correction_bias", "bias_vl")


def _load_table() -> dict[str, list[int]]:
    path = os.environ.get(_ENV)
    table: dict[str, list[int]] = {}
    if path:
        with open(path) as f:
            table = json.load(f)
        for key, perm in table.items():
            if sorted(perm) != list(range(len(perm))):
                raise ValueError(f"{_ENV}: {key} is not a permutation")
        logger.info("%s: %d MoE layers from %s", _ENV, len(table), path)
    return table


def _table() -> dict[str, list[int]]:
    global _TABLE
    table = _TABLE
    if table is None:
        with _TABLE_LOCK:
            table = _TABLE
            if table is None:
                table = _load_table()
                _TABLE = table
    return table


def preload_placement() -> None:
    """Read the table on the calling thread.

    `load_model` calls this before the loader's worker threads exist, so no
    worker is ever the table's first caller; the lock above is what makes a
    caller that does not come through `load_model` safe as well.
    """
    _table()


def placement_for(prefix: str, num_experts: int | None = None) -> list[int] | None:
    """The permutation of ``prefix``'s routed experts, or None (identity)."""
    perm = _table().get(prefix)
    if perm is None:
        return None
    if num_experts is not None and len(perm) != num_experts:
        raise ValueError(
            f"{_ENV}: {prefix} lists {len(perm)} experts, the layer has {num_experts}"
        )
    return perm


def apply_gate_placement(model: torch.nn.Module) -> None:
    """Move every placed layer's gate rows to their new expert ids. Call once,
    after the checkpoint is loaded and before weight post-processing."""
    if not _table():
        return
    done = 0
    for module in model.modules():
        gate = getattr(module, "gate", None)
        experts = getattr(module, "experts", None)
        prefix = getattr(experts, "prefix", None)
        if gate is None or prefix is None or prefix not in _table():
            continue
        if getattr(module, "_expert_placement_applied", False):
            continue
        if hasattr(gate, "tid2eid"):
            raise NotImplementedError(f"{_ENV}: {prefix} is hash-routed")
        num_experts = gate.weight.shape[0]
        perm = torch.tensor(placement_for(prefix, num_experts), dtype=torch.long)
        inverse = torch.empty_like(perm)
        inverse[perm] = torch.arange(num_experts)
        for name in _GATE_ROWS:
            param = getattr(gate, name, None)
            if param is None:
                continue
            assert param.shape[0] == num_experts, (prefix, name, tuple(param.shape))
            param.data.copy_(param.data[inverse.to(param.device)].clone())
        module._expert_placement_applied = True
        done += 1
    logger.info("%s: gate rows renumbered in %d MoE layers", _ENV, done)
