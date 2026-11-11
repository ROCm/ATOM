# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Lean host prep on the steps the V4.1 mono hi hook serves (``ATOM_DSV41_MONO_HI``).

Once the hook runs every layer of a step on the hi path (``hi_runner``), part of what V4.1's metadata builder lays
out for that step has no reader left: ATOM's compressor (the compress / write plans and their key RoPE
positions), its indexer (the score planner's packed offsets and bands, mono's walk plans) and its attention (the
per-ratio indptrs) run only in the layers the hook replaced. The hi runtime's layers derive their own bounds and plans
from the step's positions, batch ids, slots and block tables, and the DSpark draft reads the step's positions,
query starts, slots and window. On such a step this skips, on the host:

- ``_build_compress_plans`` (numpy over the requests, three buffers a ratio staged into the step's publication);
- the step planner (``ScorePlanner``: the host layout and its buffers' upload);

(the per-ratio indptrs are filled lazily, by the first attention index build that reads them: none on such a step),

and the runner (``hi_dispatch.HiDecodeModel._hi_forward``) skips the two device launches of the target step only
ATOM's K1 / K2 read: its mailboxes' ``step_begin`` and ``write_step_meta``. Everything else the builder does stays:
positions, query starts, batch ids, slots, block tables, the per-ratio visible counts, Engram's rows and staging,
the state probe and cursor, the tentative state the verdict commits.

A step is lean only when it is certainly served: a decode verify of one request at a width the hook served a
forward at (``HiLayers.served_widths``: recorded by the served forward itself, so a captured graph's width is in it
only when its capture took the hook's path, and a replay runs what the capture recorded), or of B requests at a batch
width (``HiLayers.split``: B requests verifying the query bucket's rows each, every one of them that many), no
image rows, no recomputed rows. The decision is on the host and TP-uniform (it reads the scheduler's batch, the same on every
rank), part of the step's one routing decision (``round.HiRound.service``: ``RoundService.LEAN``), which the builder
takes before each step's layout. Nothing changes when the hook is off, or on any other step.
"""

from __future__ import annotations

import logging

from atom.models.deepseek_v41.mono.hi_gate import log_once

logger = logging.getLogger("atom")


def lean_step(
    hook,
    batch,
    spans,
    running_bs,
    running_tokens,
    max_q_len,
    tentative,
    is_prefill,
    resumes,
) -> bool:
    """Whether this step's builder may skip what only ATOM's replaced layers read (module docstring). Host-side,
    TP-uniform: the scheduler's batch and the forward's shape."""
    if (
        hook is None
        or not hook.runs_every_layer
        or batch.is_dummy_run
        or is_prefill
        or not tentative
    ):
        return False
    if resumes is not None or max_q_len is None or len(spans) != running_bs:
        return False
    if running_bs == 1:
        if (
            running_tokens != max_q_len
            or max_q_len not in hook.served_widths
            or spans[0].length > max_q_len
        ):
            return False
    elif (getattr(hook, "split", {}).get(running_tokens) != (running_bs, max_q_len)
          or running_tokens not in hook.served_widths or any(s.length != max_q_len for s in spans)):  # fmt: skip
        return False  # (B requests: rectangular, each the bucket's rows, as the batch program lays them out)
    # an image row (``_token_mask``): the forward would carry an image mask and not take mono's path
    data = getattr(batch, "multimodal_data", None)
    return not (data and any(data.get(s.request_id) for s in spans))


def check_batch_layout(hook, spans, running_tokens, max_q_len) -> None:
    """A decode at one of the hook's batch widths (``HiLayers.split``: B requests of the bucket's rows) runs
    the hi runtime's ``Scenario(B, rows)``, which takes request b's rows at b x rows: every request but the last must verify
    the bucket's rows (a shorter last one ends in padding rows). Raise on any other layout (a ragged verify).
    """
    split = getattr(hook, "split", {}).get(running_tokens)
    if split is None or split[0] == 1 or split[1] != max_q_len:
        return
    if any(s.length != max_q_len for s in spans[:-1]) or (
        spans and spans[-1].length > max_q_len
    ):
        raise RuntimeError(f"V4.1 mono hi: a ragged decode ({[s.length for s in spans]} rows a request) at the batch "
                           f"width {running_tokens} (hi_compose.ROWS): the batch programs take {max_q_len} "
                           "rows a request")  # fmt: skip


def install(hook) -> bool:
    """The lean host prep for ``hook`` (``HiRound.service``'s ``LEAN`` steps); False when a layer runs on ATOM (every
    step lays out in full)."""
    if not hook.runs_every_layer:
        log_once(
            logging.WARNING,
            "V4.1 mono hi: no lean host prep: layers %s run on ATOM",
            sorted(i for i, p in hook.plan.items() if not p.hi),
        )
        return False
    hook.lean_host = True
    log_once(
        logging.INFO,
        "V4.1 mono hi: lean host prep on served steps: no compress plans, step planners, ATOM mailbox "
        "step_begin or K1 / K2 step metadata",
    )
    return True
