# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Which MTP forwards of several requests take the mono path.

With MTP a verify step of G requests is G groups of ``S = k + 1`` rows, group
``g`` request ``g``'s tokens; under a captured graph the step is padded to the
capture size, its trailing groups' state slots -1. The group runner
(``group_runner``) runs the groups in one grouped launch per layer, each
group's rows as the single-request verify runs them. The draft steps that
follow are ``n·S`` rows at step 0 (the verify's rows of the ``n`` real
requests, never padded) and ``n`` rows at each later step, one per request
(``draft_runner``).

A forward of one request stays with ``dispatch.step_supported`` /
``draft_step_supported``; these take G (or n) of 2 .. ``max_groups``. Each
accepts exactly the widths its runner's ``prepare`` builds: a refused build
turns mono off for good (``LazyRunner``).
"""

import torch

from atom.utils.forward_context import get_forward_context


def _metadata(draft: bool):
    """The step's attention metadata, or None if the forward is not a decode
    step with its QSA, PLE (the target's) and verify metadata, or with
    ``draft`` not the drafter's."""
    fwd = get_forward_context()
    c = fwd.context
    if c is None or c.is_prefill or fwd.ubatch_slices is not None:
        return None
    if draft and not c.is_draft:
        return None
    md = fwd.attn_metadata
    if getattr(md, "qsa_metadata", None) is None:
        return None
    if not draft and getattr(md, "ple_metadata", None) is None:
        return None
    gdn = getattr(md, "gdn_metadata", None)
    if gdn is None or gdn.num_decodes != 0:
        return None
    return md


def group_step_supported(
    input_ids: torch.Tensor, inputs_embeds, verify_rows: int, max_groups: int
) -> bool:
    """Is this forward a verify step of G = 2 .. ``max_groups`` groups of
    ``verify_rows`` (k + 1) rows, at most G of them requests (the rest
    padding)?"""
    rows, s = input_ids.numel(), verify_rows
    if inputs_embeds is not None or s < 2 or rows % s:
        return False
    groups = rows // s
    if not 2 <= groups <= max_groups:
        return False
    md = _metadata(draft=False)
    if md is None:
        return False
    gdn = md.gdn_metadata
    return (
        md.max_seqlen_q == s
        and 1 <= gdn.num_spec_decodes <= groups
        and gdn.spec_state_indices_tensor is not None
        and gdn.num_accepted_tokens is not None
    )


def group_draft_step_supported(
    input_ids: torch.Tensor, inputs_embeds, verify_rows: int, max_groups: int
) -> bool:
    """Is this drafter forward a draft step after a verify of n = 2 ..
    ``max_groups`` requests: step 0 over their ``n·(k + 1)`` rows, or with
    k >= 2 a later step, one row each?"""
    rows, s = input_ids.numel(), verify_rows
    if inputs_embeds is not None or s < 2:
        return False
    md = _metadata(draft=True)
    if md is None:
        return False
    # the target's verify metadata: the draft steps keep its GDN part
    n = md.gdn_metadata.num_spec_decodes
    if not 2 <= n <= max_groups:
        return False
    if md.max_seqlen_q == s:
        return rows == n * s
    return s > 2 and md.max_seqlen_q == 1 and rows == n
