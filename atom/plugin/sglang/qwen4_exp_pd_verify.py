# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

"""Opt-in single-node byte verification of every Native Flash PD buffer.

ATOM_FLASH_PD_VERIFY_DIR must name the same local directory on P and D.
This intentionally synchronizes and copies data to CPU; use for correctness
diagnostics, never performance measurement. No wire metadata is changed.
"""

from __future__ import annotations

import functools
import hashlib
import json
import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)


def hash_rows(tensor, indices):
    import torch

    selected = tensor.index_select(0, indices.to(tensor.device)).contiguous()
    raw = selected.view(torch.uint8).cpu().numpy().tobytes()
    return {"sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)}


def register_verify_views(pool, layers, is_draft):
    """Keep tensors for all raw pointer descriptors; don't dereference on CPU."""
    page_size = pool.page_size
    views = {}
    for layer in layers:
        layer_id = 0 if is_draft else int(layer.layer_num)
        for kind, tensor in zip(("k", "v"), pool.get_kv_buffer(layer_id)):
            assert tensor.is_contiguous()
            paged = tensor.view(-1, page_size, *tensor.shape[1:])
            views[tensor.data_ptr()] = (f"kv.{layer_id}.{kind}", paged)
    tensors, ids = pool._atom_pd_qsa
    half = len(tensors) // 2
    for i, (tensor, layer_id) in enumerate(zip(tensors, ids)):
        kind = "raw" if i < half else "compressed"
        views[tensor.data_ptr()] = (f"qsa.{layer_id}.{kind}", tensor)
    pool._atom_pd_verify_views = views


def verify_request(scheduler, req, role):
    import torch
    from sglang.srt.disaggregation.base.conn import StateType
    from sglang.srt.disaggregation.checksum import (
        is_health_check_req,
        page_indices_for_request,
    )
    from sglang.srt.disaggregation.utils import _is_fake_transfer
    from sglang.srt.runtime_context import get_parallel

    if is_health_check_req(req) or _is_fake_transfer(req):
        return
    queue = (
        scheduler.disagg_prefill_bootstrap_queue
        if role == "prefill"
        else scheduler.disagg_decode_prealloc_queue
    )
    pool = queue.token_to_kv_pool
    if not hasattr(pool, "_atom_pd_verify_views"):
        return
    args = queue.kv_manager.kv_args
    views = dict(pool._atom_pd_verify_views)
    draft = queue.draft_token_to_kv_pool
    if draft is not None:
        views.update(
            {
                ptr: ("draft." + label, tensor)
                for ptr, (label, tensor) in draft._atom_pd_verify_views.items()
            }
        )
    mamba = scheduler.req_to_token_pool.mamba_pool
    for field, tensor, _, layer_id in mamba._iter_transfer_state_entries():
        views[tensor.data_ptr()] = (f"mamba.{field}.{layer_id}", tensor)

    pages = page_indices_for_request(scheduler, req, len(req.origin_input_ids))
    req_pool = scheduler.req_to_token_pool
    slots = (
        req_pool.translate_mamba_indices(
            req_pool.req_index_to_mamba_index_mapping[req.kv.req_pool_idx]
        )
        .to(dtype=torch.int64)
        .reshape(-1)
        .contiguous()
    )
    records = {}

    def capture(prefix, pointers, item_lengths, indices):
        for i, (ptr, item_bytes) in enumerate(zip(pointers, item_lengths)):
            if ptr not in views:
                raise RuntimeError(f"Unverified Flash PD pointer: {prefix}[{i}]")
            label, tensor = views[ptr]
            if tensor[0].nbytes != item_bytes:
                raise RuntimeError(f"Flash PD byte stride differs for {label}")
            records[f"{prefix}.{i}:{label}"] = hash_rows(tensor, indices)

    # Wait for all source computation / destination operations before reading.
    # The D hook runs after transfer completion and before process_prebuilt.
    torch.cuda.synchronize()
    capture("kv", args.kv_data_ptrs, args.kv_item_lens, pages)
    for i, kind in enumerate(args.state_types):
        if kind == StateType.MAMBA:
            indices = slots
        elif kind == StateType.QSA_COMPRESSED:
            indices = pages
        else:
            raise RuntimeError(f"Flash PD verifier lacks state indexing for {kind}")
        capture(kind.value, args.state_data_ptrs[i], args.state_item_lens[i], indices)

    rank = get_parallel().tp_rank
    root = Path(os.environ["ATOM_FLASH_PD_VERIFY_DIR"])
    root.mkdir(parents=True, exist_ok=True)
    stem = f"{int(req.bootstrap_room)}.tp{rank}"
    result = {
        "bootstrap_room": int(req.bootstrap_room),
        "rank": rank,
        "role": role,
        "prompt_tokens": len(req.origin_input_ids),
        "pages": pages.cpu().tolist(),
        "slots": slots.cpu().tolist(),
        "records": records,
    }
    if role == "decode":
        expected = json.loads((root / f"{stem}.prefill.json").read_text())
        result["mismatches"] = [
            key for key in records if records[key] != expected["records"].get(key)
        ]
        result["matched"] = records == expected["records"]
    destination = root / f"{stem}.{role}.json"
    temporary = destination.with_suffix(".tmp")
    temporary.write_text(json.dumps(result, indent=2) + "\n")
    temporary.replace(destination)
    if role == "decode":
        if not result["matched"]:
            raise RuntimeError(
                f"Flash PD state mismatch: {destination}: {result['mismatches']}"
            )
        logger.info(
            "ATOM Flash PD byte verification passed: room=%s rank=%s tensors=%s bytes=%s",
            req.bootstrap_room,
            rank,
            len(records),
            sum(x["bytes"] for x in records.values()),
        )


def install_verify_hooks():
    if not os.environ.get("ATOM_FLASH_PD_VERIFY_DIR"):
        return
    from sglang.srt.disaggregation.decode import SchedulerDisaggregationDecodeMixin
    from sglang.srt.disaggregation.prefill import SchedulerDisaggregationPrefillMixin

    original_send = SchedulerDisaggregationPrefillMixin.send_kv_chunk
    if getattr(original_send, "_atom_flash_verify", False):
        return
    original_verify = getattr(
        SchedulerDisaggregationDecodeMixin, "_verify_kv_checksums", None
    )
    if original_verify is None:
        raise RuntimeError(
            "ATOM_FLASH_PD_VERIFY_DIR requires SGLang's _verify_kv_checksums "
            "hook, which is unavailable in this runtime. Use the bundled "
            "SGLang v0.5.22 release image or unset ATOM_FLASH_PD_VERIFY_DIR."
        )

    @functools.wraps(original_send)
    def send(self, req, last_chunk=False, end_idx=None):
        if last_chunk:
            verify_request(self, req, "prefill")
        return original_send(self, req, last_chunk=last_chunk, end_idx=end_idx)

    @functools.wraps(original_verify)
    def verify(self, batch):
        original_verify(self, batch)
        for req in batch.reqs:
            verify_request(self, req, "decode")

    send._atom_flash_verify = True
    SchedulerDisaggregationPrefillMixin.send_kv_chunk = send
    SchedulerDisaggregationDecodeMixin._verify_kv_checksums = verify
