# SPDX-License-Identifier: MIT
"""Temporary three-event observation of one Kimi PAGE checkpoint, per rank."""

import hashlib
import json
import os

PREFIX = "PR3501_CHECKPOINT "
IMAGE_BYTES = 56_171_520


def pack_bytes(chunks, size):
    """Concatenate byte views without numeric conversion or padding."""
    packed = bytearray(size)
    offset = 0
    for chunk in chunks:
        chunk = memoryview(chunk).cast("B")
        end = offset + len(chunk)
        if end > size:
            raise ValueError("checkpoint views exceed the image")
        packed[offset:end] = chunk
        offset = end
        del chunk
    if offset != size:
        raise ValueError("checkpoint views do not fill the image")
    return bytes(packed)


def image_bytes(views, size):
    import torch

    def chunks():
        for view in views:
            if not view.is_contiguous():
                raise ValueError("checkpoint observation needs contiguous views")
            yield view.detach().reshape(-1).view(torch.uint8).cpu().numpy()

    return pack_bytes(chunks(), size)


class KDACheckpointBoundary:
    def __init__(self, rank):
        self.rank = rank
        self.count = 0
        self.reference = None
        self.identity = None

    def before(self, builder, batch):
        ops = batch.state_maintenance_ops
        stores, restores = ops.checkpoint_stores, ops.checkpoint_restores
        if batch.is_dummy_run or not (stores or restores):
            return None
        runner, spec = (
            builder.model_runner,
            builder.model_runner.state_runtime.checkpoint_spec,
        )
        cfg = runner.config
        if not (
            runner.enforce_eager
            and cfg.compilation_config.level == 0
            and cfg.speculative_config is None
            and cfg.tensor_parallel_size == cfg.tp_world_size == 8
            and cfg.pipeline_parallel_size == 1
            and cfg.parallel_config.data_parallel_size == 1
            and runner.hf_text_config.model_type == "kimi_linear"
            and 0 <= self.rank < 8
            and spec is not None
            and spec.image_bytes == IMAGE_BYTES
            and spec.layout_id.startswith("kda-paged-state-v1:")
            and spec.layout_id.endswith(":tp=8:spec=0:carry=all")
        ):
            raise RuntimeError("checkpoint observer outside the fixed eager Kimi scope")
        if (
            batch.total_seqs_num != 1
            or len(stores) + len(restores) != 1
            or self.count >= 3
        ):
            raise RuntimeError(
                "checkpoint observer requires one store then two restores"
            )
        storing = bool(stores)
        if storing != (self.count == 0):
            raise RuntimeError("checkpoint observation order changed")
        op = (stores or restores)[0]
        identity = (tuple(op.unit_ids), op.layout_id, op.total_bytes)
        record = {
            "rank": self.rank,
            "event": ("store", "restore_1", "restore_2")[self.count],
            "request_id": str(batch.req_ids[0]),
            "prefill": bool(batch.has_prefill),
            "cached": int(batch.num_cached_tokens[0]),
            "scheduled": int(batch.num_scheduled_tokens[0]),
            "unit_ids": list(op.unit_ids),
            "layout_id": op.layout_id,
            "image_bytes": spec.image_bytes,
            "op_bytes": op.total_bytes,
            "slot": (
                (op.scratch_slot if op.scratch_slot >= 0 else op.src_slot)
                if storing
                else op.dst_slot
            ),
        }
        incoming = (
            None
            if storing
            else image_bytes(builder.page_unit_views(op.unit_ids), spec.image_bytes)
        )
        return builder, record, identity, incoming

    def after(self, ticket, metadata, positions):
        if ticket is None:
            return
        builder, record, identity, incoming = ticket
        size = record["image_bytes"]
        if self.count == 0:
            self.reference = image_bytes(
                builder.page_unit_views(record["unit_ids"]), size
            )
            self.identity = identity
            record["page_sha256"] = hashlib.sha256(self.reference).hexdigest()
        else:
            md = getattr(metadata, "kda_metadata", None)
            if md is None:
                md = metadata.gdn_metadata
            read = md.non_spec_state_indices_in_tensor
            if read is None:
                read = md.non_spec_state_indices_tensor
            read_slots = read.tolist()
            write_slots = md.non_spec_state_indices_tensor.tolist()
            initial = md.has_initial_state.tolist()
            cu = md.non_spec_query_start_loc.tolist()
            if len(read_slots) != 1 or len(write_slots) != 1 or len(initial) != 1:
                raise RuntimeError(
                    "checkpoint observer requires one actual KDA read row"
                )
            slot = int(read_slots[0])
            restored = image_bytes(builder.state_entry_views(slot), size)
            controls = {
                "read_slot": slot,
                "write_slot": int(write_slots[0]),
                "has_initial_state": bool(initial[0]),
                "query_start_loc": cu,
                "replayssm_cursor": int(builder.model_runner.replayssm_write_pos[slot]),
                "position_span": [
                    int(positions[0]),
                    int(positions[-1]),
                    positions.numel(),
                ],
                "num_prefills": md.num_prefills,
                "num_actual_tokens": md.num_actual_tokens,
            }
            expected = dict(
                controls,
                read_slot=record["slot"],
                write_slot=record["slot"],
                has_initial_state=True,
                query_start_loc=[0, 512],
                replayssm_cursor=-1,
                position_span=[512, 1023, 512],
                num_prefills=1,
                num_actual_tokens=512,
            )
            record.update(
                identity_matches=identity == self.identity,
                page_sha256=hashlib.sha256(incoming).hexdigest(),
                restored_sha256=hashlib.sha256(restored).hexdigest(),
                page_unchanged=incoming == self.reference,
                restored_matches_incoming=restored == incoming,
                restored_matches_reference=restored == self.reference,
                controls=controls,
                controls_expected=controls == expected,
            )
        line = (
            PREFIX + json.dumps(record, separators=(",", ":"), allow_nan=False) + "\n"
        ).encode()
        if len(line) > 2048:
            raise RuntimeError("checkpoint observation exceeds its per-record bound")
        # One pipe write keeps rank records intact; 24 * 2048 is below 64 KiB.
        if os.write(1, line) != len(line):
            raise RuntimeError("incomplete checkpoint observation write")
        self.count += 1
