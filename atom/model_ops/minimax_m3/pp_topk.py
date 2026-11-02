# SPDX-License-Identifier: MIT
"""Carry MiniMax top-k selections across PP boundaries, scoped to one forward.

Only logical block IDs cross the boundary. Physical page tables belong to the
receiving stage, and TP ranks own different index heads, so the IDs must also
bypass the hidden-state send/all-gather optimization.
"""

import logging

import torch

logger = logging.getLogger("atom")

PP_TOPK_INDICES = "minimax_m3_topk_indices"
PP_TOPK_METADATA = "minimax_m3_topk_metadata"


def transfer_signature(key: tuple) -> tuple:
    # Device ordinal, physical block-table width and strides are stage-local.
    # Keep the phase, query geometry/dtype, sequence count and selector policy.
    return key[:3] + key[6:]


class MiniMaxM3PPTopK:
    def __init__(self, start_layer, end_layer, num_layers, schedule):
        """schedule is (global layer ID, sparse ordinal, reuse) from the model."""
        self.start_layer = start_layer
        self.end_layer = end_layer

        def source_at(boundary):
            source = None
            for layer, ordinal, reuse in schedule:
                if layer >= boundary:
                    if not reuse:
                        return None
                    if source is None:
                        raise ValueError(
                            f"MiniMax-M3 reuse layer {layer} has no computing layer"
                        )
                    return source
                if not reuse:
                    source = (layer, ordinal)
            return None

        # Look through dense-only stages: they must relay a selection until
        # the next sparse layer, even if they never consume it themselves.
        self.recv_source = source_at(start_layer)
        self.send_source = source_at(end_layer) if end_layer < num_layers else None
        if self.recv_source is not None or self.send_source is not None:
            logger.info(
                "MiniMax-M3 PP top-k relay: layers [%d,%d), recv=%s, send=%s",
                start_layer,
                end_layer,
                self.recv_source,
                self.send_source,
            )

    def begin(self, sparse_metadata, intermediate_tensors):
        # PP is eager. Metadata can come from a reused ring slot; never import
        # a prior step's memo, including after a failed/empty previous forward.
        state = {}
        sparse_metadata._index_topk_cache_state = state
        tensors = {} if intermediate_tensors is None else intermediate_tensors.tensors
        indices = tensors.pop(PP_TOPK_INDICES, None)
        metadata = tensors.pop(PP_TOPK_METADATA, None)
        if self.recv_source is None:
            if indices is not None or metadata is not None:
                raise RuntimeError("Unexpected MiniMax-M3 PP top-k payload")
            return
        if indices is None or metadata is None:
            raise RuntimeError(
                f"MiniMax-M3 PP stage starting at layer {self.start_layer} "
                f"requires top-k from layer {self.recv_source[0]}"
            )
        if metadata["source"] != self.recv_source:
            raise RuntimeError("MiniMax-M3 PP top-k computing layer mismatch")
        if indices.dtype != torch.int32 or indices.ndim != 3:
            raise RuntimeError("Invalid MiniMax-M3 PP top-k tensor")
        state["topk"] = {
            "key": None,
            "pp_signature": metadata["signature"],
            "value": (indices, None, None),
            "layer_num": self.recv_source[0],
            "sparse_layer_ordinal": self.recv_source[1],
        }

    def attach(self, sparse_metadata, intermediate_tensors):
        if self.send_source is None:
            return
        entry = getattr(sparse_metadata, "_index_topk_cache_state", {}).get("topk")
        if (
            entry is None
            or (entry["layer_num"], entry["sparse_layer_ordinal"]) != self.send_source
        ):
            raise RuntimeError(
                f"MiniMax-M3 PP stage ending at layer {self.end_layer} "
                f"did not produce top-k from layer {self.send_source[0]}"
            )
        # A dense-only stage can forward an entry before it has a local key.
        signature = (
            transfer_signature(entry["key"])
            if entry["key"] is not None
            else entry["pp_signature"]
        )
        intermediate_tensors.tensors[PP_TOPK_INDICES] = entry["value"][0]
        intermediate_tensors.tensors[PP_TOPK_METADATA] = {
            "source": self.send_source,
            "signature": signature,
        }


def bind_received_topk(entry, key, sparse_metadata, *, nvfp4, mxfp4):
    """Bind received IDs to the local memo and rebuild local physical pages."""
    if entry["pp_signature"] != transfer_signature(key):
        raise RuntimeError("MiniMax-M3 PP top-k forward geometry/policy mismatch")
    indices = entry["value"][0]
    mode, query_shape = key[:2]
    rows = key[6][0] * key[11] if mode == "decode" else query_shape[0]
    if tuple(indices.shape) != (key[10], rows, key[7]):
        raise RuntimeError("MiniMax-M3 PP top-k shape does not match this forward")
    if indices.device != key[3]:
        raise RuntimeError("MiniMax-M3 PP top-k is on the wrong receiving device")
    sparse_bt = sparse_ctx = None
    if not nvfp4:
        from atom.model_ops.minimax_m3.index_topk import emit_sparse_block_table

        md = sparse_metadata.prefill if mode == "prefill" else sparse_metadata.decode
        sparse_bt, sparse_ctx = emit_sparse_block_table(
            indices,
            md.block_table,
            md.cu_seqlens_q if mode == "prefill" else md.seq_lens,
            md.context_lens if mode == "prefill" else md.seq_lens,
            max_query_len=key[11],
            decode=mode == "decode",
            pages_per_block=1 if mxfp4 else 8,
            block_page_stride=1 if mxfp4 else 8,
        )
    entry["key"] = key
    entry["value"] = (indices, sparse_bt, sparse_ctx)
