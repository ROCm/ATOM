# SPDX-License-Identifier: MIT
"""Producer-only ATOM adapter for the TileRT V4.1 protocol.

The first implementation sends synchronously on the KV worker. It retains
PAGE/STATE ownership through both GPU packing and RDMA completion and never
retries a write after admission. TileRT owns decode and its HTTP interface.
"""

import socket
from dataclasses import dataclass

from atom.kv_transfer.disaggregation.base import (
    KVConnectorBase,
    KVConnectorSchedulerBase,
)
from atom.kv_transfer.disaggregation.types import ConnectorMetadata


@dataclass
class TileRTRequest:
    req_id: str | int
    rid: str
    prompt_ids: list[int]
    blocks: list[int]
    slot: int
    host: str
    port: int
    sampling: dict | None


class TileRTMetadata(ConnectorMetadata):
    WORK_FIELDS = ConnectorMetadata.WORK_FIELDS + ("requests",)

    def __init__(self):
        super().__init__()
        self.requests = []


def require_producer(config):
    cfg = config.kv_transfer_config
    if cfg.get("kv_role") != "kv_producer":
        raise ValueError(
            "TileRT is an external decoder; ATOM supports only kv_producer"
        )
    if config.pipeline_parallel_size != 1 or config.decode_context_parallel_size != 1:
        raise ValueError("Initial TileRT adapter requires PP=1 and DCP=1")
    return cfg


class TileRTConnectorScheduler(KVConnectorSchedulerBase):
    is_producer = True
    dispatch_on_idle = True

    def __init__(self, config):
        self.cfg = require_producer(config)
        self.pending, self.claims = {}, set()

    def get_num_new_matched_tokens(self, seq):
        return 0, False

    def update_state_after_alloc(self, seq):
        pass

    def request_finished(self, seq):
        from tilert.pd_vllm.wire import derive_rid

        seq.kv_transfer_params_output = None
        if getattr(seq, "leave_reason", None) == "aborted":
            return
        params = seq.kv_transfer_params or {}
        host = params.get("tilert_host") or self.cfg.get("tilert_host")
        if not host:
            raise ValueError("TileRT request is missing tilert_host")
        request = TileRTRequest(
            seq.id,
            params.get("rid") or derive_rid(str(seq.id)),
            list(seq.prompt_token_ids),
            list(seq.block_table),
            seq.state_slot,
            host,
            int(params.get("tilert_ctrl_port", self.cfg.get("tilert_ctrl_port", 5556))),
            params.get("sampling"),
        )
        if len(seq.output_tokens) != 1:
            raise ValueError("TileRT prefill must generate exactly one anchor token")
        seq.kv_transfer_params_output = {
            "rid": request.rid,
            "first_token_id": int(seq.output_tokens[0]),
        }
        self.claims.add(seq.id)
        self.pending[seq.id] = request

    def build_connector_meta(self):
        meta = TileRTMetadata()
        meta.requests = list(self.pending.values())
        self.pending.clear()
        return meta

    def should_defer_free(self, seq):
        return seq.id in self.claims

    def send_finished(self, req_id):
        self.claims.discard(req_id)

    def abort_send(self, seq):
        # A dispatched synchronous sender may still be reading PAGE/STATE.
        # Its TP completion, rather than client cancellation, releases them.
        if seq.id not in self.claims:
            self.pending.pop(seq.id, None)

    def source_blocks_released(self, seq):
        self.pending.pop(seq.id, None)


class TileRTConnector(KVConnectorBase):
    is_producer = True

    def __init__(self, config):
        self.cfg = require_producer(config)
        from aiter.dist.parallel_state import get_tp_group

        self.rank = get_tp_group().rank_in_group
        self.max_seq_len = int(self.cfg.get("tilert_max_seq_len", config.max_model_len))
        self.finished = set()

    @classmethod
    def kv_budget_reserve_bytes(cls, config):
        # Every TP rank reserves the same amount so PAGE counts stay equal.
        from tilert.pd_vllm.profiles.dsv41_rocm import wire_bytes

        return wire_bytes(
            int(
                config.kv_transfer_config.get(
                    "tilert_max_seq_len", config.max_model_len
                )
            )
        )

    def register_kv_caches(self, kv_caches, transfer_tensors=None, num_blocks=None):
        if self.rank != 0:
            return
        from tilert.pd_vllm.transport import make_transport

        from atom.utils.network import get_ip

        from .codec import AtomWireEncoder

        if transfer_tensors is None or transfer_tensors.external_cache is None:
            raise ValueError("TileRT requires the V4.1 attention cache")
        self.encoder = AtomWireEncoder(
            transfer_tensors.external_cache, self.max_seq_len
        )
        self.transport = make_transport("mooncake")
        self.transport.init(get_ip())
        stg = self.encoder.staging
        self.transport.register(stg.data_ptr(), stg.numel(), stg.device.index)

    def start_load_kv(self, metadata):
        for req in metadata.requests:
            if self.rank == 0:
                self._send(req)
            # Quorum remains TP-wide even though KV is replicated and rank0 sends.
            self.finished.add(req.req_id)

    def _send(self, req):
        from tilert.pd_vllm import wire

        sections = self.encoder.encode(req)
        with socket.create_connection((req.host, req.port), timeout=60) as conn:
            conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            hello = wire.recv_msg(conn)
            expected = {
                "magic": wire.MAGIC,
                "protocol_version": wire.PROTOCOL_VERSION,
                "layout_version": 2041,
                "transport": "mooncake",
                "senders": 1,
            }
            for name, value in expected.items():
                if hello.get(name) != value:
                    raise ValueError(f"TileRT incompatible {name}: {hello.get(name)!r}")
            if len(req.prompt_ids) > int(hello["max_seq_len"]):
                raise ValueError("Prompt exceeds TileRT decoder capacity")
            wire.send_msg(
                conn,
                {
                    "rid": req.rid,
                    "rank": 0,
                    "seq_len": len(req.prompt_ids),
                    "last_prompt_token": req.prompt_ids[-1],
                    "sampling": req.sampling,
                    "prompt_token_ids": req.prompt_ids,
                    "admission_window_s": 60,
                },
            )
            ack = wire.recv_msg(conn)
            if (
                ack.get("accepted") is not True
                or ack.get("rid") != req.rid
                or ack.get("rank") != 0
                or type(ack.get("generation")) is not int
            ):
                raise RuntimeError(f"TileRT admission failed: {ack}")
            src, dst, size = self.encoder.profile.rdma_plan(
                hello, sections, 0, len(req.prompt_ids), self.encoder.staging.data_ptr()
            )
            self.transport.write(hello, src, dst, size)
            wire.send_msg(conn, wire.done_msg(req.rid, 0, ack["generation"]))

    def get_finished(self):
        finished, self.finished = self.finished, set()
        return finished, set()
