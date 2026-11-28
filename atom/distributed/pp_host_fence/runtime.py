# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

"""Source-bound PP8 K3 IPC transport. Failures abort the transport.

No retries, per-message fallback, process kill, or resource controller. Importing
this module does not allocate a device buffer or form a group.
"""

import hashlib
import json
import logging
import os
import socket
import threading
from datetime import timedelta
from pathlib import Path

import torch

from atom.mono.runtime.hip_ipc import HipRuntime
from atom.mono.runtime.uncached import DeviceBytes
from atom.utils import envs

from .async_core import AsyncSender, Receiver
from .gloo_channel import GlooChannel
from .k3_edge_schema import SPLIT, edge_schemas, routes
from .tensor_backend import RawRing, ReceiveOps, SendOps

logger = logging.getLogger("atom")
IPC_EDGES = (0, 1, 2, 4, 5, 6)
_ACTIVE = None
_PROCESS_ADAPTERS = []


def _digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def agree(rows, phase):
    if len(rows) != 8 or [row.get("rank") for row in rows] != list(range(8)):
        raise ValueError("PP IPC collective rank order differs at " + phase)
    errors = [
        {"rank": row["rank"], "error": row.get("error")}
        for row in rows
        if row.get("error")
    ]
    if errors:
        raise RuntimeError("PP IPC all-rank refusal at " + phase + ": " + repr(errors))


def validate_topology(rows):
    agree(rows, "config/topology")
    keys = ("configDigest", "schemas", "sourceBindingSha256", "runtime")
    if any(row[key] != rows[0][key] for row in rows for key in keys):
        raise ValueError("PP IPC ranks disagree on source/config/schema/runtime")
    identities = [row["identity"] for row in rows]
    links = routes(identities)
    expected = ["hip-ipc-memory" if rank in IPC_EDGES else "rccl" for rank in range(7)]
    if [link["transport"] for link in links] != expected:
        raise ValueError("PP IPC adapter requires exactly contiguous 4+4 hosts")
    if len({(x["hostname"], x["bootId"]) for x in identities}) != 2:
        raise ValueError("PP IPC adapter requires exactly two actual hosts/boots")
    if any(
        x["ppRank"] != rank or x["globalRank"] != rank or x["localDevice"] != rank % 4
        for rank, x in enumerate(identities)
    ):
        raise ValueError("PP IPC adapter requires the source-bound rank/device layout")
    return links


class Adapter:
    def __init__(self, rank):
        self.rank, self.caller = rank, threading.get_ident()
        self.sender = self.receiver = self.ring = self.send_ops = None
        self.groups = {}
        self.closed = False
        self.failed = False
        _PROCESS_ADAPTERS.append(self)

    def _caller(self):
        if threading.get_ident() != self.caller or self.closed or self.failed:
            raise RuntimeError("PP IPC adapter requires its live original caller")

    def send(self, tensors, owned):
        self._caller()
        if self.sender is None:
            raise RuntimeError("this stage has no outgoing IPC edge")
        try:
            return self.sender.submit(tensors, owned)
        except Exception:
            self.failed = True
            raise

    def recv(self):
        self._caller()
        if self.receiver is None:
            raise RuntimeError("this stage has no incoming IPC edge")
        try:
            return self.receiver.recv()
        except Exception:
            self.failed = True
            raise

    def flush(self):
        self._caller()
        # ACKs are posted in recv(), so only the outgoing worker needs to drain
        # them. Do not block the model caller on flushing its incoming ACKs:
        # callers can flush stages in order rather than all at once.
        if self.sender is not None:
            try:
                self.sender.flush()
            except Exception:
                self.failed = True
                raise

    def close(self):
        self._caller()
        try:
            if self.sender is not None:
                token = self.sender.close()
                self.send_ops.close_mapping(token)
        except Exception:
            self.failed = True
            raise
        self.closed = True
        # Incoming ring, ACK-work buffers and process groups remain strongly
        # referenced. No raw-ring free, incoming mapping close or group destroy.
        # This is normal model process exit only; model reload is unsupported.


def initialize_collective(
    torch, pp, device, preflight, schemas, runtime, device_bytes_type
):
    """Agree on every initialization phase before permitting payload copies."""
    rank = pp.rank_in_group

    def gather(row):
        output = [None] * 8
        torch.distributed.all_gather_object(output, row, group=pp.cpu_group)
        return output

    rows = gather(dict(rank=rank, **preflight))
    links = validate_topology(rows)
    adapter = Adapter(rank)
    adapter.identities, adapter.links = [row["identity"] for row in rows], links
    adapter.schema_id = rows[0]["configDigest"]
    # All default-global ranks call all groups in identical order. Each group
    # has exactly one outgoing Python worker and one receiver caller at runtime.
    for edge in IPC_EDGES:
        group = torch.distributed.new_group(
            ranks=[edge, edge + 1], backend="gloo", timeout=timedelta(seconds=60)
        )
        if rank in (edge, edge + 1):
            adapter.groups[edge] = group
    error = None
    try:
        if rank - 1 in IPC_EDGES:
            adapter.ring = RawRing(
                torch, device, schemas[rank - 1], runtime, device_bytes_type
            )
    except Exception as exc:
        logger.exception("PP IPC initialization phase failed")
        error = type(exc).__name__ + ": " + str(exc)
    allocations = gather(
        {
            "rank": rank,
            "error": error,
            "ringBytes": (
                0 if adapter.ring is None else adapter.ring.metadata["ringBytes"]
            ),
        }
    )
    agree(allocations, "allocation completion")
    # No handle is published until every allocation has completed its memset
    # and device fence. Failed/partially allocated rings stay process-owned.
    error, publication = None, None
    try:
        if adapter.ring is not None:
            publication = adapter.ring.publish(rank - 1, adapter.schema_id)
    except Exception as exc:
        logger.exception("PP IPC initialization phase failed")
        error = type(exc).__name__ + ": " + str(exc)
    publications = gather({"rank": rank, "error": error, "publication": publication})
    agree(publications, "handle publication")
    for peer_rank, row in enumerate(publications):
        published = row["publication"]
        if (peer_rank - 1 in IPC_EDGES) != (published is not None):
            raise ValueError("unexpected publication for this rank topology")
    error = None
    try:
        if rank in IPC_EDGES:
            adapter.send_ops = SendOps(
                torch,
                device,
                schemas[rank],
                runtime,
                publications[rank + 1]["publication"],
                rank,
                adapter.schema_id,
            )
        if adapter.ring is not None:
            incoming = GlooChannel(torch, adapter.groups[rank - 1], pp.ranks[rank - 1])
            adapter.receiver = Receiver(
                rank - 1,
                schemas[rank - 1],
                adapter.schema_id,
                incoming,
                ReceiveOps(torch, device, adapter.ring),
            )
    except Exception as exc:
        logger.exception("PP IPC initialization phase failed")
        error = type(exc).__name__ + ": " + str(exc)
    agree(gather({"rank": rank, "error": error}), "handle open")
    # Mapping agreement precedes every outgoing worker. An init failure never
    # sends/copies a payload and never falls back to RCCL on a local edge.
    error = None
    try:
        if adapter.send_ops is not None:
            outgoing = GlooChannel(torch, adapter.groups[rank], pp.ranks[rank + 1])
            adapter.sender = AsyncSender(
                rank, schemas[rank], adapter.schema_id, outgoing, adapter.send_ops
            )
    except Exception as exc:
        logger.exception("PP IPC initialization phase failed")
        error = type(exc).__name__ + ": " + str(exc)
    agree(gather({"rank": rank, "error": error}), "worker creation")
    adapter.ring_bytes = (
        0 if adapter.ring is None else adapter.ring.metadata["ringBytes"]
    )
    logger.info(
        "PP_HOST_FENCE_INIT %s",
        json.dumps(
            {
                "rank": rank,
                "ringBytes": adapter.ring_bytes,
                "slots": 3,
                "sourceBindingSha256": rows[0]["sourceBindingSha256"],
                "configDigest": adapter.schema_id,
                "incoming": "ipc" if adapter.receiver is not None else "native-or-none",
                "outgoing": "ipc" if adapter.sender is not None else "native-or-none",
                "identities": adapter.identities,
            }
        ),
    )
    return adapter


def initialize(runner, pp):
    global _ACTIVE
    if not envs.ATOM_PP_HOST_FENCE_IPC:
        return None
    if _ACTIVE is not None:
        raise RuntimeError("PP IPC does not support in-process model reload")
    device, cfg = runner.device, runner.config
    error = None
    preflight, schemas = {}, None
    try:
        binding_path = Path(__file__).with_name("SOURCE-BINDING.json")
        binding = json.loads(binding_path.read_text())
        atom_root = Path(__file__).resolve().parents[3]
        for relative, expected in binding["activeSourceSha256"].items():
            if _digest(atom_root / relative) != expected:
                raise ValueError("PP IPC active source differs: " + relative)
        if (
            torch.distributed.get_world_size() != 8
            or pp.world_size != 8
            or list(pp.ranks) != list(range(8))
            or torch.distributed.get_rank() != pp.rank_in_group
            or cfg.tensor_parallel_size != 1
            or cfg.pipeline_parallel_size != 8
            or cfg.pp_size_local != 4
            or cfg.max_num_batched_tokens != 16384
            or cfg.max_num_seqs != 256
            or cfg.gpu_memory_utilization != 0.88
            or cfg.enforce_eager is not True
            or cfg.torch_dtype != torch.bfloat16
            or not (
                cfg.kv_transfer_config is None
                or (type(cfg.kv_transfer_config) is dict and not cfg.kv_transfer_config)
            )
        ):
            raise ValueError("PP IPC requires fixed #772 eager PP8/TP1 4+4 config")
        if os.environ.get("VLLM_PP_LAYER_PARTITION") != ",".join(map(str, SPLIT)):
            raise ValueError("PP IPC requires exact source-bound layer split")
        if Path("/proc/sys/kernel/numa_balancing").read_text().strip() != "0":
            raise ValueError("PP IPC requires NUMA balancing zero")
        main_path, draft_path = (
            Path(cfg.model) / "config.json",
            Path(cfg.speculative_config.model) / "config.json",
        )
        if (
            _digest(main_path) != binding["mainConfigSha256"]
            or _digest(draft_path) != binding["draftConfigSha256"]
        ):
            raise ValueError(
                "model/draft config differs from the source-bound reference"
            )
        main, draft = json.loads(main_path.read_text()), json.loads(
            draft_path.read_text()
        )
        schemas = edge_schemas(main, draft)
        relay = runner.pp_aux_relay
        rank = pp.rank_in_group
        if (
            relay is None
            or list(relay.spec.layer_ids) != draft["target_layer_ids"]
            or relay.spec.hidden_size != 7168
            or (rank < 7 and len(relay.outgoing_ids) != schemas[rank].aux_taps)
            or (rank > 0 and len(relay.incoming_ids) != schemas[rank - 1].aux_taps)
        ):
            raise ValueError("actual aux relay disagrees with per-edge schema")
        runtime_version = {"torch": torch.__version__, "hip": torch.version.hip}
        if runtime_version != binding["runtime"]:
            raise ValueError("PP IPC requires source-bound Torch/HIP runtime")
        properties = torch.cuda.get_device_properties(device)
        if properties.gcnArchName.split(":", 1)[0] != binding["architecture"]:
            raise ValueError(
                "PP IPC has only been prepared for the bound GPU architecture"
            )
        identity = {
            "ppRank": rank,
            "globalRank": torch.distributed.get_rank(),
            "hostname": socket.gethostname(),
            "bootId": Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
            "gpuUuid": str(properties.uuid),
            "localDevice": device.index,
        }
        metadata = [schema.metadata() for schema in schemas]
        config_digest = hashlib.sha256(
            json.dumps(
                {
                    "schemas": metadata,
                    "main": binding["mainConfigSha256"],
                    "draft": binding["draftConfigSha256"],
                    "split": SPLIT,
                },
                sort_keys=True,
            ).encode()
        ).hexdigest()
        preflight = {
            "identity": identity,
            "runtime": runtime_version,
            "configDigest": config_digest,
            "schemas": metadata,
            "sourceBindingSha256": _digest(binding_path),
        }
    except Exception as exc:
        logger.exception("PP IPC initialization phase failed")
        error = type(exc).__name__ + ": " + str(exc)
    preflight["error"] = error
    _ACTIVE = initialize_collective(
        torch, pp, device, preflight, schemas, HipRuntime(), DeviceBytes
    )
    return _ACTIVE


def active():
    if not envs.ATOM_PP_HOST_FENCE_IPC:
        return None
    if _ACTIVE is None:
        raise RuntimeError("PP IPC enabled before all-rank initialization completed")
    return _ACTIVE
