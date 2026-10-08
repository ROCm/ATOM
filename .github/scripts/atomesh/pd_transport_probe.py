"""Bounded, model-free transport probe for an existing two-node allocation.

Run only via ATOMESH_TRANSPORT_ONLY=1 in the normal P/D harness. No package
installation, scheduler operation, network mutation or TCP payload fallback.
GPU work is opt-in (ATOMESH_TRANSPORT_GPU=1); imports stay in supervised children.
ATOMESH_TRANSPORT_NIXL_READ_ONLY=1 additionally requires an explicit locally
validated UCX selection and agreeing peers; it skips all MoRI work and size parsing.
ATOMESH_TRANSPORT_UCX_AUTO_SINGLE=1 instead agrees on one fresh peer-inventory
candidate, only with GPU=1 and READ_ONLY=1, without an explicit selection.
Results describe pinned-image transport, NOT fixed-main vLLM or model PD support.
Exit 0 means collection completed, not that transport passed; inspect summary.json.

Public NIXL wrapper APIs implement one 4096-byte GPU READ (rank 1 pulls rank 0).
Metadata/ack files use the existing shared RUN_DIR, never carry payload bytes.
MoRI uses IOEngine.register_torch_tensor, as in its existing io/test_engine.py;
only registration is tested, in fresh processes per size, not a model or transfer.
"""

import argparse
import base64
import hashlib
import importlib.metadata
import ipaddress
import json
import os
import re
import resource
import signal
import socket
import subprocess
import sys
import time
import traceback
from pathlib import Path

FIXED_SHA = "b22494cc0cb4bd9db4a62fb107d92429a4a3249d"
IMAGE_DIGEST = "sha256:659b28319fef4ea0e3d8f33e25b4c35d6f663f5d818a5b2fa06dceaf859234e4"
TINY_BYTES = 4096
MAX_MORI_BYTES = 2493186048
STAGE_SECONDS = 60
TOTAL_SECONDS = 600


class Unknown(RuntimeError):
    """A prerequisite or an observation is missing; not a transport failure."""


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(value, indent=2) + "\n")
    tmp.replace(path)


def publish_once(path, value):
    """Atomically create protocol evidence without replacing a previous run."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".{os.getpid()}.tmp")
    try:
        with tmp.open("x") as stream:
            stream.write(json.dumps(value, indent=2) + "\n")
        try:
            os.link(tmp, path)
        except FileExistsError:
            raise Unknown(f"conflicting protocol evidence: {path.name}") from None
    finally:
        tmp.unlink(missing_ok=True)


def read_text(path):
    try:
        return Path(path).read_text().strip()
    except OSError as exc:
        return {"error": str(exc)}


def transport_env():
    return {
        k: v
        for k, v in sorted(os.environ.items())
        if k.startswith(("UCX_", "NIXL_", "MORI_", "NCCL_"))
        or k in ("HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES")
    }


def inventory():
    ports = []
    for port in sorted(Path("/sys/class/infiniband").glob("*/ports/*")):
        gids = []
        for gid in sorted((port / "gids").glob("*")):
            gids.append(
                {
                    "index": gid.name,
                    "gid": read_text(gid),
                    "type": read_text(port / "gid_attrs/types" / gid.name),
                    "ndev": read_text(port / "gid_attrs/ndevs" / gid.name),
                }
            )
        ports.append(
            {
                "device": port.parent.parent.name,
                "port": port.name,
                "state": read_text(port / "state"),
                "link_layer": read_text(port / "link_layer"),
                "gids": gids,
            }
        )
    commands = []
    inventory_commands = [
        ["ip", "-j", "address", "show"],
        ["rdma", "link", "show"],
        ["ibv_devinfo"],
        ["ucx_info", "-v"],
        ["ucx_info", "-d"],
    ]
    if os.environ.get("ATOMESH_TRANSPORT_UCX_AUTO_SINGLE") == "1":
        # Build-stage path is evidence-backed, not guaranteed in the runtime image.
        inventory_commands += [
            ["/usr/local/ucx/bin/ucx_info", "-v"],
            ["/usr/local/ucx/bin/ucx_info", "-d"],
        ]
    for argv in inventory_commands:
        try:
            p = subprocess.run(
                argv, capture_output=True, text=True, timeout=6, check=False
            )
            commands.append(
                {
                    "argv": argv,
                    "rc": p.returncode,
                    "stdout": p.stdout,
                    "stderr": p.stderr,
                }
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            commands.append({"argv": argv, "status": "UNKNOWN", "error": str(exc)})
    packages = {}
    for name in ("nixl-rocm", "nixl", "mori", "amd_mori", "torch", "vllm"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    return {
        "hostname": socket.gethostname(),
        "rank": int(os.environ["NODE_RANK"]),
        "allocation_ips": os.environ["IPADDRS"].split(","),
        "job_id": os.environ["SLURM_JOB_ID"],
        "ports": ports,
        "commands": commands,
        "environment": transport_env(),
        "packages": packages,
        "python": sys.version,
        "memlock": resource.getrlimit(resource.RLIMIT_MEMLOCK),
        "image_requested": os.environ.get("DOCKER_IMAGE"),
        "expected_image_digest": IMAGE_DIGEST,
        "survey_source_sha": os.environ.get("ATOMESH_VLLM_SOURCE_SHA"),
        "scope": "PINNED_IMAGE_TRANSPORT_ONLY_NOT_SOURCE_BUILT_VLLM",
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "model_loaded": False,
    }


def validated_selection(spec, inv):
    """Never infer GID=1 from NCCL; validate an explicit UCX request locally."""
    match = re.fullmatch(r"([A-Za-z0-9_.-]+):([1-9][0-9]*)@([0-9]+)", spec)
    if not match:
        raise Unknown("selection must be DEVICE:PORT@GID_INDEX")
    device, port, index = match.groups()
    for entry in inv["ports"]:
        if (entry["device"], entry["port"]) != (device, port):
            continue
        if not str(entry["state"]).startswith("4: ACTIVE"):
            raise Unknown("requested RDMA port is not observed ACTIVE")
        for gid in entry["gids"]:
            if gid["index"] != index:
                continue
            try:
                address = ipaddress.IPv6Address(gid["gid"])
            except (ValueError, TypeError, ipaddress.AddressValueError):
                raise Unknown("requested GID is not readable") from None
            if address.is_unspecified:
                raise Unknown("requested GID is zero")
            if entry["link_layer"] == "Ethernet":
                if not isinstance(gid["ndev"], str) or not gid["ndev"]:
                    raise Unknown("RoCE GID has no observed netdev")
                if not isinstance(gid["type"], str) or "RoCE" not in gid["type"]:
                    raise Unknown("RoCE GID type is not observed")
            return {"UCX_NET_DEVICES": f"{device}:{port}", "UCX_IB_GID_INDEX": index}
    raise Unknown("requested NIC/port/GID is absent from readonly inventory")


def current_selection(spec):
    """Re-read only the chosen sysfs tuple; never search for a replacement."""
    device_port, index = spec.split("@")
    device, port = device_port.split(":")
    path = Path("/sys/class/infiniband") / device / "ports" / port
    return {
        "device": device,
        "port": port,
        "state": read_text(path / "state"),
        "link_layer": read_text(path / "link_layer"),
        "gid": {
            "index": index,
            "gid": read_text(path / "gids" / index),
            "type": read_text(path / "gid_attrs/types" / index),
            "ndev": read_text(path / "gid_attrs/ndevs" / index),
        },
    }


def auto_candidates(inv):
    candidates = {}
    for port in inv["ports"]:
        for gid in port["gids"]:
            spec = f"{port['device']}:{port['port']}@{gid['index']}"
            if port["link_layer"] != "Ethernet" or gid["type"] != "RoCE v2":
                continue
            try:
                validated_selection(spec, {"ports": [{**port, "gids": [gid]}]})
            except Unknown:
                continue
            if spec in candidates:
                raise Unknown("duplicate inventory tuple")
            candidates[spec] = {
                **{key: value for key, value in port.items() if key != "gids"},
                "gid": gid,
            }
    return candidates


def evidence_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def auto_single_probe(inv, root, out, rank, run, summary):
    """Agree on one fresh sysfs candidate, then validate it without retries."""
    context = {
        "job_id": os.environ["SLURM_JOB_ID"],
        "run_token": os.environ["ATOMESH_RUN_TOKEN"],
        "image": os.environ["DOCKER_IMAGE"],
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "source_sha": FIXED_SHA,
        "allocation_ips": os.environ["IPADDRS"].split(","),
        "phase": os.environ.get("ATOMESH_EXECUTION_PHASE", "benchmark"),
        "gpu": "1",
        "nixl_read_only": "1",
        "auto_single": "1",
    }
    identity = {
        **context,
        "rank": rank,
        "hostname": socket.gethostname(),
        "node_ip": context["allocation_ips"][rank],
        "nonce": os.urandom(16).hex(),
    }

    def publish(stage, **payload):
        document = {**identity, "created_at": time.time(), **payload}
        publish_once(root / f"auto-{stage}-{rank}.json", document)
        return document

    def receive(stage):
        peer = wait_json(root / f"auto-{stage}-{1-rank}.json", timeout=45)
        if (
            any(peer.get(key) != value for key, value in context.items())
            or peer.get("rank") != 1 - rank
            or peer.get("node_ip") != context["allocation_ips"][1 - rank]
            or not isinstance(peer.get("hostname"), str)
            or not peer["hostname"]
            or peer["hostname"] == identity["hostname"]
            or not isinstance(peer.get("nonce"), str)
            or re.fullmatch(r"[0-9a-f]{32}", peer["nonce"]) is None
            or not isinstance(peer.get("created_at"), (int, float))
            or not -5 <= time.time() - peer["created_at"] <= 45
        ):
            raise Unknown(f"stale or mismatched auto-{stage} peer")
        return peer

    summary["stages"]["mori-register"] = {
        "status": "UNKNOWN",
        "classification": "NOT_TESTED",
        "reason": "NIXL READ-only control",
    }
    summary["stages"]["nixl-rdma-gpu-read"] = {
        "status": "UNKNOWN",
        "classification": "NOT_TESTED",
        "reason": "single-candidate agreement and both selected backends required",
    }
    local = publish("inventory", inventory=inv)
    peer = receive("inventory")
    candidates, remote_candidates = auto_candidates(inv), auto_candidates(
        peer["inventory"]
    )
    common = candidates.keys() & remote_candidates.keys()

    def order(spec):
        device_port, gid = spec.split("@")
        device, port = device_port.split(":")
        return device, int(port), int(gid)

    evidence = {
        "rule": "lexicographic device, numeric port, numeric GID index; one attempt",
        "inventory_digests": {
            str(rank): evidence_digest(local),
            str(1 - rank): evidence_digest(peer),
        },
        "nonces": {str(rank): identity["nonce"], str(1 - rank): peer["nonce"]},
        "local_candidates": sorted(candidates, key=order),
        "peer_candidates": sorted(remote_candidates, key=order),
        "selection": min(common, key=order) if common else None,
    }
    write_json(out / "auto-selection.json", evidence)
    # Retain original settings as separately labelled evidence, never selected proof.
    run("nixl-original-create", "nixl-create")
    if not common:
        raise Unknown("no common ACTIVE RoCE v2 nonzero-GID tuple")
    selection = evidence["selection"]
    evidence.update(
        local_entry=candidates[selection], peer_entry=remote_candidates[selection]
    )
    write_json(out / "auto-selection.json", evidence)
    agreement = {
        key: evidence[key] for key in ("selection", "inventory_digests", "nonces")
    }
    publish("selection", agreement=agreement)
    remote = receive("selection")
    if remote["nonce"] != peer["nonce"] or remote.get("agreement") != agreement:
        raise Unknown("peer did not confirm fresh single-candidate agreement")
    refreshed = current_selection(selection)
    write_json(out / "auto-selection-recheck.json", refreshed)
    if refreshed != candidates[selection]:
        raise Unknown("selected tuple changed; no replacement attempted")
    overrides = validated_selection(selection, inv)
    selected = run("nixl-selected-create", "nixl-create", overrides)
    publish("ready", agreement=agreement, eligible=selected["status"] == "PASS")
    remote = receive("ready")
    if (
        remote["nonce"] != peer["nonce"]
        or remote.get("agreement") != agreement
        or remote.get("eligible") is not True
        or selected["status"] != "PASS"
    ):
        raise Unknown("selected backend failed or peer readiness disagreed; no retry")
    session = {
        **context,
        "agreement": agreement,
        "hosts": {str(rank): identity["hostname"], str(1 - rank): peer["hostname"]},
    }
    run(
        "nixl-rdma-gpu-read",
        "nixl-transfer",
        {
            **overrides,
            "UCX_TLS": "rc,rocm_copy,rocm_ipc",
            "ATOMESH_TRANSPORT_READ_SESSION": json.dumps(session, sort_keys=True),
        },
        exchange=root / f"auto-read-{evidence_digest(session)}",
    )


def wait_json(path, timeout=45):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if path.exists():
            return json.loads(path.read_text())
        time.sleep(0.1)
    raise Unknown(f"peer did not publish {path.name} within {timeout}s")


def load_nixl():
    # Same ROCm wrapper used by vllm.distributed.nixl_utils, without importing vLLM.
    from nixl_rocm._api import nixl_agent, nixl_agent_config

    return nixl_agent, nixl_agent_config


def checkpoint(path, stage, **extra):
    write_json(path, {"stage": stage, "status": "RUNNING", **extra})


def nixl_probe(args):
    agent_class, config_class = load_nixl()
    checkpoint(args.result, "backend_create", environment=transport_env())
    agent = agent_class(
        f"survey-{os.environ['SLURM_JOB_ID']}-{args.rank}",
        config_class(backends=["UCX"]),
    )
    if "UCX" not in agent.get_plugin_list() or "UCX" not in agent.backends:
        raise Unknown("UCX plugin/backend unavailable")
    details = {
        "plugins": agent.get_plugin_list(),
        "backend_params": agent.get_backend_params("UCX"),
        "api_source": sys.modules[agent_class.__module__].__file__,
    }
    if args.child == "nixl-create":
        del agent
        return {"status": "PASS", "meaning": "BACKEND_CREATE_ONLY", **details}

    import torch

    if not torch.version.hip or not torch.cuda.is_available():
        raise Unknown("ROCm GPU unavailable")
    torch.cuda.set_device(0)
    tensor = torch.arange(TINY_BYTES, dtype=torch.int32, device="cuda").to(torch.uint8)
    if args.rank == 1:
        tensor.zero_()
    torch.cuda.synchronize()
    registration = None
    handle = None
    remote = None
    try:
        checkpoint(args.result, "gpu_register")
        registration = agent.register_memory(tensor, backends=["UCX"])
        exchange = Path(args.exchange)
        session = json.loads(os.environ.get("ATOMESH_TRANSPORT_READ_SESSION", "null"))
        protocol_write = publish_once if session is not None else write_json
        binding = {"session": session} if session is not None else {}
        protocol_write(
            exchange / f"metadata-{args.rank}.json",
            {
                **binding,
                "hostname": socket.gethostname(),
                "rank": args.rank,
                "node_ip": os.environ["IPADDRS"].split(",")[args.rank],
                "metadata": base64.b64encode(agent.get_agent_metadata()).decode(),
                "pointer": tensor.data_ptr(),
                "bytes": TINY_BYTES,
                "gpu": tensor.get_device(),
            },
        )
        peer = wait_json(exchange / f"metadata-{1 - args.rank}.json")
        if peer["hostname"] == socket.gethostname() or peer["rank"] == args.rank:
            raise Unknown("distinct-node identity not established")
        if session is not None and (
            peer.get("session") != session
            or peer.get("rank") != 1 - args.rank
            or peer.get("hostname") != session["hosts"][str(1 - args.rank)]
            or peer.get("node_ip") != session["allocation_ips"][1 - args.rank]
        ):
            raise Unknown("READ metadata identity mismatch")
        if peer["bytes"] != TINY_BYTES:
            raise Unknown("peer payload size differs")
        remote = agent.add_remote_agent(base64.b64decode(peer["metadata"]))
        checkpoint(args.result, "gpu_read", peer_hostname=peer["hostname"])
        if args.rank == 1:
            local_desc = agent.get_xfer_descs(tensor)
            remote_desc = agent.get_xfer_descs(
                [(peer["pointer"], TINY_BYTES, peer["gpu"])], "VRAM"
            )
            handle = agent.initialize_xfer(
                "READ", local_desc, remote_desc, remote, backends=["UCX"]
            )
            state = agent.transfer(handle)
            deadline = time.monotonic() + 20
            while state == "PROC" and time.monotonic() < deadline:
                time.sleep(0.001)
                state = agent.check_xfer_state(handle)
            if state == "PROC":
                raise Unknown("GPU READ did not complete within 20s")
            if state != "DONE":
                raise RuntimeError(f"GPU READ state={state}")
            torch.cuda.synchronize()
            expected = torch.arange(TINY_BYTES, dtype=torch.int32).to(torch.uint8)
            if not torch.equal(tensor.cpu(), expected):
                raise RuntimeError("GPU destination payload mismatch")
            protocol_write(
                exchange / "verified.json",
                {
                    **binding,
                    "status": "PASS",
                    "bytes": TINY_BYTES,
                    "operation": "READ",
                    "source": peer["hostname"],
                    "destination": socket.gethostname(),
                    "payload_verified": True,
                    "direct_gpu_zero_copy": "NOT_ESTABLISHED_HOST_STAGING_POSSIBLE",
                    "sha256": hashlib.sha256(
                        tensor.cpu().numpy().tobytes()
                    ).hexdigest(),
                },
            )
        receipt = wait_json(exchange / "verified.json")
        if session is not None and (
            receipt.get("session") != session
            or receipt.get("source") != session["hosts"]["0"]
            or receipt.get("destination") != session["hosts"]["1"]
            or receipt.get("status") != "PASS"
            or receipt.get("bytes") != TINY_BYTES
            or receipt.get("operation") != "READ"
            or receipt.get("payload_verified") is not True
        ):
            raise Unknown("READ receipt identity or payload evidence mismatch")
        return {
            "status": "PASS",
            "meaning": "TINY_GPU_READ_NOT_MODEL_PD",
            "receipt": receipt,
            **details,
        }
    finally:
        # A hung native cleanup is also covered by the parent stage deadline.
        if handle is not None:
            agent.release_xfer_handle(handle)
        if remote is not None:
            agent.remove_remote_agent(remote)
        if registration is not None:
            agent.deregister_memory(registration, backends=["UCX"])
        del agent


def mori_probe(args):
    import torch
    from mori.io import BackendType, IOEngine, IOEngineConfig, RdmaBackendConfig

    if not torch.version.hip or not torch.cuda.is_available():
        raise Unknown("ROCm GPU unavailable")
    torch.cuda.set_device(0)
    checkpoint(args.result, "mori_backend_create")
    engine = IOEngine(
        key=f"survey-mori-{args.rank}-{args.size}",
        config=IOEngineConfig(host=os.environ["IPADDRS"].split(",")[args.rank], port=0),
    )
    engine.create_backend(
        BackendType.RDMA,
        RdmaBackendConfig(
            qp_per_transfer=1,
            post_batch_size=-1,
            num_worker_threads=1,
            enable_notification=False,
        ),
    )
    memory = None
    try:
        checkpoint(args.result, "gpu_allocate", bytes=args.size)
        tensor = torch.empty(args.size, dtype=torch.uint8, device="cuda")
        torch.cuda.synchronize()
        checkpoint(args.result, "mori_gpu_register", bytes=args.size)
        memory = engine.register_torch_tensor(tensor)
        if memory is None:
            raise RuntimeError("MoRI registration returned None")
        # Serialize to exercise the same registration descriptor path as vLLM.
        packed_bytes = len(memory.pack())
        return {
            "status": "PASS",
            "meaning": "REGISTRATION_ONLY_NOT_TRANSFER",
            "bytes": args.size,
            "descriptor_bytes": packed_bytes,
        }
    finally:
        if memory is not None:
            engine.deregister_memory(memory)
        engine.remove_backend(BackendType.RDMA)
        del engine


def child(args):
    result = {"status": "UNKNOWN"}
    try:
        result = mori_probe(args) if args.child == "mori-register" else nixl_probe(args)
    except (Unknown, ImportError) as exc:
        result = {
            "status": "UNKNOWN",
            "classification": "BLOCKED_ENV",
            "error": repr(exc),
        }
    except Exception as exc:  # noqa: BLE001 -- preserve evidence at process boundary
        traceback.print_exc()
        result = {
            "status": "FAIL",
            "classification": "OBSERVED_PROBE_FAILURE",
            "error": repr(exc),
        }
    previous = (
        json.loads(Path(args.result).read_text()) if Path(args.result).exists() else {}
    )
    write_json(args.result, {**previous, **result, "environment": transport_env()})


def stop_group(process):
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        process.wait(timeout=2)
    except subprocess.TimeoutExpired:
        pass
    # Kill descendants even when the process-group leader already exited.
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    process.wait(timeout=2)


def supervise(argv, log, timeout, env):
    with log.open("w") as stream:
        process = subprocess.Popen(
            argv,
            stdout=stream,
            stderr=subprocess.STDOUT,
            env=env,
            start_new_session=True,
        )
        try:
            return process.wait(timeout=timeout), False
        except subprocess.TimeoutExpired:
            return None, True
        finally:
            stop_group(process)


def parse_sizes(raw):
    sizes = [int(x) for x in raw.split(",")]
    if not 1 <= len(sizes) <= 3 or len(set(sizes)) != len(sizes):
        raise ValueError("supply 1-3 distinct MoRI registration byte sizes")
    if any(not 1 <= size <= MAX_MORI_BYTES for size in sizes):
        raise ValueError(f"MoRI registration sizes must be 1..{MAX_MORI_BYTES}")
    return sizes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--child", choices=("nixl-create", "nixl-transfer", "mori-register")
    )
    parser.add_argument("--rank", type=int)
    parser.add_argument("--result")
    parser.add_argument("--exchange")
    parser.add_argument("--size", type=int, default=4096)
    args = parser.parse_args()
    if args.child:
        child(args)
        return 0

    rank = int(os.environ["NODE_RANK"])
    ips = os.environ["IPADDRS"].split(",")
    if rank not in (0, 1) or len(ips) != 2 or len(set(ips)) != 2:
        raise ValueError(
            "transport diagnostic requires exactly two distinct allocated nodes"
        )
    if os.environ.get("ATOMESH_VLLM_SOURCE_SHA") != FIXED_SHA:
        raise ValueError("transport survey must retain fixed-main source identity")
    if IMAGE_DIGEST not in os.environ.get("DOCKER_IMAGE", ""):
        raise ValueError("transport survey requires the recorded digest-pinned image")
    auto_single = os.environ.get("ATOMESH_TRANSPORT_UCX_AUTO_SINGLE", "0")
    if auto_single not in ("0", "1"):
        raise ValueError("ATOMESH_TRANSPORT_UCX_AUTO_SINGLE must be 0 or 1")
    if auto_single == "1" and (
        os.environ.get("ATOMESH_TRANSPORT_GPU") != "1"
        or os.environ.get("ATOMESH_TRANSPORT_NIXL_READ_ONLY") != "1"
        or os.environ.get("ATOMESH_TRANSPORT_UCX_SELECTION", "")
        or not os.environ.get("ATOMESH_RUN_TOKEN", "").strip()
    ):
        raise ValueError(
            "AUTO_SINGLE requires GPU=1, READ_ONLY=1, run token, no explicit selection"
        )
    read_only = os.environ.get("ATOMESH_TRANSPORT_NIXL_READ_ONLY", "0")
    if read_only not in ("0", "1"):
        raise ValueError("ATOMESH_TRANSPORT_NIXL_READ_ONLY must be 0 or 1")
    sizes = (
        []
        if read_only == "1"
        else parse_sizes(os.environ.get("ATOMESH_TRANSPORT_MORI_BYTES", "4096"))
    )
    gpu = os.environ.get("ATOMESH_TRANSPORT_GPU", "0")
    if gpu not in ("0", "1"):
        raise ValueError("ATOMESH_TRANSPORT_GPU must be 0 or 1")
    if read_only == "1" and (
        gpu != "1"
        or (
            auto_single != "1"
            and not os.environ.get("ATOMESH_TRANSPORT_UCX_SELECTION", "")
        )
    ):
        raise ValueError("NIXL READ-only requires GPU=1 and an explicit UCX selection")
    root = (
        Path(os.environ["RUN_DIR"])
        / "transport-diagnostic"
        / os.environ.get("ATOMESH_EXECUTION_PHASE", "benchmark")
    )
    out = root / f"rank-{rank}"
    out.mkdir(parents=True, exist_ok=False)
    summary = {
        "collection": "RUNNING",
        "model_pd": "NOT_TESTED",
        "stages": {},
        "scope": "PINNED_IMAGE_TRANSPORT_ONLY",
        "rank": rank,
        "stage_timeout_seconds": STAGE_SECONDS,
        "total_timeout_seconds": TOTAL_SECONDS,
    }

    def interrupted(signum, _frame):
        raise Unknown(f"collection interrupted by signal {signum}")

    for sig in (signal.SIGALRM, signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, interrupted)
    signal.alarm(TOTAL_SECONDS)
    try:
        inv = inventory()
        write_json(out / "inventory.json", inv)

        def run(name, kind, overrides=None, size=4096, exchange=None):
            result_path = out / f"{name}.json"
            argv = [
                sys.executable,
                str(Path(__file__).resolve()),
                "--child",
                kind,
                "--rank",
                str(rank),
                "--result",
                str(result_path),
                "--exchange",
                str(exchange if exchange is not None else root / name),
                "--size",
                str(size),
            ]
            env = {**os.environ, **(overrides or {})}
            write_json(
                out / f"{name}.launch.json",
                {
                    "argv": argv,
                    "process_local_overrides": overrides or {},
                    "diagnostic_not_original": bool(overrides),
                },
            )
            rc, timed_out = supervise(argv, out / f"{name}.log", STAGE_SECONDS, env)
            result = json.loads(result_path.read_text()) if result_path.exists() else {}
            if (
                timed_out
                or rc != 0
                or result.get("status") not in ("PASS", "FAIL", "UNKNOWN")
            ):
                result.update(
                    status="UNKNOWN",
                    classification=(
                        "BOUNDED_TIMEOUT" if timed_out else "ABNORMAL_CHILD_EXIT"
                    ),
                    return_code=rc,
                )
            result["child_return_code"] = rc
            result["process_group_cleanup"] = "SIGNALLED_PARENT_REAPED"
            write_json(result_path, result)
            summary["stages"][name] = result
            write_json(out / "summary.json", summary)
            return result

        if auto_single == "1":
            auto_single_probe(inv, root, out, rank, run, summary)
            summary["collection"] = "COMPLETED"
            return 0

        original = run("nixl-original-create", "nixl-create")
        selection = os.environ.get("ATOMESH_TRANSPORT_UCX_SELECTION", "")
        overrides = {}
        eligible = original["status"] == "PASS"
        if selection:
            try:
                overrides = validated_selection(selection, inv)
                selected = run("nixl-selected-create", "nixl-create", overrides)
                eligible = selected["status"] == "PASS"
            except Unknown as exc:
                eligible = False
                summary["stages"]["nixl-selected-create"] = {
                    "status": "UNKNOWN",
                    "classification": "SELECTION_NOT_JUSTIFIED",
                    "error": str(exc),
                }
        # Both nodes must agree before transfer; a one-sided create is insufficient.
        write_json(
            root / f"ready-{rank}.json",
            {
                "hostname": socket.gethostname(),
                "eligible": eligible,
                "gpu": gpu,
                "selection": selection,
                **({"nixl_read_only": read_only} if read_only == "1" else {}),
            },
        )
        try:
            peer = wait_json(root / f"ready-{1-rank}.json")
        except Unknown as exc:
            peer = {"gpu": "0", "eligible": False}
            summary["peer_readiness"] = {"status": "UNKNOWN", "error": str(exc)}
        if (
            gpu == "1"
            and peer["gpu"] == "1"
            and eligible
            and peer["eligible"]
            and peer["hostname"] != socket.gethostname()
            and peer["selection"] == selection
            and peer.get("nixl_read_only", "0") == read_only
        ):
            # Force an RDMA payload path; unlike original create this is diagnostic.
            # UCX rc may use UD for wireup. No tcp, self or shared-memory payload.
            run(
                "nixl-rdma-gpu-read",
                "nixl-transfer",
                {**overrides, "UCX_TLS": "rc,rocm_copy,rocm_ipc"},
            )
        else:
            summary["stages"]["nixl-rdma-gpu-read"] = {
                "status": "UNKNOWN",
                "classification": "NOT_TESTED",
                "reason": "GPU opt-in, both backend creates and distinct hosts required",
            }
        if read_only == "1":
            summary["stages"]["mori-register"] = {
                "status": "UNKNOWN",
                "classification": "NOT_TESTED",
                "reason": "NIXL READ-only control",
            }
        elif gpu == "1":
            for size in sizes:
                run(f"mori-register-{size}", "mori-register", size=size)
        else:
            summary["stages"]["mori-register"] = {
                "status": "UNKNOWN",
                "classification": "NOT_TESTED",
                "reason": "GPU opt-in disabled",
            }
        summary["collection"] = "COMPLETED"
        return 0
    except Exception as exc:  # noqa: BLE001 -- preserve evidence at process boundary
        summary["collection"] = "UNKNOWN"
        summary["collection_error"] = repr(exc)
        traceback.print_exc()
        return 2
    finally:
        signal.alarm(0)
        write_json(out / "summary.json", summary)
        print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    sys.exit(main())
