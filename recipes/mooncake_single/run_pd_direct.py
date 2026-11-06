#!/usr/bin/env python3
"""Launch one P or D instance against an existing Mooncake Store installation."""
import argparse
import json
import os
import subprocess
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--role", choices=("prefill", "decode"), required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--host-ip", required=True)
    p.add_argument("--store-ip", required=True)
    p.add_argument("--gpus", default="0,1,2,3")
    p.add_argument("--nic", default="rdma0")
    p.add_argument("--socket-interface", default="eno0")
    p.add_argument("--output", required=True)
    p.add_argument("--dry-run", action="store_true")
    a = p.parse_args()
    root = Path(__file__).resolve().parents[2]
    template = json.loads(Path(__file__).with_name("pd_direct_commands.json").read_text())
    role = template[a.role]
    command = list(role["command"])
    command[command.index("--model") + 1] = a.model
    command[command.index("--data-parallel-master-ip") + 1] = a.host_ip
    k = command.index("--kv-transfer-config") + 1
    transfer = json.loads(command[k])
    transfer["proxy_ip"] = a.host_ip
    command[k] = json.dumps(transfer)
    env = os.environ.copy()
    for name in list(env):
        if name.startswith(("ATOM_DP_PREFIX_", "ATOM_DP_SESSION_", "ATOM_KV_OFFLOAD")) or name == "ATOM_OFFLOAD_EARLY_LOAD":
            env.pop(name)
    env.update(role["environment"])
    env.update(HIP_VISIBLE_DEVICES=a.gpus, ATOM_HOST_IP=a.host_ip,
               NCCL_SOCKET_IFNAME=a.socket_interface, GLOO_SOCKET_IFNAME=a.socket_interface)
    env["PYTHONPATH"] = str(root) + (":" + os.environ["PYTHONPATH"] if os.environ.get("PYTHONPATH") else "")
    if a.role == "prefill":
        env["ATOM_KV_OFFLOAD_EXTRA_CONFIG"] = json.dumps({
            "mooncake_store.master": f"{a.store_ip}:19861",
            "mooncake_store.metadata": f"http://{a.store_ip}:19862/metadata",
            "mooncake_store.protocol": "rdma",
            "mooncake_store.rdma_devices": a.nic,
            "mooncake_store.local_hostname": a.host_ip,
            "mooncake_store.load_pool_mib": 1024,
            "mooncake_store.save_pool_mib": 256,
        })
    record = {"command": command, "environment": {k: env[k] for k in set(role["environment"]) | {
        "HIP_VISIBLE_DEVICES", "ATOM_HOST_IP", "PYTHONPATH", "NCCL_SOCKET_IFNAME", "GLOO_SOCKET_IFNAME"}}}
    if a.dry_run:
        print(json.dumps(record, indent=2))
        return
    output = Path(a.output)
    output.mkdir(parents=True, exist_ok=False)
    (output / "launch.json").write_text(json.dumps(record, indent=2))
    with (output / "server.log").open("w") as log:
        raise SystemExit(subprocess.call(command, env=env, cwd=root, stdout=log, stderr=subprocess.STDOUT))


if __name__ == "__main__":
    main()
