"""Bind a functional contract to native srtctl configuration on an allocated node."""

import argparse
import copy
import json
import os
import re
import shlex
from pathlib import Path

import yaml


def configurations(
    contract, model_path, run_dir, experiment_dir, fabric, uid, router_image
):
    roles = {}
    for role in ("prefill", "decode"):
        spec = contract["roles"][role]
        roles[role] = {
            "nodes": 1 if role == "prefill" else "colocate",
            "workers": 1,
            "gpus": 4,
            "args": copy.deepcopy(spec["args"]),
            "env": {
                key: (
                    fabric[str(value).removeprefix("@fabric.")]
                    if str(value).startswith("@fabric.")
                    else str(value)
                )
                for key, value in spec["env"].items()
            },
        }
    environment = {
        key: (
            fabric[str(value).removeprefix("@fabric.")]
            if str(value).startswith("@fabric.")
            else str(value)
        )
        for key, value in contract["environment"].items()
    }
    environment["VLLM_HOST_IP"] = fabric["host-ip"]
    command = (
        "python3 /experiment/pd_recipe_router_ab.py benchmark "
        "--contract /experiment/contract.json "
        f"--model-path {shlex.quote(str(Path(model_path).resolve()))} "
        '--router "http://${SRT_FRONTEND_HOST}:${SRT_FRONTEND_PORT}" '
        '--workers "http://${SRT_PREFILL_ENDPOINTS}" "http://${SRT_DECODE_ENDPOINTS}" '
        "--output /logs/functional"
    )
    recipe = {
        "schema": 2,
        "name": contract["case_name"],
        "model": {
            "path": str(model_path),
            "container": contract["image"],
            "precision": "MXFP4",
        },
        "resources": {"gpu_type": "mi355x", "gpus_per_node": 8},
        "slurm": {"time_limit": "01:30:00"},
        "frontend": {
            "type": "atomesh",
            "container_image": router_image,
            "enable_multiple_frontends": False,
            "args": {"policy": "dp_sticky"},
        },
        "engine": {
            "type": "vllm",
            "connector": "mooncake",
            "set_visible_devices": True,
        },
        "roles": roles,
        "environment": environment,
        "health_check": {"interval_seconds": 10, "max_attempts": 270},
        "telemetry": {"enabled": False},
        "observability": {
            "enabled": False,
            "tachometer": {"enabled": False},
            "nsys": {"enabled": False},
        },
        "benchmark": {"type": "custom", "command": command},
        "output": {"log_dir": str(run_dir / "logs")},
        "container_mounts": {
            str(experiment_dir): "/experiment",
            str(run_dir / "hf-cache"): "/hf_hub_cache",
        },
    }
    cluster = {
        "launcher": "docker",
        "docker_gpus": "",
        "docker_cap_add": ["SYS_NICE"],
        "visible_devices_env": "HIP_VISIBLE_DEVICES",
        "network_interface": fabric["socket-ifname"],
        "docker_args": [
            "--user",
            uid,
            "--device=/dev/kfd",
            "--device=/dev/dri",
            "--device=/dev/infiniband",
            "--cap-add=IPC_LOCK",
        ],
    }
    cluster["default_mounts"] = {str(run_dir / "bin"): "/opt/atomesh-ci-bin"}
    cluster["default_bash_preamble"] = "export PATH=/opt/atomesh-ci-bin:$PATH"
    # Match the existing Spur container's writable cache paths for its required UID.
    cache = f"/tmp/atomesh-cache-{run_dir.name}"
    container_env = {
        "HOME": f"/tmp/atomesh-home-{run_dir.name}",
        "XDG_CACHE_HOME": cache,
        "TORCHINDUCTOR_CACHE_DIR": f"{cache}/torchinductor",
        "AITER_CACHE_DIR": f"{cache}/aiter",
        "AITER_JIT_DIR": f"{cache}/aiter/jit",
        "FLYDSL_RUNTIME_CACHE_DIR": f"{cache}/flydsl",
        "USER": os.environ.get("USER", uid.split(":")[0]),
        "LOGNAME": os.environ.get("USER", uid.split(":")[0]),
    }
    for key, value in container_env.items():
        cluster["docker_args"].extend(["-e", f"{key}={value}"])
    for group in ("VIDEO_GID", "RENDER_GID"):
        if os.environ.get(group):
            cluster["docker_args"].extend(["--group-add", os.environ[group]])
    for host, target in (
        (os.environ.get("HOST_IONIC", ""), "/usr/lib/x86_64-linux-gnu/libionic.so.1"),
        (
            "/usr/lib/x86_64-linux-gnu/libibverbs/libionic-rdmav34.so",
            "/usr/lib/x86_64-linux-gnu/libibverbs/libionic-rdmav34.so",
        ),
    ):
        if host and Path(host).exists():
            cluster["docker_args"].extend(["-v", f"{host}:{target}:ro"])
    return recipe, cluster


def verified_router_image(contract, manifest):
    build = contract["router_build"]
    if (
        manifest.get("commit") != build["source_sha"]
        or manifest.get("lockfile_sha256") != build["lockfile_sha256"]
        or manifest.get("base_image") != contract["router_images"]["atomesh"]
        or manifest.get("runtime_verified") is not True
        or manifest.get("profile") != build["profile"]
        or not manifest.get("cargo", "").startswith(
            "cargo " + build["rust_toolchain"] + " "
        )
        or not re.fullmatch(r"[0-9a-f]{64}", manifest.get("binary_sha256", ""))
    ):
        raise ValueError("Router runtime does not match the reviewed build contract")
    image = manifest.get("image_id", "")
    if not re.fullmatch(r"sha256:[0-9a-f]{64}", image):
        raise ValueError("A verified local router image ID is required")
    return image


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--uid", required=True)
    args = parser.parse_args()
    contract = json.loads(args.contract.read_text())
    experiment = args.run_dir / "experiment"
    experiment.mkdir(exist_ok=True)
    (args.run_dir / "hf-cache/hub").mkdir(parents=True, exist_ok=True)
    (args.run_dir / "logs").mkdir(exist_ok=True)
    (experiment / "contract.json").write_text(json.dumps(contract, indent=2))
    source = Path(__file__).with_name("pd_recipe_router_ab.py")
    (experiment / source.name).write_text(source.read_text())
    fabric = {
        "host-ip": os.environ["ATOMESH_NODE_IP"],
        "socket-ifname": os.environ["MORI_SOCKET_IFNAME"],
        "rdma-devices": os.environ["NCCL_IB_HCA"],
        "rdma-gid-index": os.environ["NCCL_IB_GID_INDEX"],
        "rdma-tc": os.environ["NCCL_IB_TC"],
    }
    router_runtime = json.loads(
        (args.run_dir / "router-build/runtime.json").read_text()
    )
    router_image = verified_router_image(contract, router_runtime)
    recipe, cluster = configurations(
        contract,
        args.model_path,
        args.run_dir,
        experiment,
        fabric,
        args.uid,
        router_image,
    )
    (args.run_dir / "recipe.yaml").write_text(yaml.safe_dump(recipe, sort_keys=False))
    (args.run_dir / "srtslurm.yaml").write_text(
        yaml.safe_dump(cluster, sort_keys=False)
    )
    (args.run_dir / "sa-flow-contract.json").write_text(
        json.dumps(
            {
                "source": contract["source"],
                "case": contract["case_name"],
                "srtctl_sha": "df706bc365e00bea74e5babe87b4049f624341ea",
                "router_runtime": router_runtime,
                "execution_backend": (
                    "gpu-pool" if os.environ.get("GPOOL_LEASE") else "Spur"
                ),
                "gpu_pool_lease": os.environ.get("GPOOL_LEASE"),
                "shared_sa_path": "schema loader, runtime, SweepOrchestrator, worker/frontend builders, port allocator, readiness, registry cleanup",
                "differences": [
                    (
                        "gpool lease and direct SSH replace SA sbatch"
                        if os.environ.get("GPOOL_LEASE")
                        else "Spur allocation replaces SA sbatch"
                    ),
                    "srtctl DockerLauncher replaces pyxis/enroot",
                    "custom functionality workload replaces AgentX performance/eval",
                    "power collection disabled for functional-only validation",
                ],
                "budget": "90 minutes including a pinned-source router build, up to 45 minutes model readiness, functionality/capture and cleanup; no performance window",
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
