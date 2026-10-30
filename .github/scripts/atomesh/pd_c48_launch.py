"""Pinned C48 diagnostic launcher; explicit recipe AL is never recomputed."""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.metadata
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path


def draft_path(model: str) -> Path:
    return Path(model).parent.parent / "Inferact/Kimi-K3-DSpark"


def render(config: dict, role: str, model: str, host: str, proxy: str,
           port: int, discovery: int, trace_dir: str | None = None,
           lmcache_port: int | None = None,
           phase: str = "combined") -> tuple[list[str], dict[str, str]]:
    args = copy.deepcopy(config["roles"][role]["args"])
    args["speculative-config"]["model"] = str(draft_path(model))
    if phase == "eval":
        spec = args["speculative-config"]
        spec["rejection_sample_method"] = "standard"
        spec.pop("synthetic_acceptance_length", None)
        spec.pop("synthetic_acceptance_rates", None)
    transfer = args["kv-transfer-config"]
    connectors = transfer.get("kv_connector_extra_config", {}).get("connectors", [transfer])
    for connector in connectors:
        if connector["kv_connector"] == "MoRIIOConnector":
            if connector["kv_connector_extra_config"].get("read_mode") is not True:
                raise ValueError("C48 Kimi-K3 requires explicit MoRIIO read_mode=true")
            connector["kv_connector_extra_config"].update(
                proxy_ip=proxy, proxy_ping_port=discovery, http_port=port)
        elif connector["kv_connector"] == "LMCacheMPConnector":
            if lmcache_port is None:
                raise ValueError("C48 LMCacheMPConnector requires the sidecar --lmcache-port")
            connector["kv_connector_extra_config"]["lmcache.mp.port"] = lmcache_port
    args.update(host="0.0.0.0", port=port, **{"worker-cls": "pd_c48_worker.ManifestWorker"})
    if "profiler-config" in args:
        if not trace_dir or role != "prefill":
            raise ValueError("C48 profiler requires an explicit Prefill trace directory")
        args["profiler-config"]["torch_profiler_dir"] = trace_dir
    argv = [sys.executable, "-m", "vllm.entrypoints.cli.main", "serve", model]
    for key, value in args.items():
        if isinstance(value, bool):
            argv.append("--" + ("" if value else "no-") + key)
        else:
            argv += ["--" + key, json.dumps(value, separators=(",", ":"))
                     if isinstance(value, (dict, list)) else str(value)]
    env = {str(k): str(v) for k, v in config["roles"][role]["env"].items()}
    # TensorWave exposes ionic_* HCAs; bootstrap follows the allocated host IP.
    # The Slurm launcher supplies these platform values to every case equally.
    for key in ("NCCL_IB_HCA", "NCCL_SOCKET_IFNAME", "GLOO_SOCKET_IFNAME"):
        env.pop(key, None)
    env["GLOO_SOCKET_IFNAME"] = os.environ.get("MORI_SOCKET_IFNAME", "")
    env["VLLM_HOST_IP"] = host
    env["PYTHONPATH"] = str(Path(__file__).parent) + os.pathsep + os.environ.get("PYTHONPATH", "")
    env["C48_DIAG_ROLE"] = role
    env["C48_DIAG_PTPC"] = str(int("quantization-config" in args))
    env["C48_DIAG_AL"] = str(args["speculative-config"].get("synthetic_acceptance_length", ""))
    env["C48_DIAG_DRAFT_TOKENS"] = str(args["speculative-config"]["num_speculative_tokens"])
    return argv, env


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--role", choices=["prefill", "decode"], required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--host", required=True)
    parser.add_argument("--proxy", required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--discovery", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--lmcache-port", type=int)
    parser.add_argument("--execute", action="store_true")
    options = parser.parse_args()
    config = json.loads(options.config.read_text())
    argv, additions = render(config, options.role, options.model, options.host,
                             options.proxy, options.port, options.discovery,
                             str(options.output.parent.parent / "traces" / options.role),
                             options.lmcache_port, os.environ.get("ATOMESH_EXECUTION_PHASE", "combined"))
    additions["C48_DIAG_DIR"] = str(options.output.parent / "model-manifests")
    record = {"config": str(options.config), "config_sha256": hashlib.sha256(options.config.read_bytes()).hexdigest(),
              "role": options.role, "argv": argv, "environment_overrides": additions,
              "diagnostic": "Preserve explicit synthetic AL; verify the recorded read-only RDMA startup source mount"}
    record["observation_module_sha256"] = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in Path(__file__).parent.glob("pd_c48_*.py")
    }
    options.output.parent.mkdir(parents=True, exist_ok=True)
    if options.execute:
        import vllm
        from pd_c48_port_patch import verify_port_patch
        record["vllm_file"] = vllm.__file__
        record["versions"] = {}
        for name in ("vllm", "torch", "triton", "aiter", "mori", "amd-aiter", "amd_mori", "fastsafetensors", "lmcache",
                     "mooncake-transfer-engine-rocm"):
            try:
                record["versions"][name] = importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError:
                record["versions"][name] = None
        engine = "3966924d5" if "-n3966-" in config["name"] else "81198e97b"
        assert engine in record["versions"]["vllm"], record["versions"]
        record["engine_startup_patch"] = verify_port_patch(Path(vllm.__file__).parent)
        from pd_c48_quant_patch import verify_quant_patch
        record["online_quant_loading_patch"] = verify_quant_patch(Path(vllm.__file__).parent)
        record["loading_memory_evidence"] = {"interval_seconds": 2, "active_only_during_load_model": True}
        record["gpu_state_before_worker_start"] = {}
        command = shutil.which("rocm-smi")
        if command:
            try:
                result = subprocess.run([command, "--showmeminfo", "vram", "--showpids", "--json"],
                                        text=True, capture_output=True, timeout=20, check=False)
                record["gpu_state_before_worker_start"] = {"returncode": result.returncode,
                    "stdout": result.stdout[:65536], "stderr": result.stderr[:4096]}
            except subprocess.TimeoutExpired:
                record["gpu_state_before_worker_start"] = {"timeout": True}
        else:
            record["gpu_state_before_worker_start"] = {"unavailable": "rocm-smi"}
        record["model_files"] = {}
        for root in (Path(options.model), draft_path(options.model)):
            assert (root / "config.json").is_file(), root
            record["model_files"][str(root)] = {
                p.name: {"bytes": p.stat().st_size,
                         "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
                for p in root.iterdir() if p.is_file() and p.suffix == ".json"}
            metadata = {"config": json.loads((root / "config.json").read_text()), "shards": {}}
            for checkpoint in sorted(root.glob("*.safetensors")):
                with checkpoint.open("rb") as handle:
                    header_size = int.from_bytes(handle.read(8), "little")
                    if header_size > 128 * 1024 * 1024:
                        raise ValueError(f"Invalid safetensors header size: {checkpoint.name}")
                    header = json.loads(handle.read(header_size))
                metadata["shards"][checkpoint.name] = {"bytes": checkpoint.stat().st_size, "header": header}
            metadata_path = options.output.parent / (root.name + "-checkpoint-metadata.json")
            metadata_path.write_text(json.dumps(metadata) + "\n")
            record["model_files"][str(root)]["checkpoint_metadata_artifact"] = str(metadata_path)
        record["platform_environment"] = {key: value for key, value in os.environ.items()
            if key.startswith(("NCCL_", "MORI_", "HSA_", "HIP_", "CUDA_", "GLOO_", "MC_", "MOONCAKE_", "VLLM_MOONCAKE_"))}
    options.output.write_text(json.dumps(record, indent=2) + "\n")
    print("[C48] " + json.dumps(record), flush=True)
    if options.execute:
        env = os.environ.copy()
        env.update(additions)
        os.execvpe(argv[0], argv, env)


if __name__ == "__main__":
    main()
