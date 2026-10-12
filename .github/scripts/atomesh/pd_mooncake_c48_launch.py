"""Launch the archived C48 engine arguments with Mooncake transport/offload."""
import argparse
import copy
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys


def render(config, role, model, host, port):
    args = copy.deepcopy(config["roles"][role]["args"])
    args["speculative-config"]["model"] = str(
        Path(model).parent.parent / "Inferact/Kimi-K3-DSpark"
    )
    args.update(host="0.0.0.0", port=port)
    argv = [sys.executable, "-m", "vllm.entrypoints.cli.main", "serve", model]
    for key, value in args.items():
        if isinstance(value, bool):
            argv.append("--" + ("" if value else "no-") + key)
        else:
            argv += ["--" + key, json.dumps(value, separators=(",", ":"))
                     if isinstance(value, (dict, list)) else str(value)]
    env = {str(k): str(v) for k, v in config["roles"][role]["env"].items()}
    # Device and interface names follow the actual TensorWave allocation.
    for key in ("NCCL_IB_HCA", "NCCL_SOCKET_IFNAME", "GLOO_SOCKET_IFNAME"):
        env.pop(key, None)
    env["GLOO_SOCKET_IFNAME"] = os.environ.get("MORI_SOCKET_IFNAME", "")
    env["VLLM_HOST_IP"] = host
    return argv, env


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--role", choices=["prefill", "decode"], required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--host", required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    options = parser.parse_args()
    config = json.loads(options.config.read_text())
    argv, additions = render(config, options.role, options.model, options.host, options.port)
    record = {
        "config_sha256": hashlib.sha256(options.config.read_bytes()).hexdigest(),
        "role": options.role, "argv": argv, "environment_overrides": additions,
    }
    if options.execute:
        import vllm
        version = importlib.metadata.version("vllm")
        assert "81198e97b" in version, version
        assert vllm.__file__.startswith("/tmp/mooncake-c48-site/"), vllm.__file__
        record.update(vllm_version=version, vllm_file=vllm.__file__)
        record["model_files"] = {}
        for root in (Path(options.model), Path(options.model).parent.parent / "Inferact/Kimi-K3-DSpark"):
            assert (root / "config.json").is_file(), root
            record["model_files"][str(root)] = {
                p.name: {"bytes": p.stat().st_size,
                         "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
                for p in root.iterdir() if p.is_file() and p.suffix == ".json"
            }
        record["platform_environment"] = {
            k: v for k, v in os.environ.items()
            if k.startswith(("NCCL_", "GLOO_", "HSA_", "HIP_", "MC_", "MOONCAKE_", "VLLM_"))
            and not any(word in k for word in ("API_KEY", "TOKEN", "PASSWORD", "SECRET"))
        }
    options.output.parent.mkdir(parents=True, exist_ok=True)
    options.output.write_text(json.dumps(record, indent=2) + "\n")
    print("[Mooncake C48] " + json.dumps(record), flush=True)
    if options.execute:
        env = os.environ.copy()
        env.update(additions)
        os.execvpe(argv[0], argv, env)


if __name__ == "__main__":
    main()
