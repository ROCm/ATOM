"""Run the public DSV4.1 random profile with optional FP8 owner execution.

`plan` needs only Python's standard library. `preflight` parses the real
EngineArgs -> Config and V4.1 validation without constructing a model. `serve`
performs that preflight before exec'ing the native module (preserving spawn).
`benchmark` is the native client, against an already running server.
`--execution fp8-owner` selects the migrated EP1/full path explicitly; the
public deployment and request profile remain the same. It does not reproduce
the controlled performance harness's separate warm593 and route observers.
"""

import argparse
import hashlib
import json
import os
import subprocess
import sys
from dataclasses import fields
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
PROFILE = HERE / "dashboard_profile.json"
# These selectors can silently carry a previous experiment into another
# Python process. Infrastructure such as visible devices and ROCm paths stays.
MODEL_ENV_PREFIXES = ("DSV41_", "ATOM_", "AITER_")
CACHE_ENV = {
    "VLLM_CACHE_ROOT": "atom",
    "XDG_CACHE_HOME": "xdg",
    "AITER_JIT_DIR": "aiter_jit",
    "AITER_ROOT_DIR": "aiter_cpp",
    "TRITON_CACHE_DIR": "triton",
    "FLYDSL_RUNTIME_CACHE_DIR": "flydsl",
    "TORCHINDUCTOR_CACHE_DIR": "inductor",
    "TORCH_EXTENSIONS_DIR": "extensions",
    "AMD_COMGR_CACHE_DIR": "comgr",
}


def load_profile():
    return json.loads(PROFILE.read_text())


def cli_options(options):
    argv = []
    for name, value in options.items():
        # False engine switches have store_true/default-False semantics;
        # the real preflight checks those defaults instead of inventing a
        # --no-enable-expert-parallel flag that EngineArgs does not provide.
        if value is False or value is None:
            continue
        argv.append("--" + name.replace("_", "-"))
        if value is not True:
            argv.append(str(value))
    return argv


def clean_environment(inherited, model_source, aiter_source, cache_dir, profile):
    removed = sorted(k for k in inherited if k.startswith(MODEL_ENV_PREFIXES))
    env = {k: v for k, v in inherited.items() if k not in removed}
    # A preload can change both compilation and execution outside Python.
    # Preserve it only through a future explicit, separately recorded profile.
    if env.pop("LD_PRELOAD", None) is not None:
        removed.append("LD_PRELOAD")
    env.update(profile["environment"])
    env["PYTHONPATH"] = os.pathsep.join((str(model_source), str(aiter_source)))
    env.update({key: str(cache_dir / leaf) for key, leaf in CACHE_ENV.items()})
    return env, sorted(removed)


def owner_environment(args, env, profile):
    """Only the service gets these switches; configuration probes stay native."""
    if args.execution != "fp8-owner" or args.action == "benchmark":
        return env
    output = args.output / "owner"
    return dict(
        env,
        **profile["owner_execution"]["environment"],
        PYTHONPATH=os.pathsep.join(
            (str(args.model_source / "tools/dsv41/owner_bootstrap"), env["PYTHONPATH"])
        ),
        DSV41_OWNER_SERVING_OUTPUT=str(output),
        DSV41_OWNER_SERVING_CONTROL=str(output / "capacity.txt"),
        DSV41_OWNER_SERVING_COMPILE_CACHE=str(args.cache_dir / "atom"),
    )


def commands(args, profile):
    server = [
        args.python,
        "-m",
        "atom.entrypoints.openai_server",
        "--model",
        args.model,
        *cli_options(profile["engine"]),
        "--host",
        args.host,
        "--server-port",
        str(args.port),
    ]
    client_host = "127.0.0.1" if args.host == "0.0.0.0" else args.host
    if ":" in client_host and not client_host.startswith("["):
        client_host = "[" + client_host + "]"
    client = [
        args.python,
        "-m",
        "atom.benchmarks.benchmark_serving",
        "--model",
        args.model,
        "--base-url",
        f"http://{client_host}:{args.port}",
        *cli_options(profile["benchmark"]),
        "--save-result",
        "--result-dir",
        str(args.output),
        "--result-filename",
        "benchmark.json",
    ]
    return server, client


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_identity(path, files):
    head = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
    )
    return {
        "root": str(path),
        "head": head.stdout.strip() if head.returncode == 0 else None,
        # Explicit entry files, not a whole-tree census or source freeze.
        "entry_sha256": {name: digest(path / name) for name in files},
    }


def manifest(args, profile, env, removed):
    server, client = commands(args, profile)
    model_config = Path(args.model) / "config.json"
    return {
        "status": "prepared_not_executed",
        "profile": profile,
        "profile_sha256": digest(PROFILE),
        "execution": args.execution,
        "runner_sha256": digest(Path(__file__)),
        "atom": source_identity(
            args.model_source,
            (
                "atom/config.py",
                "atom/model_engine/arg_utils.py",
                "atom/models/deepseek_v41/config.py",
                "atom/models/deepseek_v41/runtime.py",
                "atom/models/deepseek_v41/model.py",
                "atom/models/deepseek_v41/moe.py",
                "atom/models/deepseek_v41/mono/config.py",
                "atom/model_ops/fused_moe/comm_fused_moe.py",
                "atom/benchmarks/benchmark_serving.py",
            )
            + (
                (
                    "atom/models/deepseek_v41/owner_service_contract.py",
                    "atom/models/deepseek_v41/owner_service/__init__.py",
                    "atom/models/deepseek_v41/owner_service/owner_serving.py",
                    "atom/models/deepseek_v41/owner_service/owner_cost_attention_chain.py",
                    "atom/models/deepseek_v41/owner_service/owner_head_attention.py",
                    "atom/models/deepseek_v41/owner_service/lifecycle.py",
                    "atom/model_ops/deepseek_v41/owner_attention.py",
                    "tools/dsv41/owner_bootstrap/sitecustomize.py",
                    "tools/dsv41/owner_bootstrap/owner_bootstrap_scope.py",
                )
                if args.execution == "fp8-owner"
                else ()
            ),
        ),
        "aiter": source_identity(args.aiter_source, ("aiter/__init__.py",)),
        "model": args.model,
        "checkpoint_config_sha256": (
            digest(model_config) if model_config.is_file() else None
        ),
        "server_command": server,
        "benchmark_command": client,
        "environment": {
            k: v
            for k, v in env.items()
            if k in (*profile["environment"], "PYTHONPATH", *CACHE_ENV)
            or k.startswith(MODEL_ENV_PREFIXES)
        },
        "removed_environment_names": removed,
        "scope": "Configuration adaptation; no loaded model, actual dispatch, GPU, performance or accuracy evidence.",
    }


def write_new(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        stream.write(json.dumps(value, indent=2) + "\n")


def preflight(args, profile):
    """Use installed production APIs, without weights, a runner or GPU kernels."""
    import torch

    import atom
    from atom.config import Config
    from atom.model_engine.arg_utils import EngineArgs
    from atom.models.deepseek_v41.config import (
        build_attention_topology,
        validate_runtime_config,
    )
    from atom.utils.arg_parser import FlexibleArgumentParser

    if not Path(atom.__file__).resolve().is_relative_to(args.model_source):
        raise ValueError("ATOM was imported outside --model-source")
    if torch.cuda.is_initialized():
        raise RuntimeError("GPU was initialized before configuration preflight")
    engine = profile["engine"]
    parser = EngineArgs.add_cli_args(FlexibleArgumentParser())
    parsed = EngineArgs.from_cli_args(
        parser.parse_args(["--model", args.model, *cli_options(engine)])
    )
    mismatches = {
        key: {"expected": value, "actual": getattr(parsed, key)}
        for key, value in engine.items()
        if getattr(parsed, key) != value
    }
    if mismatches:
        raise ValueError(f"EngineArgs differs from profile: {mismatches}")
    kwargs = parsed._get_engine_kwargs()
    names = {f.name for f in fields(Config) if f.init}
    config = Config(args.model, **{k: v for k, v in kwargs.items() if k in names})
    if config.hf_config.architectures != ["DeepseekV41ForCausalLM"]:
        raise ValueError("The profile requires DeepSeek-V4.1")
    validate_runtime_config(config)
    spec = config.speculative_config
    if (
        spec is None
        or spec.method != "dspark"
        or spec.num_speculative_tokens != 5
        or spec.synthetic_acceptance_rates is not None
        or config.enable_expert_parallel
        or config.enable_dp_attention
        or config.parallel_config.data_parallel_size != 1
        or config.kv_cache_dtype != "fp8"
        or config.index_cache_dtype != "fp4"
        or config.tensor_parallel_size != 4
        or config.max_num_seqs != 512
        or config.compilation_config.level != 3
        or config.compilation_config.cudagraph_mode.name != "FULL"
    ):
        raise ValueError("Resolved Config differs from the TP4/FP8/real-DSpark profile")
    if any(k.startswith("DSV41_") for k in os.environ):
        raise ValueError("Owner experiment environment leaked into the preflight")
    if args.execution == "fp8-owner":
        from atom.models.deepseek_v41.owner_service_contract import (
            owner_prefill_enabled,
        )

        if not owner_prefill_enabled(config, owner_service=True, retained_prefill=True):
            raise ValueError("Resolved Config does not admit FP8 owner prefill")
    if torch.cuda.is_initialized():
        raise RuntimeError("Configuration preflight unexpectedly initialized GPU")
    topology = build_attention_topology(config.hf_config)
    return {
        "status": "actual_EngineArgs_Config_V41_validation_PASS",
        "engine_args": {key: getattr(parsed, key) for key in engine},
        "resolved": {
            "TP": config.tensor_parallel_size,
            "EP_enabled": config.enable_expert_parallel,
            "DP": config.parallel_config.data_parallel_size,
            "KV": config.kv_cache_dtype,
            "index": config.index_cache_dtype,
            "speculative_method": spec.method,
            "speculative_tokens": spec.num_speculative_tokens,
            "synthetic_acceptance_rates": spec.synthetic_acceptance_rates,
            "kv_cache_block_size": config.kv_cache_block_size,
            "max_model_len": config.max_model_len,
            "capture_sizes": config.capture_sizes,
            "backbone_layers": config.hf_config.num_hidden_layers,
            "topology_layers_including_draft": len(topology),
        },
        "cuda_initialized": torch.cuda.is_initialized(),
        "weights_loaded": False,
        "scope": "Configuration only. Mono and comm-fused selection still require actual loaded modules and steps.",
    }


def parse_args(argv=None):
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("action", choices=("plan", "preflight", "serve", "benchmark"))
    parser.add_argument("--model", default=load_profile()["model"])
    parser.add_argument("--model-source", type=Path, default=REPO)
    parser.add_argument("--aiter-source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument(
        "--execution", choices=("native", "fp8-owner"), default="native"
    )
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=19126)
    parser.add_argument(
        "--_preflight-worker", action="store_true", help=argparse.SUPPRESS
    )
    args = parser.parse_args(argv)
    if not 1 <= args.port <= 65535:
        parser.error("--port must be in [1, 65535]")
    for key in ("model_source", "aiter_source", "output"):
        setattr(args, key, getattr(args, key).resolve())
    args.cache_dir = (args.cache_dir or args.output / "cache").resolve()
    return args


def main(argv=None):
    args = parse_args(argv)
    profile = load_profile()
    if args._preflight_worker:
        if args.action != "preflight":
            raise ValueError("Internal worker requires preflight")
        report = preflight(args, profile)
        write_new(args.output / "preflight.json", report)
        print(json.dumps(report), flush=True)
        return
    env, removed = clean_environment(
        os.environ, args.model_source, args.aiter_source, args.cache_dir, profile
    )
    service_env = owner_environment(args, env, profile)
    record = manifest(args, profile, service_env, removed)
    if args.action == "plan":
        write_new(args.output / "plan.json", record)
        print(json.dumps(record, indent=2))
        return
    # Each action has its own immutable receipt. An attempt must use a fresh
    # output directory rather than overwrite evidence from an earlier action.
    write_new(args.output / (args.action + "_launch.json"), record)
    if args.action in ("serve", "preflight"):
        worker = [
            args.python,
            str(Path(__file__).resolve()),
            "preflight",
            "--_preflight-worker",
            "--model",
            args.model,
            "--model-source",
            str(args.model_source),
            "--aiter-source",
            str(args.aiter_source),
            "--output",
            str(args.output),
            "--execution",
            args.execution,
        ]
        # Config may query the GPU when devices are visible. Keep this probe
        # CPU-only even when serve() is launched on a GPU node; the service
        # below must retain the original device selection.
        probe_env = dict(
            env,
            ROCR_VISIBLE_DEVICES="",
            HIP_VISIBLE_DEVICES="",
            CUDA_VISIBLE_DEVICES="",
        )
        subprocess.run(worker, env=probe_env, cwd=args.model_source, check=True)
    if args.action == "preflight":
        return
    if args.action == "benchmark" and (args.output / "benchmark.json").exists():
        raise FileExistsError(args.output / "benchmark.json")
    if args.action == "serve" and args.execution == "fp8-owner":
        output = args.output / "owner"
        output.mkdir()
        with (output / "capacity.txt").open("x") as stream:
            stream.write(profile["owner_execution"]["capacity"] + "\n")
    server, client = commands(args, profile)
    command = server if args.action == "serve" else client
    os.chdir(args.model_source)
    os.execvpe(command[0], command, service_env)


if __name__ == "__main__":
    main()
