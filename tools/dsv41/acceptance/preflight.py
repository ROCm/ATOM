"""Validate the installed entry and exact server CLI before any weight loading."""

import argparse
import hashlib
import inspect
import json
import os
from dataclasses import fields
from pathlib import Path


def main():
    import run_step12_performance_r6 as benchmark_entry
    import torch
    from aiter import dtypes
    from run_rebase_graph_coverage_r6 import SERVER_COMMAND

    assert callable(benchmark_entry.main)
    assert benchmark_entry.original.SERVER_COMMAND is SERVER_COMMAND
    assert benchmark_entry.original.URL == "http://127.0.0.1:19125"

    import atom
    from atom.config import Config, CUDAGraphMode
    from atom.model_engine.arg_utils import EngineArgs
    from atom.model_engine.model_runner import ModelRunner
    from atom.model_loader import loader
    from atom.models.deepseek_v4 import make_v4_quant_config
    from atom.models.deepseek_v41 import runtime
    from atom.models.deepseek_v41.bounded_replay import late_layer_start

    candidate = os.environ["DSV41_OWNER_SERVING"] == "1"
    root = Path(os.environ["DSV41_REBASE_ATOM_SOURCE"])
    assert Path(atom.__file__).is_relative_to(root)
    assert Path(runtime.__file__).is_relative_to(root)
    assert os.environ["ATOM_DSV41_PREFILL_QUANTIZED_FFN"] == str(int(candidate))
    assert os.environ["DSV41_REBASE_SERVICE_SERVER"] == "1"
    assert ModelRunner._step12_prefill_ffn_route.enabled is candidate
    assert getattr(loader.load_weights_into_model, "_step12_loader_phase_timing", False)
    assert ModelRunner._rebase_graph_audit
    argv = (
        SERVER_COMMAND[:]
        if candidate
        else SERVER_COMMAND[: SERVER_COMMAND.index("--method")]
    )
    parser = EngineArgs.add_cli_args(argparse.ArgumentParser())
    for flag, options in (
        ("--host", {"type": str}),
        ("--server-port", {"type": int}),
        ("--disable-uvicorn-access-log", {"action": "store_true"}),
    ):
        if flag not in parser._option_string_actions:
            parser.add_argument(flag, **options)
    args = EngineArgs.from_cli_args(parser.parse_args(argv[3:]))
    kwargs = args._get_engine_kwargs()
    names = {field.name for field in fields(Config)}
    config = Config(args.model, **{k: v for k, v in kwargs.items() if k in names})
    assert config.enable_expert_parallel is candidate
    assert argv.count("--enable-expert-parallel") == int(candidate)
    assert ModelRunner._step12_owner_ep_audit["enabled"] is candidate
    assert ModelRunner._step12_owner_ep_audit["checks"] == 0
    assert config.tensor_parallel_size == 4 and config.max_num_seqs == 256
    assert config.compilation_config.level == 3
    assert config.compilation_config.cudagraph_mode == CUDAGraphMode.FULL
    assert config.kv_cache_dtype == "bf16" and config.index_cache_dtype == "fp4"
    spec = config.speculative_config
    assert (spec.num_speculative_tokens if spec else 0) == (5 if candidate else 0)
    quant = make_v4_quant_config(
        config.hf_config, online_quant_config=config.online_quant_config
    )
    routed = quant.get_layer_quant_config("layers.8.ffn.experts")
    shared = quant.get_layer_quant_config("layers.8.ffn.shared_experts.gate_up_proj")
    assert routed.quant_dtype == dtypes.fp4x2 and shared.quant_dtype == dtypes.fp8
    schema = None
    if candidate:
        from atom.models.deepseek_v41 import prefill_ffn

        assert hasattr(prefill_ffn.prequantized_owned_ffn, "__wrapped__")
        schema = str(torch.ops.aiter.v41_owned_prequantized_moe.default._schema)
        assert "original_rows" in schema and "layer_name" in schema
    flags = (
        "ATOM_DSV41_EP_PROFILED_PLACEMENT",
        "ATOM_DSV41_PREFILL_LOCAL_ROUTER",
        "ATOM_DSV41_PREFILL_ROUTE_PACKET",
    )
    assert all(os.environ[key] == str(int(candidate)) for key in flags)
    assert ModelRunner._step12_placement_service_audit == {
        "enabled": candidate,
        "checks": 0,
        "actual": None,
    }
    import sys

    support = Path(__file__).resolve().parent
    legacy_imports = [
        str(m.__file__)
        for m in list(sys.modules.values())
        if getattr(m, "__file__", None)
        and "/validation/structure_ceiling_20261024/" in str(m.__file__)
    ]
    assert not legacy_imports, legacy_imports
    report = {
        "status": "actual_entry_native_CLI_config_quant_and_hooks_before_weights_PASS",
        "candidate": candidate,
        "fixed_package_enabled": candidate,
        "fixed_placement_audit_installed": True,
        "support": str(support),
        "legacy_experiment_imports": legacy_imports,
        "benchmark_import_closure": "PASS",
        "subject": str(root),
        "server_command": argv,
        "TP": config.tensor_parallel_size,
        "EP_enabled": config.enable_expert_parallel,
        "loaded_EP_audit_installed": True,
        "level": config.compilation_config.level,
        "cudagraph_mode": str(config.compilation_config.cudagraph_mode),
        "speculative_tokens": spec.num_speculative_tokens if spec else 0,
        "expected_prefill_layers": (
            list(range(2, late_layer_start(config.hf_config) - 1)) if candidate else []
        ),
        "routed_quant": str(routed),
        "shared_quant": str(shared),
        "schema": schema,
        "source_sha256": {
            str(Path(inspect.getfile(module))): hashlib.sha256(
                Path(inspect.getfile(module)).read_bytes()
            ).hexdigest()
            for module in (runtime, loader)
        },
        "scope": "Actual installed entry and exact server EngineArgs->Config path. No model construction, weights, full Inductor stage, requests or numerical acceptance.",
    }
    path = Path(os.environ["DSV41_OWNER_SERVING_OUTPUT"]) / (
        "entry_preflight_" + os.environ["DSV41_STEP12_PREFLIGHT_LABEL"] + ".json"
    )
    path.open("x").write(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
