"""The M3.1 image CI case: the cell CASE of models_atomesh.yaml, served with
USE_ATOMESH_ENTRYPOINTS=1 as atomesh-accuracy-validation.yaml serves a model,
and scored by lm_eval's API client as pd_server_atom.sh's run_eval scores it.

`ci_case.py launch <model path> <node>`, run by checks/ci_case.env where
accept.sh runs, prints the bash that sets the profile's ATOM_ARGS and ATOM_ENV
to the cell's aggregated server. Run with no argument, in the client container,
it runs the cell's eval task against ENDPOINT and passes when the score meets
the cell's threshold.
"""

import importlib.util
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path

from mmstar import env, finish, run_check

ROOT = Path(__file__).resolve().parents[4]
MODEL = "MiniMax-M3.1-Flash-MXFP4"
SUITE = "accuracy"
CASE = "minimax-m31-flash-mxfp4-agg-tp4-mmmu"

# What expanding the file reads from the CI environment. Only the model root
# reaches the server; the rest names Slurm resources this run does not use.
CI_ENV = {
    "ATOMESH_MODEL_ROOT": "/share_nfs/models",
    "ATOMESH_SLURM_ACCOUNT": "",
    "ATOMESH_SLURM_PARTITION": "",
    "ATOMESH_SLURM_SUBMIT_RUNNER": "",
    "ATOMESH_LOG_ROOT": "",
    "ATOMESH_PD_RANK_MAPPING_POLICY": "none",
}

# server_args keys and the flags pd_server_atom.sh gives them when set.
FLAGS = {
    "max_model_len": "--max-model-len",
    "max_num_batched_tokens": "--max-num-batched-tokens",
    "online_quant_config": "--online_quant_config",
    "hf_overrides": "--hf-overrides",
    "method": "--method",
    "draft_model": "--draft-model",
    "num_speculative_tokens": "--num-speculative-tokens",
}
# Keys the aggregated server takes apart from FLAGS.
OTHER_KEYS = {
    "trust_remote_code",
    "kv_cache_dtype",
    "block_size",
    "gpu_memory_utilization",
    "enable_prefix_caching",
    "max_num_seqs",
    "extra_args",
}
NO_CUDAGRAPH = {"", "none", "false", "off", "disabled"}


def cell(node=""):
    """The case's cell, expanded by pd_matrix.py as the benchmark workflow
    expands it, on node."""
    path = ROOT / ".github/scripts/atomesh/pd_matrix.py"
    spec = importlib.util.spec_from_file_location("pd_matrix", path)
    pd_matrix = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pd_matrix)
    for name, value in CI_ENV.items():
        os.environ.setdefault(name, value)
    os.environ["ATOMESH_SINGLE_NODE"] = node or "localhost"
    (found,) = pd_matrix.build_cells(
        pd_matrix.load_config(ROOT / ".github/benchmark/models_atomesh.yaml"),
        suite=SUITE,
        model_filter={MODEL},
        case_filter={CASE},
        benchmark_kind_filter=None,
        override_image=None,
        override_benchmark_concurrency=None,
        override_eval_concurrency=None,
    )
    return found


def server(cell, node):
    """The aggregated server's environment and openai_server arguments, as
    pd_submit.sh and pd_server_atom.sh's start_aggregated build them."""
    args = cell["server_args"]
    unknown = set(args) - set(FLAGS) - OTHER_KEYS
    if unknown:
        raise ValueError(f"{cell['name']}: no launch for server_args {sorted(unknown)}")
    decode = cell["service"]["decode"]
    cudagraph = str(decode.get("cudagraph", ""))
    if cudagraph.lower() not in NO_CUDAGRAPH or decode.get("cudagraph_mode"):
        raise ValueError(f"{cell['name']}: no launch for decode cudagraph settings")
    argv = [
        "--model",
        cell["model_path"],
        "--host",
        "0.0.0.0",
        "--trust-remote-code",
        "--kv_cache_dtype",
        str(args.get("kv_cache_dtype", "fp8")),
        "--block-size",
        str(args.get("block_size", 16)),
        "--gpu-memory-utilization",
        str(args.get("gpu_memory_utilization", 0.85)),
    ]
    if not args.get("enable_prefix_caching", False):
        argv.append("--no-enable_prefix_caching")
    for key, flag in FLAGS.items():
        if args.get(key) not in (None, ""):
            argv += [flag, str(args[key])]
    argv += [
        "-tp",
        str(decode.get("tp", 8)),
        "--decode-context-parallel-size",
        str(decode.get("dcp", 1)),
    ]
    if decode.get("enable_dp_attention"):
        argv.append("--enable-dp-attention")
    argv += ["--max-num-seqs", str(args.get("max_num_seqs", 256))]
    # pd_server_atom.sh expands these unquoted: split on whitespace, not shlex.
    argv += str(args.get("extra_args", "")).split()
    argv += str(decode.get("extra_args", "")).split()
    environ = {**cell["env"]["common"], **cell["env"]["decode"]}
    environ = {k: v.replace("${ROLE_IP}", node) for k, v in environ.items()}
    environ["USE_ATOMESH_ENTRYPOINTS"] = "1"
    return [f"{k}={v}" for k, v in environ.items()], argv


def launch(model_path, node):
    found = cell(node)
    if found["model_path"] != model_path:
        raise SystemExit(
            f"{CASE} serves {found['model_path']}, the profile mounts {model_path}"
        )
    environ, argv = server(found, node)
    print("ATOM_ENV=(" + " ".join(map(shlex.quote, environ)) + ")")
    print("ATOM_ARGS=(" + " ".join(map(shlex.quote, argv)) + ")")


def lm_eval_args(cell, base_url, output):
    """The lm_eval command run_eval runs for the cell at its one concurrency."""
    acc = cell["accuracy"]
    (concurrency,) = acc["concurrency"]
    model_args = (
        f"model={cell['model_path']},base_url={base_url}/v1/{acc['endpoint']},"
        f"num_concurrent={concurrency},max_retries=3"
    )
    if acc["max_gen_toks"] is not None:
        model_args += f",max_gen_toks={acc['max_gen_toks']}"
    if acc["model_args"]:
        model_args += f",{acc['model_args']}"
    args = ["lm_eval", "--model", acc["model_type"], "--model_args", model_args]
    args += ["--tasks", acc["task"], "--num_fewshot", str(acc["fewshot"])]
    limit = os.environ.get("LIMIT") or acc["limit"]
    if limit:
        args += ["--limit", str(limit)]
    if acc["apply_chat_template"]:
        args.append("--apply_chat_template")
    if acc["gen_kwargs"]:
        args += ["--gen_kwargs", acc["gen_kwargs"]]
    return args + ["--log_samples", "--output_path", str(output)]


def main():
    found = cell()
    acc = found["accuracy"]
    output = Path(env("RUN_DIR"), "lm_eval")
    with open(f"{output}.log", "w") as log:
        subprocess.run(
            lm_eval_args(found, env("ENDPOINT"), output),
            stdout=log,
            stderr=subprocess.STDOUT,
            check=True,
        )
    results = json.loads(max(output.rglob("results_*.json")).read_text())
    score = results["results"][acc["task"]][acc["metric"]]
    threshold = float(acc["threshold"])
    result = {"cell": CASE, "task": acc["task"], "score": score, "threshold": threshold}
    failures = []
    if score < threshold:
        failures.append(f"score {score:.4f} is below the cell's {threshold:.4f}")
    finish(result, failures)


if __name__ == "__main__":
    if sys.argv[1:2] == ["launch"]:
        launch(*sys.argv[2:])
    else:
        run_check(main)
