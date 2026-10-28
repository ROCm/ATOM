"""Run the source-bound DSV41 recipe without a historical experiment directory."""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]


def environment(args):
    env = dict(os.environ)
    profile = json.loads((HERE / "r13_profile.json").read_text())
    env.update(profile["environment"])
    enabled = str(int(args.mode == "candidate"))
    for key in (
        "DSV41_OWNER_SERVING",
        "DSV41_STEP12_OWNER_EP4",
        "ATOM_DSV41_EP_PROFILED_PLACEMENT",
        "ATOM_DSV41_PREFILL_LOCAL_ROUTER",
        "ATOM_DSV41_PREFILL_ROUTE_PACKET",
        "ATOM_DSV41_PREFILL_QUANTIZED_FFN",
        "DSV41_OWNER_STABLE_GRAPHS",
        "DSV41_OWNER_PACKET_PUBLICATION",
        "DSV41_OWNER_ACCEPTED_PREFILL",
        "DSV41_OWNER_SELECTIVE_RECONCILIATION",
        "ATOM_DSV41_ROW_SHARDED_MHC",
    ):
        env[key] = enabled
    env["ATOM_DSV41_PREFILL_TAIL_MODE"] = "settled" if enabled == "1" else "off"
    output = args.output.resolve()
    source = args.model_source.resolve()
    env.update(
        DSV41_REBASE_ATOM_SOURCE=str(source),
        DSV41_ORIGIN_ATOM_SOURCE=str(source),
        DSV41_OWNER_SERVING_OUTPUT=str(output),
        DSV41_OWNER_SERVING_CONTROL=str(output / "capacity.txt"),
        DSV41_OWNER_GRAPH_PHASE_FILE=str(output / "graph_phase.txt"),
        DSV41_COMGR_RECEIPT_DIR=str(output),
        DSV41_STEP12_PREFLIGHT_LABEL=args.label,
        DSV41_REBASE_SERVICE_SERVER=str(int(args.action in ("preflight", "serve"))),
        PYTHONPATH=os.pathsep.join(
            (str(HERE / "acceptance"), str(source), str(args.aiter_source.resolve()))
        ),
    )
    if args.cache_dir:
        cache = args.cache_dir.resolve()
        env.update(
            DSV41_OWNER_SERVING_COMPILE_CACHE=str(cache / "atom"),
            VLLM_CACHE_ROOT=str(cache / "atom"),
            XDG_CACHE_HOME=str(cache / "xdg"),
        )
    return env


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument(
        "action", choices=("preflight", "client-preflight", "benchmark", "serve")
    )
    parser.add_argument("--mode", choices=("candidate", "native"), default="candidate")
    parser.add_argument("--model-source", type=Path, default=REPO)
    parser.add_argument("--aiter-source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--label", default="recipe")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    env = environment(args)
    if args.action == "serve":
        (args.output / "graph_phase.txt").write_text("warmup\n")
        (args.output / "capacity.txt").write_text(
            "packed\n" if args.mode == "candidate" else "native\n"
        )
        command = [sys.executable, str(HERE / "acceptance/serve.py")]
    else:
        name = {
            "preflight": "preflight.py",
            "client-preflight": "preflight_client.py",
            "benchmark": "run_step12_performance_r6.py",
        }[args.action]
        if args.action == "preflight":
            (args.output / "graph_phase.txt").write_text("warmup\n")
        command = [sys.executable, str(HERE / "acceptance" / name)]
    raise SystemExit(subprocess.call(command, env=env))


if __name__ == "__main__":
    main()
