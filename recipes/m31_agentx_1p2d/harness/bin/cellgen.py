#!/usr/bin/env python3
"""cellgen.py: build one M3.1 NVFP4 P/D AgentX (or GSM8K) cell in main's CI format.

Writes <run>/config.json (a models_atomesh.yaml-shaped config: main's defaults and
backends from the snapshot, plus one model and one suite entry), runs the
snapshot's own pd_matrix.py on it, and writes the expanded cell to <run>/cell.json.
The launcher (pd_submit.sh export block -> pd_slurm_job.sh --spur-worker ->
pd_server_atom.sh) then runs it exactly as CI would.

Role env files (--p-env/--d-env/--common-env) are bash files sourced in a clean
shell; every variable they set lands in env.prefill / env.decode / env.common.
A file may set AGX_ARGS: extra server flags for that role (appended last, so they
win argparse); it is not exported. Values keep ${ROLE_IP} / ${HANDSHAKE_PORT} /
${WORKER_RDMA_DEVICES} for the launcher when written in single quotes.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import yaml

MODEL_NAME = "MiniMax-M3.1-Flash-NVFP4"
CG_LADDER = [1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 14, 16, 20, 24, 28, 32, 40, 48, 56, 64,
             80, 96, 112, 128, 144, 160, 176, 192, 208, 224, 240, 256, 288, 320, 360,
             400, 448, 512]
TOPO = {  # topology -> (layout, decode workers, display name)
    "1p1d": ("single_node", 1, "1P1D-PP4-TP4-DSPARK7-NVFP4-MCSTORE"),
    "1p2d": ("packed_nodes", 2, "1P2D-PP4-2xTP4-DSPARK7-NVFP4-MCSTORE2N"),
    "1p3d": ("packed_nodes", 3, "1P3D-PP4-3xTP4-DSPARK7-NVFP4-MCSTORE2N"),
}
SKIP_ENV = {"PWD", "OLDPWD", "SHLVL", "_", "HOME", "PATH", "TERM", "HOSTNAME", "AGX_ARGS"}


# Read a role file. Two conventions:
#  - recipe-audit's: the file declares <X>_ENV_NAMES (the role env) and <X>_SERVER_ARGS (its server
#    flags) as bash arrays, and reads inputs (AL, CONC, D_NICS, P_STORE_LAYOUT, owner plans) from the env;
#  - plain: every variable the file sets is role env; AGX_ARGS holds extra server flags.
DUMP = r"""
set +u
source "$1" >/dev/null || exit 3
named=0
for v in P_ENV_NAMES D_ENV_NAMES C_ENV_NAMES AGX_ENV_NAMES; do
  if declare -p "$v" >/dev/null 2>&1; then
    named=1
    eval "names=(\"\${$v[@]}\")"
    for n in "${names[@]}"; do printf 'ENV\t%s\t%s\0' "$n" "${!n}"; done
  fi
done
for v in P_SERVER_ARGS D_SERVER_ARGS C_SERVER_ARGS; do
  if declare -p "$v" >/dev/null 2>&1; then
    eval "args=(\"\${$v[@]}\")"
    for a in "${args[@]}"; do printf 'ARG\t%s\0' "$a"; done
  fi
done
[[ -n "${AGX_ARGS:-}" ]] && for a in ${AGX_ARGS}; do printf 'ARG\t%s\0' "$a"; done
if [[ "${named}" == 0 ]]; then
  env -0 | while IFS= read -r -d '' kv; do printf 'ALL\t%s\t%s\0' "${kv%%=*}" "${kv#*=}"; done
fi
"""


def source_env(path: str | None, inputs: dict[str, str]) -> tuple[dict[str, str], list[str]]:
    """(role env, server args) of a role file; see DUMP."""
    if not path:
        return {}, []
    base = {"HOME": os.environ.get("HOME", "/tmp"), "PATH": "/usr/bin:/bin", **inputs}
    clean = subprocess.run(["bash", "-c", "env -0"], env=base, capture_output=True, check=True).stdout
    before = dict(item.decode().split("=", 1) for item in clean.split(b"\0") if b"=" in item)
    out = subprocess.run(["bash", "-c", DUMP, "_", path], env=base, capture_output=True, check=False)
    if out.returncode != 0:
        raise SystemExit(f"sourcing {path} failed rc={out.returncode}: {out.stderr.decode()[-2000:]}")
    env, args = {}, []
    for item in out.stdout.split(b"\0"):
        if not item:
            continue
        kind, _, rest = item.decode().partition("\t")
        if kind == "ARG":
            args.append(rest)
            continue
        name, _, value = rest.partition("\t")
        # A plain file's role env: what it set or changed (an input echoed back unchanged is not).
        if kind == "ENV" or (kind == "ALL" and before.get(name) != value and name not in SKIP_ENV
                             and not name.startswith("BASH_FUNC_")):
            env[name] = value
    return env, args


# Flags the launcher builds itself from the cell; a role file's copy is dropped.
LAUNCHER_FLAGS = {"--model", "--tensor-parallel-size", "-tp", "--pipeline-parallel-size", "--kv_cache_dtype",
                  "--kv-cache-dtype", "--block-size", "--max-model-len", "--kv-transfer-config", "--method",
                  "--draft-model", "--num-speculative-tokens", "--spec-decode-acceptance-length", "--host",
                  "--port", "--server-port"}
NO_VALUE = {"--trust-remote-code", "--enable-prefix-caching", "--no-enable_prefix_caching", "--enforce-eager"}
CAPACITY = {"--max-num-seqs": "seqs", "--max-num-batched-tokens": "mnbt", "--gpu-memory-utilization": "mem",
            "--cudagraph-capture-sizes": "cg"}


def split_args(args: list[str]) -> tuple[dict[str, str], list[str]]:
    """Capacity values the cell carries, and the flags passed through (one shell word each)."""
    cap, rest, i = {}, [], 0
    while i < len(args):
        tok = args[i]
        name, eq, val = tok.partition("=")
        if not tok.startswith("-"):
            raise SystemExit(f"server argument {tok!r} has no flag")
        if name in NO_VALUE:
            if name not in LAUNCHER_FLAGS and name not in ("--trust-remote-code", "--enable-prefix-caching"):
                rest.append(name)
            i += 1
            continue
        if not eq:
            if i + 1 >= len(args):
                raise SystemExit(f"server flag {tok} has no value")
            val = args[i + 1]
            i += 2
        else:
            i += 1
        if name in CAPACITY:
            cap[CAPACITY[name]] = val
        elif name not in LAUNCHER_FLAGS:
            word = "".join(val.split()) if val.lstrip().startswith(("{", "[")) else val
            if any(c.isspace() for c in word):
                raise SystemExit(f"server flag {name} value {val!r} cannot pass the launcher's word splitting")
            rest += [name, word]
    return cap, rest


def cudagraph(max_seqs: int) -> str:
    sizes = [s for s in CG_LADDER if s < max_seqs] + [max_seqs]
    return "[" + ",".join(str(s) for s in sizes) + "]"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--snap", required=True, help="snapshot worktree (its launcher expands the cell)")
    ap.add_argument("--run", required=True, help="run dir")
    ap.add_argument("--topo", required=True, choices=sorted(TOPO))
    ap.add_argument("--nodes", required=True, help="comma-separated hostnames, node A first")
    ap.add_argument("--image", required=True)
    ap.add_argument("--conc", type=int, required=True)
    ap.add_argument("--mode", choices=("agentx", "gsm8k"), default="agentx")
    ap.add_argument("--duration", type=int, default=3600)
    ap.add_argument("--warmup-per-lane", type=int, default=10)
    ap.add_argument("--al", default="4.04", help="forced acceptance length; '' or off = real acceptance")
    ap.add_argument("--model-path", default=os.environ.get("AGX_MODEL_PATH",
                                                           "/share_nfs/models/amd/MiniMax-M3.1-Flash-Quark-MXFP4"))
    ap.add_argument("--store-plan", required=True, help="store_plan.py JSON")
    ap.add_argument("--store-pools", default="per_nic", choices=("per_nic", "shared"))
    ap.add_argument("--evict", default="", help="HIGH/RATIO, default 0.89/0.02 (draft window) or 0.88/0.02")
    ap.add_argument("--draft-window", type=int, default=1)
    ap.add_argument("--pp-inflight", default="8", help="ATOM_PP_MAX_INFLIGHT ('' = unset)")
    ap.add_argument("--p-env")
    ap.add_argument("--d-env")
    ap.add_argument("--common-env")
    ap.add_argument("--common-args", default="--index-cache-dtype nvfp4 --spec-decode-kv-cache-dtype fp8")
    ap.add_argument("--p-args", default="")
    ap.add_argument("--d-args", default="")
    ap.add_argument("--p-mnbt", type=int, default=0, help="prefill MNBT (0 = role file, else 8192)")
    ap.add_argument("--d-mnbt", type=int, default=0, help="decode MNBT (0 = role file, else ATOM's default)")
    ap.add_argument("--p-seqs", type=int, default=0, help="prefill max_num_seqs (0 = 2 x conc)")
    ap.add_argument("--d-seqs", type=int, default=0, help="per-decode max_num_seqs (0 = auto)")
    ap.add_argument("--mem", default="", help="gpu_memory_utilization for both roles (default 0.9)")
    ap.add_argument("--router-policy", default="", help="default: random for 1p1d, cache_aware otherwise")
    ap.add_argument("--balance-abs", default="2")
    ap.add_argument("--admission", default='{"enabled":true,"max_pending_kv_fraction":0.15,"num_reserved_decode_tokens":512}')
    ap.add_argument("--gsm8k-limit", type=int, default=50)
    ap.add_argument("--gsm8k-conc", type=int, default=32)
    ap.add_argument("--gsm8k-threshold", type=float, default=0.88)
    ap.add_argument("--label", default="", help="suffix for the display topology")
    args = ap.parse_args()

    snap, run = Path(args.snap), Path(args.run)
    layout, n_decode, display = TOPO[args.topo]
    if args.label:
        display += "-" + args.label.upper()
    nodes = args.nodes.split(",")
    with open(args.store_plan) as handle:
        plan = json.load(handle)
    with open(snap / ".github/benchmark/models_atomesh.yaml") as handle:
        main_cfg = yaml.safe_load(handle)

    al_in = "real" if args.al in ("", "off", "none") or args.mode == "gsm8k" else args.al
    inputs = {
        "MODEL": args.model_path, "AL": al_in, "CONC": str(args.conc), "D_NICS": "launcher",
        "P_STORE_LAYOUT": args.store_pools, "MOONCAKE_STORE_OWNERS": plan["owners"],
        "MOONCAKE_STORE_DECODE_OWNERS": plan.get("decode_owners", ""),
    }
    p_env, p_raw = source_env(args.p_env, inputs)
    d_env, d_raw = source_env(args.d_env, inputs)
    c_env, c_raw = source_env(args.common_env, inputs)
    p_cap, p_rest = split_args(p_raw)
    d_cap, d_rest = split_args(d_raw)
    c_cap, c_rest = split_args(c_raw)
    p_args, d_args, c_args = " ".join(p_rest), " ".join(d_rest), " ".join(c_rest)
    if not plan.get("decode_owners"):
        p_env.pop("MOONCAKE_STORE_DECODE_OWNERS", None)
    if p_env.get("MOONCAKE_STORE_DECODE_OWNERS") == "":
        p_env.pop("MOONCAKE_STORE_DECODE_OWNERS")

    draft = 1 if args.draft_window else 0
    high, ratio = (args.evict.split("/") if args.evict else (("0.89", "0.02") if draft else ("0.88", "0.02")))
    store_env = {
        "MOONCAKE_STORE": "1",
        "MOONCAKE_STORE_POOLS": args.store_pools,
        "MOONCAKE_STORE_OWNERS": plan["owners"],
        "MOONCAKE_STORE_PAGE_CACHE_DROP_DIRS": "${ATOMESH_MODEL_ROOT}",
        "MOONCAKE_STORE_COMPACT_TIMEOUT": "1800",
        "MOONCAKE_STORE_EVICTION_HIGH_WATERMARK": high,
        "MOONCAKE_STORE_EVICTION_RATIO": ratio,
        "OFFLOAD_PROFILE": "1",
        "OFFLOAD_MIN_LOAD_TOKENS": "0",
    }
    if plan.get("decode_owners"):
        store_env["MOONCAKE_STORE_DECODE_OWNERS"] = plan["decode_owners"]
    prefill_env = {
        "VLLM_PP_LAYER_PARTITION": "16,16,16,12",
        "ATOM_MOONCAKE_STORE_DRAFT_WINDOW": str(draft),
        "PREFILL_KV_TRANSFER_CONFIG": (
            '{"kv_connector":"mooncake","kv_role":"kv_producer","proxy_ip":"${ROLE_IP}",'
            '"handshake_port":${HANDSHAKE_PORT},"protocol":"rdma","enable_chunked_transfer":true,'
            '"chunked_admission_timeout_s":1500}'
        ),
        **store_env,
    }
    if args.pp_inflight:
        prefill_env["ATOM_PP_MAX_INFLIGHT"] = args.pp_inflight
    decode_env = {
        "ATOM_M3_INDEXER_CP": "1",
        "DECODE_KV_TRANSFER_CONFIG": (
            '{"kv_connector":"mooncake","kv_role":"kv_consumer","proxy_ip":"${ROLE_IP}",'
            '"handshake_port":${HANDSHAKE_PORT},"protocol":"rdma","ib_device":"${WORKER_RDMA_DEVICES}",'
            f'"pd_admission":{args.admission}}}'
        ),
    }
    if layout == "single_node":
        prefill_env["HIP_VISIBLE_DEVICES"] = "0,1,2,3"
        decode_env["HIP_VISIBLE_DEVICES"] = "4,5,6,7"
    common_env = {
        "AITER_LOG_LEVEL": "WARNING",
        "PYTHONNOUSERSITE": "1",
        "PYTHONHASHSEED": "0",
        "MC_DISABLE_HIP": "1",
        "MC_NUM_QP_PER_EP": "1",
        "NCCL_IB_DISABLE": "1",
        "ATOM_PD_MINIMAX_LANDING": "1",
        "ATOM_FUSE_MXFP8_SHARED_INTO_MXFP4": "0",
        "ATOM_TOKENIZE_CACHE_MB": "2048",
        "HF_HUB_CACHE": "/tmp/hf-hub",
        "HF_HUB_DOWNLOAD_TIMEOUT": "120",
        "ATOMESH_TORCH_PROFILE_DURATION": "0",
    }
    common_env.update(c_env)
    prefill_env.update(p_env)
    decode_env.update(d_env)

    conc = args.conc
    if args.d_seqs:
        d_seqs = args.d_seqs
    elif "seqs" in d_cap:
        d_seqs = int(d_cap["seqs"])
    elif n_decode == 1:
        d_seqs = min(2 * conc, 512)
    else:
        d_seqs = min(conc, 512)
    p_seqs = args.p_seqs or int(p_cap.get("seqs", 0)) or min(2 * conc, 1024)
    p_mnbt = args.p_mnbt or int(p_cap.get("mnbt", 0)) or 8192
    d_mnbt = args.d_mnbt or int(d_cap.get("mnbt", 0)) or None
    d_cg = d_cap.get("cg") if not args.d_seqs and "cg" in d_cap else cudagraph(d_seqs)
    # A role's own utilization comes last in its flags, so it wins argparse over the common one.
    mem = args.mem or c_cap.get("mem") or "0.9"
    if "mem" in p_cap:
        p_args = f"{p_args} --gpu-memory-utilization {p_cap['mem']}".strip()
    if "mem" in d_cap:
        d_args = f"{d_args} --gpu-memory-utilization {d_cap['mem']}".strip()
    # GSM8K always runs real acceptance (the launcher's eval phase drops it too).
    al = "" if args.al in ("", "off", "none") or args.mode == "gsm8k" else args.al
    common_args = " ".join(x for x in (args.common_args, c_args) if x)
    model_path = args.model_path

    suite = {
        "name": f"m31agx-{args.topo}-c{conc}-{args.mode}",
        "topology": args.topo,
        "display_topology": display,
        "pd_worker_layout": layout,
        "nodes": ",".join(nodes),
        "isl": [1048576],
        "osl": 1024,
        "concurrency": [conc],
        "prefill": {
            "workers": 1,
            "tp": 1,
            "cudagraph": "[1]",
            "extra_args": " ".join(x for x in (
                f"--pipeline-parallel-size 4 --enforce-eager --max-num-batched-tokens {p_mnbt}",
                args.p_args, p_args) if x),
        },
        "decode": {
            "workers": n_decode,
            "tp": 4,
            "cudagraph": d_cg,
            "extra_args": " ".join(x for x in (args.d_args, d_args) if x),
        },
        "router": {"policy": args.router_policy or ("random" if n_decode == 1 else "cache_aware")},
        "server": {
            "common_args": {
                "trust_remote_code": True,
                "kv_cache_dtype": "nvfp4",
                "block_size": 128,
                "enable_prefix_caching": True,
                "gpu_memory_utilization": float(mem),
                "max_model_len": 1048576,
                "max_num_seqs": p_seqs,
                "decode_max_num_seqs": d_seqs,
                "decode_max_num_batched_tokens": d_mnbt or "",
                "method": "dspark",
                "draft_model": f"{model_path}/dspark",
                "num_speculative_tokens": 7,
                "spec_decode_acceptance_length": al,
                "extra_args": common_args,
            }
        },
        "env": {"common": common_env, "prefill": prefill_env, "decode": decode_env},
    }
    if suite["router"]["policy"] == "cache_aware":
        suite["router"]["balance_abs_threshold"] = int(args.balance_abs)
    if args.mode == "agentx":
        suite["benchmark"] = {
            "kind": "aiperf_agentic",
            "scenario": "agentx",
            "public_dataset": "semianalysis_cc_traces_weka_062126",
            "apply_chat_template": True,
            "max_context_length": 1048576,
            "num_dataset_entries": 393,
            "benchmark_duration": args.duration,
            "warmup_requests_per_lane": args.warmup_per_lane,
            "trace_idle_gap_cap_seconds": 300,
            "warmup_grace_period": 1800,
            "trajectory_start_min_ratio": 0.25,
            "trajectory_start_max_ratio": 0.75,
            "failed_request_threshold": 0.10,
            "slice_duration": 1.0,
            "cancel_drain_timeout": 300,
            "http_tcp_user_timeout": 900000,
            "dataset_configuration_timeout": 1800,
            "service_profile_configure_timeout": 1800,
            "wait_server_timeout": 7200,
            "wait_router_timeout": 600,
        }
        suite["run_eval"] = False
    else:
        suite["benchmark"] = {"kind": "random", "wait_server_timeout": 7200, "wait_router_timeout": 600}
        suite["run_eval"] = True
        suite["eval_only"] = True
        suite["eval_concurrency"] = [args.gsm8k_conc]
        suite["accuracy"] = {
            "task": "gsm8k",
            "fewshot": 20,
            "limit": args.gsm8k_limit or None,
            "model_type": "local-chat-completions",
            "endpoint": "chat/completions",
            "max_gen_toks": 16384,
            "apply_chat_template": True,
            "fewshot_as_multiturn": True,
            "threshold": args.gsm8k_threshold,
        }
    cfg = {
        "defaults": main_cfg["defaults"],
        "backends": main_cfg["backends"],
        "models": {
            MODEL_NAME: {
                "backend": "atom",
                "model_path": model_path,
                "precision": "MXFP4",
                "suites": {"agx": [suite]},
            }
        },
    }
    (run / "config.json").write_text(json.dumps(cfg, indent=1))
    env = dict(os.environ)
    env.update({
        "ATOMESH_SLURM_ACCOUNT": "",
        "ATOMESH_SLURM_PARTITION": "amd-spur",
        "ATOMESH_SLURM_SUBMIT_RUNNER": "atomesh-cicd",
        "ATOMESH_LOG_ROOT": str(run / "logs"),
        "ATOMESH_PD_RANK_MAPPING_POLICY": "none",
        "ATOMESH_MODEL_ROOT": os.environ.get("AGX_MODEL_ROOT", "/share_nfs/models"),
    })
    out = run / "cells.all.json"
    proc = subprocess.run(
        [sys.executable, str(snap / ".github/scripts/atomesh/pd_matrix.py"), "--config", str(run / "config.json"),
         "--suite", "agx", "--model", MODEL_NAME, "--image", args.image, "--output", str(out)],
        env=env, capture_output=True, text=True, check=False,
    )
    sys.stdout.write(proc.stdout)
    if proc.returncode != 0:
        sys.stderr.write(proc.stderr)
        return proc.returncode
    cells = json.loads(out.read_text())["include"]
    if len(cells) != 1:
        print(f"expected one cell, got {[c['id'] for c in cells]}", file=sys.stderr)
        return 2
    cell = cells[0]
    if cell["num_nodes"] != len(nodes) and not (layout == "single_node" and cell["num_nodes"] == 1):
        print(f"cell needs {cell['num_nodes']} node(s), given {nodes}", file=sys.stderr)
        return 2
    (run / "cell.json").write_text(json.dumps(cell, indent=2))
    print(f"cell {cell['id']} nodes={cell['nodes']} num_nodes={cell['num_nodes']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
