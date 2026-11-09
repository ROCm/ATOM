"""Single-node ROCm V4.1: shared LMCache MP L1, optional Mooncake Store."""

import argparse
import datetime
import json
import os
import pathlib
import signal
import socket
import subprocess
import time

p = argparse.ArgumentParser()
p.add_argument("--arm", choices=["local", "store"], required=True)
p.add_argument("--model", required=True)
p.add_argument("--host-ip", default="127.0.0.1")
p.add_argument("--gpus", default="0,1,2,3")
p.add_argument("--port", type=int, default=22300)
p.add_argument("--l1-gib", type=int, default=256)
p.add_argument("--store-gib", type=int, default=512)
p.add_argument("--buffer-gib", type=int, default=1)
p.add_argument("--gpu-util", type=float, default=0.65)
p.add_argument("--master-bin", default="mooncake_master")
p.add_argument("--output", type=pathlib.Path, required=True)
a = p.parse_args()
a.arm = "single-" + a.arm
a.role = "prefill"
a.ip = a.host_ip
a.rank_start = 0
cfg = {
    "ip": a.ip,
    "gpus": [int(v) for v in a.gpus.split(",")],
    "api": a.port,
    "dp_master": a.port + 100,
    "dp_base": a.port + 120,
}
assert (
    len(cfg["gpus"]) == 4 and len(set(cfg["gpus"])) == 4
), "four distinct GPU IDs required"
assert a.l1_gib > 0 and a.store_gib > 0 and a.buffer_gib > 0
mp_port = a.port + 300
out = a.output.resolve()
out.mkdir(parents=True, exist_ok=False)
env = {k: v for k, v in os.environ.items() if not k.startswith("MORI_")}
env.update(
    ATOM_DISABLE_MORI_EP="1",
    ATOM_ENABLE_METRICS_REQUEST_CONTEXT="0",
    HIP_VISIBLE_DEVICES=",".join(map(str, cfg["gpus"])),
    ATOM_HOST_IP=a.ip,
    NCCL_IB_DISABLE="1",
    ATOM_DISABLE_MMAP="true",
    AITER_LOG_LEVEL="WARNING",
    OMP_NUM_THREADS="4",
    ATOM_DSPARK_DISABLE_COMPILE="1",
    ATOM_MONO_ENABLE="0",
    PYTHONFAULTHANDLER="1",
    PYTHONDONTWRITEBYTECODE="1",
    MC_DISABLE_HIP="1",
    ATOM_ENABLE_METRICS_DEVICE_TIMER="1",
    PYTHONUNBUFFERED="1",
    ATOM_DP_SESSION_AFFINITY="1",
    ATOM_DP_PREFIX_SPREAD="1",
    ATOM_DP_PREFIX_POLICY="idle_ratio",
    ATOM_DP_PREFIX_IDLE_TOKENS="32768",
    ATOM_DP_PREFIX_OWNER_BACKLOG_TOKENS="65536",
    ATOM_DP_PREFIX_RATIO="0",
    ATOM_DP_PREFIX_SIMILAR_TOKENS="16384",
    ATOM_OFFLOAD_EARLY_LOAD="1",
    ATOM_USE_CUSTOM_ALL_GATHER="0",
)
requested_stop = False
children = []
reason = None


def stop(sig, frame):
    global requested_stop
    requested_stop = True


signal.signal(signal.SIGTERM, stop)
signal.signal(signal.SIGINT, stop)


def start(name, cmd):
    with (out / (name + ".log")).open("ab", buffering=0) as f:
        child = subprocess.Popen(
            cmd,
            stdout=f,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            env=env,
            start_new_session=True,
        )
    children.append((name, child))
    return child


try:
    if a.arm == "single-store":
        master = start(
            "master",
            [
                a.master_bin,
                "--rpc_address=" + a.ip,
                "--rpc_port=" + str(a.port + 261),
                "--enable_http_metadata_server=true",
                "--http_metadata_server_host=" + a.ip,
                "--http_metadata_server_port=" + str(a.port + 262),
                "--metrics_host=" + a.ip,
                "--metrics_port=" + str(a.port + 263),
            ],
        )
        for _ in range(60):
            assert master.poll() is None, "Mooncake master exited; inspect master.log"
            try:
                with socket.create_connection((a.ip, a.port + 261), timeout=1):
                    break
            except OSError:
                time.sleep(1)
        else:
            raise TimeoutError("Mooncake master startup")
    if a.role == "prefill":
        env.update(
            ATOM_KV_OFFLOAD="lmcache_mp",
            ATOM_KV_OFFLOAD_EXTRA_CONFIG=json.dumps(
                {
                    "lmcache.mp.host": "tcp://127.0.0.1",
                    "lmcache.mp.port": mp_port,
                    "lmcache.mp.model_revision": "1bae0e1088993b9b82de08f59de5be5a34e42170",
                }
            ),
            LMCACHE_CHUNK_SIZE="256",
            OFFLOAD_MIN_LOAD_TOKENS="8192",
            OFFLOAD_MIN_SAVE_TOKENS="8192",
            OFFLOAD_MAX_PENDING_SAVES="8",
            OFFLOAD_PROFILE="1",
        )
        mp = [
            "lmcache",
            "server",
            "--host",
            "127.0.0.1",
            "--port",
            str(mp_port),
            "--http-host",
            "127.0.0.1",
            "--http-port",
            str(mp_port + 1),
            "--prometheus-port",
            str(mp_port + 2),
            "--chunk-size",
            "256",
            "--null-block-id",
            "-1",
            "--separate-object-groups",
            "--supported-transfer-mode",
            "lmcache_driven",
            "--l1-size-gb",
            str(a.l1_gib),
            "--l1-use-lazy",
            "--l1-init-size-gb",
            "4",
            "--eviction-policy",
            "LRU",
            "--l1-read-ttl-seconds",
            "900",
            "--eviction-trigger-watermark",
            "0.98",
            "--enable-extra-logging",
        ]
        if a.arm == "single-store":
            mp += [
                "--l2-adapter",
                json.dumps(
                    {
                        "type": "mooncake_store",
                        "num_workers": 4,
                        "local_hostname": a.ip,
                        "metadata_server": "http://"
                        + a.ip
                        + ":"
                        + str(cfg["api"] + 262)
                        + "/metadata",
                        "master_server_addr": a.ip + ":" + str(cfg["api"] + 261),
                        "protocol": "tcp",
                        "local_buffer_size": str(a.buffer_gib * 1024**3),
                        "global_segment_size": str(a.store_gib * 1024**3),
                    }
                ),
            ]
        child = start("mp", mp)
        for _ in range(120):
            assert child.poll() is None, "LMCache exited during startup"
            try:
                with socket.create_connection(("127.0.0.1", mp_port), timeout=1):
                    break
            except OSError:
                time.sleep(1)
        else:
            raise TimeoutError("LMCache startup")
    role = "kv_producer" if a.role == "prefill" else "kv_consumer"
    api = cfg["api"]
    pc = [
        "-tp",
        "4",
        "--data-parallel-master-ip",
        a.ip,
        "--data-parallel-master-port",
        str(cfg["dp_master"]),
        "--data-parallel-base-port",
        str(cfg["dp_base"]),
    ]
    seqs, tokens = (64, 16384) if a.role == "prefill" else (64, 1024)
    cmd = [
        "python3",
        "-m",
        "atom.entrypoints.openai_server",
        "--model",
        a.model,
        "--trust-remote-code",
        *pc,
        "--enable-dp-attention",
        "--moe-backend",
        "standard",
        "--all2all-backend",
        "none",
        "--dp-load-balance",
        "least_requests",
        "--server-port",
        str(api),
        "--kv_cache_dtype",
        "bf16",
        "--index-cache-dtype",
        "fp8",
        "--gpu-memory-utilization",
        str(a.gpu_util),
        "--max-num-seqs",
        str(seqs),
        "--max-num-batched-tokens",
        str(tokens),
        "--attn-prefill-chunk-size",
        "16384",
        "--enable_prefix_caching",
        "--block-size",
        "16",
        "--state-checkpoint-interval-tokens",
        "8192",
        "--level",
        "3",
        "--cudagraph-mode",
        "FULL",
        "--cudagraph-capture-sizes",
        "[1,2,4,8,16,24,32,40,48,64]",
        "--method",
        "dspark",
        "--num-speculative-tokens",
        "5",
        "--tool-call-parser",
        "dsml_v41",
        "--kv-transfer-config",
        json.dumps({"kv_role": "offload", "kv_connector": "lmcache_mp"}),
    ]
    k = cmd.index("--kv-transfer-config")
    del cmd[k : k + 2]
    (out / "launch.json").write_text(
        json.dumps(
            {
                "command": cmd,
                "environment": {
                    k: v
                    for k, v in env.items()
                    if k.startswith(
                        (
                            "ATOM",
                            "MORI",
                            "NCCL",
                            "LMCACHE",
                            "OFFLOAD",
                            "HIP",
                            "MC_",
                            "GLOO",
                        )
                    )
                },
                "role": a.role,
                "rank_start": a.rank_start,
                "start_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            },
            indent=2,
        )
    )
    child = start("engine", cmd)
    (out / "pids.json").write_text(json.dumps({k: p.pid for k, p in children}))
    while not requested_stop:
        exits = [(k, p.returncode) for k, p in children if p.poll() is not None]
        if exits:
            reason = {"child_exits": exits}
            break
        time.sleep(2)
except BaseException as e:  # noqa: BLE001
    reason = {"exception": repr(e)}
finally:
    (out / "exit.json").write_text(
        json.dumps(
            {
                "reason": reason,
                "requested_stop": requested_stop,
                "time_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            },
            indent=2,
        )
    )
    for name, child in reversed(children):
        if child.poll() is None:
            os.killpg(child.pid, signal.SIGTERM)
    for name, child in reversed(children):
        try:
            child.wait(timeout=10)
        except subprocess.TimeoutExpired:
            os.killpg(child.pid, signal.SIGKILL)
            child.wait()
raise SystemExit(0 if requested_stop else 1)
