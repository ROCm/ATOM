# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

"""E2E test: MORI-IO P2P GPU-to-GPU hidden state transfer.

Tests correctness (data integrity) and speed (latency + throughput).
Runs producer on GPU 4, consumer on GPU 0, matching the Eagle3 SDDD layout.

Usage (inside docker with GPU access):
    python tests/integration/test_mori_io_e2e.py
"""

import multiprocessing as mp
import os
import shutil
import sys
import time

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from atomrl.transfer.eagle_mooncake_store import (
    HIDDEN_STATES_STORAGE_DTYPE,
)

SRC_GPU = 4
DST_GPU = 0
HIDDEN_DIM = 7168
NUM_AUX_LAYERS = 3
NUM_TRAINING_LAYERS = NUM_AUX_LAYERS - 1
TRAINING_HIDDEN_SIZE = NUM_TRAINING_LAYERS * HIDDEN_DIM

TEST_CONFIGS = [
    {"name": "short_seq", "seq_len": 512, "iters": 20},
    {"name": "medium_seq", "seq_len": 2048, "iters": 10},
    {"name": "long_seq", "seq_len": 8192, "iters": 5},
]


def _clean_shm():
    for d in ["/dev/shm/mori_io", "/dev/shm/mori_io_ready"]:
        if os.path.exists(d):
            shutil.rmtree(d)


def _producer_process(cfg, result_queue):
    """Producer: create tensors on SRC_GPU, put via MoriIOStore."""
    try:
        torch.cuda.set_device(SRC_GPU)
        from atomrl.transfer.mori_io_store import MoriIOStore

        store = MoriIOStore(
            role="producer",
            src_gpu=SRC_GPU,
            dst_gpu=DST_GPU,
            max_seq_len=cfg["seq_len"] + 256,
            hidden_dim=HIDDEN_DIM,
            num_aux_layers=NUM_AUX_LAYERS,
        )
        store.setup()

        seq_len = cfg["seq_len"]
        iters = cfg["iters"]
        latencies = []

        for i in range(iters):
            hidden_states = torch.randn(
                seq_len,
                TRAINING_HIDDEN_SIZE,
                dtype=HIDDEN_STATES_STORAGE_DTYPE,
                device=f"cuda:{SRC_GPU}",
            )
            input_ids = torch.arange(
                i * seq_len,
                (i + 1) * seq_len,
                dtype=torch.int64,
                device=f"cuda:{SRC_GPU}",
            )
            last_hidden_states = torch.randn(
                seq_len,
                HIDDEN_DIM,
                dtype=HIDDEN_STATES_STORAGE_DTYPE,
                device=f"cuda:{SRC_GPU}",
            )

            # Compute checksum before transfer
            hs_sum = hidden_states.float().sum().item()
            ids_sum = input_ids.float().sum().item()
            lhs_sum = last_hidden_states.float().sum().item()

            torch.cuda.synchronize(SRC_GPU)
            t0 = time.perf_counter()

            key = f"test_{cfg['name']}_{i}"
            store.put(
                key=key,
                hidden_states=hidden_states,
                input_ids=input_ids,
                last_hidden_states=last_hidden_states,
            )

            torch.cuda.synchronize(SRC_GPU)
            t1 = time.perf_counter()
            latencies.append(t1 - t0)

            result_queue.put(
                {
                    "type": "put_done",
                    "key": key,
                    "iter": i,
                    "hs_sum": hs_sum,
                    "ids_sum": ids_sum,
                    "lhs_sum": lhs_sum,
                    "shapes": {
                        "hidden_states": tuple(hidden_states.shape),
                        "input_ids": tuple(input_ids.shape),
                        "last_hidden_states": tuple(last_hidden_states.shape),
                    },
                    "dtypes": {
                        "hidden_states": str(hidden_states.dtype),
                        "input_ids": str(input_ids.dtype),
                        "last_hidden_states": str(last_hidden_states.dtype),
                    },
                    "latency_ms": latencies[-1] * 1000,
                }
            )

        total_bytes_per_iter = (
            seq_len * TRAINING_HIDDEN_SIZE * 2  # bfloat16
            + seq_len * 8  # int64
            + seq_len * HIDDEN_DIM * 2  # bfloat16
        )
        avg_lat = sum(latencies[1:]) / max(len(latencies) - 1, 1)
        throughput_gbps = (total_bytes_per_iter / avg_lat) / 1e9

        result_queue.put(
            {
                "type": "producer_summary",
                "name": cfg["name"],
                "seq_len": seq_len,
                "iters": iters,
                "avg_latency_ms": avg_lat * 1000,
                "throughput_gbps": throughput_gbps,
                "total_bytes_per_iter": total_bytes_per_iter,
            }
        )

        store.close()
    except (OSError, RuntimeError, ValueError) as e:
        import traceback

        result_queue.put(
            {
                "type": "error",
                "role": "producer",
                "error": str(e),
                "tb": traceback.format_exc(),
            }
        )


def _consumer_process(cfg, result_queue):
    """Consumer: get tensors from MoriIOStore, verify checksums."""
    try:
        torch.cuda.set_device(DST_GPU)
        from atomrl.transfer.mori_io_store import MoriIOStore

        store = MoriIOStore(
            role="consumer",
            src_gpu=SRC_GPU,
            dst_gpu=DST_GPU,
            max_seq_len=cfg["seq_len"] + 256,
            hidden_dim=HIDDEN_DIM,
            num_aux_layers=NUM_AUX_LAYERS,
        )
        store.setup()

        seq_len = cfg["seq_len"]
        iters = cfg["iters"]
        correctness_results = []

        for i in range(iters):
            key = f"test_{cfg['name']}_{i}"
            shapes = {
                "hidden_states": (seq_len, TRAINING_HIDDEN_SIZE),
                "input_ids": (seq_len,),
                "last_hidden_states": (seq_len, HIDDEN_DIM),
            }
            dtypes = {
                "hidden_states": HIDDEN_STATES_STORAGE_DTYPE,
                "input_ids": torch.int64,
                "last_hidden_states": HIDDEN_STATES_STORAGE_DTYPE,
            }

            torch.cuda.synchronize(DST_GPU)
            t0 = time.perf_counter()

            output = store.get(
                key=key,
                shapes=shapes,
                dtypes=dtypes,
                device=torch.device(f"cuda:{DST_GPU}"),
            )

            torch.cuda.synchronize(DST_GPU)
            t1 = time.perf_counter()

            hs_sum = output.hidden_states.float().sum().item()
            ids_sum = output.input_ids.float().sum().item()
            lhs_sum = output.last_hidden_states.float().sum().item()

            correctness_results.append(
                {
                    "key": key,
                    "iter": i,
                    "hs_sum": hs_sum,
                    "ids_sum": ids_sum,
                    "lhs_sum": lhs_sum,
                    "get_latency_ms": (t1 - t0) * 1000,
                    "hs_device": str(output.hidden_states.device),
                    "ids_device": str(output.input_ids.device),
                    "hs_shape": tuple(output.hidden_states.shape),
                    "ids_shape": tuple(output.input_ids.shape),
                    "lhs_shape": tuple(output.last_hidden_states.shape),
                }
            )

            store.remove_eagle3_tensors(key, has_last_hidden_states=True)

        result_queue.put(
            {
                "type": "consumer_results",
                "results": correctness_results,
            }
        )

        store.close()
    except (OSError, RuntimeError, ValueError) as e:
        import traceback

        result_queue.put(
            {
                "type": "error",
                "role": "consumer",
                "error": str(e),
                "tb": traceback.format_exc(),
            }
        )


def run_test(cfg):
    print(f"\n{'='*70}")
    print(f"  Test: {cfg['name']}  (seq_len={cfg['seq_len']}, iters={cfg['iters']})")
    print(f"  Transfer: GPU {SRC_GPU} → GPU {DST_GPU}")
    print(f"{'='*70}")

    _clean_shm()

    result_queue = mp.Queue()

    producer = mp.Process(target=_producer_process, args=(cfg, result_queue))
    consumer = mp.Process(target=_consumer_process, args=(cfg, result_queue))

    consumer.start()
    time.sleep(0.2)
    producer.start()

    producer.join(timeout=120)
    consumer.join(timeout=120)

    if producer.is_alive():
        producer.kill()
        print("  ERROR: producer timed out")
        return False
    if consumer.is_alive():
        consumer.kill()
        print("  ERROR: consumer timed out")
        return False

    # Collect results
    producer_puts = {}
    consumer_results = None
    producer_summary = None
    errors = []

    while not result_queue.empty():
        msg = result_queue.get_nowait()
        if msg["type"] == "error":
            errors.append(msg)
        elif msg["type"] == "put_done":
            producer_puts[msg["key"]] = msg
        elif msg["type"] == "producer_summary":
            producer_summary = msg
        elif msg["type"] == "consumer_results":
            consumer_results = msg["results"]

    if errors:
        for e in errors:
            print(f"  ERROR ({e['role']}): {e['error']}")
            print(e["tb"])
        return False

    if consumer_results is None:
        print("  ERROR: no consumer results received")
        return False

    # Verify correctness
    all_correct = True
    for cr in consumer_results:
        key = cr["key"]
        pp = producer_puts.get(key)
        if pp is None:
            print(f"  FAIL: {key} — no producer data")
            all_correct = False
            continue

        hs_match = abs(pp["hs_sum"] - cr["hs_sum"]) < abs(pp["hs_sum"]) * 1e-3 + 1e-6
        ids_match = pp["ids_sum"] == cr["ids_sum"]
        lhs_match = (
            abs(pp["lhs_sum"] - cr["lhs_sum"]) < abs(pp["lhs_sum"]) * 1e-3 + 1e-6
        )

        if hs_match and ids_match and lhs_match:
            print(
                f"  PASS: {key}  "
                f"put={pp['latency_ms']:.1f}ms  get={cr['get_latency_ms']:.1f}ms  "
                f"device={cr['hs_device']}"
            )
        else:
            print(f"  FAIL: {key}")
            if not hs_match:
                print(
                    f"    hidden_states sum: producer={pp['hs_sum']:.4f} consumer={cr['hs_sum']:.4f}"
                )
            if not ids_match:
                print(
                    f"    input_ids sum: producer={pp['ids_sum']:.4f} consumer={cr['ids_sum']:.4f}"
                )
            if not lhs_match:
                print(
                    f"    last_hidden_states sum: producer={pp['lhs_sum']:.4f} consumer={cr['lhs_sum']:.4f}"
                )
            all_correct = False

    # Print speed summary
    if producer_summary:
        ps = producer_summary
        total_mb = ps["total_bytes_per_iter"] / 1024**2
        print("\n  --- Speed Summary ---")
        print(f"  Payload per transfer: {total_mb:.1f} MB")
        print(f"  Avg put latency (excl warmup): {ps['avg_latency_ms']:.2f} ms")
        print(f"  Throughput: {ps['throughput_gbps']:.2f} GB/s")

    _clean_shm()
    return all_correct


def main():
    mp.set_start_method("spawn", force=True)

    print("MORI-IO E2E Transfer Test")
    print(f"Producer GPU: {SRC_GPU}, Consumer GPU: {DST_GPU}")
    print(f"Hidden dim: {HIDDEN_DIM}, Aux layers: {NUM_AUX_LAYERS}")

    all_pass = True
    for cfg in TEST_CONFIGS:
        ok = run_test(cfg)
        if not ok:
            all_pass = False

    print(f"\n{'='*70}")
    if all_pass:
        print("  ALL TESTS PASSED")
    else:
        print("  SOME TESTS FAILED")
    print(f"{'='*70}")
    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
