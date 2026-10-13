#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Read-only ROCm metrics with PCI mapping and interval throttle residency."""

import argparse
import ctypes
import ctypes.util
import json
import os
import signal
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path


def check(status, operation):
    if status != 0:
        raise RuntimeError(f"{operation} returned status {status}")


def pci_string(value):
    return f"{value >> 32:04x}:{(value >> 8) & 255:02x}:{(value >> 3) & 31:02x}.{value & 7}"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--mapping", action="store_true")
    parser.add_argument("--interval", type=float, default=2)
    parser.add_argument("--samples", type=int, default=0, help="0 means until SIGTERM")
    args = parser.parse_args()
    if args.interval <= 0 or args.samples < 0 or not (args.mapping or args.output):
        parser.error("Provide --mapping or --output, interval >0 and samples >=0")
    rocm = Path(os.environ.get("ROCM_PATH", "/opt/rocm"))
    sys.path.insert(0, str(rocm / "libexec/rocm_smi"))
    from rsmiBindings import rsmi_gpu_metrics_t

    smi_path = ctypes.util.find_library("rocm_smi64") or str(
        rocm / "lib/librocm_smi64.so"
    )
    smi = ctypes.CDLL(smi_path)
    smi.rsmi_init.argtypes = [ctypes.c_uint64]
    check(smi.rsmi_init(0), "rsmi_init")
    try:
        count = ctypes.c_uint32()
        check(smi.rsmi_num_monitor_devices(ctypes.byref(count)), "device count")
        if count.value == 0:
            raise RuntimeError("No ROCm SMI devices visible")
        devices = []
        for index in range(count.value):
            pci = ctypes.c_uint64()
            check(smi.rsmi_dev_pci_id_get(index, ctypes.byref(pci)), "PCI id")
            devices.append(
                {"smi_index": index, "pci_id": pci.value, "pci": pci_string(pci.value)}
            )
        if args.mapping:
            hip_path = ctypes.util.find_library("amdhip64") or str(
                rocm / "lib/libamdhip64.so"
            )
            hip = ctypes.CDLL(hip_path)
            hip_count = ctypes.c_int()
            check(hip.hipGetDeviceCount(ctypes.byref(hip_count)), "HIP device count")
            ranks = []
            for rank in range(hip_count.value):
                bus = ctypes.create_string_buffer(64)
                check(hip.hipDeviceGetPCIBusId(bus, len(bus), rank), "HIP PCI bus")
                ranks.append({"local_rank": rank, "pci": bus.value.decode()})
            print(
                json.dumps(
                    {
                        "HIP_VISIBLE_DEVICES": os.environ.get("HIP_VISIBLE_DEVICES"),
                        "hip_ranks": ranks,
                        "smi_devices": devices,
                    },
                    indent=2,
                )
            )
            if hip_count.value != 4:
                raise RuntimeError(
                    "This SP4 experiment requires exactly four HIP-visible GPUs"
                )
            return

        stop = threading.Event()
        signal.signal(signal.SIGTERM, lambda *_: stop.set())
        signal.signal(signal.SIGINT, lambda *_: stop.set())
        counters = (
            "prochot_residency_acc",
            "ppt_residency_acc",
            "socket_thm_residency_acc",
            "vr_thm_residency_acc",
            "hbm_thm_residency_acc",
        )
        fields = (
            "temperature_hotspot",
            "temperature_mem",
            "current_gfxclk",
            "current_socket_power",
            "accumulation_counter",
            *counters,
        )
        previous = {}
        sample = 0
        with args.output.open("w", buffering=1) as output:
            while not stop.is_set():
                row = {
                    "time": time.time(),
                    "utc": datetime.now(timezone.utc).isoformat(),
                    "gpus": [],
                }
                for device in devices:
                    metrics = rsmi_gpu_metrics_t()
                    status = smi.rsmi_dev_gpu_metrics_info_get(
                        device["smi_index"], ctypes.byref(metrics)
                    )
                    entry = dict(**device, status=status)
                    if status == 0:
                        entry.update({key: getattr(metrics, key) for key in fields})
                        old = previous.get(device["pci"])
                        if old is not None:
                            interval = (
                                entry["accumulation_counter"]
                                - old["accumulation_counter"]
                            )
                            for key in counters:
                                delta = entry[key] - old[key]
                                entry[key + "_interval_pct"] = (
                                    100 * delta / interval
                                    if interval > 0 and 0 <= delta <= interval
                                    else None
                                )
                        previous[device["pci"]] = entry
                    row["gpus"].append(entry)
                output.write(json.dumps(row) + "\n")
                if any(gpu["status"] != 0 for gpu in row["gpus"]):
                    raise RuntimeError(
                        "GPU metrics unavailable; inspect status in telemetry.jsonl"
                    )
                sample += 1
                if args.samples and sample >= args.samples:
                    break
                stop.wait(args.interval)
    finally:
        smi.rsmi_shut_down()


if __name__ == "__main__":
    main()
