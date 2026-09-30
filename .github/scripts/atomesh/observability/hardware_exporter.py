"""One sysfs hardware exporter per node; no inference-engine imports.

Use --register to resolve a worker's HIP-visible devices once, in its launch
 environment. The long-lived exporter only reads sysfs and registration JSON.
"""

from __future__ import annotations

import argparse
import ctypes
import ctypes.util
import json
import math
import re
import signal
import socket
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

# key: (Prometheus metric, display title, display unit, sysfs divisor)
METRICS = {
    "sclk": ("atom_gpu_sclk_mhz", "Core clock", "MHz", 1e6),
    "mclk": ("atom_gpu_mclk_mhz", "Memory clock", "MHz", 1e6),
    "junction": ("atom_gpu_junction_celsius", "GPU hotspot temperature", "°C", 1e3),
    "mem": ("atom_gpu_memory_celsius", "HBM temperature", "°C", 1e3),
    "power": ("atom_gpu_power_watts", "GPU power", "W", 1e6),
    "power_cap": ("atom_gpu_power_cap_watts", "Power limit", "W", 1e6),
    "busy": ("atom_gpu_busy_percent", "GPU busy", "%", 1),
    "memory_busy": ("atom_gpu_memory_busy_percent", "Memory busy", "%", 1),
    "vram_used": ("atom_gpu_vram_used_bytes", "VRAM used", "GiB", 1),
    "vram_total": ("atom_gpu_vram_total_bytes", "VRAM capacity", "GiB", 1),
}


def positive_seconds(value):
    result = float(value)
    if not math.isfinite(result) or result < 0.001:
        raise argparse.ArgumentTypeError("interval must be finite and >= 0.001 seconds")
    return result


def pci_address(value):
    value = value.lower()
    if not re.fullmatch(r"[0-9a-f]{4}:[0-9a-f]{2}:[0-9a-f]{2}\.[0-7]", value):
        raise ValueError(f"Invalid PCI address: {value}")
    return value


def hip_devices(count):
    """Honor HIP/ROCR visibility exactly as the worker will; never guess indices."""
    library = ctypes.util.find_library("amdhip64") or "libamdhip64.so"
    hip = ctypes.CDLL(library)
    hip.hipGetDeviceCount.argtypes = [ctypes.POINTER(ctypes.c_int)]
    hip.hipGetDeviceCount.restype = ctypes.c_int
    hip.hipDeviceGetPCIBusId.argtypes = [ctypes.c_char_p, ctypes.c_int, ctypes.c_int]
    hip.hipDeviceGetPCIBusId.restype = ctypes.c_int
    available = ctypes.c_int()
    rc = hip.hipGetDeviceCount(ctypes.byref(available))
    if rc or not 0 < count <= available.value:
        raise RuntimeError(
            f"HIP device discovery failed: rc={rc}, visible={available.value}, requested={count}"
        )
    result = []
    for index in range(count):
        address = ctypes.create_string_buffer(32)
        rc = hip.hipDeviceGetPCIBusId(address, len(address), index)
        if rc:
            raise RuntimeError(f"HIP PCI lookup failed for device {index}: rc={rc}")
        result.append(pci_address(address.value.decode()))
    return result


def discover(root):
    """Return physical AMD GPU sensors keyed by BDF, deduplicating DRM aliases."""
    devices = {}
    for card in sorted(root.glob("card[0-9]*")):
        if not re.fullmatch(r"card\d+", card.name):
            continue
        device = (card / "device").resolve()
        try:
            bdf = pci_address(device.name)
            hwmon = next(
                h
                for h in (device / "hwmon").glob("hwmon*")
                if (h / "name").read_text().strip() == "amdgpu"
            )
        except (OSError, ValueError, StopIteration):
            continue
        if bdf in devices:
            continue
        sensors = {
            "busy": device / "gpu_busy_percent",
            "memory_busy": device / "mem_busy_percent",
            "vram_used": device / "mem_info_vram_used",
            "vram_total": device / "mem_info_vram_total",
        }
        for label in hwmon.glob("*_label"):
            try:
                name = label.read_text().strip()
            except OSError:
                continue
            stem = label.name.removesuffix("_label")
            if (stem.startswith("freq") and name in {"sclk", "mclk"}) or (
                stem.startswith("temp") and name in {"junction", "mem"}
            ):
                sensors[name] = hwmon / (stem + "_input")
            if stem.startswith("power") and name == "PPT":
                sensors["power"] = hwmon / (stem + "_input")
                if not sensors["power"].exists():
                    sensors["power"] = hwmon / (stem + "_average")
                sensors["power_cap"] = hwmon / (stem + "_cap")
        # Older amdgpu drivers expose power1 without a label.
        if "power" not in sensors:
            sensors["power"] = hwmon / (
                "power1_input"
                if (hwmon / "power1_input").exists()
                else "power1_average"
            )
            sensors["power_cap"] = hwmon / "power1_cap"
        devices[bdf] = (card.name, sensors)
    return devices


def registrations(directory):
    roles = {}
    errors = 0
    for path in sorted(directory.glob("*.json")):
        try:
            registration = json.loads(path.read_text())
            role = registration["role"]
            if role not in {"prefill", "decode", "standalone"}:
                raise ValueError("Invalid GPU role")
            for bdf in registration["devices"]:
                roles.setdefault(pci_address(bdf), set()).add(role)
        except (OSError, ValueError, KeyError, TypeError):
            errors += 1
    return {bdf: "+".join(sorted(values)) for bdf, values in roles.items()}, errors


def line(name, value, labels=None):
    encoded = ",".join(
        f"{key}={json.dumps(str(value), ensure_ascii=False)}"
        for key, value in (labels or {}).items()
    )
    return f"{name}{{{encoded}}} {value}\n"


class Sampler:
    def __init__(
        self, root, *, hostname, interval, registration_dir=None, pci_roles=None
    ):
        self.root = root
        self.hostname = hostname
        self.interval = interval
        self.registration_dir = registration_dir
        self.pci_roles = pci_roles
        self.body = b""
        self.updated = 0.0
        self.lock = threading.Lock()

    def sample(self):
        started = time.monotonic()
        devices = discover(self.root)
        roles, registration_errors = (
            registrations(self.registration_dir)
            if self.registration_dir
            else (self.pci_roles, 0)
        )
        if roles is None:  # Explicit --all-devices mode only.
            roles = {bdf: "standalone" for bdf in devices}
        output = [
            line("atom_hardware_registration_errors", registration_errors),
            line("atom_hardware_selected_devices", len(roles)),
        ]
        for bdf, role in sorted(roles.items()):
            card, sensors = devices.get(bdf, ("unavailable", {}))
            labels = {
                "hostname": self.hostname,
                "pci_bdf": bdf,
                "card": card,
                "gpu_role": role,
            }
            output.append(line("atom_gpu_info", 1, labels))
            for key, (metric, _, _, divisor) in METRICS.items():
                value = None
                try:
                    value = float(sensors[key].read_text().strip()) / divisor
                    if (
                        not math.isfinite(value)
                        or value < 0
                        or (key in {"busy", "memory_busy"} and value > 100)
                    ):
                        value = None
                except (OSError, ValueError, KeyError):
                    pass
                output.append(
                    line(
                        "atom_gpu_sensor_available",
                        int(value is not None),
                        {**labels, "sensor": key},
                    )
                )
                if value is not None:
                    output.append(line(metric, value, labels))
        output.extend(
            [
                line("atom_hardware_sample_timestamp_seconds", time.time()),
                line(
                    "atom_hardware_sample_duration_seconds", time.monotonic() - started
                ),
                line("atom_hardware_sample_interval_seconds", self.interval),
            ]
        )
        # Keep each Prometheus family contiguous, with an explicit gauge type.
        exposition = []
        previous = None
        for sample in sorted(output):
            metric = sample.split("{", 1)[0]
            if metric != previous:
                exposition.append(f"# TYPE {metric} gauge\n")
                previous = metric
            exposition.append(sample)
        with self.lock:
            self.body = "".join(exposition).encode()
            self.updated = time.monotonic()

    def snapshot(self):
        with self.lock:
            return self.body, time.monotonic() - self.updated <= max(
                5, 3 * self.interval
            )


def handler_for(sampler):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path not in {"/metrics", "/health"}:
                self.send_error(404)
                return
            body, fresh = sampler.snapshot()
            if self.path == "/health":
                body = b"ok\n" if fresh else b"stale\n"
            self.send_response(200 if fresh else 503)
            self.send_header("Content-Type", "text/plain; version=0.0.4; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *_):
            pass

    return Handler


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--register",
        type=Path,
        help="Write a worker's HIP-visible PCI mapping, then exit",
    )
    mode.add_argument("--registration-dir", type=Path)
    mode.add_argument(
        "--all-devices",
        action="store_true",
        help="Explicitly monitor all AMD GPUs on this host",
    )
    mode.add_argument(
        "--pci-role",
        action="append",
        help="PCI_BDF=prefill|decode|standalone; repeat per GPU",
    )
    parser.add_argument(
        "--role", choices=("prefill", "decode", "standalone"), default="standalone"
    )
    parser.add_argument("--device-count", type=int, default=1)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=9108)
    parser.add_argument("--hostname", default=socket.gethostname())
    parser.add_argument("--sysfs-root", type=Path, default=Path("/sys/class/drm"))
    parser.add_argument("--sample-interval-seconds", type=positive_seconds, default=1.0)
    args = parser.parse_args()
    if args.register:
        data = {"role": args.role, "devices": hip_devices(args.device_count)}
        args.register.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.register.with_suffix(".tmp")
        temporary.write_text(json.dumps(data))
        temporary.replace(args.register)
        print(json.dumps(data), flush=True)
        return
    roles = None
    if args.pci_role:
        roles = {}
        for item in args.pci_role:
            bdf, role = item.split("=", 1)
            if (
                role not in {"prefill", "decode", "standalone"}
                or pci_address(bdf) in roles
            ):
                parser.error("Each PCI address must have one valid role")
            roles[pci_address(bdf)] = role
    sampler = Sampler(
        args.sysfs_root,
        hostname=args.hostname,
        interval=args.sample_interval_seconds,
        registration_dir=args.registration_dir,
        pci_roles=roles,
    )
    sampler.sample()
    stop = threading.Event()

    def sample_loop():
        while not stop.wait(args.sample_interval_seconds):
            try:
                sampler.sample()
            except (OSError, ValueError, KeyError, TypeError) as exc:
                print(f"[hardware] sample failed: {exc}", flush=True)

    with ThreadingHTTPServer((args.host, args.port), handler_for(sampler)) as server:
        server.timeout = 0.5
        for sig in (signal.SIGTERM, signal.SIGINT):
            signal.signal(sig, lambda *_: stop.set())
        worker = threading.Thread(target=sample_loop, daemon=True)
        worker.start()
        print(
            f"[hardware] listening on {server.server_address}, interval={args.sample_interval_seconds}s",
            flush=True,
        )
        try:
            while not stop.is_set():
                server.handle_request()
        finally:
            stop.set()
            worker.join(timeout=2)


if __name__ == "__main__":
    main()
