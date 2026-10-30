"""Verify the shared RDMA startup fix mounted over the pinned Python source."""
from __future__ import annotations

import hashlib
import os
from pathlib import Path

BASE_ENGINE_SHA = "81198e97ba7eee2a22540caaa756b7fdddcb4d93"
RELATIVE_PATH = "distributed/kv_transfer/kv_connector/v1/moriio/moriio_common.py"
BASE_SHA256 = "819cff0ed4157533a5793abf69ac609aba1176f01ca849b10adb2be37a721a48"
PATCHED_SHA256 = "23cd517ed12a0102bbedd0aeb755fae7a865f8d5f56bb2bb1f59e02d0d804268"
BEFORE = b"            local_kv_port=get_open_port(),\n"
AFTER = (
    b"            # Reserve the RDMA listener in bind(0); MoRI publishes its actual port.\n"
    b'            local_kv_port=0 if backend == "rdma" else get_open_port(),\n'
)


# The same fix ported onto vLLM 3966924d for -N3966- cases (n3966924d/manifest.json).
PORTED = {
    "73165df8f5a7f1b1bff21118aea045447538343ce6c4e65db40add424a246b22": (
        "3966924d566340e971a6da313e790d05492aa505",
        "b14418194b379d89d24ceebf167942e4a4e51c2c5f7f3727f08ad95d8643f600"),
}


def verify_port_patch(vllm_root: Path) -> dict:
    path = vllm_root / RELATIVE_PATH
    source = path.read_bytes()
    digest = hashlib.sha256(source).hexdigest()
    base_engine_sha, base_sha256 = PORTED.get(digest, (BASE_ENGINE_SHA, BASE_SHA256))
    if digest != PATCHED_SHA256 and digest not in PORTED:
        raise RuntimeError(f"C48 requires the prepared read-only source mount: {path} sha256={digest}")
    # Evidence only: this function never writes to the root-owned image package.
    readonly_mount = False
    for line in Path('/proc/self/mountinfo').read_text().splitlines():
        fields = line.split()
        if fields[4].replace('\\040', ' ') == str(path):
            readonly_mount = 'ro' in fields[5].split(',')
    return {
        "name": "mori_rdma_atomic_listener_bind",
        "base_engine_sha": base_engine_sha,
        "path": str(path),
        "base_sha256": base_sha256,
        "patched_sha256": digest,
        "observed_sha256": digest,
        "deployment": "single_file_readonly_bind_mount",
        "readonly_mount_observed": readonly_mount,
        "runtime_uid": os.geteuid(),
        "scope": "RDMA listener allocation during startup; shared by all four cases",
    }
