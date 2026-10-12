"""Apply the shared RDMA startup fix to the exact pinned vLLM Python source."""
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


def apply_port_patch(vllm_root: Path) -> dict:
    path = vllm_root / RELATIVE_PATH
    source = path.read_bytes()
    digest = hashlib.sha256(source).hexdigest()
    if digest == BASE_SHA256:
        assert source.count(BEFORE) == 1
        patched = source.replace(BEFORE, AFTER)
        assert hashlib.sha256(patched).hexdigest() == PATCHED_SHA256
        temporary = path.with_name(path.name + f".c48-{os.getpid()}.tmp")
        temporary.write_bytes(patched)
        temporary.chmod(path.stat().st_mode & 0o777)
        temporary.replace(path)
    elif digest != PATCHED_SHA256:
        raise RuntimeError(f"C48 port fix refuses unexpected vLLM source: {path} sha256={digest}")
    assert hashlib.sha256(path.read_bytes()).hexdigest() == PATCHED_SHA256
    return {
        "name": "mori_rdma_atomic_listener_bind",
        "base_engine_sha": BASE_ENGINE_SHA,
        "path": str(path),
        "base_sha256": BASE_SHA256,
        "patched_sha256": PATCHED_SHA256,
        "source_before_patch_sha256": digest,
        "scope": "RDMA listener allocation during startup; shared by all four cases",
    }
