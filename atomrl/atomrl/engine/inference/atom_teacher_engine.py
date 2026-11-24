# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

"""ATOM-based teacher inference engine for Eagle3 speculative distillation.

Runs the teacher model in a **separate subprocess** on dedicated GPUs
with tensor parallelism and optional MXFP4/FP8 quantization.  Designed
for the 4+4 GPU split strategy:

- GPUs 0-3: training ranks (FSDP2, draft model)
- GPUs 4-7: ATOM subprocess (TP=4, teacher model, forward-only)

The subprocess uses ATOM's ``AsyncLLMEngine`` (with ``RLHFModelRunner``)
which handles TP process spawning internally via ``AsyncIOProcManager``.
Hidden states (3 aux layers + last hidden) are captured via
``register_forward_hook`` on decoder layers and transferred via
**Mooncake TCP** — the same transport used by ``VllmTeacherEngine``.

Three auxiliary hidden states from layers [1, N//2-1, N-4] are captured
by ``RLHFModelRunner.configure_hidden_states()`` which registers hooks
on the decoder layers without modifying any ``@support_torch_compile``
model files.

Requires ATOM from the ``sijyang/torchspec_dev`` branch which adds:
- ``atom.rollout.async_engine.AsyncLLMEngine``
- ``atom.rollout.model_runner_ext.RLHFModelRunner``
- ``atom.rollout.engine_utility.EngineUtilityHandler``
"""

from __future__ import annotations

import errno
import json
import logging
import os
import select
import signal
import socket
import subprocess
import sys
import tempfile
import textwrap
import threading
import time
from typing import Any

import torch

logger = logging.getLogger(__name__)


def _detect_local_ip() -> str:
    """Return the route-selected local IP, falling back to hostname lookup."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as route_socket:
            route_socket.connect(("8.8.8.8", 80))
            return route_socket.getsockname()[0]
    except OSError:
        return socket.gethostbyname(socket.gethostname())


_HIDDEN_XFER_DIR = os.environ.get(
    "ATOMRL_TEACHER_HIDDEN_DIR",
    "/dev/shm/atomrl_teacher_hidden",
)
_READY_TIMEOUT_SECONDS = float(
    os.environ.get("ATOMRL_TEACHER_READY_TIMEOUT_SECONDS", "600"),
)
_CMD_FIFO_OPEN_TIMEOUT_SECONDS = 10.0
_DEFAULT_COMMAND_TIMEOUT_SECONDS = 600.0
_SHUTDOWN_COMMAND_TIMEOUT_SECONDS = 5.0
_WORKER_TERMINATE_GRACE_SECONDS = 10.0

# ---------------------------------------------------------------------------
# Worker subprocess script
#
# Runs in a separate process on the teacher GPUs.  Uses ATOM's
# AsyncLLMEngine which internally spawns tp_size ModelRunner processes
# via AsyncIOProcManager, handling NCCL distributed init correctly.
#
# Communication:
# - Named FIFOs for JSON commands + responses (small control messages)
# - Mooncake TCP for hidden state transfer (via RLHFModelRunner)
# - /dev/shm for input_ids (small, a few KB per batch)
# - /dev/shm for teacher weights (lm_head, embed, norm)
# ---------------------------------------------------------------------------

_TEACHER_WORKER_SCRIPT = textwrap.dedent("""\
import gc, json, os, sys, logging, time, socket, glob

# ---- Ensure atomrl is importable (for ATOM's fallback imports) ----
for _p in ["/root/atomrl", os.getcwd()]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

# ---- Isolate from torchrun environment ----
os.environ.pop("PYTORCH_CUDA_ALLOC_CONF", None)
for key in list(os.environ.keys()):
    if any(key.startswith(p) for p in [
        "MASTER_ADDR", "MASTER_PORT", "RANK", "LOCAL_RANK",
        "WORLD_SIZE", "LOCAL_WORLD_SIZE", "GROUP_RANK",
        "GROUP_WORLD_SIZE", "ROLE_RANK", "ROLE_WORLD_SIZE",
        "TORCHELASTIC_", "TORCH_NCCL_", "NCCL_ASYNC",
        "OMP_NUM_THREADS",
    ]):
        del os.environ[key]

# Suppress noisy logs from mooncake C++ (glog) and aiter
os.environ["GLOG_minloglevel"] = "3"
os.environ["GLOG_v"] = "0"
os.environ["MOONCAKE_LOG_LEVEL"] = "FATAL"
os.environ["AITER_LOG_LEVEL"] = "WARNING"

logging.basicConfig(stream=sys.stderr, level=logging.INFO)
logger = logging.getLogger("atom_teacher")

import torch
from transformers import AutoConfig

# Parse arguments
cmd_fifo = sys.argv[1]
resp_fifo = sys.argv[2]
model_path = sys.argv[3]
tp_size = int(sys.argv[4])
hidden_dir = sys.argv[5]
max_batch = int(sys.argv[6])
max_seq = int(sys.argv[7])
atom_args_json = sys.argv[8] if len(sys.argv) > 8 else "{}"
atom_extra = json.loads(atom_args_json)

os.makedirs(hidden_dir, exist_ok=True)

# ---- Extract weights from checkpoint (CPU only, before engine) ----
hf_config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
hf_text = getattr(hf_config, "text_config", hf_config)
num_layers = hf_text.num_hidden_layers
hidden_dim = hf_text.hidden_size
norm_eps = getattr(hf_text, "rms_norm_eps", 1e-6)

from safetensors.torch import load_file as st_load
st_files = sorted(glob.glob(os.path.join(model_path, "*.safetensors")))

lm_head_weight = embed_weight = norm_weight = None

for sf in st_files:
    tensors = st_load(sf, device="cpu")
    for name, tensor in tensors.items():
        if "lm_head" in name and "weight" in name and lm_head_weight is None:
            lm_head_weight = tensor
        if "embed_tokens" in name and "weight" in name and embed_weight is None:
            embed_weight = tensor
        if norm_weight is None and "norm.weight" in name and "layer" not in name:
            if any(p in name for p in ["model.norm.weight",
                                        "language_model.model.norm.weight"]):
                norm_weight = tensor
    del tensors
    if all(w is not None for w in [lm_head_weight, embed_weight, norm_weight]):
        break

if lm_head_weight is None or embed_weight is None:
    raise RuntimeError(
        f"Could not find lm_head/embed_tokens weights in {model_path}. "
        f"Searched {len(st_files)} safetensors files."
    )

torch.save(lm_head_weight, os.path.join(hidden_dir, "lm_head_weight.pt"))
torch.save(embed_weight, os.path.join(hidden_dir, "embed_weight.pt"))
if norm_weight is not None:
    torch.save({"weight": norm_weight, "eps": norm_eps},
               os.path.join(hidden_dir, "norm_weight.pt"))
logger.info(
    "Weights saved: lm_head=%s, embed=%s, norm=%s",
    lm_head_weight.shape, embed_weight.shape,
    norm_weight.shape if norm_weight is not None else "N/A",
)
del lm_head_weight, embed_weight, norm_weight

# ---- Create AsyncLLMEngine (handles TP internally) ----
from atom.rollout.async_engine import AsyncLLMEngine
logging.getLogger("atom").setLevel(logging.WARNING)

engine_kwargs = dict(
    tensor_parallel_size=tp_size,
    enforce_eager=atom_extra.pop("enforce_eager", True),
    trust_remote_code=atom_extra.pop("trust_remote_code", True),
    max_num_batched_tokens=atom_extra.pop("max_num_batched_tokens", 32768),
    max_num_seqs=atom_extra.pop("max_num_seqs", 64),
)
for k, v in atom_extra.items():
    engine_kwargs[k] = v

logger.info("Creating AsyncLLMEngine: %s", engine_kwargs)
engine = AsyncLLMEngine(model_path, **engine_kwargs)
logger.info("AsyncLLMEngine created successfully")

# ---- Configure hidden states extraction ----
aux_layer_ids = [1, num_layers // 2 - 1, num_layers - 4]
max_seq = engine_kwargs.get("max_model_len", max_seq)
mooncake_config = {
    "local_hostname": os.environ.get("MOONCAKE_LOCAL_HOSTNAME", "localhost"),
    "metadata_server": os.environ.get("MOONCAKE_METADATA_SERVER", ""),
    "master_server_address": os.environ.get("MOONCAKE_MASTER_SERVER", ""),
    "protocol": os.environ.get("MOONCAKE_PROTOCOL", "tcp"),
    "device_name": os.environ.get("MOONCAKE_DEVICE_NAME", ""),
    "global_segment_size": int(os.environ.get(
        "MOONCAKE_GLOBAL_SEGMENT_SIZE", str(16 * 1024**3))),
    "local_buffer_size": int(os.environ.get(
        "MOONCAKE_LOCAL_BUFFER_SIZE", str(4 * 1024**3))),
    "enable_gpu_direct": os.environ.get("MOONCAKE_ENABLE_GPU_DIRECT", "0") == "1",
    "enable_hard_pin": os.environ.get("MOONCAKE_ENABLE_HARD_PIN", "0") == "1",
    "max_seq_len": max_seq,
    "hidden_dim": hidden_dim,
}
engine.configure_hidden_states(aux_layer_ids, mooncake_config)
logger.info("Hidden states configured: aux_layers=%s", aux_layer_ids)

# ---- Signal ready ----
resp_f = open(resp_fifo, "w")
resp_f.write(json.dumps({
    "status": "ready",
    "hidden_dim": hidden_dim,
    "num_layers": num_layers,
    "aux_layer_indices": aux_layer_ids,
}) + "\\n")
resp_f.flush()

cmd_f = open(cmd_fifo, "r")
req_counter = 0

# ---- Command loop ----
for line in cmd_f:
    line = line.strip()
    if not line:
        continue
    msg = json.loads(line)
    cmd = msg["cmd"]

    if cmd == "extract_hidden":
        input_path = msg["input_path"]
        data = torch.load(input_path, map_location="cpu", weights_only=True)
        input_ids_batch = data["input_ids"]  # [B, T]
        B, T = input_ids_batch.shape
        req_counter += 1

        input_ids_list = [input_ids_batch[i].tolist() for i in range(B)]
        data_ids = [f"atom_{os.getpid()}_{req_counter}_{i}" for i in range(B)]

        t0 = time.monotonic()
        engine.generate_hidden_states(input_ids_list, data_ids)
        elapsed = time.monotonic() - t0
        logger.info("extract_hidden: B=%d, T=%d, %.2fs", B, T, elapsed)

        resp_f.write(json.dumps({
            "status": "ok", "B": B, "T": T, "D": hidden_dim,
            "mooncake_keys": data_ids,
        }) + "\\n")
        resp_f.flush()
        del data, input_ids_batch, input_ids_list

    elif cmd == "shutdown":
        try:
            engine.shutdown()
        except Exception:
            pass
        gc.collect()
        resp_f.write(json.dumps({"status": "ok"}) + "\\n")
        resp_f.flush()
        break

cmd_f.close()
resp_f.close()
""")


class AtomTeacherEngine:
    """ATOM-based teacher inference for Eagle3 speculative distillation.

    Runs the teacher model in a separate subprocess on dedicated GPUs,
    with tensor parallelism and MXFP4/FP8 quantization.  Hidden states
    (3 aux layers + last hidden) are transferred via Mooncake TCP.

    Communication protocol:
    - Named FIFOs for JSON commands (small control messages)
    - Mooncake TCP ``EagleMooncakeStore`` for hidden state transfer
    - ``/dev/shm`` files for ``input_ids`` (a few KB, not worth Mooncake)
    """

    def __init__(
        self,
        model_name: str,
        tensor_parallel_size: int = 4,
        gpu_ids: list[int] | None = None,
        *,
        mooncake_config: Any = None,
        transport: str = "mooncake",
        quantization: str = "",
        atom_config: dict[str, Any] | None = None,
        max_batch_size: int = 32,
        max_seq_len: int = 4096,
        local_device: torch.device | None = None,
    ) -> None:
        self._model_name = model_name
        self._tp_size = tensor_parallel_size
        self._gpu_ids = gpu_ids or list(range(tensor_parallel_size))
        self._hidden_dir = _HIDDEN_XFER_DIR
        self._mooncake_config = mooncake_config
        self._transport = transport
        self._quantization = quantization
        self._atom_config = atom_config or {}
        self._max_batch = max_batch_size
        self._max_seq = max_seq_len
        self._local_device = local_device or torch.device("cuda:0")

        self._proc: subprocess.Popen | None = None
        self._fifo_dir: str | None = None
        self._cmd_fifo: str | None = None
        self._resp_fifo: str | None = None
        self._cmd_f = None
        self._resp_f = None
        self._initialized = False
        self._req_counter = 0

        self._hidden_dim: int = 0
        self._num_layers: int = 0
        self._cmd_lock = threading.Lock()
        self._aux_layer_indices: list[int] = []
        self._mooncake_store: Any = None

    @property
    def is_alive(self) -> bool:
        return (
            self._initialized and self._proc is not None and self._proc.poll() is None
        )

    def _describe_worker_exit(self, exit_code: int) -> str:
        if exit_code < 0:
            sig = -exit_code
            signal_name = (
                signal.Signals(sig).name
                if sig in signal.Signals._value2member_map_
                else f"SIG{sig}"
            )
            return f"exited by signal {signal_name} ({sig})"
        return f"exited with code {exit_code}"

    def _terminate_worker(self, reason: str) -> None:
        if self._proc is None:
            return

        if self._proc.poll() is None:
            logger.warning(
                "AtomTeacherEngine: terminating worker pid=%d (%s).",
                self._proc.pid,
                reason,
            )
            try:
                os.killpg(self._proc.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            except OSError as exc:
                logger.debug("Failed to SIGTERM worker group: %s", exc)
            try:
                self._proc.wait(timeout=_WORKER_TERMINATE_GRACE_SECONDS)
            except subprocess.TimeoutExpired:
                logger.warning(
                    "AtomTeacherEngine: worker pid=%d did not exit after %.1fs; sending SIGKILL.",
                    self._proc.pid,
                    _WORKER_TERMINATE_GRACE_SECONDS,
                )
                try:
                    os.killpg(self._proc.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                except OSError as exc:
                    logger.debug("Failed to SIGKILL worker group: %s", exc)
                try:
                    self._proc.wait(timeout=5)
                except (OSError, subprocess.TimeoutExpired):
                    pass

    def _cleanup_ipc_resources(self) -> None:
        if hasattr(self, "_worker_log_f") and self._worker_log_f:
            try:
                self._worker_log_f.close()
            except OSError:
                pass
            self._worker_log_f = None
        for f in [self._cmd_f, self._resp_f]:
            try:
                if f:
                    f.close()
            except OSError:
                pass
        self._cmd_f = None
        self._resp_f = None

        for p in [self._cmd_fifo, self._resp_fifo]:
            try:
                if p and os.path.exists(p):
                    os.unlink(p)
            except OSError:
                pass
        self._cmd_fifo = None
        self._resp_fifo = None

        try:
            if self._fifo_dir and os.path.isdir(self._fifo_dir):
                os.rmdir(self._fifo_dir)
        except OSError:
            pass
        self._fifo_dir = None

    def _wait_for_ready(self, timeout_s: float) -> dict[str, Any]:
        if self._resp_fifo is None:
            raise RuntimeError("Response FIFO path is missing")
        if self._proc is None:
            raise RuntimeError("Teacher worker process is not started")

        fd = os.open(self._resp_fifo, os.O_RDONLY | os.O_NONBLOCK)
        deadline = time.monotonic() + timeout_s
        buf = ""

        try:
            while True:
                poll_code = self._proc.poll()
                if poll_code is not None:
                    raise RuntimeError(
                        f"teacher worker {self._describe_worker_exit(poll_code)} before READY",
                    )

                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError(
                        f"teacher worker did not send READY within {timeout_s:.1f}s",
                    )

                ready, _, _ = select.select([fd], [], [], min(0.5, remaining))
                if not ready:
                    continue

                try:
                    chunk = os.read(fd, 4096)
                except BlockingIOError:
                    continue
                if not chunk:
                    continue

                buf += chunk.decode("utf-8", errors="replace")
                if "\n" not in buf:
                    continue

                line, _, _ = buf.partition("\n")
                line = line.strip()
                if not line:
                    continue
                try:
                    return json.loads(line)
                except json.JSONDecodeError as exc:
                    raise RuntimeError(
                        f"teacher worker sent invalid READY payload: {line[:200]!r}",
                    ) from exc
        finally:
            os.close(fd)

    def _open_cmd_writer(self, timeout_s: float) -> Any:
        if self._cmd_fifo is None:
            raise RuntimeError("Command FIFO path is missing")
        if self._proc is None:
            raise RuntimeError("Teacher worker process is not started")

        deadline = time.monotonic() + timeout_s
        while True:
            poll_code = self._proc.poll()
            if poll_code is not None:
                raise RuntimeError(
                    f"teacher worker {self._describe_worker_exit(poll_code)} before command channel setup",
                )
            try:
                fd = os.open(self._cmd_fifo, os.O_WRONLY | os.O_NONBLOCK)
                return os.fdopen(fd, "w")
            except OSError as exc:
                if exc.errno != errno.ENXIO:
                    raise
                if time.monotonic() >= deadline:
                    raise TimeoutError(
                        f"teacher worker did not open command FIFO within {timeout_s:.1f}s",
                    ) from exc
                time.sleep(0.1)

    def _read_response_line(self, *, timeout_s: float, context: str) -> str:
        if self._resp_f is None:
            raise RuntimeError("Response FIFO is not open")
        if self._proc is None:
            raise RuntimeError("Teacher worker process is not started")

        deadline = time.monotonic() + timeout_s
        resp_fd = self._resp_f.fileno()
        while True:
            poll_code = self._proc.poll()
            if poll_code is not None:
                raise RuntimeError(
                    f"teacher worker {self._describe_worker_exit(poll_code)} while waiting for {context}",
                )

            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(
                    f"timed out waiting for {context} after {timeout_s:.1f}s",
                )

            ready, _, _ = select.select([resp_fd], [], [], min(0.5, remaining))
            if not ready:
                continue

            resp_line = self._resp_f.readline()
            if not resp_line:
                raise RuntimeError(
                    f"response FIFO closed while waiting for {context}; worker likely crashed",
                )
            return resp_line

    def start(self) -> None:
        """Start the teacher worker subprocess and set up Mooncake store."""
        if self.is_alive:
            return

        os.makedirs(self._hidden_dir, exist_ok=True)

        self._fifo_dir = tempfile.mkdtemp(prefix="atomrl_teacher_fifo_")
        self._cmd_fifo = os.path.join(self._fifo_dir, "cmd")
        self._resp_fifo = os.path.join(self._fifo_dir, "resp")
        os.mkfifo(self._cmd_fifo)
        os.mkfifo(self._resp_fifo)

        # Build environment: isolate from torchrun, set teacher GPUs
        env = os.environ.copy()
        gpu_str = ",".join(str(g) for g in self._gpu_ids)
        env["CUDA_VISIBLE_DEVICES"] = gpu_str
        env["HIP_VISIBLE_DEVICES"] = gpu_str
        env.pop("ROCR_VISIBLE_DEVICES", None)
        env.pop("PYTORCH_CUDA_ALLOC_CONF", None)
        for key in list(env.keys()):
            if any(
                key.startswith(p)
                for p in [
                    "MASTER_ADDR",
                    "MASTER_PORT",
                    "RANK",
                    "LOCAL_RANK",
                    "WORLD_SIZE",
                    "LOCAL_WORLD_SIZE",
                    "GROUP_RANK",
                    "GROUP_WORLD_SIZE",
                    "ROLE_RANK",
                    "ROLE_WORLD_SIZE",
                    "TORCHELASTIC_",
                    "TORCH_NCCL_",
                    "NCCL_ASYNC",
                    "OMP_NUM_THREADS",
                ]
            ):
                del env[key]

        env["GLOG_minloglevel"] = "3"
        env["GLOG_v"] = "0"
        env["MOONCAKE_LOG_LEVEL"] = "FATAL"
        env["AITER_LOG_LEVEL"] = "WARNING"
        if "AITER_CONFIG_GEMM_BF16" not in env:
            _bf16_cfg = ""
            aiter_root = os.environ.get("AITER_DIR", "/workspace/aiter")
            _bf16_cfg = os.path.join(
                aiter_root,
                "aiter",
                "configs",
                "model_configs",
                "kimik2_bf16_tuned_gemm.csv",
            )
            env["AITER_CONFIG_GEMM_BF16"] = _bf16_cfg

        attn_backend = os.environ.get("VLLM_ROCM_ATTN_BACKEND")
        if attn_backend:
            env["VLLM_ROCM_ATTN_BACKEND"] = attn_backend

        # Set Mooncake env vars for worker subprocess
        if self._transport == "mooncake" and self._mooncake_config is not None:
            mc = self._mooncake_config
            local_ip = _detect_local_ip()

            master_addr = getattr(mc, "master_server_address", "") or ""
            metadata_server = getattr(mc, "metadata_server", "") or ""
            protocol = getattr(mc, "protocol", "tcp") or "tcp"
            device_name = getattr(mc, "device_name", "") or ""

            env["MOONCAKE_LOCAL_HOSTNAME"] = local_ip
            env["MOONCAKE_MASTER_SERVER"] = master_addr
            env["MOONCAKE_METADATA_SERVER"] = metadata_server
            env["MOONCAKE_PROTOCOL"] = protocol
            env["MOONCAKE_DEVICE_NAME"] = device_name
            env["MOONCAKE_GLOBAL_SEGMENT_SIZE"] = str(
                mc.global_segment_size_bytes
                if hasattr(mc, "global_segment_size_bytes")
                else 17179869184
            )
            env["MOONCAKE_LOCAL_BUFFER_SIZE"] = str(
                mc.local_buffer_size_bytes
                if hasattr(mc, "local_buffer_size_bytes")
                else 4294967296
            )
            env["MOONCAKE_ENABLE_GPU_DIRECT"] = "0"
            env["MOONCAKE_ENABLE_HARD_PIN"] = "0"

            try:
                from transformers import AutoConfig as _AC

                _hf = _AC.from_pretrained(self._model_name, trust_remote_code=True)
                _hf_text = getattr(_hf, "text_config", _hf)
                _hdim = getattr(_hf_text, "hidden_size", 4096)
            except (ImportError, OSError, ValueError):
                _hdim = 4096
            from atomrl.transfer.eagle_mooncake_store import (
                calculate_eagle3_buffer_size,
            )

            worker_host_buf = calculate_eagle3_buffer_size(
                max_seq_len=self._max_seq,
                batch_size=self._max_batch,
                hidden_dim=_hdim,
                safety_margin=2.0,
            )
            env["MOONCAKE_HOST_BUFFER_SIZE"] = str(worker_host_buf)

            logger.info(
                "Mooncake env vars for ATOM worker: local=%s, master=%s, "
                "metadata=%s, protocol=%s",
                local_ip,
                master_addr,
                metadata_server,
                protocol,
            )

        logger.info(
            "AtomTeacherEngine: starting teacher worker for %s "
            "(tp=%d, gpus=%s, transport=%s, quant=%s)",
            self._model_name,
            self._tp_size,
            self._gpu_ids,
            self._transport,
            self._quantization or "none",
        )

        atom_args_json = json.dumps(self._atom_config)

        worker_log_path = os.path.join(self._hidden_dir, "atom_teacher_worker.log")
        # Subprocess log stays open until cleanup.
        self._worker_log_f = open(  # noqa: SIM115 -- lifetime handle
            worker_log_path, "w"
        )
        logger.info("AtomTeacherEngine: worker log -> %s", worker_log_path)

        self._proc = subprocess.Popen(
            [
                sys.executable,
                "-u",
                "-c",
                _TEACHER_WORKER_SCRIPT,
                self._cmd_fifo,
                self._resp_fifo,
                self._model_name,
                str(self._tp_size),
                self._hidden_dir,
                str(self._max_batch),
                str(self._max_seq),
                atom_args_json,
            ],
            stdin=subprocess.DEVNULL,
            stdout=self._worker_log_f,
            stderr=subprocess.STDOUT,
            env=env,
            start_new_session=True,
        )
        try:
            resp = self._wait_for_ready(timeout_s=_READY_TIMEOUT_SECONDS)
            if resp.get("status") != "ready":
                raise RuntimeError(f"worker reported non-ready startup status: {resp}")
            self._cmd_f = self._open_cmd_writer(
                timeout_s=_CMD_FIFO_OPEN_TIMEOUT_SECONDS,
            )
            # FIFO stays open for the worker lifetime.
            self._resp_f = open(self._resp_fifo, "r")  # noqa: SIM115 -- lifetime handle
        # Worker startup must surface arbitrary subprocess and backend failures.
        except Exception as exc:
            worker_log_tail = ""
            try:
                worker_log_path = os.path.join(
                    self._hidden_dir, "atom_teacher_worker.log"
                )
                if os.path.exists(worker_log_path):
                    with open(worker_log_path) as wlf:
                        lines = wlf.readlines()
                    worker_log_tail = "".join(lines[-50:])
            except OSError:
                pass
            logger.error(
                "AtomTeacherEngine: startup failed before READY or channel setup (%s). "
                "Root-cause hint: worker crash, missing READY, or startup timeout.\n"
                "--- worker log (last 50 lines) ---\n%s\n--- end worker log ---",
                exc,
                worker_log_tail,
            )
            self._terminate_worker("startup failure")
            self._cleanup_ipc_resources()
            self._proc = None
            self._initialized = False
            raise RuntimeError(
                f"Failed to start AtomTeacherEngine worker: {exc}",
            ) from exc

        self._hidden_dim = resp["hidden_dim"]
        self._num_layers = resp.get("num_layers", 0)
        self._aux_layer_indices = resp.get("aux_layer_indices", [])

        # Set up training-side Mooncake store
        if self._transport == "mooncake" and self._mooncake_config is not None:
            try:
                from atomrl.transfer.eagle_mooncake_store import EagleMooncakeStore
                from atomrl.transfer.mooncake_config import MooncakeConfig

                mc = self._mooncake_config
                local_ip = _detect_local_ip()

                mc_cfg = MooncakeConfig(
                    master_server_address=getattr(mc, "master_server_address", ""),
                    metadata_server=getattr(mc, "metadata_server", ""),
                    local_hostname=local_ip,
                    protocol=getattr(mc, "protocol", "tcp"),
                    device_name=getattr(mc, "device_name", ""),
                    global_segment_size=getattr(mc, "global_segment_size", "16GB"),
                    local_buffer_size=getattr(mc, "local_buffer_size", "4GB"),
                    max_seq_len=self._max_seq,
                    hidden_dim=self._hidden_dim,
                    get_retry_wait_seconds=getattr(mc, "get_retry_wait_seconds", 1.0),
                    get_retry_max_wait_seconds=getattr(
                        mc, "get_retry_max_wait_seconds", 90.0
                    ),
                )
                self._mooncake_store = EagleMooncakeStore(mc_cfg)
                self._mooncake_store.setup(self._local_device)
                logger.info("Training-side EagleMooncakeStore initialized (ATOM)")
            except Exception as e:
                logger.error("Failed to init training-side Mooncake: %s", e)
                self._terminate_worker("mooncake initialization failure")
                self._cleanup_ipc_resources()
                self._proc = None
                self._initialized = False
                raise

        self._initialized = True
        logger.info(
            "AtomTeacherEngine: teacher worker ready (pid=%d, hidden_dim=%d, "
            "aux_layers=%s, transport=%s).",
            self._proc.pid,
            self._hidden_dim,
            self._aux_layer_indices,
            self._transport,
        )

    def _send_cmd_unlocked(
        self,
        cmd: dict[str, Any],
        *,
        timeout_s: float = _DEFAULT_COMMAND_TIMEOUT_SECONDS,
    ) -> dict:
        """Send a JSON command and read the response (caller holds _cmd_lock)."""
        if not self.is_alive:
            raise RuntimeError("Teacher worker is not running")
        try:
            self._cmd_f.write(json.dumps(cmd) + "\n")
            self._cmd_f.flush()
        except BrokenPipeError as exc:
            raise RuntimeError(
                "Teacher worker command FIFO is broken; worker likely crashed",
            ) from exc
        resp_line = self._read_response_line(
            timeout_s=timeout_s,
            context=f"response to cmd={cmd.get('cmd', 'unknown')}",
        )
        return json.loads(resp_line)

    def _send_cmd(
        self,
        cmd: dict[str, Any],
        *,
        timeout_s: float = _DEFAULT_COMMAND_TIMEOUT_SECONDS,
    ) -> dict:
        """Send a JSON command and read the response (thread-safe)."""
        with self._cmd_lock:
            return self._send_cmd_unlocked(cmd, timeout_s=timeout_s)

    def extract_hidden_states(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        recv_device: torch.device | None = None,
    ) -> dict[str, torch.Tensor]:
        """Run teacher forward pass, return hidden states on training GPU.

        Args:
            input_ids: ``[B, T]`` token ids.
            attention_mask: ``[B, T]`` mask (1 = valid, 0 = pad).
            recv_device: Device for received tensors. Defaults to local GPU.

        Returns:
            Dict with ``hidden_states`` ``[B, T, 3*D]`` (3 aux layers),
            ``token_embeds`` ``[B, T, D]``,
            ``last_hidden_states`` ``[B, T, D]``,
            and ``input_ids``.
            All tensors on ``recv_device`` or ``self._local_device``.
        """
        if not self.is_alive:
            self.start()

        with self._cmd_lock:
            self._req_counter += 1
            tag = f"req_{self._req_counter}"
            input_path = os.path.join(self._hidden_dir, f"{tag}_input.pt")

            torch.save(
                {"input_ids": input_ids.cpu(), "attention_mask": attention_mask.cpu()},
                input_path,
            )

            resp = self._send_cmd_unlocked(
                {
                    "cmd": "extract_hidden",
                    "input_path": input_path,
                }
            )

        if resp.get("status") != "ok":
            raise RuntimeError(f"extract_hidden failed: {resp}")

        resp["B"]
        T = resp["T"]
        D = resp["D"]

        if recv_device is None:
            recv_device = self._local_device

        if self._transport == "mooncake":
            mooncake_keys = resp["mooncake_keys"]
            num_aux = len(self._aux_layer_indices)
            training_hidden_size = num_aux * D

            # Fetch per-request hidden states from Mooncake and stack
            all_hs = []
            all_ids = []
            all_last_hs = []

            for key in mooncake_keys:
                shapes = {
                    "hidden_states": (T, training_hidden_size),
                    "input_ids": (T,),
                    "last_hidden_states": (T, D),
                }
                dtypes = {
                    "hidden_states": torch.bfloat16,
                    "input_ids": torch.int64,
                    "last_hidden_states": torch.bfloat16,
                }

                output = self._mooncake_store.get(
                    key,
                    shapes,
                    dtypes,
                    device=recv_device,
                )

                all_hs.append(output.hidden_states)
                all_last_hs.append(output.last_hidden_states)
                all_ids.append(output.input_ids)

                self._mooncake_store.remove_eagle3_tensors(
                    key,
                    has_last_hidden_states=True,
                    has_target=False,
                )

            hidden_states = torch.stack(all_hs)  # [B, T, 3*D]
            last_hidden_states = torch.stack(all_last_hs)  # [B, T, D]
            ret_ids = torch.stack(all_ids)  # [B, T]

            # First aux layer as embed proxy (same convention as VllmTeacherEngine)
            token_embeds = hidden_states[:, :, :D].clone()

            return {
                "hidden_states": hidden_states,
                "token_embeds": token_embeds,
                "input_ids": ret_ids,
                "last_hidden_states": last_hidden_states,
            }
        else:
            raise AssertionError("MORI-IO transport not implemented for 3 aux layers")

    def get_lm_head_weight(self) -> torch.Tensor:
        """Load the teacher's lm_head.weight from shared memory."""
        if not self.is_alive:
            self.start()
        path = os.path.join(self._hidden_dir, "lm_head_weight.pt")
        for _ in range(30):
            if os.path.exists(path):
                return torch.load(path, map_location="cpu", weights_only=True)
            time.sleep(0.5)
        raise FileNotFoundError(f"lm_head_weight.pt not found at {path} after 15s.")

    def get_embed_weight(self) -> torch.Tensor:
        """Load the teacher's embed_tokens.weight from shared memory."""
        if not self.is_alive:
            self.start()
        path = os.path.join(self._hidden_dir, "embed_weight.pt")
        for _ in range(30):
            if os.path.exists(path):
                return torch.load(path, map_location="cpu", weights_only=True)
            time.sleep(0.5)
        raise FileNotFoundError(f"embed_weight.pt not found at {path} after 15s.")

    def get_norm_weight(self) -> tuple[torch.Tensor, float]:
        """Load the teacher's final norm weight and eps from shared memory."""
        if not self.is_alive:
            self.start()
        path = os.path.join(self._hidden_dir, "norm_weight.pt")
        for _ in range(30):
            if os.path.exists(path):
                data = torch.load(path, map_location="cpu", weights_only=False)
                return data["weight"], data["eps"]
            time.sleep(0.5)
        raise FileNotFoundError(f"norm_weight.pt not found at {path} after 15s.")

    def shutdown(self) -> None:
        """Terminate the teacher worker subprocess and clean up."""
        if self._proc is not None and self._proc.poll() is None:
            try:
                self._send_cmd(
                    {"cmd": "shutdown"},
                    timeout_s=_SHUTDOWN_COMMAND_TIMEOUT_SECONDS,
                )
            except (OSError, RuntimeError, ValueError) as exc:
                logger.warning(
                    "AtomTeacherEngine: graceful shutdown command failed: %s", exc
                )
            self._terminate_worker("shutdown requested")

        if self._mooncake_store is not None:
            try:
                if hasattr(self._mooncake_store, "close"):
                    self._mooncake_store.close()
                elif hasattr(self._mooncake_store, "shutdown"):
                    self._mooncake_store.shutdown()
            # Mooncake shutdown is best-effort across third-party store implementations.
            except Exception as exc:  # noqa: BLE001 -- boundary
                logger.warning(
                    "AtomTeacherEngine: failed to close Mooncake store: %s", exc
                )
            finally:
                self._mooncake_store = None

        self._cleanup_ipc_resources()
        self._proc = None
        self._initialized = False
        logger.info("AtomTeacherEngine: shutdown complete.")

    def __del__(self) -> None:
        self.shutdown()


__all__ = ["AtomTeacherEngine"]
