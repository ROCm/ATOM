# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

"""ATOM-backed inference engine with subprocess isolation for RL training.

Uses ATOM's ``LLMEngine`` and ``SamplingParams`` APIs for efficient generation.
Runs ATOM in a **separate subprocess** via
``subprocess.Popen`` to get full process isolation from torchrun's
NCCL process groups.

Communication: uses a pair of named FIFOs (named pipes) for JSON
commands/responses, bypassing stdout/stderr which can be polluted
by C++ libraries (Gloo, NCCL).
"""

from __future__ import annotations

import gc
import json
import logging
import os
import subprocess
import sys
import tempfile
import textwrap
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch

from atomrl.core.config import AtomConfig

logger = logging.getLogger(__name__)

_WEIGHT_SYNC_DIR = os.environ.get(
    "ATOMRL_WEIGHT_SYNC_DIR",
    "/dev/shm/atomrl_weight_sync",
)

_WORKER_SCRIPT = textwrap.dedent("""\
import gc, json, os, sys, logging
import numpy as np

class _NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

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

logging.basicConfig(stream=sys.stderr, level=logging.INFO)

import torch
from atom import LLMEngine, SamplingParams

# Suppress ATOM's DEBUG spam ("Engine Core Mgr: callback=NOT FOUND" etc.)
logging.getLogger("atom").setLevel(logging.WARNING)

cmd_fifo = sys.argv[1]
resp_fifo = sys.argv[2]
model_path = sys.argv[3]
gpu_mem = float(sys.argv[4])
max_model_len = int(sys.argv[5]) if sys.argv[5] != "None" else None
kv_cache_dtype = sys.argv[6]

kwargs = {
    "model": model_path,
    "gpu_memory_utilization": gpu_mem,
    "enforce_eager": True,
    "trust_remote_code": True,
    "tensor_parallel_size": 1,
    "max_num_batched_tokens": 32768,
    "max_num_seqs": 64,
}
if max_model_len is not None:
    kwargs["max_model_len"] = max_model_len
if kv_cache_dtype != "auto":
    kwargs["kv_cache_dtype"] = kv_cache_dtype

llm = LLMEngine(**kwargs)

resp_f = open(resp_fifo, "w")
resp_f.write(json.dumps({"status": "ready"}) + "\\n")
resp_f.flush()

cmd_f = open(cmd_fifo, "r")

for line in cmd_f:
    line = line.strip()
    if not line:
        continue
    msg = json.loads(line)
    cmd = msg["cmd"]

    if cmd == "generate":
        prompts = msg["prompts"]
        sp_dict = msg.get("sampling_params", {})
        atom_sp = SamplingParams(
            max_tokens=int(sp_dict.get("max_tokens", sp_dict.get("max_new_tokens", 128))),
            temperature=float(sp_dict.get("temperature", 1.0)),
            top_p=float(sp_dict.get("top_p", 1.0)),
            top_k=int(sp_dict.get("top_k", -1)),
        )
        results = llm.generate(prompts, atom_sp)
        str_results = []
        for r in results:
            if isinstance(r, dict):
                str_results.append(r.get("text", str(r)))
            elif isinstance(r, str):
                str_results.append(r)
            else:
                str_results.append(str(r))
        resp_f.write(json.dumps({"results": str_results}, cls=_NumpyEncoder) + "\\n")
        resp_f.flush()

    elif cmd == "sleep":
        if llm is not None:
            llm.close()
            del llm
            llm = None
        torch.cuda.empty_cache()
        gc.collect()
        torch.cuda.empty_cache()
        resp_f.write(json.dumps({"status": "ok"}) + "\\n")
        resp_f.flush()

    elif cmd == "wake":
        new_path = msg.get("model_path")
        if new_path:
            kwargs["model"] = new_path
        if llm is not None:
            llm.close()
            del llm
            torch.cuda.empty_cache()
            gc.collect()
        llm = LLMEngine(**kwargs)
        resp_f.write(json.dumps({"status": "ok"}) + "\\n")
        resp_f.flush()

    elif cmd == "shutdown":
        try:
            if llm is not None:
                llm.close()
            del llm
        except Exception:
            pass
        torch.cuda.empty_cache()
        gc.collect()
        resp_f.write(json.dumps({"status": "ok"}) + "\\n")
        resp_f.flush()
        break

cmd_f.close()
resp_f.close()
""")


class AtomEngine:
    """ATOM inference engine with subprocess isolation for colocated RL.

    Spawns ATOM in a completely separate process via ``subprocess.Popen``.
    Uses named FIFOs for JSON communication to avoid stdout/stderr pollution
    from C++ libraries (Gloo, NCCL).
    """

    def __init__(self, config: AtomConfig, model_name: str) -> None:
        self._config = config
        self._model_name = model_name
        self._proc: subprocess.Popen | None = None
        self._initialized = False
        self._sleeping = False
        self._weight_dir: str | None = None
        self._cmd_fifo: str | None = None
        self._resp_fifo: str | None = None
        self._cmd_f = None
        self._resp_f = None

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def is_awake(self) -> bool:
        return (
            self._initialized and self._proc is not None and self._proc.poll() is None
        )

    @property
    def is_initialized(self) -> bool:
        """Whether the worker subprocess has completed its ready handshake."""
        return self._initialized

    @property
    def is_sleeping(self) -> bool:
        """Whether the live worker has released its model resources."""
        return self._sleeping

    def set_weight_path(self, weight_dir: str | os.PathLike[str]) -> None:
        """Set the model path used by the next wake/reload operation."""
        self._weight_dir = os.fspath(weight_dir)

    def wake_up(self) -> None:
        """Ensure the inference worker is awake using the current weights."""
        if self._sleeping:
            if self._proc is not None and self._proc.poll() is None:
                model_path = self._weight_dir or self._model_name
                self._send_cmd({"cmd": "wake", "model_path": model_path})
                self._sleeping = False
                return
            self._sleeping = False
        if not self.is_awake:
            self.wake()

    def _start_worker(self, model_path: str | None = None) -> None:
        """Start the ATOM worker subprocess."""
        if self._proc is not None and self._proc.poll() is None:
            return

        path = model_path or self._weight_dir or self._model_name
        gpu_mem = self._config.gpu_memory_utilization
        env_mem = os.environ.get("ATOM_GPU_MEMORY_UTILIZATION")
        if env_mem is not None:
            gpu_mem = float(env_mem)

        gpu_id = (
            self._config.gpu_id
            if self._config.gpu_id is not None
            else int(os.environ.get("LOCAL_RANK", "0"))
        )

        fifo_dir = tempfile.mkdtemp(prefix="atomrl_fifo_")
        self._cmd_fifo = os.path.join(fifo_dir, "cmd")
        self._resp_fifo = os.path.join(fifo_dir, "resp")
        os.mkfifo(self._cmd_fifo)
        os.mkfifo(self._resp_fifo)

        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        env["VLLM_USE_V1"] = "1"
        env["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
        env["VLLM_CONFIGURE_LOGGING"] = "0"
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

        attn_backend = os.environ.get("VLLM_ROCM_ATTN_BACKEND")
        if attn_backend:
            env["VLLM_ROCM_ATTN_BACKEND"] = attn_backend

        max_model_len_str = (
            str(self._config.max_model_len)
            if self._config.max_model_len is not None
            else "None"
        )

        logger.info(
            "AtomEngine: starting ATOM worker for %s (gpu=%d, mem=%.0f%%)",
            path,
            gpu_id,
            gpu_mem * 100,
        )

        self._proc = subprocess.Popen(
            [
                sys.executable,
                "-u",
                "-c",
                _WORKER_SCRIPT,
                self._cmd_fifo,
                self._resp_fifo,
                path,
                str(gpu_mem),
                max_model_len_str,
                self._config.kv_cache_dtype,
            ],
            stdin=subprocess.DEVNULL,
            stdout=None,
            stderr=None,
            env=env,
            start_new_session=True,
        )

        # FIFO stays open for the worker lifetime.
        self._resp_f = open(self._resp_fifo, "r")  # noqa: SIM115 -- lifetime handle

        resp_line = self._resp_f.readline()
        if not resp_line:
            raise RuntimeError("ATOM worker subprocess exited before ready")
        resp = json.loads(resp_line)
        if resp.get("status") != "ready":
            raise RuntimeError(f"ATOM worker failed to start: {resp}")

        # FIFO stays open for the worker lifetime.
        self._cmd_f = open(self._cmd_fifo, "w")  # noqa: SIM115 -- lifetime handle

        self._initialized = True
        logger.info("AtomEngine: ATOM worker ready (pid=%d).", self._proc.pid)

    def _send_cmd(self, cmd: dict) -> dict:
        """Send a JSON command to the worker and read the response."""
        if self._proc is None or self._proc.poll() is not None:
            raise RuntimeError("ATOM worker is not running")
        self._cmd_f.write(json.dumps(cmd) + "\n")
        self._cmd_f.flush()
        resp_line = self._resp_f.readline()
        if not resp_line:
            raise RuntimeError("ATOM worker closed response FIFO unexpectedly")
        return json.loads(resp_line)

    def generate(
        self,
        prompts: list[str] | list[list[int]],
        sampling_params: Mapping[str, Any] | None = None,
    ) -> list[str]:
        """Generate text completions via the ATOM worker."""
        if not self.is_awake:
            self._start_worker()

        sp = dict(sampling_params or {})
        str_prompts = [p if isinstance(p, str) else str(p) for p in prompts]

        resp = self._send_cmd(
            {
                "cmd": "generate",
                "prompts": str_prompts,
                "sampling_params": sp,
            }
        )
        return resp["results"]

    def update_weights(self, state_dict: dict[str, torch.Tensor]) -> None:
        """Save a new weight snapshot for the next ``wake()`` cycle."""
        sync_dir = Path(_WEIGHT_SYNC_DIR)
        sync_dir.mkdir(parents=True, exist_ok=True)

        try:
            from safetensors.torch import save_file
        except ImportError:
            ckpt_path = sync_dir / "model_weights.pt"
            torch.save(state_dict, ckpt_path)
            self._weight_dir = str(sync_dir)
            logger.info(
                "AtomEngine.update_weights: saved %d tensors to %s",
                len(state_dict),
                ckpt_path,
            )
            return

        tensors: dict[str, torch.Tensor] = {}
        for name, tensor in state_dict.items():
            tensors[name] = tensor.contiguous().cpu()

        save_path = sync_dir / "model.safetensors"
        save_file(tensors, str(save_path))
        self._weight_dir = str(sync_dir)
        logger.info(
            "AtomEngine.update_weights: saved %d tensors to %s", len(tensors), save_path
        )

    def sleep_inprocess(self, level: int = 1) -> None:
        """Put ATOM to sleep without killing the subprocess.

        Closes the worker's ``LLMEngine`` and releases its GPU caches while
        keeping the FIFO command process alive. ``level`` is retained for the
        common rollout-engine interface; ATOM currently uses the same release
        behavior for every level.
        """
        if self._proc is None or self._proc.poll() is not None:
            return
        if self._sleeping:
            return
        try:
            self._send_cmd({"cmd": "sleep", "level": level})
            self._sleeping = True
            logger.info("AtomEngine: sleep_inprocess(level=%d) complete.", level)
        except (OSError, RuntimeError, ValueError) as exc:
            logger.warning(
                "AtomEngine: sleep_inprocess failed (%s), falling back to kill.", exc
            )
            self.sleep()

    def wake_inprocess(self) -> None:
        """Recreate ATOM's ``LLMEngine`` after in-process sleep.

        If the subprocess died or recreation fails (e.g. ROCm page fault),
        kills the subprocess, waits briefly for driver cleanup, then starts
        a fresh subprocess.
        """
        if not self._sleeping:
            if self._proc is None or self._proc.poll() is not None:
                self.wake()
            return
        if self._proc is not None and self._proc.poll() is None:
            try:
                self._send_cmd({"cmd": "wake"})
                self._sleeping = False
                logger.info("AtomEngine: wake_inprocess complete.")
                return
            except (OSError, RuntimeError, ValueError) as exc:
                logger.warning(
                    "AtomEngine: wake_inprocess failed (%s), killing subprocess and restarting.",
                    exc,
                )
                self.sleep()
                time.sleep(5)
        self._sleeping = False
        self.wake()

    def sleep(self) -> None:
        """Kill the ATOM subprocess to free all GPU memory for training.

        On ROCm, ``del llm`` within the subprocess doesn't release all
        GPU memory because the HIP context retains it.  Terminating
        the process is the only reliable way to free everything.
        """
        if self._proc is None or self._proc.poll() is not None:
            return
        try:
            self._send_cmd({"cmd": "shutdown"})
        # The command crosses into ATOM and may surface backend-specific errors.
        except Exception:  # noqa: BLE001,S110 -- best-effort cleanup boundary
            pass
        try:
            self._proc.terminate()
            self._proc.wait(timeout=10)
        except Exception:  # noqa: BLE001 -- best-effort cleanup boundary
            try:
                self._proc.kill()
            except Exception:  # noqa: BLE001,S110 -- best-effort cleanup boundary
                pass
        for f in [self._cmd_f, self._resp_f]:
            try:
                if f:
                    f.close()
            except OSError:
                pass
        self._proc = None
        self._cmd_f = None
        self._resp_f = None
        self._initialized = False
        torch.cuda.empty_cache()
        gc.collect()
        logger.info("AtomEngine: sleep complete (subprocess terminated).")

    def wake(self) -> None:
        """Start a fresh ATOM subprocess for generation.

        If ``_weight_dir`` is set (from a prior ``update_weights`` or
        weight-sync call), loads from that path instead of the original
        model. The directory is **not** cleared so that subsequent
        wake cycles (e.g. after crash recovery) still use the latest
        synced weights.
        """
        if self._proc is not None and self._proc.poll() is None:
            logger.info("AtomEngine: already awake.")
            return
        model_path = self._weight_dir or self._model_name
        self._start_worker(model_path)
        logger.info(
            "AtomEngine: wake complete (fresh subprocess, path=%s).", model_path
        )

    def shutdown(self) -> None:
        """Release all resources permanently."""
        if self._proc is not None and self._proc.poll() is None:
            try:
                self._send_cmd({"cmd": "shutdown"})
            # The command crosses into ATOM and may surface backend-specific errors.
            except Exception:  # noqa: BLE001,S110 -- best-effort cleanup boundary
                pass
            try:
                self._proc.terminate()
                self._proc.wait(timeout=10)
            except Exception:  # noqa: BLE001 -- best-effort cleanup boundary
                try:
                    self._proc.kill()
                except Exception:  # noqa: BLE001,S110 -- cleanup boundary
                    pass
        for f in [self._cmd_f, self._resp_f]:
            try:
                if f:
                    f.close()
            except OSError:
                pass
        for p in [self._cmd_fifo, self._resp_fifo]:
            try:
                if p:
                    os.unlink(p)
            except OSError:
                pass
        self._proc = None
        self._cmd_f = None
        self._resp_f = None
        self._initialized = False
        self._weight_dir = None
        logger.info("AtomEngine: shutdown complete.")
