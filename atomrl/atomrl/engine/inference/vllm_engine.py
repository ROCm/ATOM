# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

"""Native (upstream) vLLM rollout engine with subprocess isolation for RL.

This is the vanilla-vLLM counterpart of :mod:`atomrl.engine.inference.atom_engine`.
It lets a AtomRL DAPO/GRPO run use **vLLM for inference** while training with
**Lumen FSDP**, without depending on verl.

Design (mirrors ``AtomEngine`` so the trainer can drive either engine through the
same code path):

- vLLM runs in a separate ``subprocess.Popen`` for full process isolation from
  torchrun's NCCL/Gloo process groups (colocated rollout + training on the same
  GPU set, swapping GPU memory between phases).
- Communication uses a pair of named FIFOs carrying newline-delimited JSON,
  avoiding stdout/stderr pollution from C++ libraries.
- Weight sync: the trainer writes updated HF-format safetensors to
  ``ATOMRL_WEIGHT_SYNC_DIR`` (``/dev/shm`` by default); on the next ``wake`` the
  worker rebuilds the vLLM engine from that path (reliable weight reload on ROCm,
  same approach ATOM uses).

Unlike ``AtomEngine``, ``generate_with_logprobs`` returns per-sample prompt token
ids, response token ids and (optionally) per-response-token log-probs so the
trainer can build sequences exactly (no text re-tokenization mismatch) and apply
TIS/MIS rollout correction.
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

from atomrl.core.config import VLLMConfig

logger = logging.getLogger(__name__)

_WEIGHT_SYNC_DIR = os.environ.get(
    "ATOMRL_WEIGHT_SYNC_DIR",
    "/dev/shm/atomrl_weight_sync",
)


def _kill_proc_tree(pid: int) -> None:
    """Kill a process and ALL descendants (AsyncLLM spawns a separate EngineCore
    process, and that spawns GPU workers; terminating only the direct child leaks
    the EngineCore and keeps the GPU). verl avoids this via ray; we reap the
    subtree explicitly with psutil."""
    try:
        import psutil
    except ImportError:
        return
    try:
        parent = psutil.Process(pid)
    except psutil.Error:
        return
    procs = []
    try:
        procs = parent.children(recursive=True)
    except psutil.Error:
        pass
    procs.append(parent)
    for p in procs:
        try:
            p.kill()
        except (OSError, psutil.Error):
            pass
    try:
        psutil.wait_procs(procs, timeout=10)
    except psutil.Error:
        pass


# The worker subprocess. It reads JSON commands from ``cmd_fifo`` and writes JSON
# responses to ``resp_fifo``. Arguments are passed positionally (see Popen call).
_WORKER_SCRIPT = textwrap.dedent("""\
import gc, json, os, sys, logging

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
from vllm import LLM, SamplingParams

logging.getLogger("vllm").setLevel(logging.WARNING)

cmd_fifo = sys.argv[1]
resp_fifo = sys.argv[2]
cfg = json.loads(sys.argv[3])

def build_llm(model_path):
    kwargs = dict(
        model=model_path,
        gpu_memory_utilization=cfg["gpu_memory_utilization"],
        enforce_eager=cfg["enforce_eager"],
        trust_remote_code=cfg["trust_remote_code"],
        tensor_parallel_size=cfg["tensor_parallel_size"],
        max_num_batched_tokens=cfg["max_num_batched_tokens"],
        max_num_seqs=cfg["max_num_seqs"],
        enable_chunked_prefill=cfg["enable_chunked_prefill"],
        swap_space=cfg["swap_space"],
        dtype=cfg["dtype"],
    )
    if cfg.get("max_model_len") is not None:
        kwargs["max_model_len"] = cfg["max_model_len"]
    if cfg.get("kv_cache_dtype", "auto") != "auto":
        kwargs["kv_cache_dtype"] = cfg["kv_cache_dtype"]
    if cfg.get("quantization"):
        kwargs["quantization"] = cfg["quantization"]
    if cfg.get("seed") is not None:
        kwargs["seed"] = int(cfg["seed"])
    return LLM(**kwargs)

llm = build_llm(cfg["model_path"])

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
        want_logprobs = bool(msg.get("logprobs", False))
        sp = SamplingParams(
            max_tokens=int(sp_dict.get("max_tokens", sp_dict.get("max_new_tokens", 128))),
            temperature=float(sp_dict.get("temperature", 1.0)),
            top_p=float(sp_dict.get("top_p", 1.0)),
            top_k=int(sp_dict.get("top_k", -1)),
            logprobs=(1 if want_logprobs else None),
        )
        outs = llm.generate(prompts, sp)
        results = []
        for o in outs:
            comp = o.outputs[0]
            tok_ids = list(comp.token_ids)
            lps = None
            if want_logprobs and comp.logprobs is not None:
                lps = []
                for pos, tid in enumerate(tok_ids):
                    entry = comp.logprobs[pos]
                    lp = entry.get(tid)
                    lps.append(float(lp.logprob) if lp is not None else 0.0)
            results.append({
                "text": comp.text,
                "prompt_token_ids": list(o.prompt_token_ids),
                "token_ids": tok_ids,
                "logprobs": lps,
            })
        resp_f.write(json.dumps({"results": results}) + "\\n")
        resp_f.flush()

    elif cmd == "sleep":
        if llm is not None:
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
            cfg["model_path"] = new_path
        if llm is not None:
            del llm
            torch.cuda.empty_cache()
            gc.collect()
        llm = build_llm(cfg["model_path"])
        resp_f.write(json.dumps({"status": "ok"}) + "\\n")
        resp_f.flush()

    elif cmd == "reload":
        # In-place weight update (no engine rebuild): vLLM V1
        # model_runner.reload_weights(weights_path=...) loads the new
        # checkpoint from disk into the live model.
        new_path = msg.get("model_path")
        try:
            if new_path:
                cfg["model_path"] = new_path
            llm.collective_rpc("reload_weights", kwargs={"weights_path": cfg["model_path"]})
            resp_f.write(json.dumps({"status": "ok"}) + "\\n")
        except Exception as e:
            resp_f.write(json.dumps({"status": "error", "msg": str(e)}) + "\\n")
        resp_f.flush()

    elif cmd == "shutdown":
        try:
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


class VLLMEngine:
    """Upstream vLLM inference engine with subprocess isolation for colocated RL.

    API-compatible with :class:`AtomEngine` (``generate``, ``sleep``/``wake``,
    ``sleep_inprocess``/``wake_inprocess``, ``update_weights``, ``shutdown``,
    ``_send_cmd``, ``_weight_dir``, ``_sleeping``, ``_model_name``,
    ``_initialized``, ``is_awake``) so :class:`RLTrainer` can drive either engine.
    Adds :meth:`generate_with_logprobs` for TIS-aware rollout.
    """

    def __init__(self, config: VLLMConfig, model_name: str) -> None:
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

    def _worker_cfg(self, model_path: str, seed: int | None = None) -> dict[str, Any]:
        c = self._config
        gpu_mem = c.gpu_memory_utilization
        env_mem = os.environ.get("VLLM_GPU_MEMORY_UTILIZATION")
        if env_mem is not None:
            gpu_mem = float(env_mem)
        return {
            "seed": seed,
            "model_path": model_path,
            "gpu_memory_utilization": gpu_mem,
            "enforce_eager": bool(c.enforce_eager),
            "trust_remote_code": bool(c.trust_remote_code),
            "tensor_parallel_size": int(c.tensor_parallel_size),
            "max_num_batched_tokens": int(c.max_num_batched_tokens),
            "max_num_seqs": int(c.max_num_seqs),
            "enable_chunked_prefill": bool(c.enable_chunked_prefill),
            "swap_space": int(c.swap_space),
            "dtype": c.dtype,
            "max_model_len": c.max_model_len,
            "kv_cache_dtype": c.kv_cache_dtype,
            "quantization": c.quantization or "",
        }

    def _start_worker(self, model_path: str | None = None) -> None:
        if self._proc is not None and self._proc.poll() is None:
            return

        path = model_path or self._weight_dir or self._model_name
        gpu_id = (
            self._config.gpu_id
            if self._config.gpu_id is not None
            else int(os.environ.get("LOCAL_RANK", "0"))
        )

        # verl alignment: per-engine seed = base_seed + local_rank (verl uses
        # ``replica_rank + data.seed``). local_rank gives a distinct per-GPU
        # offset so the 8 colocated engines don't all sample identically.
        base_seed = getattr(self._config, "seed", None)
        worker_seed = None
        if base_seed is not None:
            worker_seed = int(base_seed) + int(os.environ.get("LOCAL_RANK", "0"))

        fifo_dir = tempfile.mkdtemp(prefix="atomrl_vllm_fifo_")
        self._cmd_fifo = os.path.join(fifo_dir, "cmd")
        self._resp_fifo = os.path.join(fifo_dir, "resp")
        os.mkfifo(self._cmd_fifo)
        os.mkfifo(self._resp_fifo)

        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        env["VLLM_USE_V1"] = "1"
        env["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
        env["VLLM_CONFIGURE_LOGGING"] = "0"
        # If ATOM is installed in the image it registers a vLLM model plugin that
        # vLLM auto-loads; on aiter API drift its import crashes our (vanilla)
        # vLLM rollout. Disable the ATOM vLLM plugin for this native path.
        env.setdefault("ATOM_DISABLE_VLLM_PLUGIN", "1")
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

        cfg_json = json.dumps(self._worker_cfg(path, seed=worker_seed))
        logger.info(
            "VLLMEngine: starting vLLM worker for %s (gpu=%s, mem=%.0f%%, seed=%s)",
            path,
            gpu_id,
            self._worker_cfg(path, seed=worker_seed)["gpu_memory_utilization"] * 100,
            worker_seed,
        )

        # Online (verl-style) AsyncLLM worker vs offline inline LLM worker.
        # Default to the online AsyncLLM path (config.online); env var overrides.
        # AsyncLLM spawns a separate EngineCore process (multiprocessing spawn),
        # which requires a standalone importable module file (not `python -c`).
        _online_default = "1" if bool(getattr(self._config, "online", True)) else "0"
        if os.environ.get("LUMEN_VLLM_ASYNC", _online_default) == "1":
            worker_py = os.path.join(
                os.path.dirname(os.path.abspath(__file__)), "vllm_async_worker.py"
            )
            popen_cmd = [
                sys.executable,
                "-u",
                worker_py,
                self._cmd_fifo,
                self._resp_fifo,
                cfg_json,
            ]
            logger.info("VLLMEngine: launching ONLINE AsyncLLM worker (%s)", worker_py)
        else:
            popen_cmd = [
                sys.executable,
                "-u",
                "-c",
                _WORKER_SCRIPT,
                self._cmd_fifo,
                self._resp_fifo,
                cfg_json,
            ]
            logger.info("VLLMEngine: launching OFFLINE LLM worker (inline)")

        self._proc = subprocess.Popen(
            popen_cmd,
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
            raise RuntimeError("vLLM worker subprocess exited before ready")
        resp = json.loads(resp_line)
        if resp.get("status") != "ready":
            raise RuntimeError(f"vLLM worker failed to start: {resp}")

        # FIFO stays open for the worker lifetime.
        self._cmd_f = open(self._cmd_fifo, "w")  # noqa: SIM115 -- lifetime handle
        self._initialized = True
        logger.info("VLLMEngine: vLLM worker ready (pid=%d).", self._proc.pid)

    def _send_cmd(self, cmd: dict) -> dict:
        if self._proc is None or self._proc.poll() is not None:
            raise RuntimeError("vLLM worker is not running")
        self._cmd_f.write(json.dumps(cmd) + "\n")
        self._cmd_f.flush()
        resp_line = self._resp_f.readline()
        if not resp_line:
            raise RuntimeError("vLLM worker closed response FIFO unexpectedly")
        return json.loads(resp_line)

    def generate(
        self,
        prompts: list[str],
        sampling_params: Mapping[str, Any] | None = None,
    ) -> list[str]:
        """Generate response text (AtomEngine-compatible)."""
        results = self.generate_with_logprobs(
            prompts, sampling_params, want_logprobs=False
        )
        return [r["text"] for r in results]

    def generate_with_logprobs(
        self,
        prompts: list[str],
        sampling_params: Mapping[str, Any] | None = None,
        want_logprobs: bool = True,
    ) -> list[dict[str, Any]]:
        """Generate and return per-sample dicts.

        Each dict has keys: ``text``, ``prompt_token_ids`` (list[int]),
        ``token_ids`` (response, list[int]), ``logprobs`` (response, list[float]
        or ``None`` when ``want_logprobs`` is False).
        """
        if not self.is_awake:
            self._start_worker()
        sp = dict(sampling_params or {})
        str_prompts = [p if isinstance(p, str) else str(p) for p in prompts]
        resp = self._send_cmd(
            {
                "cmd": "generate",
                "prompts": str_prompts,
                "sampling_params": sp,
                "logprobs": bool(want_logprobs and self._config.calculate_log_probs),
            }
        )
        return resp["results"]

    def update_weights(self, state_dict: dict[str, torch.Tensor]) -> None:
        """Save a new HF-format weight snapshot for the next ``wake`` cycle."""
        sync_dir = Path(_WEIGHT_SYNC_DIR)
        sync_dir.mkdir(parents=True, exist_ok=True)
        try:
            from safetensors.torch import save_file
        except ImportError:
            ckpt_path = sync_dir / "model_weights.pt"
            torch.save(state_dict, ckpt_path)
            self._weight_dir = str(sync_dir)
            return
        tensors = {name: t.contiguous().cpu() for name, t in state_dict.items()}
        save_file(tensors, str(sync_dir / "model.safetensors"))
        self._weight_dir = str(sync_dir)
        logger.info(
            "VLLMEngine.update_weights: saved %d tensors to %s", len(tensors), sync_dir
        )

    def reload_weights(self, weight_dir: str) -> None:
        """Update the resident engine's weights in place from ``weight_dir``.

        Uses vLLM ``collective_rpc("reload_weights", weights_path=...)`` so the
        engine is NOT rebuilt (persistent path). Falls back to a fresh build on
        the next wake if the engine isn't running yet.
        """
        self._weight_dir = weight_dir
        if not self.is_awake:
            return  # will load from _weight_dir when started
        resp = self._send_cmd({"cmd": "reload", "model_path": weight_dir})
        if resp.get("status") != "ok":
            raise RuntimeError(f"vLLM reload_weights failed: {resp.get('msg')}")
        logger.info("VLLMEngine: reloaded weights in place from %s", weight_dir)

    def sleep_inprocess(self, level: int = 1) -> None:
        """Release vLLM GPU memory between training steps.

        On ROCm, vLLM V1's in-process ``del llm`` does NOT reliably free GPU
        memory (the EngineCore workers / HIP context retain it), so it leaks and
        OOMs after a few steps with large (20K-token) KV caches. Terminating the
        subprocess is the only reliable release; the next ``wake()`` rebuilds the
        engine from the latest synced weights. So sleep == kill here.
        """
        self.sleep()

    def wake_inprocess(self) -> None:
        if not self._sleeping:
            if self._proc is None or self._proc.poll() is not None:
                self.wake()
            return
        if self._proc is not None and self._proc.poll() is None:
            try:
                model_path = self._weight_dir or self._model_name
                self._send_cmd({"cmd": "wake", "model_path": model_path})
                self._sleeping = False
                logger.info(
                    "VLLMEngine: wake_inprocess complete (path=%s).", model_path
                )
                return
            except (OSError, RuntimeError, ValueError) as exc:
                logger.warning(
                    "VLLMEngine: wake_inprocess failed (%s), restarting subprocess.",
                    exc,
                )
                self.sleep()
                time.sleep(5)
        self._sleeping = False
        self.wake()

    def sleep(self) -> None:
        """Kill the vLLM subprocess to free all GPU memory for training."""
        if self._proc is None or self._proc.poll() is not None:
            return
        _pid = self._proc.pid
        try:
            self._send_cmd({"cmd": "shutdown"})
        # The command crosses into vLLM and may surface backend-specific errors.
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
        # Reap the whole subtree (worker + AsyncLLM EngineCore + GPU workers) so
        # the online EngineCore never leaks the GPU after sleep.
        _kill_proc_tree(_pid)
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
        logger.info("VLLMEngine: sleep complete (subprocess terminated).")

    def wake(self) -> None:
        """Start a fresh vLLM subprocess for generation (loads latest weights)."""
        if self._proc is not None and self._proc.poll() is None:
            return
        model_path = self._weight_dir or self._model_name
        self._start_worker(model_path)
        logger.info("VLLMEngine: wake complete (path=%s).", model_path)

    def shutdown(self) -> None:
        if self._proc is not None and self._proc.poll() is None:
            _pid = self._proc.pid
            try:
                self._send_cmd({"cmd": "shutdown"})
            # The command crosses into vLLM and may surface backend-specific errors.
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
            _kill_proc_tree(_pid)
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
        logger.info("VLLMEngine: shutdown complete.")
