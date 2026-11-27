# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

"""Standalone vLLM v1 AsyncLLM rollout worker (online, concurrent) for AtomRL.

This is the AsyncLLM counterpart of the inline offline-`LLM` worker embedded in
``vllm_engine.py``. It is launched as a separate process via
``python -u vllm_async_worker.py <cmd_fifo> <resp_fifo> <cfg_json>``.

Why a standalone file (not an inline ``python -c`` script): vLLM v1 ``AsyncLLM``
spawns a separate EngineCore process via multiprocessing *spawn*, which requires
the launching module to be importable and guarded by ``if __name__ == '__main__'``.
An inline ``-c`` script cannot be re-imported by spawn, so AsyncLLM fails there.

Behavior matches verl's online rollout: every prompt is submitted as its own
``engine.generate(..., request_id=uuid4())`` and all are awaited concurrently
with ``asyncio.gather`` (continuous batching + concurrent scheduling), instead of
the offline ``LLM.generate(list)`` batch call.

Protocol (newline-delimited JSON over FIFOs), identical to the offline worker:
  - {"cmd":"generate","prompts":[...],"sampling_params":{...},"logprobs":bool}
      -> {"results":[{text,prompt_token_ids,token_ids,logprobs}, ...]}
  - {"cmd":"reload","model_path":...} -> {"status":"ok"}
      verl-style in-place hot weight swap: ``collective_rpc("reload_weights",
      weights_path=...)`` + ``reset_prefix_cache`` (engine stays resident, no
      rebuild). This is the persistent path used during RL training.
  - {"cmd":"sleep"}   -> {"status":"ok"}  (free engine; non-persistent fallback)
  - {"cmd":"wake","model_path":...} -> {"status":"ok"}  (rebuild if freed, else
      in-place reload; non-persistent fallback)
  - {"cmd":"shutdown"} -> {"status":"ok"} then exit

Memory model: vLLM's cumem sleep (``engine.sleep``) is unreliable on this ROCm
build ("Memory usage increased after sleeping"), so the engine is kept RESIDENT
(weights + KV at ``gpu_memory_utilization``) and coexists with FSDP training,
exactly like the current persistent path; weights are refreshed in place via
``reload_weights`` instead of being offloaded/rebuilt.
"""

import asyncio
import ctypes
import gc
import json
import logging
import os
import platform
import signal
import sys
from contextlib import ExitStack
from uuid import uuid4


def _set_pdeathsig(sig=signal.SIGTERM):
    """verl-style: receive `sig` when the parent (lumen rank) process dies, so this
    worker never lingers as an orphan holding the GPU. If the parent is already
    gone (reparented to init), exit immediately."""
    if platform.system() != "Linux":
        return
    try:
        ctypes.CDLL("libc.so.6").prctl(1, int(sig))  # PR_SET_PDEATHSIG = 1
        if os.getppid() == 1:
            os.kill(os.getpid(), signal.SIGKILL)
    except OSError:
        pass


def _kill_descendants():
    """Kill all child processes (vLLM EngineCore + GPU workers) so AsyncLLM's
    spawned EngineCore does not leak and keep the GPU when this worker exits.
    Without ray (verl uses ray for this), we must reap the subtree ourselves."""
    try:
        import psutil

        me = psutil.Process()
        kids = me.children(recursive=True)
        for c in kids:
            try:
                c.kill()
            except (OSError, psutil.Error):
                pass
        psutil.wait_procs(kids, timeout=5)
    except psutil.Error:
        pass


def _on_term(signum, frame):
    _kill_descendants()
    os._exit(0)


# Detach from torchrun's distributed env so vLLM starts its own process group.
os.environ.pop("PYTORCH_CUDA_ALLOC_CONF", None)
for _k in list(os.environ.keys()):
    if any(
        _k.startswith(p)
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
        del os.environ[_k]

logging.basicConfig(stream=sys.stderr, level=logging.WARNING)

import torch
from vllm import SamplingParams
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.v1.engine.async_llm import AsyncLLM

logging.getLogger("vllm").setLevel(logging.WARNING)


def build_engine(cfg: dict) -> AsyncLLM:
    kwargs = {
        "model": cfg["model_path"],
        "gpu_memory_utilization": cfg["gpu_memory_utilization"],
        "enforce_eager": cfg["enforce_eager"],
        "trust_remote_code": cfg["trust_remote_code"],
        "tensor_parallel_size": cfg["tensor_parallel_size"],
        "max_num_batched_tokens": cfg["max_num_batched_tokens"],
        "max_num_seqs": cfg["max_num_seqs"],
        "enable_chunked_prefill": cfg["enable_chunked_prefill"],
        "dtype": cfg["dtype"],
        "disable_log_stats": True,
    }
    if cfg.get("max_model_len") is not None:
        kwargs["max_model_len"] = cfg["max_model_len"]
    if cfg.get("kv_cache_dtype", "auto") != "auto":
        kwargs["kv_cache_dtype"] = cfg["kv_cache_dtype"]
    if cfg.get("quantization"):
        kwargs["quantization"] = cfg["quantization"]
    if cfg.get("seed") is not None:
        kwargs["seed"] = int(cfg["seed"])
    return AsyncLLM.from_engine_args(AsyncEngineArgs(**kwargs))


async def _generate_one(
    engine: AsyncLLM, prompt: str, sp: SamplingParams, want_lp: bool
) -> dict:
    """One independent request (verl-style: per-request id, concurrent)."""
    final = None
    async for out in engine.generate(prompt, sp, request_id=uuid4().hex):
        final = out
    comp = final.outputs[0]
    tok_ids = list(comp.token_ids)
    lps = None
    if want_lp and comp.logprobs is not None:
        lps = []
        for pos, tid in enumerate(tok_ids):
            entry = comp.logprobs[pos]
            lp = entry.get(tid)
            lps.append(float(lp.logprob) if lp is not None else 0.0)
    return {
        "text": comp.text,
        "prompt_token_ids": list(final.prompt_token_ids),
        "token_ids": tok_ids,
        "logprobs": lps,
    }


def _open_fifo_pair(cmd_fifo: str, resp_fifo: str):
    stack = ExitStack()
    try:
        response_file = stack.enter_context(
            open(resp_fifo, "w")  # noqa: SIM115  # ExitStack owns the FIFO handle.
        )
        command_file = stack.enter_context(
            open(cmd_fifo)  # noqa: SIM115  # ExitStack owns the FIFO handle.
        )
    except Exception:
        stack.close()
        raise
    return stack, response_file, command_file


async def serve(cmd_fifo: str, resp_fifo: str, cfg: dict) -> None:
    engine = build_engine(cfg)

    fifo_stack, resp_f, cmd_f = await asyncio.to_thread(
        _open_fifo_pair, cmd_fifo, resp_fifo
    )
    resp_f.write(json.dumps({"status": "ready"}) + "\n")
    resp_f.flush()
    loop = asyncio.get_running_loop()

    while True:
        # FIFO readline is blocking; run it off the event loop.
        line = await loop.run_in_executor(None, cmd_f.readline)
        if not line:
            await asyncio.sleep(0.02)
            continue
        line = line.strip()
        if not line:
            continue
        msg = json.loads(line)
        cmd = msg["cmd"]

        if cmd == "generate":
            prompts = msg["prompts"]
            sp_dict = msg.get("sampling_params", {})
            want_lp = bool(msg.get("logprobs", False))
            sp = SamplingParams(
                max_tokens=int(
                    sp_dict.get("max_tokens", sp_dict.get("max_new_tokens", 128))
                ),
                temperature=float(sp_dict.get("temperature", 1.0)),
                top_p=float(sp_dict.get("top_p", 1.0)),
                top_k=int(sp_dict.get("top_k", -1)),
                logprobs=(0 if want_lp else None),
            )
            str_prompts = [p if isinstance(p, str) else str(p) for p in prompts]
            results = await asyncio.gather(
                *[_generate_one(engine, p, sp, want_lp) for p in str_prompts]
            )
            resp_f.write(json.dumps({"results": list(results)}) + "\n")
            resp_f.flush()

        elif cmd == "reload":
            # verl-style in-place hot weight swap (no engine rebuild): load the
            # new checkpoint from disk into the live model on every TP worker,
            # then drop the prefix cache (KV was computed with the old weights).
            new_path = msg.get("model_path")
            if new_path:
                cfg["model_path"] = new_path
            try:
                if engine is None:
                    engine = build_engine(cfg)
                else:
                    await engine.collective_rpc(
                        "reload_weights",
                        kwargs={"weights_path": cfg["model_path"]},
                    )
                    await engine.reset_prefix_cache()
                resp_f.write(json.dumps({"status": "ok"}) + "\n")
            # Worker command protocol returns arbitrary vLLM backend failures to the controller.
            except Exception as e:  # noqa: BLE001 -- boundary
                resp_f.write(json.dumps({"status": "error", "msg": str(e)}) + "\n")
            resp_f.flush()

        elif cmd == "sleep":
            # Non-persistent fallback only: fully release the engine (cumem sleep
            # is broken on this ROCm build, so we cannot keep the proc + free VRAM).
            if engine is not None:
                engine.shutdown()
                engine = None
            torch.cuda.empty_cache()
            gc.collect()
            resp_f.write(json.dumps({"status": "ok"}) + "\n")
            resp_f.flush()

        elif cmd == "wake":
            new_path = msg.get("model_path")
            if new_path:
                cfg["model_path"] = new_path
            try:
                if engine is None:
                    engine = build_engine(cfg)
                else:
                    await engine.collective_rpc(
                        "reload_weights",
                        kwargs={"weights_path": cfg["model_path"]},
                    )
                    await engine.reset_prefix_cache()
                resp_f.write(json.dumps({"status": "ok"}) + "\n")
            # Wake command reports arbitrary engine rebuild failures over the FIFO protocol.
            except Exception as e:  # noqa: BLE001 -- boundary
                resp_f.write(json.dumps({"status": "error", "msg": str(e)}) + "\n")
            resp_f.flush()

        elif cmd == "shutdown":
            if engine is not None:
                engine.shutdown()
            _kill_descendants()  # ensure EngineCore subprocess is reaped
            torch.cuda.empty_cache()
            gc.collect()
            resp_f.write(json.dumps({"status": "ok"}) + "\n")
            resp_f.flush()
            break

    fifo_stack.close()


if __name__ == "__main__":
    # Die with the parent (lumen rank) and reap EngineCore on SIGTERM, so the
    # AsyncLLM EngineCore never leaks the GPU when the trainer is killed.
    _set_pdeathsig(signal.SIGTERM)
    signal.signal(signal.SIGTERM, _on_term)
    _cmd_fifo = sys.argv[1]
    _resp_fifo = sys.argv[2]
    _cfg = json.loads(sys.argv[3])
    try:
        asyncio.run(serve(_cmd_fifo, _resp_fifo, _cfg))
    finally:
        _kill_descendants()
