# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

"""verl-aligned Ray rollout server: vLLM AsyncLLM inside a Ray actor.

This is AtomRL's port of verl's online rollout replica stack:

* ``VLLMRayServer``  ~ verl ``vLLMHttpServer`` -- a Ray actor that owns one
  vLLM ``AsyncLLM`` engine pinned to an explicit set of GPUs, optionally serves
  a uvicorn OpenAI-compatible HTTP app, and exposes token-in/token-out
  ``generate`` + ``collective_rpc`` + ``sleep``/``wake_up`` +
  ``update_weights_from_ipc`` over Ray RPC.
* ``VLLMReplicaManager`` ~ verl ``vLLMReplica`` + ``LLMServerManager`` +
  ``CheckpointEngineManager`` -- driver-side helper that colocates one server
  actor on the same GPUs as a group of training ``AtomActorWorker``s
  (NodeAffinity + explicit ``CUDA_VISIBLE_DEVICES``), then fans out generate /
  sleep / wake / weight-update across the replicas.

8 GPUs => 8 replicas, each TP=1 (verl's default DP=8 rollout layout).

With ``tensor_parallel_size=N`` the layout becomes one engine per N actors:
32 GPUs and TP=8 give 4 replicas, each spanning one node. Models whose kernels
have no TP=1 path need this -- DeepSeek-V4 on gfx950 faults at TP=1 and runs at
TP=8 -- and it also cuts the per-GPU engine footprint by N, which is what makes
a colocated engine affordable for a very large policy.
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys
import time
from typing import Any
from uuid import uuid4

logger = logging.getLogger(__name__)
# A Ray actor configures no logging at all, so records fall through to
# `logging.lastResort`, which drops anything below WARNING. Everything
# this module says at INFO -- including the KV pool each engine actually
# got, the number that makes `gpu_memory_utilization` comparable between
# engines -- was being thrown away. Setting the level is not enough on its
# own; the records need somewhere to go. Ray relays actor stderr to the
# driver, so a stream handler is all it takes, and the guard keeps the
# driver process (where the root logger is already configured) from
# printing everything twice.
logger.setLevel(os.getenv("ATOMRL_LOGGING_LEVEL", "INFO"))
if not logger.handlers and not logging.getLogger().handlers:
    _h = logging.StreamHandler(sys.stderr)
    _h.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    )
    logger.addHandler(_h)

_VLLM_RUNTIME_ENV_KEYS = (
    "NCCL_IB_DISABLE",
    "NCCL_SOCKET_IFNAME",
    "NCCL_IB_HCA",
    "NCCL_IB_GID_INDEX",
    "NCCL_NET_GDR_LEVEL",
    "NCCL_DMABUF_ENABLE",
    "NCCL_DEBUG",
    "NCCL_DEBUG_SUBSYS",
    "NCCL_MSCCL_ENABLE",
    "RCCL_MSCCL_ENABLE",
    "VLLM_ROCM_USE_AITER",
    "VLLM_ROCM_USE_AITER_MOE",
    "VLLM_ROCM_USE_AITER_TRITON_GEMM",
    "ATOMRL_DIAG_ALL_GATHER",
    "ATOMRL_DIAG_ALL_GATHER_NUMEL",
    "ATOMRL_WEIGHT_SYNC_INTEGRITY",
)


def _copy_vllm_runtime_env(env_vars: dict[str, str]) -> None:
    for key in _VLLM_RUNTIME_ENV_KEYS:
        if key in os.environ:
            env_vars[key] = os.environ[key]


class VLLMRayServer:
    """Ray actor hosting one vLLM AsyncLLM engine on a single pinned GPU."""

    def __init__(
        self,
        model_name: str,
        engine_kwargs: dict[str, Any],
        replica_rank: int,
        http_port: int = 0,
        start_http: bool = False,
        base_seed: int | None = None,
    ) -> None:
        # CUDA_VISIBLE_DEVICES / NOSET / replica-rank / job-id are injected via
        # the actor's runtime_env by VLLMReplicaManager before this process
        # starts, so the engine and its ZMQ IPC receiver land on the right GPU.
        self.model_name = model_name
        self.engine_kwargs = dict(engine_kwargs)
        self.replica_rank = int(replica_rank)
        self.http_port = int(http_port)
        self.start_http = bool(start_http)
        self.base_seed = base_seed
        self.engine = None
        self._http_task = None
        self._http_ready = False

    def _log_kv_pool(self) -> None:
        """Say how many KV blocks this engine actually got.

        vLLM prints this at INFO and the harness captures only WARNING, so the
        number never reaches the run log -- which makes `gpu_memory_utilization`
        look comparable across engines when it is not. The two arms derive their
        pools from different baselines, so the same 0.30 can buy one engine
        several times the KV of the other, and the only honest way to compare a
        rollout's speed is to compare the pools rather than the knob.
        """
        try:
            cache = self.engine.vllm_config.cache_config
            blocks = cache.num_gpu_blocks
            logger.info(
                "VLLMRayServer[%d]: kv pool = %s blocks x %s tokens = %s tokens "
                "(gpu_memory_utilization=%s)",
                self.replica_rank,
                blocks,
                cache.block_size,
                (blocks or 0) * cache.block_size,
                self.engine_kwargs.get("gpu_memory_utilization"),
            )
        except Exception:
            logger.debug(
                "VLLMRayServer[%d]: kv pool size unavailable",
                self.replica_rank,
                exc_info=True,
            )

    async def launch(self) -> bool:
        """Build the AsyncLLM engine (and optional HTTP server)."""
        from vllm.engine.arg_utils import AsyncEngineArgs
        from vllm.v1.engine.async_llm import AsyncLLM

        # verl-aligned per-replica seed: seed = base_seed + replica_rank, so the
        # colocated engines don't all share one RNG (verl: replica_rank+data.seed).
        if self.base_seed is not None and "seed" not in self.engine_kwargs:
            self.engine_kwargs["seed"] = int(self.base_seed) + self.replica_rank

        worker_ext = "atomrl.engine.inference.vllm_colocate_worker_ext.vLLMColocateWorkerExtension"
        args = AsyncEngineArgs(
            model=self.model_name,
            worker_extension_cls=worker_ext,
            distributed_executor_backend="mp",
            **self.engine_kwargs,
        )
        logger.info(
            "VLLMRayServer[%d]: engine seed=%s",
            self.replica_rank,
            self.engine_kwargs.get("seed"),
        )
        self.engine = AsyncLLM.from_engine_args(args)
        logger.info("VLLMRayServer[%d]: AsyncLLM ready.", self.replica_rank)
        self._log_kv_pool()

        # verl-aligned: mask OOV/padded logits at engine init (critical for online
        # FP8 rollout where the requantized lm_head can otherwise emit garbage after
        # a weight update). Best-effort; tokenizer length = true vocab size.
        try:
            from transformers import AutoTokenizer

            vocab_size = len(
                AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
            )
            await self.engine.collective_rpc(
                "monkey_patch_model", kwargs={"vocab_size": vocab_size}
            )
            logger.info(
                "VLLMRayServer[%d]: monkey_patch_model applied (vocab=%d).",
                self.replica_rank,
                vocab_size,
            )
        except (ImportError, OSError, RuntimeError, ValueError) as exc:
            logger.warning(
                "VLLMRayServer[%d]: monkey_patch_model failed: %s",
                self.replica_rank,
                exc,
            )

        if self.start_http:
            try:
                await self._start_http()
            except (ImportError, OSError, RuntimeError, ValueError) as exc:
                logger.warning(
                    "VLLMRayServer[%d]: HTTP server failed: %s", self.replica_rank, exc
                )
        return True

    async def _start_http(self) -> None:
        import uvicorn
        from vllm.entrypoints.openai.api_server import build_app, init_app_state
        from vllm.entrypoints.openai.cli_args import make_arg_parser

        try:
            from vllm.utils.argparse_utils import FlexibleArgumentParser
        except ImportError:
            from vllm.utils import FlexibleArgumentParser

        parser = FlexibleArgumentParser()
        parser = make_arg_parser(parser)
        http_args = parser.parse_args([])
        http_args.model = self.model_name

        app = build_app(http_args)
        import inspect

        sig = inspect.signature(init_app_state)
        vllm_config = None
        if hasattr(self.engine, "get_vllm_config"):
            vllm_config = await self.engine.get_vllm_config()
        if "vllm_config" in sig.parameters and vllm_config is not None:
            await init_app_state(self.engine, vllm_config, app.state, http_args)
        else:
            await init_app_state(self.engine, app.state, http_args)

        cfg = uvicorn.Config(
            app, host="0.0.0.0", port=self.http_port, log_level="warning"
        )
        server = uvicorn.Server(cfg)
        self._http_task = asyncio.create_task(server.serve())
        self._http_ready = True
        logger.info("VLLMRayServer[%d]: HTTP on :%d", self.replica_rank, self.http_port)

    def ready(self) -> bool:
        return self.engine is not None

    async def generate(
        self,
        prompt: Any,
        sampling_params: dict[str, Any],
        request_id: str | None = None,
    ) -> dict[str, Any]:
        """Single-prompt generate. ``prompt`` may be a text string (vLLM tokenizes
        it, matching the FIFO/verl path) or a list of token ids (TokensPrompt)."""
        from vllm import SamplingParams
        from vllm.inputs import TokensPrompt

        want_lp = sampling_params.get("logprobs", None) is not None
        sp = SamplingParams(
            n=1,
            max_tokens=int(sampling_params.get("max_tokens", 512)),
            temperature=float(sampling_params.get("temperature", 1.0)),
            top_p=float(sampling_params.get("top_p", 1.0)),
            top_k=int(sampling_params.get("top_k", -1)),
            logprobs=(0 if want_lp else None),
            seed=sampling_params.get("seed", None),
            stop_token_ids=sampling_params.get("stop_token_ids", None),
        )
        vllm_prompt = (
            prompt
            if isinstance(prompt, str)
            else TokensPrompt(prompt_token_ids=list(prompt))
        )
        rid = request_id or uuid4().hex
        final = None
        async for out in self.engine.generate(vllm_prompt, sp, request_id=rid):
            final = out
        comp = final.outputs[0]
        tok = list(comp.token_ids)
        lps = None
        if want_lp and comp.logprobs is not None:
            lps = [float(comp.logprobs[i].get(t).logprob) for i, t in enumerate(tok)]
        out = {
            "text": comp.text,
            "prompt_token_ids": list(final.prompt_token_ids),
            "token_ids": tok,
            "logprobs": lps,
            # MILES R3 contract: vLLM returns the top-k expert ids for every
            # model-input token as [sequence_length - 1, num_layers, top_k].
            # Keep this as a numpy array across Ray; the controller packs it
            # into an int16 DataProto tensor before dispatching to Megatron.
            "routed_experts": getattr(comp, "routed_experts", None),
        }
        # Rollout Routing Replay: the expert ids this engine actually selected,
        # uint8 [prompt + response - 1, num_layers, top_k]. Row t is the routing
        # of the forward at position t, i.e. the one that produced token t+1 --
        # the same alignment as `logprobs`. Present only when the engine was
        # built with enable_return_routed_experts.
        routed = getattr(comp, "routed_experts", None)
        if routed is not None:
            out["routed_experts"] = routed
        return out

    async def generate_batch(
        self,
        prompts: list,
        sampling_params: dict[str, Any],
    ) -> list[dict[str, Any]]:
        """Submit every prompt (text or token ids) as its own concurrent request."""
        tasks = [
            self.generate(p, sampling_params, request_id=uuid4().hex) for p in prompts
        ]
        return await asyncio.gather(*tasks)

    async def generate_group(
        self,
        prompt: list[int],
        num_generations: int,
        sampling_params: dict[str, Any],
    ) -> list[dict[str, Any]]:
        """``num_generations`` completions of one prompt (AsyncLLM interleaves calls)."""
        return await self.generate_batch(
            [list(prompt)] * num_generations, sampling_params
        )

    async def collective_rpc(
        self, method: str, args: tuple = (), kwargs: dict | None = None
    ) -> Any:
        return await self.engine.collective_rpc(method, args=args, kwargs=kwargs or {})

    async def get_rdma_capabilities(self) -> Any:
        """Collect the RDMA capability contract from every TP worker."""
        return await self.engine.collective_rpc(
            "get_rdma_capabilities",
            args=(),
            kwargs={},
        )

    async def update_weights_from_ipc(
        self,
        use_shm: bool = False,
        version: int | None = None,
        prequantized_fp8: bool = False,
    ) -> list:
        """Start the in-worker IPC receiver; blocks until the sender completes.

        Returns each vLLM worker's receive summary (timing dicts on the BF16 path).
        """
        results = await self.engine.collective_rpc(
            "update_weights_from_ipc",
            kwargs={
                "use_shm": use_shm,
                "version": version,
                "prequantized_fp8": bool(prequantized_fp8),
            },
        )
        t0 = time.perf_counter()
        await self.engine.reset_prefix_cache()
        reset_s = time.perf_counter() - t0
        out = []
        for r in results or []:
            if isinstance(r, dict) and "total_s" in r:
                r = {**r, "reset_prefix_cache_s": reset_s}
            out.append(r)
        return out

    async def reload_weights_from_path(self, weight_dir: str) -> bool:
        """Reload weights from a safetensors directory (for separation mode / TP>1)."""
        await self.engine.collective_rpc(
            "reload_weights_from_safetensors", kwargs={"weight_dir": weight_dir}
        )
        await self.engine.reset_prefix_cache()
        return True

    def rdma_preflight(self, interface: str, hca: str) -> dict[str, Any]:
        from pathlib import Path

        uverbs = sorted(Path("/dev/infiniband").glob("uverbs*"))
        if not uverbs or not (Path("/sys/class/infiniband") / hca).exists():
            raise RuntimeError(
                f"RDMA unavailable in rollout container: uverbs={uverbs}, hca={hca}"
            )
        if os.environ.get("NCCL_IB_DISABLE", "0") == "1":
            raise RuntimeError("NCCL_IB_DISABLE=1 would force Socket transport")
        return {
            "replica": self.replica_rank,
            "interface": interface,
            "hca": hca,
            "uverbs": len(uverbs),
        }

    async def init_rdma_weight_group(
        self,
        master_addr: str,
        master_port: int,
        base_rank: int,
        world_size: int,
        group_name: str,
        timeout_s: int = 600,
    ) -> bool:
        await self.engine.collective_rpc(
            "init_rdma_weight_group",
            kwargs={
                "master_addr": master_addr,
                "master_port": int(master_port),
                "base_rank": int(base_rank),
                "world_size": int(world_size),
                "group_name": group_name,
                "timeout_s": int(timeout_s),
            },
        )
        return True

    async def receive_weights_rdma(
        self,
        group_name: str,
        version: int,
        verify_full_load: bool = True,
        prequantized_fp8: bool = False,
        online_staging: str = "gpu",
    ) -> Any:
        stats = await self.engine.collective_rpc(
            "receive_weights_rdma",
            kwargs={
                "group_name": group_name,
                "version": int(version),
                "verify_full_load": bool(verify_full_load),
                "prequantized_fp8": bool(prequantized_fp8),
                "online_staging": str(online_staging),
            },
        )
        await self.engine.reset_prefix_cache()
        return stats

    async def destroy_rdma_weight_group(self, group_name: str) -> bool:
        if self.engine is not None:
            await self.engine.collective_rpc(
                "destroy_rdma_weight_group",
                kwargs={"group_name": group_name},
            )
        return True

    async def stage_weights_from_http(self, source_url: str, weight_dir: str) -> bool:
        """Download one exported checkpoint onto the rollout node atomically."""
        await asyncio.to_thread(
            self._stage_weights_from_http_sync, source_url, weight_dir
        )
        return True

    @staticmethod
    def _stage_weights_from_http_sync(source_url: str, weight_dir: str) -> None:
        import json
        from concurrent.futures import ThreadPoolExecutor
        from pathlib import Path
        from urllib.parse import quote
        from urllib.request import urlopen

        target = Path(weight_dir)
        target.mkdir(parents=True, exist_ok=True)

        def _download(filename: str, destination: Path) -> None:
            url = f"{source_url.rstrip('/')}/{quote(filename)}"
            temp = destination.with_suffix(destination.suffix + ".tmp")
            with urlopen(url, timeout=1800) as response, temp.open("wb") as out:
                while chunk := response.read(16 * 1024 * 1024):
                    out.write(chunk)
            temp.replace(destination)

        index_path = target / "model.safetensors.index.json"
        _download("model.safetensors.index.json", index_path)
        index = json.loads(index_path.read_text())
        expected_files = sorted(set(index["weight_map"].values()))
        # The writer serves independent immutable shards. Parallel downloads
        # avoid limiting a 400-Gb link to one Python TCP stream.
        workers = min(8, len(expected_files))
        with ThreadPoolExecutor(max_workers=workers) as pool:
            list(
                pool.map(
                    lambda filename: _download(filename, target / filename),
                    expected_files,
                )
            )

        expected = set(expected_files)
        for old in target.glob("model-*.safetensors"):
            if old.name not in expected:
                old.unlink()
        logger.info(
            "VLLMRayServer: staged %d weight shards from %s to %s",
            len(expected_files),
            source_url,
            target,
        )

    async def sleep(self, level: int = 2) -> bool:
        await self.engine.sleep(level=level)
        return True

    async def wake_up(self, tags: list[str] | None = None) -> bool:
        await self.engine.wake_up(tags=tags)
        return True

    async def reset_prefix_cache(self) -> bool:
        await self.engine.reset_prefix_cache()
        return True

    async def pause_generation(
        self, mode: str = "keep", clear_cache: bool = True
    ) -> bool:
        """Stop scheduling until :meth:`resume_generation`, e.g. around a weight load.

        ``keep`` freezes in-flight requests instead of aborting them. With
        ``clear_cache`` vLLM also preempts them and drops their KV, so on resume
        they re-prefill prompt plus generated tokens on whatever weights are
        loaded by then.
        """
        await self.engine.pause_generation(mode=mode, clear_cache=clear_cache)
        return True

    async def resume_generation(self) -> bool:
        await self.engine.resume_generation()
        return True

    async def wait_for_requests_to_drain(self, timeout_s: float = 60.0) -> bool:
        """Best-effort wait until no unfinished requests remain."""
        deadline = asyncio.get_event_loop().time() + timeout_s
        while asyncio.get_event_loop().time() < deadline:
            n = 0
            try:
                if hasattr(self.engine, "get_num_unfinished_requests"):
                    n = self.engine.get_num_unfinished_requests()
            except RuntimeError:
                n = 0
            if not n:
                return True
            await asyncio.sleep(0.05)
        return True

    async def shutdown(self) -> bool:
        try:
            if self._http_task is not None:
                self._http_task.cancel()
        # asyncio implementations can raise loop-specific errors while closing.
        except Exception:  # noqa: BLE001,S110 -- best-effort cleanup boundary
            pass
        try:
            if self.engine is not None and hasattr(self.engine, "shutdown"):
                self.engine.shutdown()
        # vLLM extensions can raise backend-specific errors during teardown.
        except Exception:  # noqa: BLE001,S110 -- best-effort cleanup boundary
            pass
        self.engine = None
        return True


class VLLMReplicaManager:
    """Driver-side controller for a fleet of colocated ``VLLMRayServer`` actors.

    Mirrors verl's ``vLLMReplica`` placement + ``LLMServerManager`` fan-out +
    ``CheckpointEngineManager`` sleep/wake orchestration, adapted to AtomRL's
    ``RayWorkerGroup``.
    """

    def __init__(
        self,
        actor_wg,
        model_name: str,
        engine_kwargs: dict[str, Any],
        *,
        base_port: int = 8700,
        start_http: bool = False,
        max_concurrency: int = 64,
        base_seed: int | None = None,
        tensor_parallel_size: int = 1,
    ) -> None:
        self.actor_wg = actor_wg
        self.model_name = model_name
        self.engine_kwargs = dict(engine_kwargs)
        self.base_port = int(base_port)
        self.start_http = bool(start_http)
        self.max_concurrency = int(max_concurrency)
        self.base_seed = base_seed
        # One engine per TP group of training actors: 32 actors with TP=8 give 4
        # replicas of 8 GPUs each. TP=1 keeps the original one-engine-per-GPU
        # layout. The weight sync depends on this exact grouping -- actor
        # ``rank`` feeds replica ``rank // tp`` on TP rank ``rank % tp``.
        tp = int(tensor_parallel_size or 1)
        if tp <= 1:
            # Callers that only fill engine_kwargs rather than passing it here.
            tp = int(self.engine_kwargs.get("tensor_parallel_size", 1) or 1)
        self.tensor_parallel_size = max(1, tp)
        # Resolved here rather than in create() so that num_replicas is never
        # tp-times too large for anything reading it in between.
        if actor_wg.num_workers % self.tensor_parallel_size:
            raise ValueError(
                f"rollout tensor_parallel_size={self.tensor_parallel_size} must divide "
                f"the actor count {actor_wg.num_workers}"
            )
        self.num_replicas = actor_wg.num_workers // self.tensor_parallel_size
        self.servers: list = []
        self.rdma_group_name: str | None = None
        self._rdma_capabilities_group: str | None = None
        self._rdma_capabilities: tuple[dict[str, object], ...] = ()

    def create(self) -> None:
        """Create + launch server actors colocated with training/rollout workers.

        Supports TP>1: when ``engine_kwargs["tensor_parallel_size"] > 1``,
        groups consecutive workers (by their GPU IDs) into multi-GPU replicas,
        mirroring ATOMReplicaManager's grouping logic.
        """
        import ray
        from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

        job_id = ray.get_runtime_context().get_job_id()
        infos = self.actor_wg.execute_all_sync("get_colocation_info")
        logger.info("VLLMReplicaManager: colocation infos = %s", infos)

        tp = self.tensor_parallel_size
        num_workers = len(infos)
        if num_workers % tp != 0:
            raise ValueError(
                f"num_workers ({num_workers}) must be divisible by "
                f"tensor_parallel_size ({tp})"
            )
        num_replicas = max(1, num_workers // tp)
        self.num_replicas = num_replicas

        remote_cls = ray.remote(VLLMRayServer)
        for r in range(num_replicas):
            group = infos[r * tp : (r + 1) * tp]
            node_ids = {g["node_id"] for g in group}
            if len(node_ids) != 1:
                raise RuntimeError(
                    f"replica {r} spans {len(node_ids)} nodes: actors "
                    f"{r * tp}..{r * tp + tp - 1} must be colocated for a TP={tp} "
                    "engine, but Ray placed them apart"
                )
            node_id = group[0]["node_id"]
            # Visible device j of the engine is actor (r*tp + j)'s physical GPU,
            # which is what makes the sender's rank % tp the receiver's local_rank.
            gpu_ids_str = ",".join(str(g) for info in group for g in info["gpu_ids"])
            # ROCm device pinning: select the physical GPU via CUDA/HIP visible
            # devices ONLY. Do NOT also set ROCR_VISIBLE_DEVICES to the physical
            # index -- ROCR filters at a lower level, so ROCR=<phys>+HIP=<phys>
            # double-filters ("No HIP GPUs available"). We still set the NOSET
            # flags (incl. ROCR) so Ray (num_gpus=0) doesn't clear visibility.
            env_vars = {
                "CUDA_VISIBLE_DEVICES": gpu_ids_str,
                "HIP_VISIBLE_DEVICES": gpu_ids_str,
                "RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES": "1",
                "RAY_EXPERIMENTAL_NOSET_HIP_VISIBLE_DEVICES": "1",
                "RAY_EXPERIMENTAL_NOSET_ROCR_VISIBLE_DEVICES": "1",
                "NCCL_CUMEM_ENABLE": "0",
                # This is the native vLLM backend.  ATOM's out-of-tree
                # platform plugin can otherwise override ROCm platform
                # detection, and a plugin import failure leaves device_type
                # empty (AsyncLLM then fails while constructing DeviceConfig).
                "ATOM_DISABLE_VLLM_PLUGIN": "1",
                "VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS": os.getenv(
                    "VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS", "300"
                ),
                "LUMEN_REPLICA_RANK": str(r),
                "LUMEN_RAY_JOB_ID": str(job_id),
            }
            _copy_vllm_runtime_env(env_vars)
            server = remote_cls.options(
                num_gpus=0,  # pinned manually via CUDA_VISIBLE_DEVICES; no Ray GPU slot
                num_cpus=tp,  # the mp executor forks one worker process per TP rank
                name=f"lumen-vllm-replica-{r}",
                max_concurrency=self.max_concurrency,
                scheduling_strategy=NodeAffinitySchedulingStrategy(
                    node_id=node_id, soft=False
                ),
                runtime_env={"env_vars": env_vars},
            ).remote(
                model_name=self.model_name,
                engine_kwargs=self._engine_kwargs_for_replica(r, str(job_id)),
                replica_rank=r,
                http_port=self.base_port + r,
                start_http=self.start_http,
                base_seed=self.base_seed,
            )
            self.servers.append(server)

        ray.get([s.launch.remote() for s in self.servers])
        logger.info(
            "VLLMReplicaManager: launched %d colocated rollout replicas (tp=%d, workers=%d).",
            num_replicas,
            tp,
            num_workers,
        )

    def _engine_kwargs_for_replica(
        self, replica_rank: int, job_id: str
    ) -> dict[str, Any]:
        """Give each replica an isolated torch compile cache."""
        kwargs = dict(self.engine_kwargs)
        if os.getenv("ATOM_ISOLATE_TORCH_COMPILE_CACHE", "1") not in {
            "1",
            "true",
            "TRUE",
            "yes",
            "YES",
        }:
            return kwargs

        comp_cfg = dict(kwargs.get("compilation_config") or {})
        cache_root = os.getenv(
            "ATOM_TORCH_COMPILE_CACHE_ROOT",
            "/tmp/atom_torch_compile_cache",
        )
        safe_job_id = "".join(
            ch if ch.isalnum() or ch in "-_." else "_" for ch in job_id
        )
        comp_cfg["cache_dir"] = os.path.join(
            cache_root, safe_job_id, f"vllm_replica_{replica_rank}"
        )
        kwargs["compilation_config"] = comp_cfg
        logger.info(
            "VLLMReplicaManager: replica %d torch compile cache_dir=%s",
            replica_rank,
            comp_cfg["cache_dir"],
        )
        return kwargs

    # -- fan-out helpers -------------------------------------------------
    def sleep_all(self, level: int = 2) -> None:
        import ray

        ray.get([s.sleep.remote(level) for s in self.servers])

    def wake_all(self, tags: list[str] | None = None) -> None:
        import ray

        ray.get([s.wake_up.remote(tags) for s in self.servers])

    def drain_all(self) -> None:
        import ray

        ray.get([s.wait_for_requests_to_drain.remote() for s in self.servers])

    def pause_all(self, mode: str = "keep", clear_cache: bool = True) -> None:
        import ray

        ray.get([s.pause_generation.remote(mode, clear_cache) for s in self.servers])

    def resume_all(self) -> None:
        import ray

        ray.get([s.resume_generation.remote() for s in self.servers])

    def reload_weights_from_path(self, weight_dir: str) -> None:
        """Reload weights one replica at a time.

        Each TP=2 replica reads the full checkpoint on both workers.  Loading
        all replicas concurrently creates eight competing 61GB reads and can
        leave vLLM workers busy beyond the sample RPC timeout.
        """
        import ray

        for server in self.servers:
            ray.get(server.reload_weights_from_path.remote(weight_dir))

    def reload_weights_from_http(self, source_url: str, weight_dir: str) -> None:
        """Transfer once to the rollout node, then reload every local replica."""
        import ray

        if not self.servers:
            raise RuntimeError("No vLLM replicas are available for weight reload.")
        ray.get(self.servers[0].stage_weights_from_http.remote(source_url, weight_dir))
        self.reload_weights_from_path(weight_dir)

    def init_rdma_weight_group(
        self,
        actor_wg,
        *,
        interface: str,
        hca: str,
        require_rdma: bool,
        timeout_s: int,
        group_name: str,
    ) -> dict[str, Any]:
        """Create one persistent source+8-worker RCCL communicator."""
        import ray

        self._clear_rdma_capabilities()
        if require_rdma:
            checks = [actor_wg.call_single_async(0, "rdma_preflight", interface, hca)]
            checks.extend(
                server.rdma_preflight.remote(interface, hca) for server in self.servers
            )
            logger.info("RDMA preflight: %s", ray.get(checks))
        rendezvous = actor_wg.execute_rank_zero_sync("get_rdma_rendezvous", interface)
        master_addr = str(rendezvous["address"])
        master_port = int(rendezvous["port"])
        world_size = 1 + self.num_replicas * self.tensor_parallel_size
        refs = [
            actor_wg.call_single_async(
                0,
                "init_rdma_weight_group",
                master_addr,
                master_port,
                world_size,
                group_name,
                timeout_s,
            )
        ]
        for replica_rank, server in enumerate(self.servers):
            refs.append(
                server.init_rdma_weight_group.remote(
                    master_addr,
                    master_port,
                    1 + replica_rank * self.tensor_parallel_size,
                    world_size,
                    group_name,
                    timeout_s,
                )
            )
        ray.get(refs)
        self.rdma_group_name = group_name
        logger.info(
            "RDMA weight group ready: %s master=%s:%d world=%d",
            group_name,
            master_addr,
            master_port,
            world_size,
        )
        return {
            "master_addr": master_addr,
            "master_port": master_port,
            "world_size": world_size,
        }

    def _clear_rdma_capabilities(self) -> None:
        self._rdma_capabilities_group = None
        self._rdma_capabilities = ()

    def _validate_rdma_capabilities(self) -> None:
        if self._rdma_capabilities_group == self.rdma_group_name:
            return

        import ray

        from atomrl.engine.inference.rdma_protocol import RDMA_PROTOCOL_VERSION

        validated: list[dict[str, object]] = []
        for server_rank, server in enumerate(self.servers):
            try:
                rpc = server.get_rdma_capabilities
                capabilities = ray.get(rpc.remote())
            except Exception as exc:
                raise RuntimeError(
                    "RDMA capability handshake failed for "
                    f"server={server_rank} "
                    f"workers=0..{self.tensor_parallel_size - 1}: "
                    "get_rdma_capabilities RPC is unavailable or failed"
                ) from exc

            if not isinstance(capabilities, (list, tuple)):
                raise TypeError(
                    "Invalid RDMA capability response for "
                    f"server={server_rank} worker=unknown: expected a worker list"
                )
            if len(capabilities) != self.tensor_parallel_size:
                raise RuntimeError(
                    "Invalid RDMA capability response for "
                    f"server={server_rank} worker=unknown: expected "
                    f"{self.tensor_parallel_size} TP workers, got "
                    f"{len(capabilities)}"
                )

            for worker_rank, capability in enumerate(capabilities):
                identity = f"server={server_rank} worker={worker_rank}"
                if not isinstance(capability, dict):
                    raise TypeError(
                        f"Invalid RDMA capability for {identity}: expected mapping"
                    )
                if capability.get("protocol_version") != RDMA_PROTOCOL_VERSION:
                    raise RuntimeError(
                        f"Incompatible RDMA capability for {identity}: "
                        f"protocol_version={capability.get('protocol_version')!r}, "
                        f"expected {RDMA_PROTOCOL_VERSION}"
                    )
                module_path = capability.get("module_path")
                if not isinstance(module_path, str) or not module_path:
                    raise RuntimeError(
                        f"Incompatible RDMA capability for {identity}: "
                        "module_path is missing"
                    )
                for field in ("online_quant_reload", "prequantized_stream"):
                    if capability.get(field) is not True:
                        raise RuntimeError(
                            f"Incompatible RDMA capability for {identity}: "
                            f"{field}={capability.get(field)!r}, expected True; "
                            f"module_path={module_path}"
                        )
                validated.append(dict(capability))

        self._rdma_capabilities = tuple(validated)
        self._rdma_capabilities_group = self.rdma_group_name

    def start_receive_weights_rdma(
        self,
        *,
        version: int,
        verify_full_load: bool,
        prequantized_fp8: bool = False,
        online_staging: str = "gpu",
    ) -> list:
        if not self.rdma_group_name:
            raise RuntimeError("RDMA weight group has not been initialized")
        self._validate_rdma_capabilities()
        kwargs = {
            "group_name": self.rdma_group_name,
            "version": int(version),
            "verify_full_load": bool(verify_full_load),
            "prequantized_fp8": bool(prequantized_fp8),
            "online_staging": str(online_staging),
        }
        return [server.receive_weights_rdma.remote(**kwargs) for server in self.servers]

    def destroy_rdma_weight_group(self, actor_wg=None) -> None:
        import ray

        self._clear_rdma_capabilities()
        if not self.rdma_group_name:
            return
        refs = [
            server.destroy_rdma_weight_group.remote(self.rdma_group_name)
            for server in self.servers
        ]
        if actor_wg is not None:
            refs.append(actor_wg.call_single_async(0, "destroy_rdma_weight_group"))
        try:
            ray.get(refs)
        finally:
            self.rdma_group_name = None

    def shutdown(self) -> None:
        import ray

        self._clear_rdma_capabilities()
        try:
            ray.get([s.shutdown.remote() for s in self.servers])
        # Actor teardown can surface backend exceptions through Ray wrappers.
        except Exception:  # noqa: BLE001,S110 -- best-effort cleanup boundary
            pass
        for s in self.servers:
            try:
                ray.kill(s)
            except Exception:  # noqa: BLE001,S110 -- best-effort cleanup boundary
                pass
        self.servers = []
