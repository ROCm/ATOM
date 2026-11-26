# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

"""Fully-async RL trainer with decoupled Rollouter and Trainer.

Inspired by VERL's ``fully_async_policy`` — see
https://github.com/verl-project/verl/blob/main/docs/advance/fully_async.md

Architecture:
    ┌──────────┐       ┌──────────────┐       ┌──────────┐
    │ Rollouter├──────►│ MessageQueue ├──────►│ Trainer  │
    └────┬─────┘       └──────────────┘       └────┬─────┘
         │                                          │
         └──────────── WeightSync ◄────────────────┘

Modes (controlled by ``AsyncTrainingConfig``):
  a) **On-policy** (staleness=0, sync_step=1)
  b) **Stream off-policy** (staleness=0, sync_step>1)
  c) **Async stream + stale** (staleness>0, partial=False)
  d) **Async stream + partial** (staleness>0, partial=True)

In the colocated (default) layout, both rollouter and trainer threads
share the same GPU set and swap memory via optimizer offload.  In the
separated layout, they run on disjoint GPU groups.
"""

from __future__ import annotations

import gc
import logging
import os
import queue
import threading
import time
from typing import Any

import torch

import atomrl.algorithms  # noqa: F401
from atomrl.controller import RayCluster
from atomrl.core.config import AtomRLConfig
from atomrl.core.protocol import DataProto
from atomrl.core.registry import ALGORITHM_REGISTRY
from atomrl.quantization.rollout_correction import apply_rollout_correction
from atomrl.trainer.callbacks import Callback, LoggingCallback
from atomrl.trainer.message_queue import AsyncMessageQueue, SampleItem
from atomrl.trainer.weight_sync import FilesystemWeightSync
from atomrl.utils.metrics import MetricsTracker

logger = logging.getLogger(__name__)


def _algo_num_generations(config: AtomRLConfig) -> int:
    name = config.algorithm.name.lower()
    if name == "ppo":
        return 1
    if name == "dapo":
        return int(config.algorithm.dapo.num_generations)
    return int(config.algorithm.grpo.num_generations)


class AsyncRLTrainer:
    """The ``async_training`` trainer; superseded, see the warning below.

    ``train()`` generates and trains in turn on the same GPUs. Updated weights
    are pushed to ``FilesystemWeightSync``, but only ``_rollout_loop``, which
    is never started, would point the engine at them, so rollouts keep using
    the initial weights. Asynchronous training is ``RLTrainer`` with
    ``controller.ray.one_step_off_policy`` or ``controller.ray.fully_async``.
    """

    def __init__(self, config: AtomRLConfig) -> None:
        logger.warning(
            "async_training.enabled selects AsyncRLTrainer, which does not overlap "
            "rollout with training and never loads updated weights into the rollout "
            "engine. Use controller.ray.fully_async or "
            "controller.ray.one_step_off_policy instead (see "
            "atomrl-docs/docs/source/advance/async_training.md)."
        )
        self.config = config
        self.global_step: int = 0
        self.param_version: int = 0
        self.last_metrics: dict[str, float] = {}
        self.callbacks: list[Callback] = []

        self._algorithm: Any = None
        self._metrics = MetricsTracker()

        self._actor_model: torch.nn.Module | None = None
        self._ref_model: torch.nn.Module | None = None
        self._ref_on_cpu: bool = True
        self._optimizer: torch.optim.Optimizer | None = None
        self._tokenizer: Any = None
        self._dataset: Any = None
        self._atom_engine: Any = None
        self._ray_cluster: RayCluster | None = None
        self._use_atom: bool = config.policy.generation_backend.lower() == "atom"

        self._is_distributed: bool = torch.distributed.is_initialized()
        self._rank: int = torch.distributed.get_rank() if self._is_distributed else 0
        self._world_size: int = (
            torch.distributed.get_world_size() if self._is_distributed else 1
        )
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        self._device = torch.device(
            f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu"
        )

        acfg = config.async_training
        self._msg_queue = AsyncMessageQueue(
            maxsize=acfg.queue_maxsize,
            staleness_threshold=acfg.staleness_threshold,
        )
        self._weight_sync = FilesystemWeightSync(
            sync_dir=acfg.weight_sync_dir,
        )
        self._require_batches = acfg.require_batches
        self._sync_every = acfg.trigger_parameter_sync_step
        self._use_rollout_log_probs = acfg.use_rollout_log_probs
        self._partial_rollout = acfg.partial_rollout
        self._staleness = acfg.staleness_threshold

        self._rollout_thread: threading.Thread | None = None
        self._stop_rollout = threading.Event()
        self._sync_requested = threading.Event()
        self._sync_complete = threading.Event()

        self._rollout_idle_time: float = 0.0
        self._trainer_idle_time: float = 0.0

    def setup(self) -> None:
        """Initialize models, optimizer, dataset, and algorithm.

        Delegates to the same model-building logic as ``RLTrainer``.
        """
        if bool(getattr(self.config.controller.ray, "enabled", False)):
            if RayCluster is None:
                raise RuntimeError(
                    "Ray controller is enabled but RayCluster is unavailable."
                )
            self._ray_cluster = RayCluster(self.config.cluster)
            self._ray_cluster.init()

        model_name = self.config.policy.model_name
        if not model_name:
            raise ValueError("config.policy.model_name is required.")

        algo_cls: type[Any] = ALGORITHM_REGISTRY.get(self.config.algorithm.name)
        self._algorithm = algo_cls(self.config)

        quant = {}
        tq = self.config.quantization.training
        if tq.fp8:
            quant["fp8"] = tq.fp8
        quant["fp8_weight_cache"] = tq.fp8_weight_cache
        quant["lumen_norm"] = tq.lumen_norm
        quant["fused_mlp"] = tq.fused_mlp
        quant["fused_rope"] = tq.fused_rope

        from atomrl.engine.training.fsdp_backend import FSDP2Backend

        logger.info(
            "[rank %d] AsyncRLTrainer: building actor model: %s", self._rank, model_name
        )
        self._actor_model = FSDP2Backend.build_model(model_name)
        self._actor_model = FSDP2Backend.apply_lumen_optimizations(
            self._actor_model, quant
        )

        if self._is_distributed:
            fsdp_cfg = {"enabled": True}
            if (
                hasattr(self.config.policy, "training")
                and hasattr(self.config.policy.training, "fsdp_cfg")
                and self.config.policy.training.fsdp_cfg
            ):
                fsdp_cfg.update(self.config.policy.training.fsdp_cfg)
            self._actor_model = FSDP2Backend.apply_fsdp2(self._actor_model, fsdp_cfg)
        else:
            self._actor_model.to(self._device)

        self._optimizer = torch.optim.AdamW(
            [p for p in self._actor_model.parameters() if p.requires_grad],
            lr=1e-6,
            weight_decay=0.01,
        )

        from transformers import AutoTokenizer

        self._tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=True
        )
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token
        self._tokenizer.padding_side = "left"

        kl_coeff = 0.0
        algo_name = self.config.algorithm.name.lower()
        if algo_name == "dapo":
            kl_coeff = self.config.algorithm.dapo.kl_coeff
        elif algo_name == "grpo":
            kl_coeff = self.config.algorithm.grpo.kl_coeff
        elif algo_name == "ppo":
            kl_coeff = self.config.algorithm.ppo.kl_coeff

        if kl_coeff > 0.0:
            from transformers import AutoModelForCausalLM

            self._ref_model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.bfloat16,
                attn_implementation="sdpa",
                trust_remote_code=True,
            )
            self._ref_model.eval()
            for p in self._ref_model.parameters():
                p.requires_grad_(False)
        else:
            self._ref_model = None

        if self._use_atom and self._rank == 0:
            from atomrl.engine.inference.atom_engine import AtomEngine

            atom_cfg = self.config.policy.generation.atom_cfg
            self._atom_engine = AtomEngine(config=atom_cfg, model_name=model_name)
        elif self._use_atom:
            logger.info(
                "[rank %d] Skipping ATOM engine init (rank 0 handles rollout)",
                self._rank,
            )

        self._load_dataset()

        if not self.callbacks:
            self.callbacks.append(
                LoggingCallback(interval=max(1, self.config.logger.log_interval))
            )

        if self._is_distributed:
            torch.distributed.barrier()

        logger.info(
            "[rank %d] AsyncRLTrainer.setup complete: algo=%s, model=%s, "
            "staleness=%.2f, sync_every=%d, require_batches=%d",
            self._rank,
            self.config.algorithm.name,
            model_name,
            self._staleness,
            self._sync_every,
            self._require_batches,
        )

    def _load_dataset(self) -> None:
        """Load training dataset — same logic as RLTrainer."""

        dataset_path = self.config.reward.dataset
        if not dataset_path:
            self._dataset = None
            return

        from datasets import load_dataset

        if os.path.isfile(dataset_path) or os.path.isdir(dataset_path):
            if dataset_path.endswith(".parquet"):
                self._dataset = load_dataset(
                    "parquet", data_files=dataset_path, split="train"
                )
            elif dataset_path.endswith((".jsonl", ".json")):
                self._dataset = load_dataset(
                    "json", data_files=dataset_path, split="train"
                )
            else:
                self._dataset = load_dataset(dataset_path, split="train")
        else:
            self._dataset = load_dataset(dataset_path, split="train")
        logger.info("Loaded dataset: %d samples", len(self._dataset))

    # ------------------------------------------------------------------
    # Rollout thread
    # ------------------------------------------------------------------

    def _rollout_loop(self, total_samples: int) -> None:
        """Background thread: generate samples and push to queue.

        Runs until ``total_samples`` are produced or ``_stop_rollout`` is set.
        Handles parameter sync interrupts for partial rollout support.
        """

        num_generations = _algo_num_generations(self.config)
        produced = 0
        dataset_idx = 0

        while produced < total_samples and not self._stop_rollout.is_set():
            if self._sync_requested.is_set():
                idle_start = time.time()
                logger.info(
                    "Rollout: pausing for parameter sync (v%d)", self.param_version
                )
                self._sync_requested.clear()
                self._sync_complete.wait(timeout=120.0)
                self._sync_complete.clear()
                self._rollout_idle_time += time.time() - idle_start

                if self._atom_engine is not None:
                    latest = self._weight_sync.latest_path()
                    if latest:
                        self._atom_engine.set_weight_path(latest)

            prompts, gts = self._get_single_prompt(dataset_idx)
            dataset_idx += 1

            try:
                sample = self._generate_single_sample(prompts, gts, num_generations)
            # Rollout producer isolates per-sample engine and reward failures.
            except Exception as e:  # noqa: BLE001 -- boundary
                logger.warning("Rollout: generation error: %s", e)
                continue

            item = SampleItem(
                data=sample,
                param_version=self.param_version,
            )
            try:
                self._msg_queue.put(item, timeout=30.0)
                produced += 1
            except queue.Full:
                logger.warning("Rollout: queue full, dropping sample")

        logger.info(
            "Rollout: produced %d/%d samples, stopping.", produced, total_samples
        )

    def _get_single_prompt(self, idx: int) -> tuple[list[str], list[str]]:
        """Get a single prompt from the dataset."""
        import json as _json

        if self._dataset is None:
            return [f"What is {idx} + {idx + 1}?"], [str(2 * idx + 1)]

        sample = self._dataset[idx % len(self._dataset)]
        prompt_raw = (
            sample.get("prompt")
            or sample.get("question")
            or sample.get("input")
            or sample.get("problem")
            or ""
        )
        if isinstance(prompt_raw, list):
            text_parts = [
                m.get("content", "") for m in prompt_raw if isinstance(m, dict)
            ]
            prompt_text = "\n".join(text_parts)
        elif isinstance(prompt_raw, str) and prompt_raw.startswith("["):
            try:
                msgs = _json.loads(prompt_raw)
                prompt_text = "\n".join(
                    m.get("content", "") for m in msgs if isinstance(m, dict)
                )
            except (_json.JSONDecodeError, TypeError):
                prompt_text = prompt_raw
        else:
            prompt_text = str(prompt_raw)

        rm_raw = sample.get("reward_model", {})
        if isinstance(rm_raw, str):
            try:
                rm_raw = _json.loads(rm_raw)
            except (_json.JSONDecodeError, TypeError):
                rm_raw = {}
        if isinstance(rm_raw, dict) and rm_raw.get("ground_truth", "") != "":
            gt = rm_raw["ground_truth"]
        else:
            gt = (
                sample.get("answer")
                or sample.get("solution")
                or sample.get("expected_answer")
                or ""
            )

        if self._tokenizer and hasattr(self._tokenizer, "apply_chat_template"):
            raw = (
                sample.get("prompt")
                or sample.get("question")
                or sample.get("input")
                or sample.get("problem")
                or ""
            )
            if isinstance(raw, list):
                try:
                    prompt_text = self._tokenizer.apply_chat_template(
                        raw,
                        tokenize=False,
                        add_generation_prompt=True,
                    )
                except (RuntimeError, TypeError, ValueError):
                    pass

        return [prompt_text], [gt]

    def _generate_single_sample(
        self,
        prompts: list[str],
        ground_truths: list[str],
        num_generations: int,
    ) -> DataProto:
        """Generate responses for a single prompt and package as DataProto."""
        from atomrl.trainer.rl_trainer import RLTrainer

        expanded = [p for p in prompts for _ in range(num_generations)]

        algo_name = self.config.algorithm.name.lower()
        max_resp = self.config.policy.max_response_length
        max_tok = max_resp if max_resp > 0 else 2048
        sp: dict[str, Any] = {"max_tokens": max_tok}
        if algo_name == "dapo":
            sp.update({"temperature": 1.0, "top_p": 0.95})
        elif algo_name == "grpo":
            sp.update({"temperature": 0.7})

        if self._atom_engine is not None:
            self._atom_engine.wake_up()
            response_texts = self._atom_engine.generate(expanded, sampling_params=sp)
            self._atom_engine.sleep()
        else:
            raise RuntimeError("AsyncRLTrainer requires ATOM engine for rollout")

        full_texts = [p + r for p, r in zip(expanded, response_texts)]
        encoding = self._tokenizer(
            full_texts,
            padding=True,
            truncation=True,
            max_length=self.config.policy.max_total_sequence_length,
            return_tensors="pt",
        )
        sequences = encoding["input_ids"]
        seq_mask = encoding["attention_mask"]

        prompt_encoding = self._tokenizer(
            expanded,
            padding=True,
            truncation=True,
            max_length=min(1024, self.config.policy.max_total_sequence_length // 2),
            return_tensors="pt",
        )
        prompt_lengths = prompt_encoding["attention_mask"].sum(dim=1).tolist()

        response_mask = seq_mask.clone()
        for i, plen in enumerate(prompt_lengths):
            response_mask[i, :plen] = 0
        response_mask = response_mask[:, 1:]

        from atomrl.rewards.math_reward import compute_math_reward

        expanded_gts = [gt for gt in ground_truths for _ in range(num_generations)]
        rewards, _details = compute_math_reward(response_texts, expanded_gts)

        old_log_probs = None
        if self._use_rollout_log_probs:
            self._actor_model.eval()
            with torch.no_grad():
                seqs_dev = sequences.to(self._device)
                mask_dev = seq_mask.to(self._device)
                outputs = self._actor_model(input_ids=seqs_dev, attention_mask=mask_dev)
                logits = outputs.logits if hasattr(outputs, "logits") else outputs
                old_log_probs = RLTrainer._fused_token_log_probs(logits, seqs_dev).cpu()

        tensors: dict[str, torch.Tensor] = {
            "input_ids": sequences,
            "attention_mask": seq_mask,
            "rewards": rewards,
            "response_mask": response_mask,
        }
        if old_log_probs is not None:
            tensors["old_log_probs"] = old_log_probs

        return DataProto(
            tensors=tensors,
            meta={
                "responses": response_texts,
                "ground_truths": expanded_gts,
                "prompt_lengths": prompt_lengths,
                "response_lengths": [
                    int(response_mask[i].sum().item())
                    for i in range(response_mask.shape[0])
                ],
            },
        )

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------

    @staticmethod
    def _fused_token_log_probs(
        logits: torch.Tensor, target_ids: torch.Tensor
    ) -> torch.Tensor:
        """Row-chunked log-prob computation (delegates to RLTrainer)."""
        from atomrl.trainer.rl_trainer import RLTrainer

        return RLTrainer._fused_token_log_probs(logits, target_ids)

    def _train_step(self, batch: DataProto) -> dict[str, float]:
        """One gradient step on the actor model."""
        self._actor_model.train()
        sequences = batch["input_ids"].to(self._device)
        attention_mask = batch["attention_mask"].to(self._device)

        outputs = self._actor_model(input_ids=sequences, attention_mask=attention_mask)
        logits = outputs.logits if hasattr(outputs, "logits") else outputs
        token_log_probs = self._fused_token_log_probs(logits, sequences)
        batch.tensors["log_probs"] = token_log_probs

        if self.global_step < 6 and self._rank == 0:
            olp = batch.tensors.get("old_log_probs")
            nlp = token_log_probs
            if olp is not None:
                ratio = (nlp - olp).exp()
                logger.info(
                    "NaN-DEBUG train_step [step=%d] old_lp: nan=%d inf=%d min=%.4f max=%.4f | "
                    "new_lp: nan=%d inf=%d min=%.4f max=%.4f | "
                    "ratio: nan=%d inf=%d min=%.4f max=%.4f",
                    self.global_step,
                    olp.isnan().sum().item(),
                    olp.isinf().sum().item(),
                    (
                        olp[~olp.isnan()].min().item()
                        if not olp.isnan().all()
                        else float("nan")
                    ),
                    (
                        olp[~olp.isnan()].max().item()
                        if not olp.isnan().all()
                        else float("nan")
                    ),
                    nlp.isnan().sum().item(),
                    nlp.isinf().sum().item(),
                    (
                        nlp[~nlp.isnan()].min().item()
                        if not nlp.isnan().all()
                        else float("nan")
                    ),
                    (
                        nlp[~nlp.isnan()].max().item()
                        if not nlp.isnan().all()
                        else float("nan")
                    ),
                    ratio.isnan().sum().item(),
                    ratio.isinf().sum().item(),
                    (
                        ratio[~ratio.isnan()].min().item()
                        if not ratio.isnan().all()
                        else float("nan")
                    ),
                    (
                        ratio[~ratio.isnan()].max().item()
                        if not ratio.isnan().all()
                        else float("nan")
                    ),
                )

        loss, metrics = self._algorithm.compute_loss(batch)
        loss = loss.to(self._device)

        if loss.isnan():
            metrics["loss"] = float("nan")
            return metrics

        num_accum = getattr(self, "_grad_accum_steps", 1)
        (loss / num_accum).backward()
        metrics["loss"] = float(loss.detach())
        return metrics

    def _trigger_param_sync(self) -> None:
        """Push updated weights and notify rollout thread."""
        self.param_version += 1
        self._msg_queue.current_param_version = self.param_version

        try:
            sd = {k: v.data for k, v in self._actor_model.named_parameters()}
            self._weight_sync.push(sd, version=self.param_version)
        except (OSError, RuntimeError, ValueError) as e:
            logger.warning("Weight sync push failed: %s", e)

        if self._partial_rollout:
            self._sync_requested.set()
            time.sleep(0.1)
            self._sync_complete.set()

        logger.info(
            "Param sync triggered: v%d (queue=%d)",
            self.param_version,
            self._msg_queue.qsize(),
        )

    def _broadcast_batch(self, batch: DataProto | None) -> DataProto:
        """Rank 0 broadcasts a DataProto to all ranks via torch.distributed."""
        import pickle

        if not self._is_distributed or self._world_size == 1:
            return batch

        if self._rank == 0:
            data_bytes = pickle.dumps(batch)
            size_tensor = torch.tensor(
                [len(data_bytes)], dtype=torch.long, device=self._device
            )
        else:
            size_tensor = torch.tensor([0], dtype=torch.long, device=self._device)

        torch.distributed.broadcast(size_tensor, src=0)
        nbytes = int(size_tensor.item())

        if self._rank == 0:
            buf = torch.frombuffer(bytearray(data_bytes), dtype=torch.uint8).to(
                self._device
            )
        else:
            buf = torch.empty(nbytes, dtype=torch.uint8, device=self._device)

        torch.distributed.broadcast(buf, src=0)

        if self._rank != 0:
            batch = pickle.loads(buf.cpu().numpy().tobytes())

        for k in list(batch.tensors.keys()):
            batch.tensors[k] = batch.tensors[k].to(self._device)

        return batch

    def _offload_optimizer_to_cpu(self) -> None:
        """Move optimizer state tensors to CPU to free GPU memory for ATOM."""
        if self._optimizer is None:
            return
        for state in self._optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor) and v.is_cuda:
                    state[k] = v.cpu()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.debug("Optimizer offloaded to CPU")

    def _reload_optimizer_to_gpu(self) -> None:
        """Move optimizer state tensors back to GPU."""
        if self._optimizer is None:
            return
        for state in self._optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor) and not v.is_cuda:
                    state[k] = v.to(self._device)
        logger.debug("Optimizer reloaded to GPU")

    def _generate_rollout_batch(
        self, step: int, num_generations: int, dataset_idx: int
    ) -> tuple[DataProto, int]:
        """Generate a rollout batch on rank 0 using ATOM, broadcast to all ranks.

        Handles optimizer offload/reload for colocated GPU sharing.
        Returns (batch, new_dataset_idx).
        """
        num_prompts = max(1, self._require_batches)

        if self._rank == 0:
            if step > 0:
                self._offload_optimizer_to_cpu()

            all_samples: list[DataProto] = []
            for i in range(num_prompts):
                prompts, gts = self._get_single_prompt(dataset_idx + i)
                try:
                    sample = self._generate_single_sample(prompts, gts, num_generations)
                    all_samples.append(sample)
                # Batch generation continues after an individual backend sample fails.
                except Exception as e:  # noqa: BLE001 -- boundary
                    logger.warning("Generation error at idx %d: %s", dataset_idx + i, e)

            if all_samples:
                batch = DataProto.merge(all_samples)
            else:
                batch = DataProto()

            if step > 0:
                self._reload_optimizer_to_gpu()
        else:
            batch = None

        batch = self._broadcast_batch(batch)
        return batch, dataset_idx + num_prompts

    def train(self) -> None:
        """Main async training loop — staged pipeline for colocated mode.

        For colocated GPU sharing, rollout and training alternate on the same
        GPU set. Rank 0 runs the ATOM engine for generation while other ranks
        wait, then all ranks participate in the FSDP2 training step.
        """
        if self._algorithm is None or self._actor_model is None:
            raise RuntimeError("Call setup() before train().")

        num_gen = _algo_num_generations(self.config)
        total_steps = int(self.config.num_training_steps)
        mini_bs = max(1, int(self.config.policy.train_micro_batch_size))

        if self._rank == 0:
            for cb in self.callbacks:
                cb.on_train_begin(self)

        logger.info(
            "[rank %d] AsyncRLTrainer: starting staged pipeline "
            "(total_steps=%d, sync_every=%d, require_batches=%d)",
            self._rank,
            total_steps,
            self._sync_every,
            self._require_batches,
        )

        dataset_idx = 0
        local_updates = 0

        for step in range(total_steps * self._sync_every):
            step_start = time.time()
            self.global_step = step

            if self._rank == 0 and step % self._sync_every == 0:
                sync_step = step // self._sync_every
                for cb in self.callbacks:
                    cb.on_step_begin(self, sync_step)

            if self._is_distributed:
                torch.distributed.barrier()

            t0 = time.time()
            batch, dataset_idx = self._generate_rollout_batch(
                step, num_gen, dataset_idx
            )
            gen_time = time.time() - t0

            if batch.batch_size == 0:
                logger.warning("[step %d] Empty rollout batch, skipping", step)
                continue

            self._actor_model.eval()
            with torch.no_grad():
                all_ids = batch["input_ids"]
                all_mask = batch["attention_mask"]
                bs = all_ids.shape[0]
                chunk_sz = max(1, mini_bs)
                old_lp_chunks = []
                for ci in range(0, bs, chunk_sz):
                    c_ids = all_ids[ci : ci + chunk_sz]
                    c_mask = all_mask[ci : ci + chunk_sz]
                    c_out = self._actor_model(input_ids=c_ids, attention_mask=c_mask)
                    c_logits = c_out.logits if hasattr(c_out, "logits") else c_out
                    old_lp_chunks.append(self._fused_token_log_probs(c_logits, c_ids))
                    del c_out, c_logits
                batch.tensors["old_log_probs"] = torch.cat(old_lp_chunks, dim=0)
                del old_lp_chunks

            if "ref_log_probs" not in batch.tensors:
                batch.tensors["ref_log_probs"] = torch.zeros_like(
                    batch.tensors["old_log_probs"]
                )

            batch = self._algorithm.compute_advantages(batch)
            batch = apply_rollout_correction(batch, self.config)

            max_tok = int(self.config.policy.max_token_len_per_gpu)
            from atomrl.trainer.rl_trainer import RLTrainer

            dummy_trainer = RLTrainer.__new__(RLTrainer)
            dummy_trainer.config = self.config
            mini_batches = dummy_trainer._dynamic_mini_batches(batch, max_tok, mini_bs)
            self._grad_accum_steps = len(mini_batches)

            self._optimizer.zero_grad(set_to_none=True)
            metrics_accum: dict[str, float] = {}
            mb_count = 0
            for mini in mini_batches:
                m = self._train_step(mini)
                for k, v in m.items():
                    metrics_accum[k] = metrics_accum.get(k, 0.0) + v
                mb_count += 1

            if self._rank == 0 and step < 6:
                nan_grads = sum(
                    1
                    for p in self._actor_model.parameters()
                    if p.grad is not None and p.grad.isnan().any()
                )
                total_grads = sum(
                    1 for p in self._actor_model.parameters() if p.grad is not None
                )
                logger.info(
                    "NaN-DEBUG [step=%d] pre-clip: nan_grads=%d/%d",
                    step,
                    nan_grads,
                    total_grads,
                )

            nan_fixed = 0
            for p in self._actor_model.parameters():
                if p.grad is not None and p.grad.isnan().any():
                    p.grad = torch.where(
                        p.grad.isnan(), torch.zeros_like(p.grad), p.grad
                    )
                    nan_fixed += 1

            grad_norm = torch.nn.utils.clip_grad_norm_(
                self._actor_model.parameters(), max_norm=1.0
            )
            self._optimizer.step()
            self._optimizer.zero_grad(set_to_none=True)

            if nan_fixed > 0 and self._rank == 0:
                logger.warning(
                    "[step %d] Zeroed NaN grads in %d/%d params, grad_norm=%.4f",
                    step,
                    nan_fixed,
                    sum(1 for _ in self._actor_model.parameters()),
                    float(grad_norm),
                )

            if self._rank == 0 and step < 6:
                nan_params = sum(
                    1 for p in self._actor_model.parameters() if p.data.isnan().any()
                )
                logger.info(
                    "NaN-DEBUG [step=%d] post-optim: grad_norm=%.4f nan_params=%d/%d",
                    step,
                    float(grad_norm),
                    nan_params,
                    sum(1 for _ in self._actor_model.parameters()),
                )

            local_updates += 1

            if local_updates >= self._sync_every:
                self._trigger_param_sync()
                local_updates = 0

            denom = max(1, mb_count)
            metrics = {k: v / denom for k, v in metrics_accum.items()}
            step_time = time.time() - step_start
            metrics["timing/step_s"] = step_time
            metrics["timing/gen_s"] = gen_time
            metrics["async/param_version"] = float(self.param_version)

            if self._is_distributed:
                for k in list(metrics.keys()):
                    t = torch.tensor(
                        metrics[k], dtype=torch.float64, device=self._device
                    )
                    torch.distributed.all_reduce(t, op=torch.distributed.ReduceOp.AVG)
                    metrics[k] = float(t.item())

            self.last_metrics = metrics
            for k, v in metrics.items():
                self._metrics.update(k, v)

            if self._rank == 0 and (step + 1) % self._sync_every == 0:
                sync_step = step // self._sync_every
                for cb in self.callbacks:
                    cb.on_step_end(self, sync_step, metrics)

            del batch, mini_batches
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        if self._rank == 0:
            for cb in self.callbacks:
                cb.on_train_end(self)

        logger.info(
            "[rank %d] AsyncRLTrainer.train finished: %d local updates, v%d",
            self._rank,
            self.global_step + 1,
            self.param_version,
        )

    def cleanup(self) -> None:
        """Release all resources."""
        self._stop_rollout.set()
        if self._rollout_thread and self._rollout_thread.is_alive():
            self._rollout_thread.join(timeout=10.0)
        if self._atom_engine is not None:
            self._atom_engine.shutdown()
            self._atom_engine = None
        if self._actor_model is not None:
            del self._actor_model
        if self._ref_model is not None:
            del self._ref_model
        self._actor_model = None
        self._ref_model = None
        self._optimizer = None
        if self._ray_cluster is not None:
            self._ray_cluster.shutdown()
            self._ray_cluster = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        logger.info("[rank %d] AsyncRLTrainer.cleanup complete.", self._rank)


__all__ = ["AsyncRLTrainer"]
