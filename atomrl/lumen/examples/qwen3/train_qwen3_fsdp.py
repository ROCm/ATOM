# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

"""Qwen3-8B training — PyTorch FSDP + Lumen FP8 blockwise2d or MXFP4.

Supports both full-parameter causal-LM pretraining and LoRA SFT:
  - AutoModelForCausalLM + Qwen3DecoderLayer FSDP wrap policy
  - pretraining from a pretrained checkpoint or random weights from its config
  - raw text / ``{"text": ...}`` jsonl pretraining data
  - alpaca-style jsonl SFT data with an answer-only loss mask
"""

import argparse
import json
import logging
import math
import os
import time
from argparse import Namespace
from collections import Counter, defaultdict
from contextlib import nullcontext
from functools import partial

import torch
import torch.distributed as dist
from atomrl.lumen.config import LumenConfig

# Reuse the FSDP trainer's rank0 logger so startup/quant/LoRA lines share the
# `INFO:atomrl.lumen.models.fsdp:` / `INFO:atomrl.lumen.quantize:` style of the llama2 runs.
from atomrl.lumen.models.fsdp import (
    _rank0_print as rank0,
)
from atomrl.lumen.models.fsdp import (
    register_quant_optimizer_hooks,
    validate_fsdp_quant_args,
)
from torch import nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import MixedPrecision, ShardingStrategy
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from transformers.models.qwen3.modeling_qwen3 import Qwen3DecoderLayer


def _init_wandb(args, world_size):
    """Open a rank0 wandb run, or return None when --wandb-project is unset.

    The import lives here so a run without the flag never needs wandb installed.
    """
    if not args.wandb_project:
        return None
    import wandb

    return wandb.init(
        project=args.wandb_project,
        name=args.wandb_run_name,
        config={
            **vars(args),
            "world_size": world_size,
            "global_batch_size": world_size
            * args.micro_batch_size
            * args.gradient_accumulation_steps,
        },
    )


class StepProfiler:
    """Rank-0 torch profiler, enabled only when ``LUMEN_PROF_START`` is set."""

    def __init__(self, enabled):
        self.start = int(os.environ.get("LUMEN_PROF_START") or 0)
        self.enabled = enabled and self.start > 0
        self.end = int(os.environ.get("LUMEN_PROF_END") or (self.start + 3))
        self.out = os.environ.get("LUMEN_PROF_OUTPUT") or "/results/qwen3_profile.txt"
        self.trace = os.environ.get("LUMEN_PROF_TRACE") or ""
        self.copy_trace = os.environ.get("LUMEN_COPY_TRACE", "0") == "1"
        self.shapes = os.environ.get("LUMEN_PROF_SHAPES", "0") == "1"
        self.prof = None
        self._tracing = False
        self._copy_n = Counter()
        self._copy_mb = defaultdict(float)
        self._contig_n = Counter()
        self._contig_mb = defaultdict(float)
        if self.enabled:
            rank0(
                f"> Profiler armed: steps {self.start}-{self.end} -> {self.out}"
                + (" (+copy trace)" if self.copy_trace else "")
            )

    def _short_stack(self):
        import traceback

        rel = []
        for frame in traceback.extract_stack()[:-2]:
            if any(part in frame.filename for part in ("lumen/", "examples/")):
                rel.append(f"{frame.filename}:{frame.lineno} {frame.name}")
        return " <- ".join(rel[-3:]) if rel else "(non-lumen)"

    def _install_copy_patches(self):
        self._orig_copy = torch.Tensor.copy_
        self._orig_contig = torch.Tensor.contiguous
        profiler = self

        def _copy(tensor, src, *args, **kwargs):
            if profiler._tracing:
                key = profiler._short_stack()
                profiler._copy_n[key] += 1
                profiler._copy_mb[key] += (
                    tensor.nelement() * tensor.element_size() / 1e6
                )
            return profiler._orig_copy(tensor, src, *args, **kwargs)

        def _contiguous(tensor, *args, **kwargs):
            if profiler._tracing and not tensor.is_contiguous(*args, **kwargs):
                key = profiler._short_stack()
                profiler._contig_n[key] += 1
                profiler._contig_mb[key] += (
                    tensor.nelement() * tensor.element_size() / 1e6
                )
            return profiler._orig_contig(tensor, *args, **kwargs)

        torch.Tensor.copy_ = _copy
        torch.Tensor.contiguous = _contiguous

    def _dump_copy_trace(self):
        torch.Tensor.copy_ = self._orig_copy
        torch.Tensor.contiguous = self._orig_contig
        nsteps = max(1, self.end - self.start + 1)
        path = self.out.replace(".txt", "_copy_trace.txt")
        with open(path, "w") as output:
            output.write(
                f"copy_/contiguous trace, steps {self.start}-{self.end} "
                f"({nsteps} steps)\n\n"
            )
            for title, counts, megabytes in (
                ("copy_", self._copy_n, self._copy_mb),
                ("contiguous", self._contig_n, self._contig_mb),
            ):
                output.write(f"--- {title} by total MB (top 25) ---\n")
                output.write(f"{'calls/step':>10} {'MB/step':>10}  site\n")
                output.writelines(
                    f"{counts[key] / nsteps:>10.0f} "
                    f"{total_mb / nsteps:>10.1f}  {key}\n"
                    for key, total_mb in sorted(
                        megabytes.items(), key=lambda item: -item[1]
                    )[:25]
                )
                output.write("\n")
        rank0(f"> Profiler wrote copy trace {path}")

    def step_begin(self, step):
        if self.enabled and step == self.start:
            self.prof = torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA,
                ],
                record_shapes=self.shapes,
            )
            self.prof.__enter__()
            if self.copy_trace:
                self._install_copy_patches()
                self._tracing = True

    def step_end(self, step):
        if self.enabled and self.prof is not None and step == self.end:
            self._tracing = False
            self.prof.__exit__(None, None, None)
            table = self.prof.key_averages().table(
                sort_by="self_cuda_time_total", row_limit=40
            )
            with open(self.out, "w") as output:
                output.write(
                    f"Qwen3 FSDP profile, steps {self.start}-{self.end}\n\n{table}"
                )
            rank0(f"> Profiler wrote {self.out}")
            if self.shapes:
                shape_table = self.prof.key_averages(group_by_input_shape=True).table(
                    sort_by="self_cuda_time_total", row_limit=80
                )
                shape_path = self.out.replace(".txt", "_shapes.txt")
                with open(shape_path, "w") as output:
                    output.write(
                        f"Qwen3 FSDP per-shape profile, steps "
                        f"{self.start}-{self.end}\n\n{shape_table}"
                    )
                rank0(f"> Profiler wrote per-shape table {shape_path}")
            if self.trace:
                self.prof.export_chrome_trace(self.trace)
                rank0(f"> Profiler wrote chrome trace {self.trace}")
            if self.copy_trace:
                self._dump_copy_trace()
            self.prof = None


class AlpacaDataset(Dataset):
    """jsonl rows {instruction, input, output} -> Qwen3 chat, answer-only mask."""

    def __init__(self, path, tokenizer, seq_length, num_samples=None):
        self.tok = tokenizer
        self.seq_length = seq_length
        with open(path, encoding="utf-8") as dataset_file:
            self.rows = [json.loads(line) for line in dataset_file if line.strip()]
        self.n = num_samples or len(self.rows)
        self.pad_id = (
            tokenizer.pad_token_id
            if tokenizer.pad_token_id is not None
            else tokenizer.eos_token_id
        )

    def __len__(self):
        return self.n

    def _chat(self, msgs, add_gen):
        # transformers 5.x returns a BatchEncoding when tokenize=True; return_dict
        # gives the flat input_ids list.
        try:
            o = self.tok.apply_chat_template(
                msgs,
                tokenize=True,
                add_generation_prompt=add_gen,
                enable_thinking=False,
                return_dict=True,
            )
        except TypeError:
            o = self.tok.apply_chat_template(
                msgs, tokenize=True, add_generation_prompt=add_gen, return_dict=True
            )
        return list(o["input_ids"])

    def __getitem__(self, idx):
        r = self.rows[idx % len(self.rows)]
        prompt = r["instruction"].strip()
        if r.get("input", "").strip():
            prompt += "\n" + r["input"].strip()
        p_ids = self._chat([{"role": "user", "content": prompt}], add_gen=True)
        f_ids = self._chat(
            [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": r["output"]},
            ],
            add_gen=False,
        )
        mask = [0] * len(p_ids) + [1] * max(0, len(f_ids) - len(p_ids))
        ids = list(f_ids)
        L = self.seq_length + 1
        ids, mask = ids[:L], mask[:L]
        if len(ids) < L:
            pad = L - len(ids)
            ids += [self.pad_id] * pad
            mask += [0] * pad
        return {"input_ids": torch.LongTensor(ids), "loss_mask": torch.LongTensor(mask)}


def build_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--model-name-or-path", required=True)
    p.add_argument("--tokenizer-name-or-path", default=None)
    p.add_argument("--task", choices=["sft", "pretrain"], default="sft")
    p.add_argument(
        "--init-from-scratch",
        action="store_true",
        help="Build random weights from --model-name-or-path's config (pretrain only).",
    )
    p.add_argument(
        "--mode",
        choices=["bf16", "fp8_blockwise2d", "mxfp4"],
        default="fp8_blockwise2d",
    )
    p.add_argument("--train-data-path", required=True)
    p.add_argument("--val-data-path", default=None)
    p.add_argument("--seq-length", type=int, default=2048)
    p.add_argument("--micro-batch-size", type=int, default=1)
    p.add_argument("--gradient-accumulation-steps", type=int, default=1)
    p.add_argument("--max-steps", type=int, default=200)
    p.add_argument("--lr", type=float, default=4e-4)
    p.add_argument("--min-lr", type=float, default=0.0)
    p.add_argument("--lr-warmup-steps", type=int, default=0)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--max-grad-norm", type=float, default=0.3)
    p.add_argument(
        "--lora-rank",
        type=int,
        default=None,
        help="LoRA rank. Defaults to 16 for SFT and 0 for full pretraining.",
    )
    p.add_argument("--lora-alpha", type=float, default=32.0)
    p.add_argument("--lora-dropout", type=float, default=0.1)
    p.add_argument(
        "--cache-frozen-weight",
        action="store_true",
        help="cache the frozen base weight's FP8 quant (skip per-fwd re-quant)",
    )
    p.add_argument(
        "--bpreshuffle",
        action="store_true",
        help="use the ~2.5x-faster B-preshuffle blockscale GEMM (needs --cache-frozen-weight)",
    )
    p.add_argument(
        "--sharding",
        choices=["full_shard", "shard_grad_op"],
        default="full_shard",
        help="FSDP sharding: shard_grad_op (ZeRO-2) avoids per-step param all-gather",
    )
    p.add_argument(
        "--fp8-scaling",
        choices=["blockwise2d", "delayed", "dynamic"],
        default="blockwise2d",
        help="FP8 linear scaling: blockwise2d (128x128, accurate) vs delayed/dynamic (per-tensor, faster GEMM)",
    )
    p.add_argument(
        "--aiter-attn",
        action="store_true",
        help="route SDPA attention through AITER (CK FMHA) instead of PyTorch AOTriton (hf_attn_patch)",
    )
    p.add_argument(
        "--lumen-norm",
        action="store_true",
        help="replace HF Qwen3RMSNorm with Lumen fused RMSNorm (AITER)",
    )
    p.add_argument(
        "--fuse-rope",
        action="store_true",
        help="replace HF apply_rotary_pos_emb with AITER autograd RoPE (fwd+bwd)",
    )
    p.add_argument(
        "--no-grad-checkpointing",
        dest="grad_checkpointing",
        action="store_false",
        help="disable activation checkpointing (no backward forward-recompute; more memory)",
    )
    p.add_argument(
        "--no-limit-all-gathers",
        dest="limit_all_gathers",
        action="store_false",
        help="allow FSDP to overlap consecutive all-gathers with compute (more memory)",
    )
    p.add_argument(
        "--forward-prefetch",
        action="store_true",
        help="FSDP forward_prefetch: prefetch next unit's all-gather during compute",
    )
    p.add_argument(
        "--fsdp-version",
        type=int,
        choices=[1, 2],
        default=1,
        help="FSDP version: 1 (FullyShardedDataParallel) or 2 (fully_shard)",
    )
    p.add_argument(
        "--fsdp-fp8-param-storage",
        action="store_true",
        help="FSDP2 only: store the frozen blockwise2d base weight as FP8 and "
        "all-gather it as FP8 (no per-step re-quant)",
    )
    p.add_argument(
        "--fsdp-mxfp4-comm",
        action="store_true",
        help="FSDP2 only: all-gather MXFP4-patched weights as packed FP4 "
        "(requires --mode mxfp4 --fsdp-version 2)",
    )
    p.add_argument(
        "--first-last-layers-bf16",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Keep configured edge transformer layers in BF16. Defaults on for MXFP4 pretraining.",
    )
    p.add_argument("--num-layers-at-start-in-bf16", type=int, default=0)
    p.add_argument("--num-layers-at-end-in-bf16", type=int, default=5)
    p.set_defaults(grad_checkpointing=True, limit_all_gathers=True)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--log-interval", type=int, default=1)
    p.add_argument("--eval-interval", type=int, default=50)
    p.add_argument("--val-samples", type=int, default=200)
    p.add_argument(
        "--eval-batches",
        type=int,
        default=10,
        help="Micro-batches per rank to average the validation loss over. "
        "Raise it when the val loss is the measurement, not a progress readout.",
    )
    p.add_argument(
        "--train-samples",
        type=int,
        default=0,
        help="Cap training samples (0 = use the whole corpus). Pretraining only; "
        "keeps startup tokenization proportional to the requested step count.",
    )
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument(
        "--wandb-project",
        default=None,
        help="Log the step metrics to this Weights & Biases project. "
        "Unset means wandb is never imported.",
    )
    p.add_argument(
        "--wandb-run-name",
        default=None,
        help="Run name within --wandb-project. Defaults to wandb's own.",
    )
    return p


def parse_args(argv=None):
    args = build_parser().parse_args(argv)
    if args.lora_rank is None:
        args.lora_rank = 0 if args.task == "pretrain" else 16
    if args.task == "pretrain" and args.lora_rank != 0:
        raise ValueError(
            "--task pretrain requires --lora-rank 0 (full-parameter training)"
        )
    if args.task != "pretrain" and args.init_from_scratch:
        raise ValueError("--init-from-scratch is only valid with --task pretrain")
    if args.task != "pretrain" and args.train_samples:
        raise ValueError("--train-samples is only valid with --task pretrain")
    if args.train_samples < 0:
        raise ValueError("--train-samples must be >= 0")
    if args.eval_batches < 1:
        raise ValueError("--eval-batches must be >= 1")
    if args.first_last_layers_bf16 is None:
        args.first_last_layers_bf16 = args.task == "pretrain" and args.mode == "mxfp4"
    args.linear_fp8 = args.mode == "fp8_blockwise2d"
    args.linear_fp4 = args.mode == "mxfp4"
    validate_fsdp_quant_args(args)
    return args


def _set_fsdp2_gradient_sync(model, enabled):
    """Enable the final accumulation sync and suppress earlier reduce-scatters."""
    setter = getattr(model, "set_requires_gradient_sync", None)
    if setter is None:
        raise RuntimeError("FSDP2 model does not expose set_requires_gradient_sync")
    setter(enabled, recurse=True)


def _pretrain_loss(model, batch, device):
    """Compute next-token loss for an already shifted pretraining batch."""
    input_ids = batch["input_ids"].to(device, non_blocking=True)
    labels = batch["labels"].to(device, non_blocking=True)
    # Match HuggingFace's causal-LM loss: log-softmax and its backward need
    # FP32 even when model activations are BF16.
    logits = model(input_ids=input_ids).logits.float()
    return nn.functional.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        labels.reshape(-1),
    )


def _validation_batch_count(local_batches, requested, device):
    """Return one batch count all ranks can execute without FSDP deadlock."""
    count = torch.tensor(
        min(local_batches, requested), device=device, dtype=torch.int64
    )
    if dist.is_initialized():
        dist.all_reduce(count, op=dist.ReduceOp.MIN)
    count = int(count.item())
    if count == 0:
        raise ValueError(
            "Validation loader has no full micro-batch on at least one rank; "
            "increase --val-samples or reduce --micro-batch-size"
        )
    return count


def main():
    args = parse_args()

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    global_rank = int(os.environ.get("RANK", "0"))
    # Surface atomrl.lumen.* INFO logs (quant enable, LoRA) on rank0 only, like FSDPTrainer.
    logging.basicConfig(
        level=logging.INFO if global_rank == 0 else logging.WARNING,
        format="%(levelname)s:%(name)s:%(message)s",
    )
    if not dist.is_initialized():
        dist.init_process_group("nccl")
    torch.cuda.set_device(local_rank)
    torch.manual_seed(args.seed)

    if args.init_from_scratch:
        rank0(f"> Building Qwen3 from config {args.model_name_or_path} ...")
        model_cfg = AutoConfig.from_pretrained(args.model_name_or_path)
        model_cfg.torch_dtype = torch.bfloat16
        model = AutoModelForCausalLM.from_config(
            model_cfg, torch_dtype=torch.bfloat16, attn_implementation="sdpa"
        )
    else:
        rank0(f"> Loading Qwen3 from {args.model_name_or_path} ...")
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name_or_path,
            torch_dtype=torch.bfloat16,
            attn_implementation="sdpa",
        )

    if args.fuse_rope:
        # Replace HF NEOX rope (mul+rotate_half+mul+add) with AITER autograd RoPE.
        import transformers.models.qwen3.modeling_qwen3 as _q3
        from atomrl.lumen.ops.rope import apply_rotary_qk_autograd

        def _lumen_rope(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
            return apply_rotary_qk_autograd(q, k, cos, sin)

        _q3.apply_rotary_pos_emb = _lumen_rope
        rank0("> Fused RoPE: HF apply_rotary_pos_emb -> AITER autograd RoPE")
    if args.grad_checkpointing:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        rank0("> Gradient checkpointing enabled")
    else:
        rank0(
            "> Gradient checkpointing DISABLED (more activation memory, no backward recompute)"
        )

    # ---- Lumen full-parameter/LoRA training (+ optional quantised linears) ----
    fmt, scaling, blk = "fp8_e4m3", args.fp8_scaling, 128
    use_fp8, use_fp4 = args.linear_fp8, args.linear_fp4
    cfg = LumenConfig.from_args(
        Namespace(
            linear_fp8=use_fp8,
            linear_fp4=use_fp4,
            linear_fp8_format=fmt,
            linear_fp8_scaling=scaling,
            linear_fp8_block_size=blk,
            linear_fp8_amax_algo="max",
            linear_fp8_amax_history=16,
            linear_fp8_reduce_amax=False,
            linear_fp8_activation=True,
            linear_fp8_wgrad=True,
            linear_fp8_cache_frozen_weight=args.cache_frozen_weight,
            linear_fp8_bpreshuffle=args.bpreshuffle,
            grad_quant_type=None,
            first_last_layers_bf16=args.first_last_layers_bf16,
            num_layers_at_start_in_bf16=args.num_layers_at_start_in_bf16,
            num_layers_at_end_in_bf16=args.num_layers_at_end_in_bf16,
            num_layers=model.config.num_hidden_layers,
            lumen_norm=args.lumen_norm,
            hf_attn_patch=args.aiter_attn,  # route SDPA -> AITER CK FMHA when set
            lora_rank=args.lora_rank,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
        )
    )
    _manager, model = cfg.enable(
        model
    )  # logs INFO:atomrl.lumen.quantize + INFO:atomrl.lumen.config (LoRA + trainable)
    # FSDP1 flatten needs uniform dtype: cast PEFT LoRA adapters (fp32) to bf16.
    for nme, prm in model.named_parameters():
        if "lora_" in nme and prm.dtype == torch.float32:
            prm.data = prm.data.to(torch.bfloat16)

    # LoRA can prefer SHARD_GRAD_OP because its base weights are frozen. Full
    # pretraining normally needs FULL_SHARD to shard parameters, gradients, and
    # optimizer states.
    if args.fsdp_version == 2:
        # FSDP2 (fully_shard, per-param sharding). Shards each Qwen3 decoder layer
        # via apply_fsdp2's `.layers` detection; with --fsdp-fp8-param-storage the
        # frozen blockwise2d base weights are wrapped as Blockwise2DFP8Param and
        # all-gathered as FP8 (no per-step re-quant). --fsdp-mxfp4-comm wraps
        # MXFP4 linears so param all-gather ships packed FP4.
        from atomrl.lumen.models.fsdp import apply_fsdp2

        apply_fsdp2(
            model,
            Namespace(
                linear_fp8=use_fp8,
                linear_fp4=use_fp4,
                fsdp_version=2,
                sharding_strategy=args.sharding,
                fsdp_fp8_param_storage=args.fsdp_fp8_param_storage,
                fsdp_mxfp4_comm=args.fsdp_mxfp4_comm,
            ),
        )
        rank0(
            f"> FSDP2 model ready (sharding={args.sharding}, "
            f"fp8_param_storage={args.fsdp_fp8_param_storage}, "
            f"mxfp4_comm={args.fsdp_mxfp4_comm}, grad_ckpt={args.grad_checkpointing}, "
            f"world_size={world_size})"
        )
    else:
        _shard = {
            "full_shard": ShardingStrategy.FULL_SHARD,
            "shard_grad_op": ShardingStrategy.SHARD_GRAD_OP,
        }[args.sharding]
        model = FSDP(
            model,
            auto_wrap_policy=partial(
                transformer_auto_wrap_policy, transformer_layer_cls={Qwen3DecoderLayer}
            ),
            mixed_precision=MixedPrecision(
                param_dtype=torch.bfloat16,
                reduce_dtype=torch.float32,
                buffer_dtype=torch.bfloat16,
            ),
            sharding_strategy=_shard,
            device_id=local_rank,
            limit_all_gathers=args.limit_all_gathers,
            forward_prefetch=args.forward_prefetch,
            use_orig_params=True,
        )
        rank0(
            f"> FSDP model ready (sharding={args.sharding}, limit_all_gathers={args.limit_all_gathers}, "
            f"forward_prefetch={args.forward_prefetch}, grad_ckpt={args.grad_checkpointing}, world_size={world_size})"
        )

    opt = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        betas=(0.9, 0.95),
        eps=1e-5,
        weight_decay=args.weight_decay,
    )
    # Must follow optimizer construction: without it the MXFP4 weight cache is
    # never invalidated under FSDP and every quantized layer trains against its
    # step-0 weights.
    register_quant_optimizer_hooks(model, opt, args)

    def lr_lambda(step):
        w, T, mx, mn = args.lr_warmup_steps, args.max_steps, args.lr, args.min_lr
        if step < w:
            return float(step) / max(w, 1)
        prog = float(step - w) / max(T - w, 1)
        return (mn / mx if mx > 0 else 0.0) + (
            1 - (mn / mx if mx > 0 else 0.0)
        ) * 0.5 * (1 + math.cos(math.pi * prog))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)

    tok = AutoTokenizer.from_pretrained(
        args.tokenizer_name_or_path or args.model_name_or_path
    )

    def make_loader(path, n, shuffle=True):
        if args.task == "pretrain":
            from atomrl.lumen.models.llama31.dataset import PretrainTextDataset

            ds = PretrainTextDataset(
                data_path=path,
                seq_length=args.seq_length,
                tokenizer=tok,
                is_hf_tokenizer=True,
                max_samples=(math.ceil(n / world_size) if n is not None else None),
                rank=global_rank,
                world_size=world_size,
            )
            sampler = None
        else:
            ds = AlpacaDataset(path, tok, args.seq_length, num_samples=n)
            sampler = (
                DistributedSampler(
                    ds, num_replicas=world_size, rank=global_rank, shuffle=shuffle
                )
                if world_size > 1
                else None
            )
        rank0(f"Loaded {len(ds)} samples from {path}")
        return DataLoader(
            ds,
            batch_size=args.micro_batch_size,
            sampler=sampler,
            shuffle=(shuffle and sampler is None),
            num_workers=args.num_workers,
            pin_memory=True,
            drop_last=True,
        )

    train_loader = make_loader(args.train_data_path, args.train_samples or None)
    # Validation does not shuffle: comparing two runs only means something when
    # both are scored on the same tokens, and a shuffled loader hands each
    # eval -- and each run -- a different subset of the held-out set.
    val_loader = (
        make_loader(args.val_data_path, args.val_samples, shuffle=False)
        if args.val_data_path
        else None
    )

    def loss_on(b):
        if args.task == "pretrain":
            # PretrainTextDataset already pairs token[t] with token[t+1].
            # HuggingFace's labels= path shifts internally, so use explicit CE
            # to avoid accidentally learning token[t+2].
            return _pretrain_loss(model, b, local_rank)
        ids = b["input_ids"][:, :-1].to(local_rank, non_blocking=True)
        labels = b["input_ids"][:, 1:].to(local_rank, non_blocking=True)
        lm = b["loss_mask"][:, 1:].to(local_rank, non_blocking=True).float()
        logits = model(input_ids=ids).logits.float()
        per = nn.functional.cross_entropy(
            logits.view(-1, logits.size(-1)), labels.reshape(-1), reduction="none"
        )
        return (per * lm.reshape(-1)).sum() / lm.sum().clamp(min=1)

    @torch.no_grad()
    def validate():
        model.eval()
        target_batches = _validation_batch_count(
            len(val_loader), args.eval_batches, local_rank
        )
        tot_l, nb = 0.0, 0
        for b in val_loader:
            tot_l += loss_on(b).item()
            nb += 1
            if nb >= target_batches:
                break
        model.train()
        totals = torch.tensor([tot_l, nb], device=local_rank, dtype=torch.float64)
        if dist.is_initialized():
            dist.all_reduce(totals, op=dist.ReduceOp.SUM)
        return (totals[0] / totals[1]).item()

    model.train()
    ga = args.gradient_accumulation_steps
    it = iter(train_loader)
    profiler = StepProfiler(enabled=(global_rank == 0))
    wandb_run = _init_wandb(args, world_size) if global_rank == 0 else None
    for step in range(1, args.max_steps + 1):
        profiler.step_begin(step)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        opt.zero_grad()
        acc = 0.0
        for micro in range(ga):
            try:
                b = next(it)
            except StopIteration:
                it = iter(train_loader)
                b = next(it)
            final_micro = micro == ga - 1
            if args.fsdp_version == 2 and ga > 1:
                _set_fsdp2_gradient_sync(model, final_micro)
                sync_context = nullcontext()
            elif args.fsdp_version == 1 and not final_micro:
                sync_context = model.no_sync()
            else:
                sync_context = nullcontext()
            with sync_context:
                l = loss_on(b)
                (l / ga).backward()
            acc += l.item()
        grad_norm = None
        if args.max_grad_norm > 0:
            # FSDP1 wraps the model and exposes .clip_grad_norm_; FSDP2 (fully_shard)
            # returns the bare model → clip via the param-list utility (handles DTensor).
            if args.fsdp_version == 2:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), args.max_grad_norm
                )
            else:
                grad_norm = model.clip_grad_norm_(args.max_grad_norm)
        opt.step()
        sched.step()
        torch.cuda.synchronize()
        step_time_ms = (time.perf_counter() - t0) * 1e3
        peak_mem_gib = torch.cuda.max_memory_allocated() / 2**30
        if wandb_run is not None:
            wandb_run.log(
                {
                    "loss": acc / ga,
                    "grad_norm": float(grad_norm) if grad_norm is not None else None,
                    "lr": sched.get_last_lr()[0],
                    "step_time_ms": step_time_ms,
                    "peak_mem_gib": peak_mem_gib,
                },
                step=step,
            )
        if step % args.log_interval == 0:
            # Log the grad norm: it is the cheapest signal that a precision or
            # comm variant is not actually training the same parameters.
            gn = f"{float(grad_norm):.3e}" if grad_norm is not None else "n/a"
            # Peak memory belongs next to step time: the checkpointing and
            # sharding knobs buy speed with memory, so a step time on its own
            # cannot say whether a setting is affordable.
            rank0(
                f"  step {step}/{args.max_steps} | loss {acc/ga:.4f} | grad_norm {gn} "
                f"| lr {sched.get_last_lr()[0]:.2e} | step_time_ms {step_time_ms:.1f} "
                f"| peak_mem_gib {peak_mem_gib:.1f}"
            )
        profiler.step_end(step)
        if val_loader and step % args.eval_interval == 0:
            val_loss = validate()
            rank0(f"  step {step}/{args.max_steps} | val_loss {val_loss:.4f}")
            if wandb_run is not None:
                wandb_run.log({"val_loss": val_loss}, step=step)

    rank0(f"> Training complete after {args.max_steps} steps.")
    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    main()
