# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

"""Native PyTorch FSDP baseline for Llama2 LoRA SFT (no Lumen quantization path).

Mirrors atomrl.lumen.models.llama2.fsdp.sft.FSDPTrainer's BF16 path EXACTLY (model load,
gradient checkpointing, PEFT LoRA, FSDP1 wrap policy / mixed precision, AdamW,
cosine schedule, loss + eval), but applies PEFT LoRA directly instead of going
through LumenConfig.enable. Only the dataset class is reused (identical inputs).

Purpose: a pure-PyTorch-FSDP accuracy baseline to compare against the Lumen FSDP
BF16 run under identical conditions.
"""

import argparse
import math
import os
from functools import partial

import torch
import torch.distributed as dist

# Reuse ONLY the dataset (framework-agnostic) so inputs are byte-identical to the
# Lumen run. The training/model/FSDP path below is pure PyTorch + HF + PEFT.
from atomrl.lumen.models.llama2.dataset import LLaMA2SFTDataset
from peft import LoraConfig, TaskType, get_peft_model
from torch import nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import MixedPrecision, ShardingStrategy
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from torch.utils.data import DataLoader, DistributedSampler
from transformers import AutoTokenizer, LlamaForCausalLM
from transformers.models.llama.modeling_llama import LlamaDecoderLayer


def rank0(msg):
    if not dist.is_initialized() or dist.get_rank() == 0:
        print(msg, flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model-name-or-path", required=True)
    p.add_argument("--tokenizer-name-or-path", required=True)
    p.add_argument("--train-data-path", required=True)
    p.add_argument("--val-data-path", default=None)
    p.add_argument("--seq-length", type=int, default=8192)
    p.add_argument("--micro-batch-size", type=int, default=1)
    p.add_argument("--gradient-accumulation-steps", type=int, default=1)
    p.add_argument("--max-steps", type=int, default=200)
    p.add_argument("--lr", type=float, default=4e-4)
    p.add_argument("--min-lr", type=float, default=0.0)
    p.add_argument("--lr-warmup-steps", type=int, default=0)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--max-grad-norm", type=float, default=0.3)
    p.add_argument("--lora-rank", type=int, default=16)
    p.add_argument("--lora-alpha", type=float, default=32.0)
    p.add_argument("--lora-dropout", type=float, default=0.1)
    p.add_argument("--train-samples", type=int, default=10000)
    p.add_argument("--val-samples", type=int, default=500)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--eval-interval", type=int, default=10)
    p.add_argument("--seed", type=int, default=1234)
    args = p.parse_args()

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    global_rank = int(os.environ.get("RANK", "0"))
    if not dist.is_initialized():
        dist.init_process_group("nccl")
    torch.cuda.set_device(local_rank)
    torch.manual_seed(args.seed)

    # ---- model (identical to FSDPTrainer.build_model bf16 path) ----
    rank0(f"> Loading {args.model_name_or_path} ...")
    model = LlamaForCausalLM.from_pretrained(
        args.model_name_or_path, torch_dtype=torch.bfloat16, attn_implementation="sdpa"
    )
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )

    # ---- PEFT LoRA directly (no LumenConfig) ----
    peft_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )
    model = get_peft_model(model, peft_cfg)
    for n, prm in model.named_parameters():
        if "lora_" in n and prm.dtype == torch.float32:
            prm.data = prm.data.to(torch.bfloat16)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    rank0(f"> LoRA: trainable {trainable:,}/{total:,} ({100*trainable/total:.2f}%)")

    # ---- FSDP wrap (identical policy/mp/strategy to FSDPTrainer) ----
    model = FSDP(
        model,
        auto_wrap_policy=partial(
            transformer_auto_wrap_policy, transformer_layer_cls={LlamaDecoderLayer}
        ),
        mixed_precision=MixedPrecision(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.bfloat16,
            buffer_dtype=torch.bfloat16,
        ),
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        device_id=local_rank,
        limit_all_gathers=True,
        use_orig_params=True,
    )
    rank0(f"> Native FSDP ready (world_size={world_size})")

    # ---- optimizer + cosine schedule (identical) ----
    opt = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        betas=(0.9, 0.95),
        eps=1e-5,
        weight_decay=args.weight_decay,
    )

    def lr_lambda(step):
        w, T, mx, mn = args.lr_warmup_steps, args.max_steps, args.lr, args.min_lr
        if step < w:
            return float(step) / max(w, 1)
        prog = float(step - w) / max(T - w, 1)
        cos = 0.5 * (1.0 + math.cos(math.pi * prog))
        mr = mn / mx if mx > 0 else 0.0
        return mr + (1.0 - mr) * cos

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)

    # ---- data (reuse LLaMA2SFTDataset, shuffle=True per-row path) ----
    tok = AutoTokenizer.from_pretrained(args.tokenizer_name_or_path)

    def make_loader(path, n):
        ds = LLaMA2SFTDataset(
            num_samples=n,
            data_path=path,
            seq_length=args.seq_length,
            tokenizer=tok,
            is_hf_tokenizer=True,
            shuffle=True,
            seed=args.seed,
        )
        sampler = (
            DistributedSampler(
                ds, num_replicas=world_size, rank=global_rank, shuffle=True
            )
            if world_size > 1
            else None
        )
        return DataLoader(
            ds,
            batch_size=args.micro_batch_size,
            sampler=sampler,
            shuffle=(sampler is None),
            num_workers=args.num_workers,
            pin_memory=True,
            drop_last=True,
        )

    train_loader = make_loader(args.train_data_path, args.train_samples)
    val_loader = (
        make_loader(args.val_data_path, args.val_samples)
        if args.val_data_path
        else None
    )

    def loss_on(batch):
        ids = batch["input_ids"][:, :-1].to(local_rank)
        labels = batch["input_ids"][:, 1:].to(local_rank)
        lm = batch["loss_mask"][:, 1:].to(local_rank).float()
        logits = model(input_ids=ids).logits
        per = nn.functional.cross_entropy(
            logits.view(-1, logits.size(-1)), labels.reshape(-1), reduction="none"
        )
        return (per * lm.reshape(-1)).sum() / lm.sum().clamp(min=1)

    @torch.no_grad()
    def validate():
        model.eval()
        tot, nb = 0.0, 0
        for b in val_loader:
            tot += loss_on(b).item()
            nb += 1
            if nb >= 10:
                break
        model.train()
        avg = tot / max(nb, 1)
        if dist.is_initialized():
            t = torch.tensor([avg], device="cuda")
            dist.all_reduce(t, op=dist.ReduceOp.AVG)
            avg = t.item()
        return avg

    # ---- train loop (identical to FSDPTrainer.train) ----
    model.train()
    ga = args.gradient_accumulation_steps
    it = iter(train_loader)
    for step in range(1, args.max_steps + 1):
        opt.zero_grad()
        acc = 0.0
        for _ in range(ga):
            try:
                b = next(it)
            except StopIteration:
                it = iter(train_loader)
                b = next(it)
            l = loss_on(b)
            (l / ga).backward()
            acc += l.item()
        if args.max_grad_norm > 0:
            model.clip_grad_norm_(args.max_grad_norm)
        opt.step()
        sched.step()
        rank0(
            f"  step {step}/{args.max_steps} | loss {acc/ga:.4f} | lr {sched.get_last_lr()[0]:.2e}"
        )
        if val_loader and step % args.eval_interval == 0:
            rank0(f"  step {step}/{args.max_steps} | val_loss {validate():.4f}")

    rank0("> Native FSDP training complete.")


if __name__ == "__main__":
    main()
