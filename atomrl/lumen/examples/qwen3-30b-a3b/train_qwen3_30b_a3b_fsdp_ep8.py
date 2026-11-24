# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

"""Qwen3-30B-A3B MoE Training — FSDP2 (fully_shard) + 2D Parallelism (DP×EP).

2D parallelism layout (world_size = dp_size × ep_size):
  - EP groups: ranks sharing the same DP rank form an EP group (size=ep_size).
    Within an EP group, 128 experts are sharded (128/ep_size per GPU).
    Token routing uses all-to-all within the EP group.
  - DP groups: ranks sharing the same EP rank form a DP group (size=dp_size).
    Each DP group holds the *same* set of experts — they process different data.
    FSDP2 shards shared params across the DP group.
    Expert gradients are all-reduced across the DP group.

Example: DP=2, EP=8, world_size=16
  EP group 0: ranks [0,1,2,3,4,5,6,7]     (node 0)
  EP group 1: ranks [8,9,10,11,12,13,14,15] (node 1)
  DP group 0: ranks [0,8]   DP group 1: ranks [1,9]  ... DP group 7: ranks [7,15]
  ep_rank = global_rank % ep_size     (position within EP group)
  dp_rank = global_rank // ep_size    (position within DP group)
"""

import argparse
import json
import logging
import math
import os
import time
from argparse import Namespace

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from transformers import AutoModelForCausalLM, AutoTokenizer

logger = logging.getLogger(__name__)


def _rank0_print(msg: str) -> None:
    if not dist.is_initialized() or dist.get_rank() == 0:
        logger.info(msg)


# ---------------------------------------------------------------------------
# 2D Process Group Setup: DP × EP
# ---------------------------------------------------------------------------


def create_2d_process_groups(world_size, ep_size):
    """Create EP and DP sub-groups for 2D parallelism.

    Returns (ep_group, dp_group, ep_rank, dp_rank).

    Layout: ranks are arranged in a (dp_size × ep_size) grid.
      global_rank = dp_rank * ep_size + ep_rank

    EP groups (rows):  {dp_rank * ep_size + j | j in 0..ep_size-1}
    DP groups (cols):  {i * ep_size + ep_rank | i in 0..dp_size-1}
    """
    global_rank = dist.get_rank()
    dp_size = world_size // ep_size
    ep_rank = global_rank % ep_size
    dp_rank = global_rank // ep_size

    # Create EP groups (one per DP rank)
    ep_group = None
    for d in range(dp_size):
        ranks = list(range(d * ep_size, (d + 1) * ep_size))
        g = dist.new_group(ranks)
        if global_rank in ranks:
            ep_group = g

    # Create DP groups (one per EP rank)
    dp_group = None
    for e in range(ep_size):
        ranks = [i * ep_size + e for i in range(dp_size)]
        g = dist.new_group(ranks)
        if global_rank in ranks:
            dp_group = g

    _rank0_print(f"> 2D groups created: DP={dp_size} × EP={ep_size}")
    _rank0_print(
        f"  EP groups (rows): {[list(range(d*ep_size,(d+1)*ep_size)) for d in range(dp_size)]}"
    )
    _rank0_print(
        f"  DP groups (cols): {[[i*ep_size+e for i in range(dp_size)] for e in range(ep_size)]}"
    )

    return ep_group, dp_group, ep_rank, dp_rank


# ---------------------------------------------------------------------------
# Expert Parallelism: shard MoE experts across EP group
# ---------------------------------------------------------------------------


class EPShardedMoeBlock(nn.Module):
    """Wraps Qwen3MoeSparseMoeBlock with expert-parallel sharding.

    Each GPU keeps only its local experts (128 / ep_size).
    Forward: route tokens via all-to-all within ep_group, compute local experts,
    all-to-all back.
    """

    def __init__(self, original_moe_block, ep_rank, ep_size, ep_group):
        super().__init__()
        self.num_experts = original_moe_block.num_experts
        self.top_k = original_moe_block.top_k
        self.norm_topk_prob = original_moe_block.norm_topk_prob
        self.gate = original_moe_block.gate
        self.ep_size = ep_size
        self.ep_rank = ep_rank
        self.ep_group = ep_group

        experts_per_gpu = self.num_experts // ep_size
        start = ep_rank * experts_per_gpu
        end = start + experts_per_gpu
        self.local_expert_start = start
        self.local_expert_end = end
        self.experts_per_gpu = experts_per_gpu

        self.local_experts = nn.ModuleList(
            [original_moe_block.experts[i] for i in range(start, end)]
        )
        del original_moe_block.experts

    def forward(self, hidden_states: torch.Tensor):
        batch_size, seq_len, hidden_dim = hidden_states.shape
        hidden_flat = hidden_states.view(-1, hidden_dim)
        num_tokens = hidden_flat.shape[0]

        # --- Routing ---
        router_logits = self.gate(hidden_flat)
        routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
        topk_weights, topk_indices = torch.topk(routing_weights, self.top_k, dim=-1)
        if self.norm_topk_prob:
            topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
        topk_weights = topk_weights.to(hidden_flat.dtype)

        # --- Map expert indices to target GPU within EP group ---
        target_gpus = topk_indices // self.experts_per_gpu
        local_expert_ids = topk_indices % self.experts_per_gpu

        # --- Count tokens per (source_gpu -> dest_gpu) within EP group ---
        send_counts = torch.zeros(
            self.ep_size, dtype=torch.long, device=hidden_flat.device
        )
        for dest in range(self.ep_size):
            send_counts[dest] = (target_gpus == dest).sum()

        recv_counts = torch.zeros(
            self.ep_size, dtype=torch.long, device=hidden_flat.device
        )
        dist.all_to_all_single(recv_counts, send_counts, group=self.ep_group)

        # --- Prepare send buffers: sort tokens by destination GPU ---
        send_indices = []
        send_expert_ids = []
        send_weights_list = []
        for dest in range(self.ep_size):
            mask = target_gpus == dest
            token_idx, slot_idx = torch.where(mask)
            send_indices.append(token_idx)
            send_expert_ids.append(local_expert_ids[token_idx, slot_idx])
            send_weights_list.append(topk_weights[token_idx, slot_idx])

        send_indices = torch.cat(send_indices)
        send_expert_ids_cat = torch.cat(send_expert_ids)
        send_weights_cat = torch.cat(send_weights_list)
        send_hidden = hidden_flat[send_indices]

        # --- All-to-all within EP group: exchange hidden states ---
        send_counts_list = send_counts.tolist()
        recv_counts_list = recv_counts.tolist()
        total_recv = sum(recv_counts_list)

        recv_hidden = torch.empty(
            total_recv, hidden_dim, dtype=hidden_flat.dtype, device=hidden_flat.device
        )
        send_splits = [c * hidden_dim for c in send_counts_list]
        recv_splits = [c * hidden_dim for c in recv_counts_list]
        send_list = list(send_hidden.contiguous().view(-1).split(send_splits))
        recv_list = list(recv_hidden.view(-1).split(recv_splits))
        dist.all_to_all(recv_list, send_list, group=self.ep_group)
        recv_hidden = torch.cat(recv_list).view(-1, hidden_dim)

        # All-to-all for expert IDs
        recv_expert_ids = torch.empty(
            total_recv, dtype=send_expert_ids_cat.dtype, device=hidden_flat.device
        )
        send_eid_list = list(send_expert_ids_cat.split(send_counts_list))
        recv_eid_list = list(recv_expert_ids.split(recv_counts_list))
        dist.all_to_all(recv_eid_list, send_eid_list, group=self.ep_group)
        recv_expert_ids = torch.cat(recv_eid_list)

        # All-to-all for weights
        recv_weights = torch.empty(
            total_recv, dtype=send_weights_cat.dtype, device=hidden_flat.device
        )
        send_w_list = list(send_weights_cat.split(send_counts_list))
        recv_w_list = list(recv_weights.split(recv_counts_list))
        dist.all_to_all(recv_w_list, send_w_list, group=self.ep_group)
        recv_weights = torch.cat(recv_w_list)

        # --- Compute local experts ---
        output_hidden = torch.zeros_like(recv_hidden)
        for local_idx in range(self.experts_per_gpu):
            mask = recv_expert_ids == local_idx
            if mask.any():
                expert_input = recv_hidden[mask]
                expert_output = self.local_experts[local_idx](expert_input)
                output_hidden[mask] = expert_output * recv_weights[mask].unsqueeze(-1)

        # --- All-to-all back within EP group ---
        return_hidden = torch.empty(
            send_hidden.shape[0],
            hidden_dim,
            dtype=hidden_flat.dtype,
            device=hidden_flat.device,
        )
        send_back_splits = [c * hidden_dim for c in recv_counts_list]
        recv_back_splits = [c * hidden_dim for c in send_counts_list]
        send_back_list = list(
            output_hidden.contiguous().view(-1).split(send_back_splits)
        )
        recv_back_list = list(return_hidden.view(-1).split(recv_back_splits))
        dist.all_to_all(recv_back_list, send_back_list, group=self.ep_group)
        return_hidden = torch.cat(recv_back_list).view(-1, hidden_dim)

        # --- Scatter-add results back to original token positions ---
        final_output = torch.zeros(
            num_tokens, hidden_dim, dtype=hidden_flat.dtype, device=hidden_flat.device
        )
        final_output.index_add_(0, send_indices, return_hidden)

        return final_output.view(batch_size, seq_len, hidden_dim), router_logits


def shard_experts_ep(model, ep_size, ep_rank, ep_group):
    """Replace every Qwen3MoeSparseMoeBlock with EPShardedMoeBlock."""
    count = 0
    for layer in model.model.layers:
        if hasattr(layer, "mlp") and hasattr(layer.mlp, "experts"):
            layer.mlp = EPShardedMoeBlock(layer.mlp, ep_rank, ep_size, ep_group)
            count += 1
    _rank0_print(
        f"> EP sharding: replaced {count} MoE blocks, "
        f"{128 // ep_size} local experts/layer on ep_rank {ep_rank}"
    )
    return model


# ---------------------------------------------------------------------------
# FSDP2: shard shared params across DP group via fully_shard
# ---------------------------------------------------------------------------


def apply_fsdp2_dp(model, args, dp_group, dp_size):
    """Apply FSDP2 fully_shard scoped to the DP group.

    With DP=2 EP=8, FSDP2 shards the shared layers (attention, norms,
    embeddings) across the 2 GPUs in each DP group. Expert params are NOT
    sharded by FSDP — they're already partitioned by EP.

    The DeviceMesh spans only the DP group (size=dp_size), so all-gather and
    reduce-scatter happen only between DP peers (same EP rank, different data).
    """
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard

    dist.get_process_group_ranks(dp_group)
    mesh = init_device_mesh("cuda", (dp_size,), mesh_dim_names=("dp",))

    mp_policy = MixedPrecisionPolicy(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.float32,
    )

    reshard = args.sharding != "shard_grad_op"

    n_layers = 0
    for module in model.modules():
        if hasattr(module, "layers") and isinstance(module.layers, nn.ModuleList):
            for layer in module.layers:
                fully_shard(
                    layer,
                    mesh=mesh,
                    mp_policy=mp_policy,
                    reshard_after_forward=reshard,
                )
                n_layers += 1
            break

    fully_shard(
        model,
        mesh=mesh,
        mp_policy=mp_policy,
        reshard_after_forward=reshard,
    )

    _rank0_print(
        f"> FSDP2 applied on DP group (size={dp_size}), "
        f"{n_layers} layers, reshard={reshard}"
    )
    return model


# ---------------------------------------------------------------------------
# Expert gradient sync across DP replicas
# ---------------------------------------------------------------------------


def sync_expert_grads(model, dp_group):
    """All-reduce expert gradients across the DP group.

    FSDP2 handles shared params (attention, norms, embeddings) automatically.
    But expert params are EP-partitioned and NOT wrapped by FSDP — the same
    set of experts on different DP ranks (processing different data) must have
    their gradients averaged.
    """
    expert_params = [
        p
        for n, p in model.named_parameters()
        if "local_experts" in n and p.requires_grad and p.grad is not None
    ]
    if not expert_params:
        return
    for p in expert_params:
        dist.all_reduce(p.grad, op=dist.ReduceOp.AVG, group=dp_group)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    p = argparse.ArgumentParser(
        description="Qwen3-30B-A3B MoE Training — FSDP2 + 2D Parallelism (DP×EP)"
    )
    p.add_argument("--model-name-or-path", required=True)
    p.add_argument("--tokenizer-name-or-path", default=None)
    p.add_argument("--mode", choices=["bf16", "fp8_blockwise2d"], default="bf16")
    p.add_argument("--train-data-path", required=True)
    p.add_argument("--val-data-path", default=None)
    p.add_argument("--seq-length", type=int, default=2048)
    p.add_argument("--micro-batch-size", type=int, default=1)
    p.add_argument("--gradient-accumulation-steps", type=int, default=1)
    p.add_argument("--max-steps", type=int, default=100)
    p.add_argument("--lr", type=float, default=1e-6)
    p.add_argument("--min-lr", type=float, default=0.0)
    p.add_argument("--lr-warmup-steps", type=int, default=0)
    p.add_argument(
        "--lr-decay-style", choices=["cosine", "constant"], default="constant"
    )
    p.add_argument("--weight-decay", type=float, default=0.1)
    p.add_argument("--max-grad-norm", type=float, default=1.0)
    p.add_argument("--adam-beta1", type=float, default=0.9)
    p.add_argument("--adam-beta2", type=float, default=0.98)
    p.add_argument("--adam-eps", type=float, default=1e-5)
    p.add_argument(
        "--ep-size",
        type=int,
        default=8,
        help="Expert parallelism size (must divide num_experts=128)",
    )
    p.add_argument(
        "--dp-size",
        type=int,
        default=2,
        help="Data parallelism size (world_size must equal dp_size * ep_size)",
    )
    p.add_argument(
        "--sharding",
        choices=["full_shard", "shard_grad_op"],
        default="full_shard",
        help="FSDP2 sharding: full_shard (reshard after fwd) or "
        "shard_grad_op (ZeRO-2, keep params resident)",
    )
    p.add_argument("--grad-checkpointing", action="store_true", default=True)
    p.add_argument(
        "--no-grad-checkpointing", dest="grad_checkpointing", action="store_false"
    )
    p.add_argument(
        "--fp8-scaling",
        choices=["blockwise2d", "delayed", "dynamic"],
        default="blockwise2d",
    )
    p.add_argument(
        "--fsdp-fp8-param-storage",
        action="store_true",
        help="Store weights as FP8 in FSDP2 shard " "(all-gather FP8 instead of BF16)",
    )
    p.add_argument(
        "--aiter-attn", action="store_true", help="Route SDPA via AITER CK FMHA"
    )
    p.add_argument(
        "--lumen-norm",
        action="store_true",
        help="Replace Qwen3MoeRMSNorm with Lumen fused RMSNorm",
    )
    p.add_argument(
        "--fuse-rope",
        action="store_true",
        help="Replace HF rotary with AITER autograd RoPE",
    )
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--log-interval", type=int, default=1)
    p.add_argument("--eval-interval", type=int, default=50)
    p.add_argument("--val-samples", type=int, default=200)
    p.add_argument("--seed", type=int, default=1234)
    args = p.parse_args()

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    global_rank = int(os.environ.get("RANK", "0"))

    logging.basicConfig(
        level=logging.INFO if global_rank == 0 else logging.WARNING,
        format="%(levelname)s:%(name)s:%(message)s",
    )

    if not dist.is_initialized():
        dist.init_process_group("nccl")
    torch.cuda.set_device(local_rank)
    torch.manual_seed(args.seed)

    assert (
        args.ep_size <= world_size
    ), f"ep_size={args.ep_size} > world_size={world_size}"
    assert (
        128 % args.ep_size == 0
    ), f"num_experts=128 must be divisible by ep_size={args.ep_size}"
    assert (
        world_size == args.dp_size * args.ep_size
    ), f"world_size={world_size} must equal dp_size×ep_size={args.dp_size}×{args.ep_size}"

    _rank0_print(f"> Loading Qwen3-30B-A3B from {args.model_name_or_path} ...")
    _rank0_print(
        f"> Strategy: DP={args.dp_size} × EP={args.ep_size} "
        f"(world_size={world_size}), FSDP2 sharding={args.sharding}"
    )

    # ---- Create 2D process groups ----
    ep_group, dp_group, ep_rank, dp_rank = create_2d_process_groups(
        world_size, args.ep_size
    )

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
    )

    # ---- Fused ops (before EP sharding, operates on HF module names) ----
    if args.fuse_rope:
        import transformers.models.qwen3_moe.modeling_qwen3_moe as _q3m
        from atomrl.lumen.ops.rope import apply_rotary_qk_autograd

        def _lumen_rope(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
            return apply_rotary_qk_autograd(q, k, cos, sin)

        _q3m.apply_rotary_pos_emb = _lumen_rope
        _rank0_print("> Fused RoPE: HF apply_rotary_pos_emb -> AITER autograd RoPE")

    # ---- EP sharding: distribute experts across EP group ----
    model = shard_experts_ep(model, args.ep_size, ep_rank, ep_group)

    if args.grad_checkpointing:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        _rank0_print("> Gradient checkpointing enabled")

    # ---- Precision mode: FP8 (optional) or BF16 (default) ----
    use_fp8 = args.mode == "fp8_blockwise2d"
    if use_fp8:
        from atomrl.lumen.config import LumenConfig

        cfg = LumenConfig.from_args(
            Namespace(
                linear_fp8=True,
                linear_fp8_format="fp8_e4m3",
                linear_fp8_scaling=args.fp8_scaling,
                linear_fp8_block_size=128,
                linear_fp8_amax_algo="max",
                linear_fp8_amax_history=16,
                linear_fp8_reduce_amax=False,
                linear_fp8_activation=True,
                linear_fp8_wgrad=True,
                linear_fp8_cache_frozen_weight=False,
                linear_fp8_bpreshuffle=False,
                grad_quant_type=None,
                first_last_layers_bf16=False,
                lumen_norm=args.lumen_norm,
                hf_attn_patch=args.aiter_attn,
                lora_rank=0,
                lora_alpha=0,
                lora_dropout=0.0,
            )
        )
        _manager, model = cfg.enable(model)
        _rank0_print("> Lumen FP8 blockwise2d enabled (full-param)")
    else:
        _rank0_print("> BF16 full-param training")

    # ---- FSDP2: shard shared params across DP group via fully_shard ----
    model = apply_fsdp2_dp(model, args, dp_group, args.dp_size)

    param_count = sum(p.numel() for p in model.parameters())
    trainable_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    _rank0_print(
        f"> Params on rank {global_rank}: {param_count/1e9:.2f}B total, "
        f"{trainable_count/1e6:.1f}M trainable "
        f"({100*trainable_count/param_count:.2f}%)"
    )

    # ---- Optimizer ----
    opt = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        betas=(args.adam_beta1, args.adam_beta2),
        eps=args.adam_eps,
        weight_decay=args.weight_decay,
    )

    def lr_lambda(step):
        w, T = args.lr_warmup_steps, args.max_steps
        mx, mn = args.lr, args.min_lr
        if step < w:
            return float(step) / max(w, 1)
        if args.lr_decay_style == "constant":
            return 1.0
        prog = float(step - w) / max(T - w, 1)
        ratio = mn / mx if mx > 0 else 0.0
        return ratio + (1 - ratio) * 0.5 * (1 + math.cos(math.pi * prog))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)

    # ---- Data (DP-aware: each DP rank gets different data) ----
    tok = AutoTokenizer.from_pretrained(
        args.tokenizer_name_or_path or args.model_name_or_path
    )

    def make_loader(path, n, is_train=True):
        ds = AlpacaDataset(path, tok, args.seq_length, num_samples=n)
        _rank0_print(f"Loaded {len(ds)} samples from {path}")
        sampler = DistributedSampler(
            ds,
            num_replicas=args.dp_size,
            rank=dp_rank,
            shuffle=is_train,
            seed=args.seed,
        )
        return DataLoader(
            ds,
            batch_size=args.micro_batch_size,
            sampler=sampler,
            num_workers=args.num_workers,
            pin_memory=True,
            drop_last=True,
        )

    train_loader = make_loader(args.train_data_path, None, is_train=True)
    val_loader = (
        make_loader(args.val_data_path, args.val_samples, is_train=False)
        if args.val_data_path
        else None
    )

    def loss_fn(batch):
        ids = batch["input_ids"][:, :-1].to(local_rank)
        labels = batch["input_ids"][:, 1:].to(local_rank)
        lm = batch["loss_mask"][:, 1:].to(local_rank).float()
        logits = model(input_ids=ids).logits
        per_token = F.cross_entropy(
            logits.view(-1, logits.size(-1)), labels.reshape(-1), reduction="none"
        )
        return (per_token * lm.reshape(-1)).sum() / lm.sum().clamp(min=1)

    @torch.no_grad()
    def validate():
        model.eval()
        total_loss, num_batches = 0.0, 0
        for batch in val_loader:
            total_loss += loss_fn(batch).item()
            num_batches += 1
            if num_batches >= 10:
                break
        model.train()
        avg = total_loss / max(num_batches, 1)
        if dist.is_initialized():
            t = torch.tensor([avg], device="cuda")
            dist.all_reduce(t, op=dist.ReduceOp.AVG)
            avg = t.item()
        return avg

    # ---- Training loop ----
    model.train()
    ga = args.gradient_accumulation_steps
    it = iter(train_loader)
    _rank0_print(
        f"> Starting training: {args.max_steps} steps, "
        f"micro_bs={args.micro_batch_size}, ga={ga}, "
        f"seq_len={args.seq_length}, mode={args.mode}, "
        f"DP={args.dp_size} × EP={args.ep_size}, "
        f"FSDP2 sharding={args.sharding}"
    )

    for step in range(1, args.max_steps + 1):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        opt.zero_grad()
        acc_loss = 0.0
        for _ in range(ga):
            try:
                batch = next(it)
            except StopIteration:
                it = iter(train_loader)
                batch = next(it)
            loss = loss_fn(batch)
            (loss / ga).backward()
            acc_loss += loss.item()

        # FSDP2 handles reduce-scatter for shared params across DP group.
        # Expert params need explicit all-reduce across DP replicas.
        if args.dp_size > 1:
            sync_expert_grads(model, dp_group)

        if args.max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

        opt.step()
        sched.step()
        torch.cuda.synchronize()
        step_ms = (time.perf_counter() - t0) * 1e3

        if step % args.log_interval == 0:
            _rank0_print(
                f"  step {step}/{args.max_steps} | loss {acc_loss/ga:.4f} | "
                f"lr {sched.get_last_lr()[0]:.2e} | step_time_ms {step_ms:.1f}"
            )

        if val_loader and step % args.eval_interval == 0:
            _rank0_print(f"  step {step}/{args.max_steps} | val_loss {validate():.4f}")

    _rank0_print(f"> Training complete after {args.max_steps} steps.")


if __name__ == "__main__":
    main()
