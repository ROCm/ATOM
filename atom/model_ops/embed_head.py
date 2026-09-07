# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from aiter.dist.communication_op import tensor_model_parallel_all_gather
from aiter.dist.parallel_state import get_dp_group, get_tp_group
from aiter.jit.utils.torch_guard import torch_compile_guard
from aiter.tuned_gemm import tgemm
from torch import nn

from atom.model_ops.lm_head_argmax import lm_head_argmax_pack
from atom.model_ops.utils import atom_parameter
from atom.plugin import is_plugin_mode
from atom.utils import envs
from atom.utils.decorators import mark_trace
from atom.utils.forward_context import ForwardContext, get_forward_context


@triton.jit
def _masked_embedding_kernel(
    x_ptr,
    weight_ptr,
    out_ptr,
    vocab_start_idx,
    vocab_end_idx,
    stride_w_row,
    stride_out_row,
    N,
    D,
    BLOCK_D: tl.constexpr,
):
    pid_row = tl.program_id(0)
    pid_col = tl.program_id(1)
    if pid_row >= N:
        return

    token_id = tl.load(x_ptr + pid_row)
    in_range = (token_id >= vocab_start_idx) & (token_id < vocab_end_idx)
    local_idx = token_id - vocab_start_idx

    col_start = pid_col * BLOCK_D
    cols = col_start + tl.arange(0, BLOCK_D)
    col_mask = cols < D

    emb = tl.load(
        weight_ptr + local_idx * stride_w_row + cols,
        mask=in_range & col_mask,
        other=0.0,
    )

    tl.store(out_ptr + pid_row * stride_out_row + cols, emb, mask=col_mask)


def _masked_embedding_launcher(
    x: torch.Tensor,
    weight: torch.Tensor,
    vocab_start_idx: int,
    vocab_end_idx: int,
) -> torch.Tensor:
    N = x.numel()
    D = weight.shape[1]
    BLOCK_D = 1024
    out = torch.empty(N, D, dtype=weight.dtype, device=weight.device)
    grid = (N, triton.cdiv(D, BLOCK_D))
    _masked_embedding_kernel[grid](
        x,
        weight,
        out,
        vocab_start_idx,
        vocab_end_idx,
        weight.stride(0),
        out.stride(0),
        N,
        D,
        BLOCK_D=BLOCK_D,
    )
    return out


def _masked_embedding_fake(
    x: torch.Tensor,
    weight: torch.Tensor,
    vocab_start_idx: int,
    vocab_end_idx: int,
) -> torch.Tensor:
    return torch.empty(
        x.numel(),
        weight.shape[1],
        dtype=weight.dtype,
        device=weight.device,
    )


@torch_compile_guard(gen_fake=_masked_embedding_fake)
def masked_embedding(
    x: torch.Tensor,
    weight: torch.Tensor,
    vocab_start_idx: int,
    vocab_end_idx: int,
) -> torch.Tensor:
    return _masked_embedding_launcher(x, weight, vocab_start_idx, vocab_end_idx)


def _replicated_embedding_fake(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    return torch.empty(
        x.numel(),
        weight.shape[1],
        dtype=weight.dtype,
        device=weight.device,
    )


@torch_compile_guard(gen_fake=_replicated_embedding_fake)
def replicated_embedding(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    # Keep the lookup opaque to torch.compile: inductor otherwise fuses the
    # embedding gather into the surrounding graph, which corrupts the MTP draft
    # rollout (acceptance collapses ~69%->45%) — the same reason
    # VocabParallelEmbedding routes through the masked_embedding custom op.
    #
    # Route through the masked kernel with the full-table range [0, num_rows) so
    # out-of-range ids never reach a raw gather. Under async scheduling + MTP
    # spec-decode, input_ids can transiently carry the optimistic placeholder
    # token -1 (an unresolved "assumed-accepted" draft/bonus slot, produced in
    # gpu_model_runner and read back via prepare_next_token_ids_padded's backup
    # before the deferred correction lands) — for BOTH the target and the shared
    # draft embedding. A raw F.embedding(-1) reads the row before the table ->
    # random illegal memory access. The masked load returns a zero vector for any
    # out-of-range id: bit-identical to F.embedding for every valid token, and
    # matching vLLM's VocabParallelEmbedding (which masks the same -1 to 0) so the
    # unverified -1 slots — whose output is discarded/corrected by async
    # spec-decode — see the same value native does. No accuracy change.
    return _masked_embedding_launcher(x, weight, 0, weight.shape[0])


class VocabParallelEmbedding(nn.Module):

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        prefix: str = "",
    ):
        super().__init__()
        self.prefix = prefix
        self.tp_rank = get_tp_group().rank_in_group
        self.tp_size = get_tp_group().world_size
        assert num_embeddings % self.tp_size == 0
        self.num_embeddings = num_embeddings
        self.num_embeddings_per_partition = self.num_embeddings // self.tp_size
        self.vocab_start_idx = self.num_embeddings_per_partition * self.tp_rank
        self.vocab_end_idx = self.vocab_start_idx + self.num_embeddings_per_partition
        self.weight = atom_parameter(
            torch.empty(self.num_embeddings_per_partition, embedding_dim),
        )
        self.weight.weight_loader = self.weight_loader

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        param_data = param.data
        shard_size = param_data.size(0)
        start_idx = self.tp_rank * shard_size
        loaded_weight = loaded_weight.narrow(0, start_idx, shard_size)
        assert param_data.size() == loaded_weight.size()
        param_data.copy_(loaded_weight)

    @mark_trace
    def forward(self, x: torch.Tensor):
        # Torch compile will make logical_and, mask, embedding in a fused triton kernel, but make accuracy issue in MTP.
        if self.tp_size > 1:
            y = masked_embedding(
                x, self.weight, self.vocab_start_idx, self.vocab_end_idx
            )
            y = get_tp_group().all_reduce(y, ca_fp8_quant=False)
        else:
            y = F.embedding(x, self.weight)
        return y
        # if self.tp_size > 1:
        #     mask = torch.logical_and(x >= self.vocab_start_idx, x < self.vocab_end_idx)
        #     # mask = (x >= self.vocab_start_idx) & (x < self.vocab_end_idx)
        #     x = mask * (x - self.vocab_start_idx)
        # y = F.embedding(x, self.weight)
        # if self.tp_size > 1:
        #     y.masked_fill_(~mask.unsqueeze(1), 0)
        #     y = get_tp_group().all_reduce(y, ca_fp8_quant=False)
        # return y


class ReplicatedEmbedding(nn.Module):
    """Full vocab embedding replicated on every TP rank (no sharding).

    Each rank holds the complete ``[num_embeddings, embedding_dim]`` table and
    does a purely local lookup, so the forward needs **no all-reduce** — unlike
    ``VocabParallelEmbedding``, which shards the vocab and must all-reduce the
    masked partial lookups to reconstruct the full vector.

    Trades ``(tp-1)/tp`` of the embedding's memory per rank for one fewer
    collective per embed. Use ONLY where the embedding is independent of any
    sharded ``lm_head`` (e.g. the EAGLE3 draft, whose embed/lm_head are separate
    tensors). Do NOT use for an embedding shared/tied with a TP-sharded lm_head
    or with the target model's sharded embedding.
    """

    def __init__(self, num_embeddings: int, embedding_dim: int):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.weight = atom_parameter(
            torch.empty(num_embeddings, embedding_dim),
        )
        self.weight.weight_loader = self.weight_loader

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        # Full (un-sharded) copy: every rank gets the complete table.
        assert param.data.size() == loaded_weight.size(), (
            f"ReplicatedEmbedding expects the full weight "
            f"{tuple(param.data.size())}, got {tuple(loaded_weight.size())}"
        )
        param.data.copy_(loaded_weight)

    def forward(self, x: torch.Tensor):
        return replicated_embedding(x, self.weight)


class ParallelLMHead(VocabParallelEmbedding):

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        bias: bool = False,
        **kwargs,
    ):
        super().__init__(num_embeddings, embedding_dim)
        if bias:
            self.bias = atom_parameter(
                torch.empty(self.num_embeddings_per_partition),
            )
            self.bias.weight_loader = self.weight_loader
        else:
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor):
        if not is_plugin_mode():
            forward_context: ForwardContext = get_forward_context()
            context = forward_context.context
            attn_metadata = forward_context.attn_metadata
            # context = get_context()
            if context.is_prefill and not context.is_draft:
                last_indices = attn_metadata.cu_seqlens_q[1:] - 1
                x = x[last_indices].contiguous()
            if self._can_use_dp_sharded_head(context):
                return self._dp_sharded_logits(x, envs.ATOM_DP_LM_HEAD_MODE)
        logits = tgemm.mm(x, self.weight, self.bias)
        if self.tp_size > 1:
            use_custom = envs.ATOM_USE_CUSTOM_ALL_GATHER
            logits = tensor_model_parallel_all_gather(logits, use_custom=use_custom)
            # all_logits = (
            #     [torch.empty_like(logits) for _ in range(self.tp_size)]
            #     if self.tp_rank == 0
            #     else None
            # )
            # dist.gather(logits, all_logits, 0)
            # logits = torch.cat(all_logits, -1) if self.tp_rank == 0 else None
        return logits

    def compute_argmax_token(
        self, x: torch.Tensor, *, out: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Greedy argmax token over the (TP-sharded) vocab — returns ``[N]`` token
        ids WITHOUT all-gathering the full ``[N, vocab]`` logits.

        For greedy speculative drafting only the argmax is needed, so each rank
        reduces its own vocab shard to ``(max_val, global_idx)`` and we all-gather
        just those ``[N, 2]`` (tp small) instead of the O(vocab) logits.

        On the TP path the GEMM is unchanged, so this is bitwise-identical to a
        full-logits ``argmax`` -- the values compared are the same bf16 logits
        (fp32-packed exactly), and tie-breaking matches the lowest global index
        (``torch.max`` picks the lowest local index, ``argmax`` over ranks the
        lowest rank == lowest vocab range). The DP path (``ATOM_DP_DRAFT_ARGMAX``,
        via ``_dp_sharded_logits(mode="argmax")``) reshapes the GEMM ([M, V] ->
        [dp*M, V/dp]), which ``tgemm`` may tile differently, so its logits are not
        bitwise-identical to the replicated ones and a near-tie can flip. The pick
        itself is still exact over whatever logits it is given; only the GEMM's
        rounding differs.
        """
        if out is not None:
            assert out.shape == x.shape[:-1], (
                f"argmax out has shape {tuple(out.shape)}, expected "
                f"{tuple(x.shape[:-1])}"
            )
            assert (
                out.dtype == torch.long and out.device == x.device
            ), "argmax out must be an int64 tensor on the input device"
        # Pure-DP draft: shard the vocab across the DP group instead of a
        # replicated full-vocab GEMM. Skip plugin mode -- its caller decides the
        # collective count per chunk (a mismatch would deadlock DP) -- and skip
        # the context lookup entirely when the env is off.
        if (
            envs.ATOM_DP_DRAFT_ARGMAX
            and not is_plugin_mode()
            and self._can_use_dp_sharded_argmax(get_forward_context().context)
        ):
            return self._dp_sharded_logits(x, "argmax", out)
        logits = tgemm.mm(x, self.weight, self.bias)  # [N, vocab/tp]
        if self.tp_size <= 1:
            token = logits.argmax(dim=-1)
            return token if out is None else out.copy_(token)
        # Pack (val, idx) as fp32 — idx < 2^24 is exact — and all-gather only the
        # per-rank reductions ([N, 2]) instead of the full logits.
        packed = lm_head_argmax_pack(logits, self.vocab_start_idx)
        # Custom, like the logits path above: `graph_capture()` arms only that
        # one, and a draft pass records this. The RCCL path is what made the
        # head un-capturable on HIP at TP > 1.
        use_custom = envs.ATOM_USE_CUSTOM_ALL_GATHER
        gathered = get_tp_group().all_gather(packed, dim=0, use_custom=use_custom)
        gathered = gathered.view(self.tp_size, -1, 2)
        winner = gathered[:, :, 0].argmax(dim=0)  # [N] winning rank (ties -> lowest)
        token = gathered[:, :, 1].gather(0, winner.unsqueeze(0)).squeeze(0)  # [N] fp32
        return token.to(torch.long) if out is None else out.copy_(token)

    # ------------------------------------------------------------------
    # Pure-DP sharded LM head (config ② all-gather / ③ all-to-all).
    #
    # Precondition (checked by `_can_use_dp_sharded_head`): the model TP group is size 1
    # (pure DP) and the DP group is size > 1. The lm_head weight is currently
    # replicated on every DP rank, so each rank slices out its own vocab shard
    # `weight[dp_rank*V/dp : (dp_rank+1)*V/dp]` at runtime — no weight-loader
    # change needed for this prototype (the full weight still costs VRAM; a real
    # rollout should shard it at load time).
    # ------------------------------------------------------------------
    def _can_use_dp_sharded_head(self, context) -> bool:
        """Whether this step may run the DP-sharded LM head (pure-DP decode only).

        Every DP rank must reach the SAME verdict from globally-synced state, or
        the fixed-size collective in `_dp_sharded_logits` deadlocks. The strategy
        (all-gather vs all2all) is read from ATOM_DP_LM_HEAD_MODE there.

        `is_prefill` below is per-rank, which is safe only because
        `running_tokens_are_unified` is the DP-reduced form of the same
        question: a prefilling peer drives it False on every rank, so no rank
        can answer True here while another answers False.
        """
        dp_group = get_dp_group()
        # Static: enabled, pure DP, vocab evenly shardable over a >1 DP group.
        if (
            envs.ATOM_DP_LM_HEAD_MODE not in ("allgather", "all2all")
            or self.tp_size != 1
            or dp_group.world_size <= 1
            or self.num_embeddings % dp_group.world_size != 0
        ):
            return False
        # Per-step: all ranks at one height. `is_draft` is its own question,
        # not shorthand for the flag below -- a drafter states uniformity per
        # pass, so a draft arrives with either answer. Prefill is excluded
        # because x is sliced to 1 row/seq, mismatching the pad target.
        if (
            context is None
            or getattr(context, "is_draft", False)
            or getattr(context, "is_prefill", False)
            or not getattr(context, "running_tokens_are_unified", False)
        ):
            return False
        return get_forward_context().dp_metadata is not None

    def _dp_sharded_logits(
        self, x: torch.Tensor, mode: str, out: torch.Tensor | None = None
    ) -> torch.Tensor:
        """This rank's own rows out of a DP-vocab-sharded lm_head.

        Shared front half: pad x to the DP-agreed ``running_tokens``, all-gather
        hidden across DP (fixed-size, identical on all ranks), and project onto
        this rank's vocab slice. ``mode`` then picks the exchange:

        - ``"argmax"``: reduce each shard to a packed ``(max, global_id)`` and
          all-gather only ``[Σrows, 2]``, then pick the winner -> ``[local_rows]``
          int64 ids (drafting; ``out``, if given, receives them).
        - ``"all2all"`` / ``"allgather"``: exchange full-vocab logits ->
          ``[local_rows, vocab]`` (the decode head).
        """
        dp_group = get_dp_group()
        dp_size = dp_group.world_size
        dp_rank = dp_group.rank_in_group
        vshard = self.num_embeddings // dp_size
        use_custom = envs.ATOM_USE_CUSTOM_ALL_GATHER

        local_rows = x.shape[0]
        max_rows = int(get_forward_context().context.running_tokens)
        # running_tokens is the padded height, so local_rows <= max_rows always;
        # a loud tripwire beats a silent DP-wide hang on a size mismatch.
        assert local_rows <= max_rows, (
            f"DP sharded head: local_rows={local_rows} > running_tokens={max_rows}; "
            "hidden height exceeds the DP-uniform gather bucket."
        )
        if local_rows < max_rows:
            x = torch.cat([x, x.new_zeros(max_rows - local_rows, x.shape[1])], dim=0)

        # [max_rows, dim] -> [dp_size * max_rows, dim] (rank-major concat).
        gathered = dp_group.all_gather(x.contiguous(), dim=0, use_custom=use_custom)
        w = self.weight[dp_rank * vshard : (dp_rank + 1) * vshard]  # [V/dp, dim]
        b = (
            None
            if self.bias is None
            else self.bias[dp_rank * vshard : (dp_rank + 1) * vshard]
        )
        logits_shard = tgemm.mm(gathered, w, b)  # [dp_size * max_rows, V/dp]
        start = dp_rank * max_rows

        if mode == "argmax":
            # Reduce each shard to (max, global_id); exchange only [Σrows, 2].
            packed = lm_head_argmax_pack(logits_shard, dp_rank * vshard)
            gathered_packed = dp_group.all_gather(
                packed, dim=0, use_custom=use_custom
            ).view(dp_size, dp_size * max_rows, 2)
            winner = gathered_packed[:, :, 0].argmax(dim=0)  # [Σrows] winning shard
            token = (
                gathered_packed[:, :, 1]
                .gather(0, winner.unsqueeze(0))
                .squeeze(0)
                .to(torch.long)
            )[
                start : start + local_rows
            ]  # keep own rows
            return token if out is None else out.copy_(token)

        if mode == "all2all":
            # Send each destination rank only the rows it owns; receive this
            # rank's rows on every peer's vocab shard.
            logits_shard = logits_shard.contiguous()
            recv = torch.empty_like(logits_shard)
            torch.distributed.all_to_all_single(
                recv.view(-1), logits_shard.view(-1), group=dp_group.device_group
            )
            # recv is source-major [dp_size, max_rows, vshard]; the sampler wants
            # the vocab shards along dim 1. That (dp <-> rows) swap is
            # non-contiguous, so the reshape copies -- slice to the real rows
            # first so the copy only touches local_rows, not the padded tail.
            return (
                recv.view(dp_size, max_rows, vshard)[:, :local_rows, :]
                .permute(1, 0, 2)
                .reshape(local_rows, dp_size * vshard)
            )

        # "allgather": materialise the full vocab on every rank, keep own rows.
        global_logits = dp_group.all_gather(
            logits_shard, dim=1, use_custom=use_custom
        )  # [Σrows, V]
        return global_logits[start : start + local_rows].contiguous()

    # Draft greedy argmax reuses `_dp_sharded_logits(mode="argmax")`: a draft's
    # replicated [N, V] GEMM is weight-read bound at the tiny draft M, so sharding
    # the vocab ([H, V/dp]) and exchanging only the packed [N, 2] argmax pays.
    def _can_use_dp_sharded_argmax(self, context) -> bool:
        """Whether a draft step may run the DP-sharded argmax.

        Total predicate -- returns False, never raises, outside a DP-reduced
        rectangular pure-DP draft. Every DP rank must reach the same verdict, so
        every gated value is DP-agreed: `running_tokens_are_unified` marks a
        rectangular draft, `running_tokens` is bounded past
        ATOM_DP_DRAFT_ARGMAX_MAX_ROWS (the gather outgrows the weight-read win),
        and `dp_metadata is not None` is the proof `running_tokens` was actually
        reduced across DP -- absent under SGLang dp-attention (where aiter's DP
        group is >1 but ATOM's data_parallel_size is 1) and single-GPU. That
        check also gates the raising `get_dp_group()`: dp_metadata is built via
        the DP group, so its presence means the group exists.
        """
        if not envs.ATOM_DP_DRAFT_ARGMAX or is_plugin_mode() or self.tp_size != 1:
            return False
        fc = get_forward_context()
        if (
            context is None
            or fc.dp_metadata is None
            or not getattr(context, "is_draft", False)
            or getattr(context, "is_prefill", False)
            or not getattr(context, "running_tokens_are_unified", False)
            or int(context.running_tokens) > envs.ATOM_DP_DRAFT_ARGMAX_MAX_ROWS
        ):
            return False
        dp_group = get_dp_group()
        return (
            dp_group.world_size > 1 and self.num_embeddings % dp_group.world_size == 0
        )
