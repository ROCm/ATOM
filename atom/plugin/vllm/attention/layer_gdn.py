# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.


import torch
import triton
import triton.language as tl
from einops import rearrange
from atom.model_ops.mamba_ops.causal_conv1d import (
    causal_conv1d_fn,
    causal_conv1d_update,
)

# from atom.model_ops.attentions.gdn_attn import GDNAttentionMetadata
from vllm.forward_context import get_forward_context
from vllm.distributed import get_tensor_model_parallel_world_size
from vllm.v1.attention.backend import AttentionMetadata
from vllm.v1.attention.backends.gdn_attn import GDNAttentionMetadata

# Two prefill kernel options, selected per-instance via use_vk_layout:
#
#   * vk (default for the plugin path): ATOM-vendored verbatim port of
#     vllm.model_executor.layers.fla.ops.chunk. Writes ssm_state as
#     `[N, H, V, K]` per slot — matches the layout vLLM allocates via
#     MambaStateShapeCalculator.gated_delta_net_state_shape and the layout
#     that this file's decode kernels (fused_sigmoid_gating_delta_rule_update,
#     flydsl_gdr_decode) read. Bit-exact agreement with vLLM upstream
#     verified by tests/test_chunk_gated_delta_rule_vk.py.
#
#   * kv: ATOM's original chunk_gated_delta_rule in chunk.py. Writes
#     ssm_state as `[N, H, K, V]` per slot. Used by the non-plugin
#     ATOM-native GDN backend, which pairs it with its kv-layout decode
#     kernel `fused_recurrent_gated_delta_rule`. NOT compatible with the
#     plugin path's decode kernels — included here as a developer-only
#     escape hatch (e.g. for A/B layout experiments). Selecting it on the
#     plugin path will corrupt the prefill→decode ssm_state round-trip.
from atom.model_ops.fla_ops.chunk_vk import chunk_gated_delta_rule_vk
from atom.model_ops.fla_ops.fused_sigmoid_gating import (
    fused_sigmoid_gating_delta_rule_update,
)

from atom.utils import envs

from torch import nn

USE_FLYDSL_GDR = envs.ATOM_USE_FLYDSL_GDR
try:
    from aiter.ops.flydsl.linear_attention_kernels import flydsl_gdr_decode
except ImportError:
    USE_FLYDSL_GDR = False
    print(
        "Failed to import flydsl_gdr_decode. Please make sure you have the latest version of aiter installed."
    )


class ChunkGatedDeltaRule(nn.Module):
    """Prefill kernel wrapper.

    The choice between vk- and kv-layout chunk kernels is resolved at
    construction time so .forward stays branch-free. See the module-level
    import comment for the layout semantics and the constraint that the
    plugin path's decode kernels are vk-only.
    """

    def __init__(self, use_vk_layout: bool = True) -> None:
        super().__init__()
        self.use_vk_layout = use_vk_layout
        self._fla_chunk_gated_delta_rule = chunk_gated_delta_rule_vk

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        initial_state: torch.Tensor,
        output_final_state: bool,
        cu_seqlens: torch.LongTensor | None = None,
        use_qk_l2norm_in_kernel: bool = True,
        o: torch.Tensor | None = None,
        num_decodes: int = 0,
        num_decode_tokens: int = 0,
    ):
        # `num_decodes` / `num_decode_tokens`: GDN mixed-batch hint that
        # `cu_seqlens` is the ORIGINAL non-spec cumulative sequence lengths
        # with a leading decode-only prefix of `num_decodes` sequences /
        # `num_decode_tokens` tokens to skip. Passing them all the way
        # down lets the chunk_vk prologue (`prepare_chunk_indices`,
        # `prepare_chunk_offsets`, `prepare_rebased_cu_seqlens`) reuse
        # its `@tensor_cache` entries across forward calls — the cache
        # key is keyed on `cu_seqlens` identity, not content, so handing
        # in the cache-stable metadata tensor avoids the D2H syncs that
        # `.tolist()` inside the prologue would otherwise trigger.
        return self._fla_chunk_gated_delta_rule(
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            initial_state=initial_state,
            output_final_state=output_final_state,
            cu_seqlens=cu_seqlens,
            use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
            o=o,
            num_decodes=num_decodes,
            num_decode_tokens=num_decode_tokens,
        )


@triton.jit
def fused_gdn_gating_kernel(
    g,
    beta_output,
    A_log,
    a,
    b,
    dt_bias,
    seq_len,
    NUM_HEADS: tl.constexpr,
    stride_a_batch,
    stride_b_batch,
    beta: tl.constexpr,
    threshold: tl.constexpr,
    BLK_HEADS: tl.constexpr,
):
    i_b, i_s, i_d = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    head_off = i_d * BLK_HEADS + tl.arange(0, BLK_HEADS)
    out_off = i_b * seq_len * NUM_HEADS + i_s * NUM_HEADS + head_off
    mask = head_off < NUM_HEADS
    blk_A_log = tl.load(A_log + head_off, mask=mask)
    blk_a = tl.load(a + i_b * stride_a_batch + head_off, mask=mask)
    blk_b = tl.load(b + i_b * stride_b_batch + head_off, mask=mask)
    blk_bias = tl.load(dt_bias + head_off, mask=mask)
    # If the model is loaded in fp16, without the .float() here, A might be -inf
    x = blk_a.to(tl.float32) + blk_bias.to(tl.float32)
    softplus_x = tl.where(
        beta * x <= threshold, (1 / beta) * tl.log(1 + tl.exp(beta * x)), x
    )
    blk_g = -tl.exp(blk_A_log.to(tl.float32)) * softplus_x
    tl.store(g + out_off, blk_g.to(g.dtype.element_ty), mask=mask)
    # compute beta_output = sigmoid(b)
    blk_beta_output = tl.sigmoid(blk_b.to(tl.float32))
    tl.store(
        beta_output + out_off,
        blk_beta_output.to(beta_output.dtype.element_ty),
        mask=mask,
    )


def fused_gdn_gating(
    A_log: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    dt_bias: torch.Tensor,
    beta: float = 1.0,
    threshold: float = 20.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Fused computation of g and beta for Gated Delta Net.
    g = -self.A_log.float().exp() * F.softplus(a.float() + self.dt_bias)
    beta_output = b.sigmoid()
    TODO maybe use torch.compile to replace this triton kernel
    """
    batch, num_heads = a.shape
    seq_len = 1
    grid = (batch, seq_len, triton.cdiv(num_heads, 8))
    g = torch.empty(1, batch, num_heads, dtype=torch.float32, device=a.device)
    beta_output = torch.empty(1, batch, num_heads, dtype=b.dtype, device=b.device)
    fused_gdn_gating_kernel[grid](
        g,
        beta_output,
        A_log,
        a,
        b,
        dt_bias,
        seq_len,
        num_heads,
        a.stride(0),
        b.stride(0),
        beta,
        threshold,
        8,
        num_warps=1,
    )
    return g, beta_output


class GatedDeltaNet(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_k_heads: int,
        num_v_heads: int,
        head_k_dim: int,
        head_v_dim: int,
        key_dim: int,
        value_dim: int,
        dt_bias: torch.Tensor,
        A_log: torch.Tensor,
        conv1d,
        activation,
        layer_num: int = 0,
        use_vk_layout: bool = True,
        **kwargs,
    ):
        """
        Args:
            use_vk_layout: When True (default), prefill writes ssm_state in
                vLLM's `[V, K]`-per-head layout — required for this file's
                decode kernels (fused_sigmoid_gating_delta_rule_update,
                flydsl_gdr_decode) to read it correctly. The vLLM ssm_state
                allocator (MambaStateShapeCalculator.gated_delta_net_state_shape)
                also uses this layout. Set False ONLY for developer
                experiments — the plugin path's decode kernels are vk-only,
                so kv prefill would corrupt the prefill→decode round-trip.
        """
        super().__init__()
        self.layer_num = layer_num

        self.tp_size = get_tensor_model_parallel_world_size()
        self.conv1d = conv1d
        self.activation = activation
        self.A_log = A_log
        self.dt_bias = dt_bias
        self.key_dim = key_dim
        self.value_dim = value_dim
        self.hidden_size = hidden_size
        self.num_k_heads = num_k_heads
        self.num_v_heads = num_v_heads
        self.head_k_dim = head_k_dim
        self.head_v_dim = head_v_dim
        self.use_vk_layout = use_vk_layout
        self.chunk_gated_delta_rule = ChunkGatedDeltaRule(use_vk_layout=use_vk_layout)

    def rearrange_mixed_qkv(self, mixed_qkv):
        if mixed_qkv is None:
            return None, None, None
        query, key, value = torch.split(
            mixed_qkv,
            [
                self.key_dim // self.tp_size,
                self.key_dim // self.tp_size,
                self.value_dim // self.tp_size,
            ],
            dim=-1,
        )
        query, key = map(
            lambda x: rearrange(x, "l (h d) -> 1 l h d", d=self.head_k_dim),
            (query, key),
        )
        value = rearrange(value, "l (h d) -> 1 l h d", d=self.head_v_dim)
        return query.contiguous(), key.contiguous(), value.contiguous()

    # ----- Recurrent-attention path helpers -------------------------------
    # Three call sites for the gated-delta-rule recurrence, one per request
    # type. Splitting them out (instead of the prior single if/elif/else
    # block inside `forward`) lets `forward` route a mixed batch — spec +
    # decode + prefill in the same call — to all three in turn, then
    # concatenate their outputs token-wise. Each helper owns its own
    # ssm_state plumbing so the dispatcher stays simple.
    #
    # Signature difference vs `atom.model_ops.attention_gdn.GatedDeltaNet`:
    # the decode and spec kernels here (`fused_sigmoid_gating_delta_rule_update`)
    # fuse the `g`/`beta` computation inside the kernel, so those helpers
    # take `a`/`b` directly. The prefill kernel (`chunk_gated_delta_rule_vk`)
    # does not fuse gating, so `gdr_prefill` consumes the pre-computed
    # `g`/`beta` from `fused_gdn_gating`.

    def gdr_prefill(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        ssm_state: torch.Tensor,
        has_initial_state: torch.Tensor,
        non_spec_state_indices: torch.Tensor,
        query_start_loc: torch.Tensor,
        o: torch.Tensor | None = None,
        num_decodes: int = 0,
        num_decode_tokens: int = 0,
    ) -> torch.Tensor:
        """Chunked-prefill path. Python-side gather of the initial state
        out of `ssm_state`, zero out slots whose sequences have no prior
        state, run the chunk kernel, scatter the final state back.

        `o` is an optional pre-allocated inplace output buffer; pass it
        only when the prefill output can be written straight into the
        caller's `core_attn_out` slice without an intermediate copy.

        `query_start_loc` is the ORIGINAL non-spec cumulative sequence
        lengths (cache-stable across forward calls). `num_decodes` and
        `num_decode_tokens` tell the chunk-vk prologue to skip the
        leading decode-only prefix; the prologue does the rebase
        internally under `@tensor_cache`, avoiding a per-call D2H sync.
        """
        # from aiter.ops.flydsl.linear_attention_prefill_kernels import flydsl_gdr_prefill
        from aiter.ops.triton.gated_delta_net.gated_delta_rule import (
            chunk_gated_delta_rule_opt_vk,
        )

        initial_state = ssm_state[non_spec_state_indices].contiguous()
        initial_state[~has_initial_state, ...] = 0
        core_attn_out, last_recurrent_state = chunk_gated_delta_rule_opt_vk(
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            initial_state=initial_state,
            output_final_state=True,
            cu_seqlens=query_start_loc,
            use_qk_l2norm_in_kernel=True,
            o=o,
            num_decodes=num_decodes,
            num_decode_tokens=num_decode_tokens,
        )
        ssm_state[non_spec_state_indices] = last_recurrent_state.to(ssm_state.dtype)
        return core_attn_out

    def gdr_decode(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        a: torch.Tensor,
        b: torch.Tensor,
        ssm_state: torch.Tensor,
        non_spec_state_indices: torch.Tensor,
        query_start_loc: torch.Tensor,
        o: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Single-token decode path (one token per request, no spec).
        Uses the fused sigmoid-gating + delta-rule kernel which reads/
        writes ssm_state inplace via the slot table.

        When `ATOM_USE_FLYDSL_GDR` is set and aiter is importable, takes
        the flydsl decode fast-path instead. `o` is required when the
        flydsl path is selected (it has no `output_final_state` return).
        """
        if USE_FLYDSL_GDR:
            assert o is not None, "flydsl_gdr_decode requires a pre-allocated o"
            core_attn_out = o
            # flydsl expects (seq, batch, num_v_heads, head_v_dim).
            q_perm = q.permute(1, 0, 2, 3)
            # NOTE: a and b use the (batch, seq, num_v_heads) layout.
            flydsl_gdr_decode(
                query=q_perm,
                key=k,
                value=v,
                a=a.unsqueeze(1),
                b=b.unsqueeze(1),
                dt_bias=self.dt_bias,
                A_log=self.A_log,
                indices=non_spec_state_indices,
                state=ssm_state,
                out=core_attn_out,
                use_qk_l2norm=True,
                need_shuffle_state=False,
                stream=torch.cuda.current_stream(),
            )
            return core_attn_out

        core_attn_out, _ = fused_sigmoid_gating_delta_rule_update(
            A_log=self.A_log,
            a=a,
            b=b,
            dt_bias=self.dt_bias,
            q=q,
            k=k,
            v=v,
            o=o,
            initial_state=ssm_state,
            inplace_final_state=True,
            cu_seqlens=query_start_loc,
            ssm_state_indices=non_spec_state_indices,
            use_qk_l2norm_in_kernel=True,
        )
        return core_attn_out

    def gdr_spec(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        a: torch.Tensor,
        b: torch.Tensor,
        ssm_state: torch.Tensor,
        spec_state_indices: torch.Tensor,
        query_start_loc: torch.Tensor,
        num_accepted_tokens: torch.Tensor,
    ) -> torch.Tensor:
        """Multi-query speculative-decode path. Walks only the accepted
        prefix of each candidate (via `num_accepted_tokens`) and updates
        ssm_state inplace at the slot pointed to by column 0 of
        `spec_state_indices`.
        """
        core_attn_out, _ = fused_sigmoid_gating_delta_rule_update(
            A_log=self.A_log,
            a=a,
            b=b,
            dt_bias=self.dt_bias,
            q=q,
            k=k,
            v=v,
            initial_state=ssm_state,
            inplace_final_state=True,
            cu_seqlens=query_start_loc,
            ssm_state_indices=spec_state_indices,
            num_accepted_tokens=num_accepted_tokens,
            use_qk_l2norm_in_kernel=True,
        )
        return core_attn_out

    def forward(
        self,
        mixed_qkv: torch.Tensor,
        b: torch.Tensor,
        a: torch.Tensor,
        core_attn_out: torch.Tensor,
        layer_name: str,
    ):
        """
        Core attention computation (called by custom op).
        """
        forward_context = get_forward_context()
        attn_metadata: AttentionMetadata = forward_context.attn_metadata

        if attn_metadata is None:
            # V1 profile run
            core_attn_out.zero_()
            return core_attn_out

        assert isinstance(attn_metadata, dict)
        attn_metadata = attn_metadata[layer_name]
        assert isinstance(attn_metadata, GDNAttentionMetadata)
        has_initial_state = attn_metadata.has_initial_state
        spec_query_start_loc = attn_metadata.spec_query_start_loc
        non_spec_query_start_loc = attn_metadata.non_spec_query_start_loc
        spec_sequence_masks = attn_metadata.spec_sequence_masks
        spec_token_indx = attn_metadata.spec_token_indx
        non_spec_token_indx = attn_metadata.non_spec_token_indx
        spec_state_indices_tensor = (
            attn_metadata.spec_state_indices_tensor
        )  # noqa: E501
        non_spec_state_indices_tensor = (
            attn_metadata.non_spec_state_indices_tensor
        )  # noqa: E501
        compilation_config = forward_context.no_compile_layers
        self_kv_cache = compilation_config[layer_name].kv_cache
        conv_state = self_kv_cache[0].transpose(-1, -2)
        ssm_state = self_kv_cache[1]
        num_actual_tokens = attn_metadata.num_actual_tokens
        num_accepted_tokens = attn_metadata.num_accepted_tokens

        mixed_qkv = mixed_qkv[:num_actual_tokens]
        b = b[:num_actual_tokens]
        a = a[:num_actual_tokens]

        # 1. Convolution sequence transformation
        conv_weights = self.conv1d.weight.view(
            self.conv1d.weight.size(0), self.conv1d.weight.size(2)
        )

        if spec_sequence_masks is not None:
            if attn_metadata.num_prefills == 0 and attn_metadata.num_decodes == 0:
                mixed_qkv_spec = mixed_qkv
                mixed_qkv_non_spec = None
            else:
                mixed_qkv_spec = mixed_qkv.index_select(0, spec_token_indx)
                mixed_qkv_non_spec = mixed_qkv.index_select(0, non_spec_token_indx)
        else:
            mixed_qkv_spec = None
            mixed_qkv_non_spec = mixed_qkv

        # 1.1: Process the multi-query part
        if spec_sequence_masks is not None:
            query_spec, key_spec, value_spec = causal_conv1d_update(
                mixed_qkv_spec,
                conv_state,
                conv_weights,
                self.num_k_heads * self.head_k_dim // self.tp_size,
                self.num_v_heads * self.head_v_dim // self.tp_size,
                self.conv1d.bias,
                self.activation,
                conv_state_indices=spec_state_indices_tensor[:, 0][
                    : attn_metadata.num_spec_decodes
                ],
                num_accepted_tokens=num_accepted_tokens,
                query_start_loc=spec_query_start_loc,
                max_query_len=spec_state_indices_tensor.size(-1),
                validate_data=False,
            )
            num_tokens_spec = query_spec.shape[0]
            query_spec = query_spec.view(1, num_tokens_spec, -1, self.head_k_dim)
            key_spec = key_spec.view(1, num_tokens_spec, -1, self.head_k_dim)
            value_spec = value_spec.view(1, num_tokens_spec, -1, self.head_v_dim)

        # 1.2: Process the remaining part
        if attn_metadata.num_prefills > 0:
            mixed_qkv_non_spec_T = mixed_qkv_non_spec.transpose(0, 1)
            # - "cache_indices" updates the conv_state cache in positions
            #   pointed to by "state_indices_tensor"
            query_non_spec, key_non_spec, value_non_spec = causal_conv1d_fn(
                mixed_qkv_non_spec_T,
                conv_weights,
                self.conv1d.bias,
                activation=self.activation,
                conv_states=conv_state,
                has_initial_state=has_initial_state,
                cache_indices=non_spec_state_indices_tensor,
                query_start_loc=non_spec_query_start_loc,
                k_dim_size=self.num_k_heads * self.head_k_dim // self.tp_size,
                v_dim_size=self.num_v_heads * self.head_v_dim // self.tp_size,
                metadata=attn_metadata,
            )
        elif attn_metadata.num_decodes > 0:
            query_non_spec, key_non_spec, value_non_spec = causal_conv1d_update(
                mixed_qkv_non_spec,
                conv_state,
                conv_weights,
                self.num_k_heads * self.head_k_dim // self.tp_size,
                self.num_v_heads * self.head_v_dim // self.tp_size,
                self.conv1d.bias,
                self.activation,
                conv_state_indices=non_spec_state_indices_tensor[
                    : attn_metadata.num_actual_tokens
                ],
                validate_data=True,
            )
        else:
            mixed_qkv_non_spec = None

        if attn_metadata.num_prefills > 0 or attn_metadata.num_decodes > 0:
            num_tokens_nonspec = query_non_spec.shape[0]
            query_non_spec = query_non_spec.view(
                1, num_tokens_nonspec, -1, self.head_k_dim
            )
            key_non_spec = key_non_spec.view(1, num_tokens_nonspec, -1, self.head_k_dim)
            value_non_spec = value_non_spec.view(
                1, num_tokens_nonspec, -1, self.head_v_dim
            )

        # Gating: pre-computed only for the prefill path (chunk kernel does
        # not fuse gating). The decode/spec kernels fuse gating internally
        # from `a`/`b`/`A_log`/`dt_bias`, so they don't need `g`/`beta`.
        if attn_metadata.num_prefills > 0:
            g, beta = fused_gdn_gating(self.A_log, a, b, self.dt_bias)
            if spec_sequence_masks is not None:
                g_non_spec = g.index_select(1, non_spec_token_indx)
                beta_non_spec = beta.index_select(1, non_spec_token_indx)
            else:
                g_non_spec = g
                beta_non_spec = beta
        else:
            g_non_spec = None
            beta_non_spec = None

        # For the decode helper, `a`/`b` need to be in non-spec token order
        # (the kernel walks them via `cu_seqlens=non_spec_query_start_loc`).
        # When spec is active, gather the non-spec rows out of the full
        # mixed `a`/`b`; otherwise the whole tensor is already non-spec.
        if (
            attn_metadata.num_decodes > 0
            and spec_sequence_masks is not None
            and non_spec_token_indx is not None
        ):
            a_non_spec = a.index_select(0, non_spec_token_indx)
            b_non_spec = b.index_select(0, non_spec_token_indx)
        else:
            a_non_spec = a
            b_non_spec = b

        num_decodes = attn_metadata.num_decodes
        num_decode_tokens = attn_metadata.num_decode_tokens
        num_prefills = attn_metadata.num_prefills

        # 2. Recurrent attention — dispatch to per-request-type helpers.
        # The three sub-paths (spec multi-query, single-token decode,
        # chunked prefill) are independent and may all fire in the same
        # call when the batch is mixed. All run on the default stream;
        # an earlier multi-stream design was tried but the event/stream
        # bookkeeping overhead outweighed the overlap gain.

        core_attn_out_decode = None
        core_attn_out_prefill = None

        # 2.1: Chunked-prefill path.
        #
        # Disjoint-region invariants when decode also runs:
        #   * `ssm_state` rows: decode uses
        #     `non_spec_state_indices_tensor[:num_decodes]`, prefill
        #     uses `non_spec_state_indices_tensor[num_decodes:]`.
        #   * `core_attn_out` slices: decode writes `[:num_decode_tokens]`,
        #     prefill writes `[num_decode_tokens:num_actual_tokens]`.
        if num_prefills > 0:
            # Prefill `o` target. When spec is absent the prefill output
            # goes straight into `core_attn_out` (either the full
            # `[:num_actual_tokens]` slice when there are no decode
            # tokens, or the tail past the decode tokens otherwise).
            # With spec active the merge below needs a separately-
            # allocated buffer.
            if spec_sequence_masks is None:
                prefill_o = core_attn_out[
                    num_decode_tokens:num_actual_tokens
                ].unsqueeze(0)
            else:
                prefill_o = None

            # Pass the ORIGINAL `non_spec_query_start_loc` (the
            # metadata's tensor, cache-stable across forward calls)
            # plus `num_decodes` / `num_decode_tokens`. The chunk-vk
            # prologue rebases it under `@tensor_cache`, avoiding a
            # per-call `[num_decodes:] - num_decode_tokens` subtraction
            # kernel and the `.tolist()` D2H syncs that the old ad-hoc
            # rebase would otherwise trigger inside
            # `prepare_chunk_indices` / `prepare_chunk_offsets`.
            core_attn_out_prefill = self.gdr_prefill(
                q=query_non_spec[:, num_decode_tokens:],
                k=key_non_spec[:, num_decode_tokens:],
                v=value_non_spec[:, num_decode_tokens:],
                g=(
                    g_non_spec[:, num_decode_tokens:]
                    if g_non_spec is not None
                    else None
                ),
                beta=(
                    beta_non_spec[:, num_decode_tokens:]
                    if beta_non_spec is not None
                    else None
                ),
                ssm_state=ssm_state,
                has_initial_state=(
                    has_initial_state[num_decodes:]
                    if has_initial_state is not None
                    else None
                ),
                non_spec_state_indices=non_spec_state_indices_tensor[num_decodes:],
                query_start_loc=non_spec_query_start_loc,
                o=prefill_o,
                num_decodes=num_decodes,
                num_decode_tokens=num_decode_tokens,
            )

        # 2.2: Spec multi-query path.
        if spec_sequence_masks is not None:
            core_attn_out_spec = self.gdr_spec(
                q=query_spec,
                k=key_spec,
                v=value_spec,
                a=a,
                b=b,
                ssm_state=ssm_state,
                spec_state_indices=spec_state_indices_tensor,
                query_start_loc=spec_query_start_loc[
                    : attn_metadata.num_spec_decodes + 1
                ],
                num_accepted_tokens=num_accepted_tokens,
            )
        else:
            core_attn_out_spec = None

        # 2.3: Single-token decode path. vLLM convention: decode tokens
        # come first in the non-spec slice (one per request), followed
        # by variable-length prefill tokens.
        if num_decodes > 0:
            # Decode `o` slice. The decode kernel writes `[num_decode_tokens,
            # H, V]` token-major into `core_attn_out` when spec is absent —
            # the slice `[:num_decode_tokens]` is the right region whether
            # or not prefill also fires, because the prefill helper above
            # writes into `[num_decode_tokens:num_actual_tokens]`. With spec
            # active the merge below needs to scatter via `index_copy_` to
            # spec-interleaved positions, so allocate a separate buffer.
            if spec_sequence_masks is None:
                decode_o = core_attn_out[:num_decode_tokens]
            else:
                decode_o = query_non_spec.new_empty(
                    (
                        num_decode_tokens,
                        value_non_spec.shape[2],
                        value_non_spec.shape[3],
                    )
                )
            core_attn_out_decode = self.gdr_decode(
                q=query_non_spec[:, :num_decode_tokens],
                k=key_non_spec[:, :num_decode_tokens],
                v=value_non_spec[:, :num_decode_tokens],
                a=a_non_spec[:num_decode_tokens],
                b=b_non_spec[:num_decode_tokens],
                ssm_state=ssm_state,
                non_spec_state_indices=non_spec_state_indices_tensor[:num_decodes],
                query_start_loc=non_spec_query_start_loc[: num_decodes + 1],
                o=decode_o,
            )

        # Stitch the decode and prefill outputs back into a single
        # non-spec tensor in the same token order they came in. When spec
        # is absent, both helpers already wrote inplace into adjacent
        # slices of `core_attn_out` (decode → `[:num_decode_tokens]`,
        # prefill → `[num_decode_tokens:num_actual_tokens]`), so no cat is
        # needed — `core_attn_out[:num_actual_tokens]` already holds the
        # stitched non-spec result. With spec active both helpers wrote
        # into separate fresh buffers (or a `None` o for prefill) and we
        # need to cat them for the `index_copy_` in the merge step below.
        if (
            spec_sequence_masks is not None
            and core_attn_out_decode is not None
            and core_attn_out_prefill is not None
        ):
            # decode helper returned `[num_decode_tokens, H, V]` (flat
            # alloc, spec-active path); promote to `[1, ..., H, V]` for cat.
            if core_attn_out_decode.dim() == 3:
                core_attn_out_decode_4d = core_attn_out_decode.unsqueeze(0)
            else:
                core_attn_out_decode_4d = core_attn_out_decode
            core_attn_out_non_spec = torch.cat(
                (core_attn_out_decode_4d, core_attn_out_prefill), dim=1
            )
        elif core_attn_out_decode is not None:
            core_attn_out_non_spec = core_attn_out_decode
        elif core_attn_out_prefill is not None:
            core_attn_out_non_spec = core_attn_out_prefill
        else:
            core_attn_out_non_spec = None

        # 3. Merge core attention output. Three cases:
        #   (a) spec + non-spec coexist → scatter via index_copy_ into a
        #       fresh `[num_actual_tokens]` buffer.
        #   (b) spec only → copy spec output into core_attn_out.
        #   (c) non-spec only → the helpers already wrote into core_attn_out
        #       via their inplace `o` slices (no copy needed).
        if spec_sequence_masks is not None and core_attn_out_non_spec is not None:
            merged_out = torch.empty(
                (1, num_actual_tokens, *core_attn_out_spec.shape[2:]),
                dtype=core_attn_out_non_spec.dtype,
                device=core_attn_out_non_spec.device,
            )
            merged_out.index_copy_(1, spec_token_indx, core_attn_out_spec)
            # core_attn_out_non_spec may be 3-D (flat decode-only) or 4-D
            # (prefill or cat). Promote to 4-D for index_copy_ on dim 1.
            non_spec_4d = (
                core_attn_out_non_spec.unsqueeze(0)
                if core_attn_out_non_spec.dim() == 3
                else core_attn_out_non_spec
            )
            merged_out.index_copy_(1, non_spec_token_indx, non_spec_4d)
            core_attn_out[:num_actual_tokens] = merged_out.squeeze(0)
        elif spec_sequence_masks is not None:
            core_attn_out[:num_actual_tokens] = core_attn_out_spec.squeeze(0)

        # Zero padding tail for CUDA graph replay safety
        if num_actual_tokens < core_attn_out.shape[0]:
            core_attn_out[num_actual_tokens:].zero_()

        return core_attn_out
