import copy
import logging

import torch
from torch import nn
from torch.profiler import record_function

from atom.config import CompilationLevel
from atom.distributed.pcp_utils import (
    get_pcp_world_size,
    pcp_allgather_rerange,
    pcp_pad_dense,
    pcp_pad_len,
    pcp_round_robin_split,
)
from atom.spec_decode.drafter import Drafter
from atom.spec_decode.eagle3_kv_builder import Eagle3DraftBuilder
from atom.utils import envs
from atom.utils.forward_context import get_forward_context

logger = logging.getLogger("atom")


def _pcp_active_for_draft_model(draft_model: nn.Module) -> bool:
    # DeepSeek V2/DSA draft models share this sparse-MLA PCP gate.
    from atom.models.deepseek_v2 import _pcp_active

    if _pcp_active():
        return True

    if draft_model.__class__.__name__ != "DeepseekV4MTP":
        return False

    from atom.models.deepseek_v4 import _pcp_active as _pcp_active_v4

    return _pcp_active_v4()


class EagleProposer(Drafter):
    """Serial speculative drafter: plain MTP and EAGLE3.

    Drafts ``mtp_k`` tokens by running the draft model in a python loop (one
    heavyweight backbone pass per drafted token). The block-parallel DSpark
    flavor is its sibling ``DSparkProposer``; both share the ``Drafter`` base.
    """

    def __init__(self, atom_config, device: torch.device, runner):
        super().__init__(atom_config, device, runner)
        # GLM-5.2 draft index sharing: step 0 runs the MTP indexer, steps 1+
        # reuse sparse_kv_indices_buffer via skip_topk + compact_topk_indices.
        # Gated on method=mtp, DSA index_topk and the config flag, so other
        # draft backends are unchanged. (DSpark is DSparkProposer, not this
        # class, so it cannot reach here.)
        draft_hf = self.speculative_config.draft_model_hf_config
        mtp_inner = getattr(self.model, "model", None)
        self._share_mtp_indices = (
            self.speculative_config.method == "mtp"
            and getattr(draft_hf, "index_share_for_mtp_iteration", False)
            and hasattr(draft_hf, "index_topk")
            and mtp_inner is not None
            and hasattr(mtp_inner, "set_skip_topk")
        )
        if self._share_mtp_indices:
            logger.info(
                "MTP draft index_share_for_mtp_iteration enabled: "
                "step 0 computes indexer top-k, steps 1+ reuse the buffer."
            )

    def _resolve_mtp_k(self) -> int:
        return self.speculative_config.num_speculative_tokens or 0

    def _build_draft_model(self, model_class) -> nn.Module:
        draft_model_hf_config = self.speculative_config.draft_model_hf_config
        if self.speculative_config.method == "eagle3":
            # Eagle3 draft has its own architecture, so build it from the
            # draft hf_config. Disable torch.compile for the draft to avoid
            # Dynamo tracing issues with the separate KV cache binding.
            # Shallow-copy instead of deepcopy: with MLA targets (K2.6), the
            # atom_config holds non-picklable cuda.Stream objects under
            # downstream fields that deepcopy can't traverse. We only mutate
            # hf_config and compilation_config.level on the draft, so
            # isolating just those two attrs is sufficient.
            draft_atom_config = copy.copy(self.config)
            draft_atom_config.hf_config = draft_model_hf_config
            draft_atom_config.compilation_config = copy.copy(
                self.config.compilation_config
            )
            draft_atom_config.compilation_config.level = CompilationLevel.NO_COMPILATION
            # Draft attention layer_num must continue from the target model's
            # layer count so it maps to the correct kv_cache_data entry.
            model = model_class(
                draft_atom_config,
                layer_offset=self.config.hf_config.num_hidden_layers,
            )
            # MHA draft (e.g. K2.5 LlamaForCausalLMEagle3): owns an independent
            # non-MLA KV cache via Eagle3DraftBuilder, attached to the runner.
            # MLA draft (e.g. K2.6 EAGLE 3.1): same MLA shape as target, so
            # it piggybacks on the target's MLA pool (model_runner accounts
            # for the +1 draft layer via num_nextn_predict_layers default).
            draft_is_mla = bool(getattr(draft_model_hf_config, "kv_lora_rank", None))
            if not draft_is_mla:
                self.runner.eagle3_draft_builder = Eagle3DraftBuilder(
                    self.runner, draft_model_hf_config
                )
            return model

        return model_class(self.config)

    def arm_aux_capture(self, target_model: nn.Module) -> None:
        """EAGLE3: arm the target's aux hidden-state capture, entirely
        drafter-owned so neither ModelRunner nor any model wrapper carries aux
        code.

        Arming an eagle3 target means telling its forward which layers to emit as
        ``(hidden, aux_list)`` — that state (``self.aux_hidden_state_layers``)
        necessarily lives on the model, and is also the vLLM ``SupportsEagle3``
        contract. So the drafter itself routes to the model that owns it: it
        unwraps any ``.model`` wrapper (e.g. TBO ``UBatchWrapper``) to call
        ``set_aux_hidden_state_layers`` on the real model, while installing the
        tuple-strip hook on the OUTERMOST ``target_model`` — the object whose
        forward yields the final (for TBO, concatenated) output. The hook copies
        each aux tensor into a fixed drafter buffer in-place (cudagraph-safe) and
        returns plain ``hidden``, so every ModelRunner call site gets a bare
        Tensor and ``aux_for`` (base) reads the buffers.

        No-op for plain MTP.
        """
        spec = self.speculative_config
        if spec.method != "eagle3" or not spec.use_aux_hidden_state:
            return
        # Unwrap to the model that owns eagle3 aux-layer state; the strip hook
        # still goes on target_model (the outermost forward).
        aux_model = target_model
        while not hasattr(aux_model, "set_aux_hidden_state_layers") and hasattr(
            aux_model, "model"
        ):
            aux_model = aux_model.model
        aux_ids = spec.eagle3_aux_layer_ids
        if not aux_ids and hasattr(aux_model, "get_eagle3_aux_hidden_state_layers"):
            aux_ids = list(aux_model.get_eagle3_aux_hidden_state_layers())
        if not aux_ids:
            return
        aux_model.set_aux_hidden_state_layers(tuple(aux_ids))
        hidden_size = self.config.hf_config.hidden_size
        self._aux_buffers = [
            torch.zeros(
                self.max_num_tokens, hidden_size, device=self.device, dtype=self.dtype
            )
            for _ in aux_ids
        ]
        self._captures_aux = True
        target_model.register_forward_hook(self._aux_strip_hook)
        logger.info(f"Eagle3 aux hidden state layers: {aux_ids}")

    def _aux_strip_hook(self, module: nn.Module, inputs, output):
        """Forward hook on the target model: strip ``(hidden, aux_list)`` into the
        fixed aux buffers and return plain ``hidden``. Pass through any non-tuple
        output (plain hidden / IntermediateTensors on non-last PP ranks)."""
        if not (isinstance(output, tuple) and len(output) == 2):
            return output
        hidden, aux_list = output
        # strict: buffers are sized from the CONFIGURED aux ids, aux_list is what
        # the target actually installed. A lenient zip would leave a trailing
        # buffer at its zeros init and silently lose acceptance.
        for buf, aux in zip(self._aux_buffers, aux_list, strict=True):
            buf[: aux.shape[0]].copy_(aux)
        return hidden

    def propose(
        self,
        # [num_tokens]
        target_token_ids: torch.Tensor,
        # [num_tokens]
        target_positions: torch.Tensor,
        # [num_tokens, hidden_size]
        target_hidden_states: torch.Tensor,
        # [batch]
        num_reject_tokens: torch.Tensor,
        next_token_ids: torch.Tensor,
        last_token_indices: torch.Tensor,
    ) -> torch.Tensor:

        forward_context = get_forward_context()
        context = forward_context.context
        attn_metadata = forward_context.attn_metadata
        bs = context.batch_size
        context.is_draft = True

        assert self.runner is not None

        input_ids = target_token_ids
        # input_ids[last_token_indices] = next_token_ids
        input_ids.scatter_(0, last_token_indices, next_token_ids)
        positions = target_positions + 1

        # Drafter-owned aux: our own capture buffers, row-aligned to the target
        # hidden states we're drafting from.
        aux_hidden_states = self.aux_for(target_hidden_states)

        # Eagle3: project concatenated aux hidden states through fc
        if aux_hidden_states is not None:
            concat_aux = torch.cat(aux_hidden_states, dim=-1)
            hidden_states = self.model.combine_hidden_states(concat_aux)
        else:
            hidden_states = target_hidden_states

        draft_token_ids = torch.empty(
            bs, self.mtp_k, dtype=next_token_ids.dtype, device=next_token_ids.device
        )
        if envs.ATOM_DEBUG_FORCE_SKIP_DRAFT_MODEL:
            draft_token_ids.fill_(-1)
        var = self.runner.forward_vars
        target_uses_mla = self.runner.use_mla
        # Eaale3 only support mha currently
        draft_uses_mha = hasattr(self.runner, "eagle3_draft_builder")

        # Eagle3 MHA reuses target metadata, but the target may be MLA.  Keep
        # write slots sized to this draft pass, and when prefix cache is active
        # restore logical block ids: MLA prefill expands block_tables by
        # block_ratio for its physical block_size=1 pool, while the draft MHA
        # cache is indexed by runner.block_size blocks.
        if draft_uses_mha:
            attn_metadata.slot_mapping = var["slot_mapping"].gpu[: len(input_ids)]
            attn_metadata.block_tables = var["block_tables"].gpu[:bs]
        else:
            # Make MLA draft slot_mapping == q rows.
            attn_metadata.slot_mapping = attn_metadata.slot_mapping[: len(input_ids)]

        # Backends that expose flat per-seq kv_indices/kv_indptr (MLA, MHA)
        # wire them through eagle's mid-step block; V4 has block_tables +
        # context_lens instead (its v4_kv_indices_{swa,csa,hca} are per-token
        # non-equivalent). Hoisted out of the loop so the value is bound for
        # every iteration (used at i>=1 too, even though i==0 sets it).
        has_flat_kv = "kv_indices" in var

        for i in range(self.mtp_k):
            with record_function(f"draft[{i}/{self.mtp_k} bs={bs}]"):
                # Re-sync DP token
                self._refresh_dp_metadata(forward_context, input_ids.shape[0])
                # ---- Prefill Context Parallel (draft i==0 prefill) --------
                # The draft's first pass is a prefill that reuses the target's
                # 1/pcp-reindexed attn_metadata, so it must run on this rank's
                # 1/pcp query shard (input_ids / positions / previous hidden) and
                # all-gather the draft hidden back to full token order before the
                # last-token sampling gather. Later draft steps are decode
                # (is_prefill False) and run full — identical to the non-PCP path.
                # `input_ids` / `positions` / `hidden_states` themselves stay full
                # so the post-i==0 decode-metadata setup (which indexes with the
                # full `last_token_indices`) is unchanged.
                pcp_draft_prefill = i == 0 and _pcp_active_for_draft_model(self.model)
                if pcp_draft_prefill:
                    pcp_ws = get_pcp_world_size()
                    n_global_draft = input_ids.shape[0]
                    n_pad = pcp_pad_len(n_global_draft, pcp_ws) - n_global_draft
                    d_input_ids = pcp_round_robin_split(
                        pcp_pad_dense(input_ids, n_pad), pcp_ws
                    )
                    d_positions = pcp_round_robin_split(
                        pcp_pad_dense(positions, n_pad), pcp_ws
                    )
                    d_hidden = pcp_round_robin_split(
                        pcp_pad_dense(hidden_states, n_pad), pcp_ws
                    )
                else:
                    d_input_ids, d_positions, d_hidden = (
                        input_ids,
                        positions,
                        hidden_states,
                    )
                # index_share_for_mtp_iteration: step 0 runs the MTP indexer;
                # steps 1+ skip it and read the compacted sparse_kv buffer.
                if self._share_mtp_indices and i == 0:
                    self.model.model.set_skip_topk(False)
                ret_hidden_states = self.model(
                    input_ids=d_input_ids,
                    positions=d_positions,
                    hidden_states=d_hidden,
                )
                if pcp_draft_prefill:
                    ret_hidden_states = pcp_allgather_rerange(
                        ret_hidden_states, pcp_ws
                    )[:n_global_draft]
                if self._share_mtp_indices and i == 0:
                    self.model.model.set_skip_topk(True)
                    self.model.model.compact_topk_indices(last_token_indices)

                sample_hidden_states = (
                    torch.index_select(ret_hidden_states, 0, last_token_indices)
                    if i == 0
                    else ret_hidden_states
                )
                # Distributed argmax (all-gather [N, 2] not [N, vocab]) when the
                # draft supports it; token-identical to compute_logits().argmax().
                if self._draft_argmax_fused:
                    new_draft_ids = self.model.compute_draft_token(sample_hidden_states)
                else:
                    logits = self.model.compute_logits(sample_hidden_states)
                    new_draft_ids = logits.argmax(dim=-1)
                draft_token_ids[:, i] = new_draft_ids

                if i < self.mtp_k - 1:
                    do_attn_metadata_update = (
                        not context.is_prefill
                        # TODO: FIX this condition after we support3 attention head numbers=32
                        and self.runner.attn_metadata_builder.num_attention_heads != 32
                    )
                    if i == 0:
                        i0_max_seqlen_q = attn_metadata.max_seqlen_q
                        attn_metadata.max_seqlen_q = 1
                        slot_mapping = var["slot_mapping"].gpu[
                            : bs * attn_metadata.max_seqlen_q
                        ]
                        cu_seqlens_q = var["cu_seqlens_q"].gpu[: bs + 1]
                        attn_metadata.cu_seqlens_q = cu_seqlens_q
                        attn_metadata.slot_mapping = slot_mapping
                        if has_flat_kv:
                            kv_indptr = var["kv_indptr"].gpu[: bs + 1]
                            kv_indices = var["kv_indices"].gpu
                            attn_metadata.kv_indptr = kv_indptr
                            attn_metadata.kv_indices = kv_indices
                        if target_uses_mla:
                            kv_last_page_lens = var["kv_last_page_lens"].gpu[:bs]
                            attn_metadata.kv_last_page_lens = kv_last_page_lens
                            # Sparse (DSA) MLA decode packs KV per token at
                            # page_size=1, so it reads the all-1s
                            # sparse_kv_last_page_lens (NOT the dense per-block
                            # buffer, which makes the asm kernel over-read past
                            # the written sparse-index region -> illegal access).
                            # The draft reuses the target's attn_metadata but
                            # drops to max_seqlen_q=1, so it must re-point this to
                            # the per-seq all-1s slice itself.
                            if "sparse_kv_last_page_lens" in var:
                                attn_metadata.sparse_kv_last_page_lens = var[
                                    "sparse_kv_last_page_lens"
                                ].gpu[:bs]
                        # block_tables, context_lens, and sparse_kv_indptr are
                        # needed by both MHA and MLA+sparse attention
                        attn_metadata.block_tables = var["block_tables"].gpu[:bs]
                        attn_metadata.context_lens = var["context_lens"].gpu[:bs]
                        if "sparse_kv_indptr" in var:
                            attn_metadata.sparse_kv_indptr = var[
                                "sparse_kv_indptr"
                            ].gpu[: bs + 1]
                        cu_seqlens_q[: bs + 1] = self.arrange_bs[: bs + 1]
                        if target_uses_mla and has_flat_kv:
                            # MLA: block_size=1, kv_indptr tracks tokens
                            kv_indptr[1 : bs + 1] -= torch.cumsum(
                                num_reject_tokens, dim=0
                            )
                        if positions.ndim == 1:
                            positions = torch.index_select(
                                positions, 0, last_token_indices
                            )
                        else:
                            # MRoPE positions keep the token axis last (e.g.
                            # [3, num_tokens] for Qwen3.5), so select columns
                            # instead of indexing dim 0.
                            positions = torch.index_select(
                                positions, positions.ndim - 1, last_token_indices
                            )
                        context.is_prefill = False

                    # update metadata
                    attn_metadata.max_seqlen_k += 1
                    fuse_mtp = positions.ndim == 1 and getattr(
                        self.runner.attn_metadata_builder,
                        "fuse_mtp_decode_position_update",
                        False,
                    )
                    if fuse_mtp:
                        mtp_decode_kwargs = {
                            "update_context_lens": True,
                            "positions_out": positions,
                        }
                    else:
                        attn_metadata.context_lens[:bs] += 1
                        positions += 1
                        mtp_decode_kwargs = {}
                    workinfos = self.runner.attn_metadata_builder.prepare_mtp_decode(
                        bs,
                        (
                            attn_metadata.max_seqlen_q
                            if not do_attn_metadata_update
                            else i0_max_seqlen_q
                        ),
                        attn_metadata.max_seqlen_k,
                        positions,
                        only_update=do_attn_metadata_update,
                        num_reject_tokens=num_reject_tokens if i == 0 else None,
                        **mtp_decode_kwargs,
                    )
                    for k, v in workinfos.items():
                        attn_metadata.__dict__[k] = v
                    if has_flat_kv and "slot_mapping" not in workinfos:
                        # MLA/MHA path: slot derived from flat kv_indices.
                        raw_slots = kv_indices[kv_indptr[1 : bs + 1] - 1]
                        builder = self.runner.attn_metadata_builder
                        if getattr(builder, "dcp_world_size", 1) > 1:
                            # DCP round-robin: only rank (context_len-1) % W owns
                            # this draft token; other ranks' kv_indptr didn't grow,
                            # so raw_slots would point at a stale slot. Emit -1.
                            ctx = attn_metadata.context_lens[:bs]
                            owned = ((ctx - 1) % builder.dcp_world_size) == builder.dcp_rank
                            slot_mapping[:] = torch.where(owned, raw_slots, -1)
                        else:
                            slot_mapping[:] = raw_slots

                    input_ids = new_draft_ids
                    hidden_states = sample_hidden_states

        # self.runner.debug(f"final {draft_token_ids=}")
        # [batch_size, mtp_k]
        return draft_token_ids
