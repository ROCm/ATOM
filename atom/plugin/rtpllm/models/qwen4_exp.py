"""Qwen3.8-Flash-Next adapter for RTP-LLM's ATOM plugin mode."""

from __future__ import annotations

import inspect
import json
import logging
import os
from types import SimpleNamespace

import torch
from rtp_llm.config.model_config import ModelConfig, ssm_state_dtype_str_to_data_type
from rtp_llm.models.hybrid_kv_cache import build_hybrid_kv_cache_spec_descs
from rtp_llm.ops import HybridAttentionType, KVCacheSpecType
from rtp_llm.ops.compute_ops import PyModelOutputs
from rtp_llm.utils.model_weight import W

from atom.plugin.rtpllm.models.qwen3_5 import (
    ATOMQwen35Moe,
    _ATOMAttnPyObj,
    _ATOMQwen35MoeRuntime,
)

logger = logging.getLogger("atom.plugin.rtpllm.models")


class _Qwen4ExpAttnPyObj(_ATOMAttnPyObj):
    # Graph replay refreshes and consumes device block tables only. The host
    # mirrors are not read by this plugin's prepare_cuda_graph hook.
    requires_host_block_table_mirror = False

    def prepare_cuda_graph(self, attn_inputs) -> None:
        if isinstance(attn_inputs, dict):
            attn_inputs = attn_inputs["full"]
        super().prepare_cuda_graph(attn_inputs)


class _ATOMQwen4ExpRuntime(_ATOMQwen35MoeRuntime):
    """Reuse RTP's input/position bridge with QSA and PLE cache metadata."""

    def __init__(self, model_config: ModelConfig, *args, **kwargs) -> None:
        from atom.plugin.rtpllm.utils.qwen4_exp_context import RTPQwen4ExpContext

        # Bind by the parent's parameter name, not a positional offset. The
        # runtime base does not retain max_generate_batch_size as an attribute.
        bound = inspect.signature(_ATOMQwen35MoeRuntime.__init__).bind(
            self, model_config, *args, **kwargs
        )
        max_batch = bound.arguments.get("max_generate_batch_size")
        self._rtp_forward_context_cls = RTPQwen4ExpContext
        super().__init__(model_config, *args, **kwargs)
        self._rtp_model_config = model_config
        if model_config.mm_model_config.is_multimodal:
            # RTP owns the vision tower, so prepare_model clears ATOM's
            # multimodal_config and the language model has no image_token_id.
            self._image_token_id = self._read_image_token_id(model_config.ckpt_path)
        if getattr(model_config, "atom_qwen4_exp_mtp_target", False) or getattr(
            model_config, "is_mtp", False
        ):
            # Keep the parent's capture-bucket cap for large Graph workspaces.
            # Only the small position buffers need full request capacity.
            if max_batch is None:
                raise ValueError("Qwen4Exp MTP request capacity was not provided")
            max_batch = int(max_batch)
            if max_batch <= 0:
                raise ValueError("Qwen4Exp MTP requires positive request capacity")
            self._rtp_mtp_request_capacity = max_batch
        self._atom_attn_pyobj = _Qwen4ExpAttnPyObj(self)

    def copy_plugin_cache_blocks(self, mapping: torch.Tensor, tags: list[str]) -> None:
        """Mirror RTP's tagged block relocations into QSA and PLE side pools.

        RTP invokes this before eager forward *and* graph replay. A prefix hit
        keeps the same physical block; only copy-on-write relocations appear
        in ``mapping``. The RTP-owned KV/GDN pools are copied separately.
        """
        if mapping.device.type != "cpu" or mapping.ndim != 2 or mapping.shape[1] != 3:
            raise ValueError("plugin cache copy requires a CPU int32 [N,3] mapping")
        if mapping.dtype != torch.int32:
            raise ValueError("plugin cache copy requires int32 block IDs")
        if not mapping.numel():
            return
        side_pools: dict[str, list[torch.Tensor]] = {"full": [], "linear": []}
        qsa_layers = [
            module
            for module in self._rtp_layer_maps[1].values()
            if getattr(module, "is_qsa_attention", False)
        ]
        if qsa_layers:
            self._rtp_forward_context_cls._bind_qsa_caches(
                self, qsa_layers, int(self.kv_cache.get_seq_size_per_block("full"))
            )
            for module in qsa_layers:
                side_pools["full"].extend(
                    (module.raw_key_cache, module.compressed_key_cache)
                )
                if module.rope_position_cache is not None:
                    side_pools["full"].append(module.rope_position_cache)
        self._rtp_forward_context_cls._bind_ple_caches(
            self, self.model, self._rtp_layer_maps[0]
        )
        if hasattr(self, "_rtp_ple_ngram"):
            side_pools["linear"].extend((self._rtp_ple_ngram, self._rtp_ple_conv))
        copies = mapping.tolist()
        # Validate the entire mapping before touching any side pool. RTP may
        # chain copies, so the second pass must still preserve row order.
        for row, src, dst in copies:
            if row < 0 or row >= len(tags):
                raise ValueError(f"plugin cache copy has invalid tag row {row}")
            for pool in side_pools.get(tags[row], ()):
                if src < 0 or dst < 0 or src >= pool.shape[0] or dst >= pool.shape[0]:
                    raise ValueError("plugin cache copy has out-of-range block ID")
        for row, src, dst in copies:
            for pool in side_pools.get(tags[row], ()):
                if src != dst:
                    pool[dst].copy_(pool[src])

    def prepare_fmha_impl(self, inputs, is_cuda_graph: bool = False):
        tagged = inputs.attention_inputs
        if not isinstance(tagged, dict):
            return super().prepare_fmha_impl(inputs, is_cuda_graph)
        if "linear" not in tagged or "full" not in tagged:
            raise ValueError("Qwen4Exp Graph requires linear/full cache tags")
        # The parent prewarm needs a single attention input. RTP's graph
        # runner passes the whole tag map at replay, so keep the full map on
        # the original input and mark both tag entries as graph inputs.
        if is_cuda_graph:
            for attn_inputs in tagged.values():
                attn_inputs.is_cuda_graph = True
        return super().prepare_fmha_impl(
            SimpleNamespace(attention_inputs=tagged["linear"]), is_cuda_graph
        )

    def _ensure_cuda_graph_prewarmed(self) -> None:
        super()._ensure_cuda_graph_prewarmed()
        buffers = getattr(self, "_cg_meta_bufs", None)
        if buffers is None:
            return
        model = getattr(self, "model", None)
        if model is not None:
            self._rtp_forward_context_cls._bind_ple_caches(
                self, model, self._rtp_layer_maps[0]
            )
        if "qsa_compressed_slots" in buffers:
            return
        is_target = bool(
            getattr(
                getattr(self, "_rtp_model_config", None),
                "atom_qwen4_exp_mtp_target",
                False,
            )
        )
        is_mtp_draft = bool(
            getattr(getattr(self, "_rtp_model_config", None), "is_mtp", False)
        )
        mtp_tokens_per_request = (
            int(self._rtp_model_config.gen_num_per_cycle) + 1
            if is_target or is_mtp_draft
            else 1
        )
        if (is_target or is_mtp_draft) and mtp_tokens_per_request not in (2, 3, 4):
            raise ValueError("Qwen4Exp MTP supports 1, 2, or 3 draft steps")
        max_batch = int(self._cg_max_num_tokens)
        request_capacity = int(getattr(self, "_rtp_mtp_request_capacity", max_batch))
        device = self._get_model_device()
        qsa_layers = [
            module
            for module in self._rtp_layer_maps[1].values()
            if getattr(module, "is_qsa_attention", False)
        ]
        if qsa_layers:
            self._rtp_forward_context_cls._bind_qsa_caches(
                self,
                qsa_layers,
                int(self.kv_cache.get_seq_size_per_block("full")),
            )

        def add_qsa_buffers(graph_buffers, num_tokens, position_capacity):
            graph_buffers["seq_id_i32"] = graph_buffers["seq_id"].to(torch.int32)
            graph_buffers["qwen4_positions_i32"] = torch.empty(
                position_capacity, device=device, dtype=torch.int32
            )
            for name, dtype in (
                ("qsa_positions_i64", torch.int64),
                ("qsa_slots_i64", torch.int64),
                ("qwen4_has_initial", torch.bool),
                ("qsa_compressed_slots", torch.int64),
            ):
                graph_buffers[name] = torch.empty(
                    num_tokens, device=device, dtype=dtype
                )
            physical_block_size = int(self.kv_cache.get_seq_size_per_block("full"))
            max_physical_blocks = (
                int(self._cg_max_seq_len) + physical_block_size - 1
            ) // physical_block_size + num_tokens // max_batch
            graph_buffers["physical_block_table_i32"] = torch.empty(
                max_batch,
                max_physical_blocks,
                device=device,
                dtype=torch.int32,
            )

        # The same draft runtime is wrapped by both the Q-token draft-prefill
        # graph and the one-token draft-decode graph. Never replace the decode
        # geometry in the parent's buffers with Q-strided request indices.
        add_qsa_buffers(buffers, max_batch, request_capacity)
        if mtp_tokens_per_request == 1:
            return
        max_tokens = max_batch * mtp_tokens_per_request
        parent_table = buffers["block_table_i32"]
        table_width = parent_table.shape[1] + mtp_tokens_per_request - 1
        # RTP's MTP decode graph can carry Q reserved state columns even
        # though it has only one token per request. Keep its indices 1-token
        # wide, but give the table enough columns for the upstream copy.
        buffers["block_table_i32"] = torch.empty(
            max_batch, table_width, device=device, dtype=torch.int32
        )
        mtp_buffers = buffers.copy()
        mtp_buffers["mtp_window_width"] = mtp_tokens_per_request
        mtp_buffers["block_table_i32"] = torch.empty(
            max_batch,
            table_width,
            device=device,
            dtype=torch.int32,
        )
        mtp_buffers["query_start_loc"] = torch.arange(
            0,
            max_tokens + 1,
            mtp_tokens_per_request,
            device=device,
            dtype=torch.int32,
        )
        mtp_buffers["seq_id"] = torch.arange(
            max_batch, device=device, dtype=torch.int64
        ).repeat_interleave(mtp_tokens_per_request)
        for name, dtype in (
            ("block_col", torch.int32),
            ("block_col_i64", torch.int64),
            ("slot_base", torch.int32),
            ("token_offset", torch.int32),
            ("slot_mapping", torch.int64),
        ):
            mtp_buffers[name] = torch.empty(max_tokens, device=device, dtype=dtype)
        add_qsa_buffers(
            mtp_buffers, max_tokens, request_capacity * mtp_tokens_per_request
        )
        if is_target:
            # One flag per index-kernel tile; each replay overwrites it.
            # Graph padding (reserved block 0) is not counted.
            mtp_buffers["target_invalid_live_rows"] = torch.empty(
                ((max_batch + 127) // 128,), device=device, dtype=torch.int32
            )
            mtp_buffers["target_read_slots_i32"] = torch.empty(
                (max_batch, mtp_tokens_per_request), device=device, dtype=torch.int32
            )
            mtp_buffers["target_write_slots_i32"] = torch.empty(
                (max_batch, mtp_tokens_per_request), device=device, dtype=torch.int32
            )
            mtp_buffers["target_sequence_masks"] = torch.ones(
                max_batch, device=device, dtype=torch.bool
            )
            mtp_buffers["target_accepted_tokens"] = torch.ones(
                max_batch, device=device, dtype=torch.int32
            )
        self._cg_mtp_meta_bufs = mtp_buffers

    def _graph_buffers_for_inputs(self, attn_inputs):
        if getattr(attn_inputs, "is_target_verify", False) or (
            getattr(getattr(self, "_rtp_model_config", None), "is_mtp", False)
            and getattr(attn_inputs, "is_prefill", False)
        ):
            buffers = getattr(self, "_cg_mtp_meta_bufs", None)
            if buffers is None and (
                getattr(attn_inputs, "is_cuda_graph", False)
                or torch.cuda.is_current_stream_capturing()
            ):
                raise RuntimeError("Qwen4Exp MTP Graph buffers were not prewarmed")
            return buffers
        return getattr(self, "_cg_meta_bufs", None)

    def forward(self, inputs, fmha_impl=None):
        tagged = inputs.attention_inputs
        if isinstance(tagged, dict):
            if "linear" not in tagged or "full" not in tagged:
                raise ValueError(
                    f"Qwen4Exp requires linear/full cache tags, got {list(tagged)}"
                )
            # RTP exposes hybrid attention as a tag map. The Qwen3.5 bridge
            # expects one PyAttentionInputs; use the recurrent-state tag for
            # its GDN metadata and retain the full tag for QSA below.
            inputs = SimpleNamespace(
                input_ids=inputs.input_ids,
                input_hiddens=inputs.input_hiddens,
                combo_position_ids=inputs.combo_position_ids,
                embedding_inputs=inputs.embedding_inputs,
                multimodal_inputs=inputs.multimodal_inputs,
                attention_inputs=tagged["linear"],
                attention_inputs_by_tag=tagged,
                bert_embedding_inputs=inputs.bert_embedding_inputs,
            )
        if bool(getattr(fmha_impl, "is_cuda_graph", False)):
            inputs.attention_inputs.is_cuda_graph = True
        from atom.config import use_custom_atom_config

        with use_custom_atom_config(self.model.atom_config):
            if bool(
                getattr(
                    getattr(
                        getattr(self, "_rtp_model_config", None),
                        "mm_model_config",
                        None,
                    ),
                    "is_multimodal",
                    False,
                )
            ):
                outputs = self._forward_with_vision_positions(inputs)
            else:
                outputs = super().forward(inputs, fmha_impl)
        if not getattr(self._rtp_model_config, "atom_qwen4_exp_mtp_target", False):
            return outputs
        hc = outputs.hidden_states.flatten(1)
        mixed, _ = self.model.model.hyper_connection_mixer.mix(hc)
        return PyModelOutputs(mixed, hc)

    def _forward_with_vision_positions(self, inputs):
        from rtp_llm.models_py.modules.base.common.multimodal_embedding import (
            MultimodalEmbeddingInjector,
        )

        model_device = self._get_model_device()
        input_ids = inputs.input_ids
        if input_ids is not None and input_ids.numel() > 0:
            input_ids = input_ids.to(device=model_device, non_blocking=True)
        token_num = self._get_token_num(inputs, input_ids)
        logical_positions = self._extract_positions(inputs, model_device, token_num)
        rope_positions = self._rope_positions(inputs, model_device, token_num)
        features = inputs.multimodal_inputs.multimodal_features
        inputs_embeds = None
        if input_ids is None or input_ids.numel() == 0:
            inputs_embeds = inputs.input_hiddens
            if inputs_embeds is not None and inputs_embeds.numel() > 0:
                inputs_embeds = inputs_embeds.to(
                    device=model_device,
                    dtype=self._get_model_dtype(),
                    non_blocking=True,
                )
        elif features:
            inputs_embeds = MultimodalEmbeddingInjector()(
                self._embed_vision_input_ids(input_ids),
                features,
                inputs.multimodal_inputs.mm_features_locs,
            )
            # RTP uses out-of-vocabulary placeholders for image patch rows.
            # Embeddings are replaced above, but PLE also hashes input_ids and
            # persists its n-gram history across decode steps. Match ATOM's
            # canonical image token there without altering RTP's input tensor.
            input_ids = self._canonical_vision_input_ids(
                input_ids,
                features,
                inputs.multimodal_inputs.mm_features_locs,
            )

        with self._rtp_forward_context_cls.bind(
            model=self.model,
            runtime=self,
            inputs=inputs,
            positions=logical_positions,
            layer_maps=self._rtp_layer_maps,
            cg_max_seq_len=int(self._cg_max_seq_len),
            cg_bufs=self._graph_buffers_for_inputs(inputs.attention_inputs),
        ):
            hidden_states = self.model(
                input_ids=input_ids,
                positions=rope_positions,
                intermediate_tensors=None,
                inputs_embeds=inputs_embeds,
            )
        return PyModelOutputs(hidden_states)

    def _embed_vision_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        # RTP uses out-of-vocabulary placeholder IDs for image patches. TP>1
        # masks them, but TP=1 uses F.embedding and would fail before the image
        # features replace those positions. The existing masked implementation
        # preserves zero embeddings for placeholders in both configurations.
        if self.model.embed_tokens.tp_size == 1:
            from atom.model_ops.embed_head import replicated_embedding

            return replicated_embedding(input_ids, self.model.embed_tokens.weight)
        return self.model.embed_input_ids(input_ids)

    def _canonical_vision_input_ids(self, input_ids, features, locations):
        image_token_id = self._image_token_id
        canonical = input_ids.clone()
        locs = locations.to(device="cpu", dtype=torch.long).view(-1).tolist()
        if len(locs) != len(features):
            raise ValueError("Qwen4Exp vision feature locations do not match features")
        for feature, loc in zip(features, locs):
            if feature is not None and feature.numel():
                canonical.narrow(0, loc, feature.shape[0]).fill_(image_token_id)
        return canonical

    @staticmethod
    def _read_image_token_id(ckpt_path: str) -> int:
        with open(os.path.join(ckpt_path, "config.json"), encoding="utf-8") as file:
            return int(json.load(file)["image_token_id"])

    def _extract_positions(self, inputs, model_device, token_num):
        # The draft consumes target HC states, but its QSA cache still uses
        # logical token positions. MRoPE's T row can differ after an image.
        is_multimodal = bool(
            getattr(
                getattr(
                    getattr(self, "_rtp_model_config", None), "mm_model_config", None
                ),
                "is_multimodal",
                False,
            )
        )
        attn_inputs = inputs.attention_inputs
        is_target_verify = bool(getattr(attn_inputs, "is_target_verify", False))
        is_mtp_draft_prefill = bool(
            getattr(getattr(self, "_rtp_model_config", None), "is_mtp", False)
            and attn_inputs.is_prefill
        )
        if (
            not attn_inputs.is_prefill or is_target_verify or is_mtp_draft_prefill
        ) and (
            torch.cuda.is_current_stream_capturing()
            or getattr(attn_inputs, "is_cuda_graph", False)
        ):
            # RTP's BERT position buffer has max_seq_len * batch capacity and
            # is not refreshed for decode replay. The device sequence length
            # is refreshed every step, including when the graph is replayed.
            lengths = (
                attn_inputs.prefix_lengths_device
                if is_mtp_draft_prefill
                else attn_inputs.sequence_lengths_plus_1_device
            )
            tokens_per_request = (
                int(self._rtp_model_config.gen_num_per_cycle) + 1
                if is_target_verify or is_mtp_draft_prefill
                else 1
            )
            if lengths is None or lengths.numel() * tokens_per_request != token_num:
                raise ValueError(
                    "Qwen4Exp Graph requires per-row device sequence lengths"
                )
            graph_buffers = self._graph_buffers_for_inputs(attn_inputs)
            if graph_buffers is not None:
                if graph_buffers["qwen4_positions_i32"].numel() < token_num:
                    raise ValueError(
                        "Qwen4Exp Graph position buffer is too small: "
                        f"capacity={graph_buffers['qwen4_positions_i32'].numel()}, "
                        f"tokens={token_num}, rows={lengths.numel()}, "
                        f"window={tokens_per_request}"
                    )
                positions = graph_buffers["qwen4_positions_i32"][:token_num]
                if is_mtp_draft_prefill or is_target_verify:
                    base = lengths if is_mtp_draft_prefill else lengths - 1
                    for token in range(tokens_per_request):
                        torch.add(
                            base,
                            token,
                            out=positions[token::tokens_per_request],
                        )
                else:
                    torch.sub(lengths, 1, out=positions)
                return positions
            if tokens_per_request > 1:
                raise RuntimeError("Qwen4Exp MTP Graph requires prewarmed positions")
            return lengths.to(device=model_device, dtype=torch.int32) - 1
        if is_multimodal:
            if is_mtp_draft_prefill:
                starts = attn_inputs.prefix_lengths_device
                width = int(self._rtp_model_config.gen_num_per_cycle) + 1
                query_lengths = getattr(attn_inputs, "input_lengths", None)
                fixed_window = (
                    starts is not None
                    and query_lengths is not None
                    and starts.numel() * width == token_num
                    and query_lengths.numel() == starts.numel()
                    and bool(
                        torch.all(
                            (query_lengths >= 0) & (query_lengths <= width)
                        ).item()
                    )
                )
                if fixed_window:
                    logical = (
                        starts.to(device=model_device, dtype=torch.int32)[:, None]
                        + torch.arange(width, device=model_device, dtype=torch.int32)[
                            None, :
                        ]
                    ).reshape(-1)
                else:
                    # Initial MTP prompt prefill is ragged, not a fixed
                    # verification window. Use actual per-request lengths.
                    logical = self._logical_positions(attn_inputs, model_device)
            else:
                logical = self._logical_positions(attn_inputs, model_device)
            if logical is None or logical.numel() != token_num:
                raise ValueError(
                    "Qwen4Exp requires one logical cache position per token"
                )
            return logical
        raw_positions = getattr(attn_inputs, "combo_position_ids", None)
        if raw_positions is None or raw_positions.numel() == 0:
            raw_positions = getattr(inputs, "combo_position_ids", None)
        if (
            raw_positions is not None
            and token_num > 0
            and raw_positions.numel() == 3 * token_num
        ):
            return self._parse_rtp_mrope_positions(
                raw_positions, model_device, token_num
            )[0]
        positions = super()._extract_positions(inputs, model_device, token_num)
        # The text QSA path needs one logical position per token. RTP stores
        # MRoPE triples in token-major order, also for text-only checkpoints.
        if token_num > 0 and positions.numel() == 3 * token_num:
            positions = self._parse_rtp_mrope_positions(
                positions, model_device, token_num
            )[0]
        return positions.contiguous()

    def _logical_positions(self, attn_inputs, model_device):
        is_target_verify = bool(getattr(attn_inputs, "is_target_verify", False))
        if is_target_verify:
            lengths = attn_inputs.sequence_lengths_plus_1_device
            if lengths is None:
                return None
            width = int(self._rtp_model_config.gen_num_per_cycle) + 1
            base = lengths.to(device=model_device, dtype=torch.int32) - 1
            return (
                base[:, None]
                + torch.arange(width, device=model_device, dtype=torch.int32)[None, :]
            ).reshape(-1)
        if not attn_inputs.is_prefill:
            next_lengths = getattr(attn_inputs, "sequence_lengths_plus_1_device", None)
            if next_lengths is not None and next_lengths.numel() > 0:
                return (
                    next_lengths.to(device=model_device, dtype=torch.int32) - 1
                ).contiguous()
        return self._build_positions_from_attention_inputs(attn_inputs, model_device)

    @staticmethod
    def _parse_rtp_mrope_positions(positions, model_device, token_num):
        positions = positions.to(device=model_device, dtype=torch.int32)
        # RTP's gatherer uses [token, T/H/W]. Prioritize that documented
        # layout when N == 3 makes both 2-D shapes superficially identical.
        if positions.ndim == 2 and positions.shape == (token_num, 3):
            return positions.transpose(0, 1).contiguous()
        if positions.ndim == 2 and positions.shape == (3, token_num):
            return positions.contiguous()
        if positions.numel() != 3 * token_num:
            raise ValueError("Qwen4Exp MRoPE requires three positions per token")
        return positions.reshape(token_num, 3).transpose(0, 1).contiguous()

    @staticmethod
    def _rope_positions(inputs, model_device, token_num):
        attn_inputs = inputs.attention_inputs
        # RTP's CudaGraphRunner::updateInputs explicitly copies the live
        # combo_position_ids into the captured buffer before every replay.
        # Unlike its BERT position buffer, these MRoPE rows are not frozen.
        positions = getattr(attn_inputs, "combo_position_ids", None)
        if positions is None or positions.numel() == 0:
            positions = getattr(inputs, "combo_position_ids", None)
        if positions is None or positions.numel() == 0:
            raise ValueError("Qwen4Exp vision requires RTP MRoPE position IDs")
        return _ATOMQwen4ExpRuntime._parse_rtp_mrope_positions(
            positions, model_device, token_num
        )


class ATOMQwen4Exp(ATOMQwen35Moe):
    """Use ATOM's Qwen4Exp compute with RTP scheduling and token sampling."""

    @classmethod
    def _create_config(cls, ckpt_path: str) -> ModelConfig:
        with open(os.path.join(ckpt_path, "config.json"), encoding="utf-8") as file:
            root = json.load(file)
        text = root["text_config"]
        config = ModelConfig()
        config.ckpt_path = ckpt_path
        if text.get("dtype", root.get("dtype", "bfloat16")) != "bfloat16":
            raise ValueError("Qwen4Exp RTP plugin currently requires BF16 weights")
        # RTP's init_precision_config recomputes data_type from config_dtype.
        # Setting data_type alone is discarded and silently reverts to FP16.
        config.config_dtype = "bf16"
        config.data_type = "bf16"
        config.attn_config.head_num = int(text["num_attention_heads"])
        config.attn_config.kv_head_num = int(text["num_key_value_heads"])
        config.attn_config.size_per_head = int(text["head_dim"])
        config.num_layers = int(text["num_hidden_layers"])
        config.hidden_size = int(text["hidden_size"])
        config.hc_mult = int(text["hc_count"])
        config.vocab_size = int(text["vocab_size"])
        config.max_seq_len = int(text["max_position_embeddings"])
        config.tie_word_embeddings = bool(text.get("tie_word_embeddings", False))

        rope = text["rope_parameters"]
        config.attn_config.rope_config.style = 1
        config.attn_config.rope_config.base = float(rope["rope_theta"])
        config.partial_rotary_factor = float(rope["partial_rotary_factor"])
        config.attn_config.rope_config.dim = int(
            config.attn_config.size_per_head * config.partial_rotary_factor
        )
        config.layernorm_eps = float(text["rms_norm_eps"])
        config.norm_type = "rmsnorm"
        config.has_pre_decoder_layernorm = False
        # Qwen3.8 ends in the hyper-connection mixer, not a separate final RMSNorm.
        config.has_post_decoder_layernorm = False
        config.activation_type = "SiGLU"

        if root.get("language_model_only", True) is False:
            from rtp_llm.models.mrope_utils import apply_mrope_section

            config.attn_config.rope_config.style = 7
            apply_mrope_section(
                config.attn_config.rope_config,
                rope["mrope_section"],
                model_name="Qwen4Exp",
                interleaved=bool(rope.get("mrope_interleaved", True)),
            )
            config.mm_model_config.is_multimodal = True
            config.mm_model_config.mm_position_ids_style = 2
            config.mm_model_config.mm_sep_tokens = [
                [int(root["vision_start_token_id"]), int(root["vision_end_token_id"])]
            ]
            config.mm_related_params.config["ckpt_path"] = ckpt_path

        config.moe_k = int(text["num_experts_per_tok"])
        config.expert_num = int(text["num_experts"])
        config.moe_inter_size = int(text["moe_intermediate_size"])
        config.inter_size = int(text.get("shared_expert_intermediate_size", 0))
        config.n_shared_experts = int(config.inter_size > 0)
        config.has_moe_norm = bool(text.get("norm_topk_prob", True))
        config.moe_style = 2 if config.n_shared_experts else 1
        config.moe_layer_index = list(range(config.num_layers))

        config.hybrid_attention_config.enable_hybrid_attention = True
        config.hybrid_attention_config.hybrid_attention_types = [
            (
                HybridAttentionType.LINEAR
                if layer == "linear_attention"
                else HybridAttentionType.NONE
            )
            for layer in text["layer_types"]
        ]
        linear = config.linear_attention_config
        for key in (
            "linear_conv_kernel_dim",
            "linear_key_head_dim",
            "linear_num_key_heads",
            "linear_num_value_heads",
            "linear_value_head_dim",
        ):
            setattr(linear, key, int(text[key]))
        linear.ssm_state_dtype = ssm_state_dtype_str_to_data_type(
            text.get("mamba_ssm_dtype", "float32")
        )
        return config

    @classmethod
    def _post_build_model_config(cls, model_config: ModelConfig) -> None:
        model_config.kv_cache_spec_descs = build_hybrid_kv_cache_spec_descs(
            model_config.hybrid_attention_config.hybrid_attention_types,
            KVCacheSpecType.MHA,
        )

    @classmethod
    def _apply_kv_cache_config(cls, model_config: ModelConfig, kv_cache_config) -> None:
        """Budget QSA/PLE side state before RTP sizes its paged cache pools."""
        with open(
            os.path.join(model_config.ckpt_path, "config.json"), encoding="utf-8"
        ) as file:
            text = json.load(file)["text_config"]
        block_size = int(
            model_config.attn_config.tokens_per_block
            or kv_cache_config.seq_size_per_block
        )
        ratio = int(text["indexer_compress_ratio"])
        index_dim = int(text["indexer_head_dim"])
        layer_types = (
            text.get("mtp", {}).get("layer_types", [])
            if model_config.is_mtp
            else text["layer_types"]
        )
        if len(layer_types) != model_config.num_layers:
            raise ValueError("Qwen4Exp QSA layer count differs from RTP model layers")
        full_layers = sum(layer == "full_attention" for layer in layer_types)
        if block_size <= 0 or ratio <= 0 or block_size % ratio or index_dim <= 0:
            raise ValueError(
                "Qwen4Exp QSA cache requires positive, ratio-aligned block geometry"
            )
        if getattr(kv_cache_config, "multi_task_prompt", None) or getattr(
            kv_cache_config, "multi_task_prompt_str", None
        ):
            raise ValueError("Qwen4Exp RTP plugin does not support cached task prompts")
        # RTP's plugin block-copy callback must accompany reuse_cache so QSA/PLE
        # side pools follow copy-on-write relocations.
        model_config.atom_qwen4_exp_prefix_cache_enabled = bool(
            kv_cache_config.reuse_cache
        )

        # _bind_qsa_caches allocates both BF16 arrays for *every* FULL block
        # on this rank: [block, 1, dim] and [block/ratio, 1, dim].
        qsa_bytes = full_layers * (block_size + block_size // ratio) * index_dim * 2

        # _ple_metadata allocates one n-gram and one convolution window for
        # every LINEAR state slot. Both pools are also outside RTP's KV pool.
        ple_layer_ids = [] if model_config.is_mtp else text.get("ple_layer_ids", [])
        if len(ple_layer_ids) > 1:
            raise ValueError("Qwen4Exp RTP plugin currently supports one PLE layer")
        ple_bytes = 0
        if ple_layer_ids:
            ngram_size = int(text["ngram_size"])
            conv_state_len = (int(text["ple_conv_kernel_size"]) - 1) * ngram_size
            hc_hidden_size = int(text["hidden_size"]) * int(text["hc_count"])
            if ngram_size < 2 or conv_state_len < 0 or hc_hidden_size <= 0:
                raise ValueError("Qwen4Exp PLE cache geometry is invalid")
            ple_bytes = (ngram_size - 1) * 8 + (
                hc_hidden_size
                * conv_state_len
                * torch.empty((), dtype=model_config.compute_dtype).element_size()
            )
        # QSA needs the three-axis RoPE position of every image token when
        # compressing index keys. This side cache follows RTP's physical blocks.
        rope_bytes = (
            full_layers
            * block_size
            * 3
            * torch.empty((), dtype=torch.int64).element_size()
            if model_config.mm_model_config.is_multimodal
            else 0
        )
        model_config.auxiliary_paged_cache_bytes_per_block = (
            qsa_bytes + ple_bytes + rope_bytes
        )

    def _create_python_model(self):
        from atom.model_loader.loader import WeightsMapper, load_model_in_plugin_mode
        from atom.plugin.prepare import _set_framework_backbone, prepare_model
        from atom.plugin.rtpllm.utils.topk_compat import install_rtp_topk_compat

        target_device = torch.device(self.device)
        target_dtype = self.model_config.compute_dtype
        old_dtype = torch.get_default_dtype()
        old_device = torch.get_default_device()
        try:
            torch.set_default_device(target_device)
            torch.set_default_dtype(target_dtype)
            _set_framework_backbone("rtpllm")
            install_rtp_topk_compat()
            atom_model = prepare_model(config=self, engine="rtpllm")
            atom_model = atom_model.to(target_device)
            if getattr(self.model_config, "atom_qwen4_exp_mtp_target", False):
                from atom.plugin.rtpllm.models.qwen4_exp_mtp_hooks import (
                    bind_qwen4_exp_mtp_hooks,
                )

                bind_qwen4_exp_mtp_hooks(atom_model)
                atom_model.model.return_hc_state = True
            if getattr(self.model_config, "atom_qwen4_exp_prefix_cache_enabled", False):
                from atom.plugin.rtpllm.utils.gdn_checkpoint import (
                    install_rtp_gdn_checkpoint_writer,
                    install_rtp_native_prefill_snapshot,
                )
                from atom.plugin.rtpllm.utils.qwen4_exp_prefix_cache import (
                    bind_ple_checkpoint_hooks,
                )

                install_rtp_gdn_checkpoint_writer()
                install_rtp_native_prefill_snapshot()
                bind_ple_checkpoint_hooks(atom_model)
            load_model_in_plugin_mode(
                model=atom_model,
                config=atom_model.atom_config,
                prefix="model.",
                weights_mapper=WeightsMapper(
                    orig_to_new_prefix={"model.language_model.": "model."}
                ),
            )
            named = dict(atom_model.named_parameters(remove_duplicate=False))
            required = {
                W.embedding: "model.embed_tokens.weight",
                W.lm_head: "lm_head.weight",
            }
            for key, name in required.items():
                param = named.get(name)
                if param is None:
                    raise RuntimeError(f"Qwen4Exp checkpoint did not load {name}")
                self.weight.set_global_weight(key, param.detach())
        finally:
            torch.set_default_dtype(old_dtype)
            torch.set_default_device(old_device)

        self.py_model = _ATOMQwen4ExpRuntime(
            model_config=self.model_config,
            parallelism_config=self.parallelism_config,
            weights=self.weight,
            max_generate_batch_size=self.max_generate_batch_size,
            fmha_config=self.fmha_config,
            py_hw_kernel_config=self.hw_kernel_config,
            device_resource_config=self.device_resource_config,
            atom_model=atom_model,
        )
        logger.info("Created ATOM Qwen4Exp runtime for RTP plugin mode")
        return self.py_model


class _ATOMQwen4ExpMTPRuntime(_ATOMQwen4ExpRuntime):
    """Bridge RTP's flat HC hand-off to ATOM's reusable QSA draft layer."""

    @staticmethod
    def _draft_vision_embeddings(input_ids, embed_tokens, is_multimodal):
        if not is_multimodal or embed_tokens.tp_size != 1:
            return None
        # RTP's speculative padding can contain -1. The TP>1 embedding
        # masks invalid IDs, but TP=1 otherwise uses a raw gather and faults
        # before the padding row is skipped.
        from atom.model_ops.embed_head import replicated_embedding

        return replicated_embedding(input_ids, embed_tokens.weight)

    def prepare_fmha_impl(self, inputs, is_cuda_graph: bool = False):
        tagged = inputs.attention_inputs
        if isinstance(tagged, dict):
            if "full" not in tagged:
                raise ValueError("Qwen4Exp MTP requires the full cache tag")
            if is_cuda_graph:
                tagged["full"].is_cuda_graph = True
            inputs = SimpleNamespace(attention_inputs=tagged["full"])
        return _ATOMQwen35MoeRuntime.prepare_fmha_impl(self, inputs, is_cuda_graph)

    def forward(self, inputs, fmha_impl=None):
        tagged = inputs.attention_inputs
        if isinstance(tagged, dict):
            if "full" not in tagged:
                raise ValueError("Qwen4Exp MTP requires the full cache tag")
            inputs = SimpleNamespace(
                input_ids=inputs.input_ids,
                input_hiddens=inputs.input_hiddens,
                combo_position_ids=inputs.combo_position_ids,
                embedding_inputs=inputs.embedding_inputs,
                multimodal_inputs=inputs.multimodal_inputs,
                attention_inputs=tagged["full"],
                attention_inputs_by_tag=tagged,
                bert_embedding_inputs=inputs.bert_embedding_inputs,
            )
        else:
            # PyModelInputs is a fixed pybind class, not a dynamic namespace.
            inputs = SimpleNamespace(
                input_ids=inputs.input_ids,
                input_hiddens=inputs.input_hiddens,
                combo_position_ids=inputs.combo_position_ids,
                embedding_inputs=inputs.embedding_inputs,
                multimodal_inputs=inputs.multimodal_inputs,
                attention_inputs=tagged,
                attention_inputs_by_tag={"full": tagged},
                bert_embedding_inputs=inputs.bert_embedding_inputs,
            )
        if bool(getattr(fmha_impl, "is_cuda_graph", False)):
            inputs.attention_inputs.is_cuda_graph = True
        input_ids = inputs.input_ids
        model_device = self._get_model_device()
        if input_ids.device != model_device:
            input_ids = input_ids.to(device=model_device, non_blocking=True)
        is_multimodal = bool(
            getattr(
                getattr(self._rtp_model_config, "mm_model_config", None),
                "is_multimodal",
                False,
            )
        )
        if is_multimodal:
            # The draft does not run ViT, but it still embeds RTP's image
            # placeholder rows and hashes their IDs in its own PLE state.
            image_token_id = self._image_token_id
            input_ids = torch.where(
                input_ids >= self._rtp_model_config.vocab_size,
                image_token_id,
                input_ids,
            )
        hc_input = inputs.input_hiddens
        if hc_input is None or hc_input.numel() == 0:
            raise ValueError("Qwen4Exp MTP requires target HC hidden states")
        expected_width = int(
            self._rtp_model_config.hidden_size * self._rtp_model_config.hc_mult
        )
        if hc_input.ndim != 2 or hc_input.shape != (input_ids.numel(), expected_width):
            raise ValueError(
                "Qwen4Exp MTP HC input must have shape "
                f"({input_ids.numel()}, {expected_width}), got {tuple(hc_input.shape)}"
            )
        if hc_input.device != model_device:
            hc_input = hc_input.to(device=model_device, non_blocking=True)
        positions = self._extract_positions(inputs, model_device, input_ids.numel())
        rope_positions = (
            self._rope_positions(inputs, model_device, input_ids.numel())
            if is_multimodal
            else positions
        )
        with self._rtp_forward_context_cls.bind(
            model=self.model,
            runtime=self,
            inputs=inputs,
            positions=positions,
            layer_maps=self._rtp_layer_maps,
            cg_max_seq_len=int(self._cg_max_seq_len),
            cg_bufs=self._graph_buffers_for_inputs(inputs.attention_inputs),
        ):
            # QSA cache slots use the RTP logical position; the draft's RoPE
            # rotates one token ahead, as in ATOM's standalone/SGLang MTP.
            from atom.config import use_custom_atom_config
            from atom.utils.forward_context import get_forward_context

            get_forward_context().context.is_draft = True
            with use_custom_atom_config(self.model.atom_config):
                draft_embeds = self._draft_vision_embeddings(
                    input_ids, self.model.model.embed_tokens, is_multimodal
                )
                hc = self.model(
                    input_ids, rope_positions + 1, hc_input, inputs_embeds=draft_embeds
                )
        hc = hc.flatten(1)
        mixed = self.model._head_input(hc)
        return PyModelOutputs(mixed, hc)


class ATOMQwen4ExpMTP(ATOMQwen4Exp):
    """Single reusable QSA draft layer stored in the target checkpoint."""

    @classmethod
    def speculative_weight_alias_names(cls, target_model, draft_model_config):
        if not isinstance(target_model, ATOMQwen4Exp) or isinstance(
            target_model, ATOMQwen4ExpMTP
        ):
            raise TypeError("Qwen4Exp MTP requires an ATOMQwen4Exp target")
        for name in ("ckpt_path", "vocab_size", "hidden_size", "hc_mult", "data_type"):
            if getattr(target_model.model_config, name) != getattr(
                draft_model_config, name
            ):
                raise ValueError(f"Qwen4Exp MTP cannot share {name} with target")
        return (W.embedding, W.lm_head)

    @classmethod
    def _create_config(cls, ckpt_path: str) -> ModelConfig:
        config = super()._create_config(ckpt_path)
        with open(os.path.join(ckpt_path, "config.json"), encoding="utf-8") as file:
            text = json.load(file)["text_config"]
        mtp = text.get("mtp", {})
        if text.get("mtp_num_hidden_layers") != 1 or mtp.get("layer_types") != [
            "full_attention"
        ]:
            raise ValueError("Qwen4Exp MTP requires one full-attention draft layer")
        config.num_layers = 1
        config.hybrid_attention_config.hybrid_attention_types = [
            HybridAttentionType.NONE
        ]
        config.moe_layer_index = [0]
        config.is_mtp = True
        return config

    def _create_python_model(self):
        from atom.model_loader.loader import load_model_in_plugin_mode
        from atom.plugin.prepare import _set_framework_backbone, prepare_model

        owner = self._weight_alias_owner
        if owner is None or owner.py_model is None:
            raise RuntimeError("Qwen4Exp MTP target must be loaded before its draft")
        target = owner.py_model.model
        device = torch.device(self.device)
        old_dtype = torch.get_default_dtype()
        old_device = torch.get_default_device()
        try:
            torch.set_default_device(device)
            torch.set_default_dtype(self.model_config.compute_dtype)
            _set_framework_backbone("rtpllm")
            atom_model = prepare_model(config=self, engine="rtpllm").to(device)
            # Share the target's embedding and head before loading mtp.*.
            # prepare_model already constructed temporary draft parameters,
            # so this avoids persistent duplicates, not peak load-time HBM.
            atom_model.model.embed_tokens.weight = target.model.embed_tokens.weight
            atom_model.lm_head.weight = target.lm_head.weight
            # The draft owns physical RTP cache layer 0; its checkpoint layer
            # remains mtp.layers.0 and its RoPE semantics remain unchanged.
            atom_model.model.layers[0].self_attn.layer_num = 0
            load_model_in_plugin_mode(
                model=atom_model,
                config=atom_model.atom_config,
                prefix="model.",
                spec_decode=True,
            )
            for name in (W.embedding, W.lm_head):
                self.weight.set_global_weight(
                    name, owner.weight.get_global_weight(name)
                )
        finally:
            torch.set_default_dtype(old_dtype)
            torch.set_default_device(old_device)

        self.py_model = _ATOMQwen4ExpMTPRuntime(
            model_config=self.model_config,
            parallelism_config=self.parallelism_config,
            weights=self.weight,
            max_generate_batch_size=self.max_generate_batch_size,
            fmha_config=self.fmha_config,
            py_hw_kernel_config=self.hw_kernel_config,
            device_resource_config=self.device_resource_config,
            atom_model=atom_model,
        )
        logger.info("Created ATOM Qwen4Exp MTP runtime for RTP plugin mode")
        return self.py_model
