"""Text-only Qwen3.8-Flash-Next adapter for RTP-LLM's ATOM plugin mode."""

from __future__ import annotations

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
    def prepare_cuda_graph(self, attn_inputs) -> None:
        if isinstance(attn_inputs, dict):
            attn_inputs = attn_inputs["full"]
        super().prepare_cuda_graph(attn_inputs)


class _ATOMQwen4ExpRuntime(_ATOMQwen35MoeRuntime):
    """Reuse RTP's input/position bridge with QSA and PLE cache metadata."""

    def __init__(self, model_config: ModelConfig, *args, **kwargs) -> None:
        from atom.plugin.rtpllm.utils.qwen4_exp_context import RTPQwen4ExpContext

        self._rtp_forward_context_cls = RTPQwen4ExpContext
        super().__init__(model_config, *args, **kwargs)
        self._rtp_model_config = model_config
        self._atom_attn_pyobj = _Qwen4ExpAttnPyObj(self)

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
        if buffers is None or "qsa_compressed_slots" in buffers:
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
        tokens_per_request = (
            int(self._rtp_model_config.gen_num_per_cycle) + 1
            if is_target or is_mtp_draft
            else 1
        )
        if (is_target or is_mtp_draft) and tokens_per_request not in (2, 3, 4):
            raise ValueError("Qwen4Exp MTP supports 1, 2, or 3 draft steps")
        max_batch = int(self._cg_max_num_tokens)
        max_tokens = max_batch * tokens_per_request
        device = self._get_model_device()
        if tokens_per_request > 1:
            # Target verification and MTP draft prefill capture gamma + 1
            # tokens/request; the parent only prewarms ordinary decode.
            parent_table = buffers["block_table_i32"]
            buffers["block_table_i32"] = torch.empty(
                max_batch,
                parent_table.shape[1] + tokens_per_request - 1,
                device=device,
                dtype=torch.int32,
            )
            buffers["query_start_loc"] = torch.arange(
                0, max_tokens + 1, tokens_per_request, device=device, dtype=torch.int32
            )
            buffers["seq_id"] = torch.arange(
                max_batch, device=device, dtype=torch.int64
            ).repeat_interleave(tokens_per_request)
            for name, dtype in (
                ("block_col", torch.int32),
                ("block_col_i64", torch.int64),
                ("slot_base", torch.int32),
                ("token_offset", torch.int32),
                ("slot_mapping", torch.int64),
            ):
                buffers[name] = torch.empty(max_tokens, device=device, dtype=dtype)
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
        buffers["seq_id_i32"] = buffers["seq_id"].to(torch.int32)
        buffers["qsa_positions_i64"] = torch.empty(
            max_tokens, device=device, dtype=torch.int64
        )
        buffers["qwen4_positions_i32"] = torch.empty(
            max_tokens, device=device, dtype=torch.int32
        )
        buffers["qsa_slots_i64"] = torch.empty(
            max_tokens, device=device, dtype=torch.int64
        )
        buffers["qwen4_has_initial"] = torch.empty(
            max_tokens, device=device, dtype=torch.bool
        )
        buffers["qsa_compressed_slots"] = torch.empty(
            max_tokens, device=device, dtype=torch.int64
        )
        if is_target:
            buffers["target_read_slots_i32"] = torch.empty(
                (max_batch, tokens_per_request), device=device, dtype=torch.int32
            )
            buffers["target_write_slots_i32"] = torch.empty(
                (max_batch, tokens_per_request), device=device, dtype=torch.int32
            )
            buffers["target_sequence_masks"] = torch.ones(
                max_batch, device=device, dtype=torch.bool
            )
            buffers["target_accepted_tokens"] = torch.ones(
                max_batch, device=device, dtype=torch.int32
            )
        kv_cache = self.kv_cache
        physical_block_size = int(kv_cache.get_seq_size_per_block("full"))
        max_physical_blocks = (
            int(self._cg_max_seq_len) + physical_block_size - 1
        ) // physical_block_size + tokens_per_request
        buffers["physical_block_table_i32"] = torch.empty(
            max_tokens, max_physical_blocks, device=device, dtype=torch.int32
        )

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
        from atom.config import use_custom_atom_config

        with use_custom_atom_config(self.model.atom_config):
            outputs = super().forward(inputs, fmha_impl)
        if not getattr(self._rtp_model_config, "atom_qwen4_exp_mtp_target", False):
            return outputs
        hc = outputs.hidden_states.flatten(1)
        self._mtp_target_hidden_states = hc
        mixed, _ = self.model.model.hyper_connection_mixer.mix(hc)
        return PyModelOutputs(mixed, hc)

    def get_mtp_target_hidden_states(self, num_tokens: int):
        hidden = getattr(self, "_mtp_target_hidden_states", None)
        if hidden is None:
            return None
        if num_tokens >= 0 and num_tokens > hidden.shape[0]:
            raise ValueError("Qwen4Exp MTP requested more HC rows than produced")
        return hidden if num_tokens < 0 else hidden[:num_tokens]

    def _extract_positions(self, inputs, model_device, token_num):
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
            graph_buffers = getattr(self, "_cg_meta_bufs", None)
            if graph_buffers is not None:
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
        positions = super()._extract_positions(inputs, model_device, token_num)
        # RTP may package text MRoPE as three equal rows. The QSA slot-index
        # bridge is text-only and needs one logical position per token.
        if positions.ndim == 1 and token_num > 0 and positions.numel() == 3 * token_num:
            positions = positions.reshape(3, token_num)
        if positions.ndim == 2:
            if positions.shape[0] != 3:
                raise ValueError("Qwen4Exp RTP text positions require three MRoPE rows")
            positions = positions[0]
        return positions.contiguous()


class ATOMQwen4Exp(ATOMQwen35Moe):
    """Use ATOM's Qwen4Exp compute with RTP scheduling and token sampling.

    The plugin integration is text-only. Image/video position metadata is a
    separate feature.
    """

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
        # RTP's prefix cache only retains its owned KV/linear-state pools. A
        # prefix hit would skip the QSA index and PLE history writes while
        # reusing their physical slots, so those separate pools cannot yet be
        # restored correctly. Per-request reuse_cache is ANDed with this flag.
        if getattr(kv_cache_config, "multi_task_prompt", None) or getattr(
            kv_cache_config, "multi_task_prompt_str", None
        ):
            raise ValueError("Qwen4Exp RTP plugin does not support cached task prompts")
        kv_cache_config.reuse_cache = False

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
        model_config.auxiliary_paged_cache_bytes_per_block = qsa_bytes + ple_bytes

    def _create_python_model(self):
        from atom.model_loader.loader import WeightsMapper, load_model_in_plugin_mode
        from atom.plugin.prepare import _set_framework_backbone, prepare_model

        target_device = torch.device(self.device)
        target_dtype = self.model_config.compute_dtype
        old_dtype = torch.get_default_dtype()
        old_device = torch.get_default_device()
        try:
            torch.set_default_device(target_device)
            torch.set_default_dtype(target_dtype)
            _set_framework_backbone("rtpllm")
            atom_model = prepare_model(config=self, engine="rtpllm")
            atom_model = atom_model.to(target_device)
            if getattr(self.model_config, "atom_qwen4_exp_mtp_target", False):
                atom_model.model.return_hc_state = True
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
            inputs.attention_inputs_by_tag = {"full": tagged}
        if bool(getattr(fmha_impl, "is_cuda_graph", False)):
            inputs.attention_inputs.is_cuda_graph = True
        input_ids = inputs.input_ids
        model_device = self._get_model_device()
        if input_ids.device != model_device:
            input_ids = input_ids.to(device=model_device, non_blocking=True)
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
        with self._rtp_forward_context_cls.bind(
            model=self.model,
            runtime=self,
            inputs=inputs,
            positions=positions,
            layer_maps=self._rtp_layer_maps,
            cg_max_seq_len=int(self._cg_max_seq_len),
            cg_bufs=getattr(self, "_cg_meta_bufs", None),
        ):
            # QSA cache slots use the RTP logical position; the draft's RoPE
            # rotates one token ahead, as in ATOM's standalone/SGLang MTP.
            from atom.config import use_custom_atom_config

            with use_custom_atom_config(self.model.atom_config):
                hc = self.model(input_ids, positions + 1, hc_input)
        hc = hc.flatten(1)
        self._mtp_target_hidden_states = hc
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
            # The checkpoint has only one embedding and head. Share the target's
            # parameters before loading mtp.* so no duplicate HBM is allocated.
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
