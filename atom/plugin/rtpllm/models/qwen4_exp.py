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

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        from atom.plugin.rtpllm.utils.qwen4_exp_context import RTPQwen4ExpContext

        self._rtp_forward_context_cls = RTPQwen4ExpContext
        self._rtp_layer_maps = RTPQwen4ExpContext.collect_layer_maps(self.model)
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
        max_tokens = int(self._cg_max_num_tokens)
        device = self._get_model_device()
        buffers["seq_id_i32"] = torch.arange(
            max_tokens, device=device, dtype=torch.int32
        )
        buffers["qsa_positions_i64"] = torch.empty(
            max_tokens, device=device, dtype=torch.int64
        )
        buffers["qwen4_positions_i32"] = torch.empty(
            max_tokens, device=device, dtype=torch.int32
        )
        buffers["qsa_slots_i64"] = torch.empty(
            max_tokens, device=device, dtype=torch.int64
        )
        buffers["qwen4_previous_slots_i32"] = torch.empty(
            max_tokens, device=device, dtype=torch.int32
        )
        buffers["qwen4_has_initial"] = torch.empty(
            max_tokens, device=device, dtype=torch.bool
        )
        buffers["qsa_compressed_slots"] = torch.empty(
            max_tokens, device=device, dtype=torch.int64
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
        return super().forward(inputs, fmha_impl)

    def _extract_positions(self, inputs, model_device, token_num):
        attn_inputs = inputs.attention_inputs
        if not attn_inputs.is_prefill and (
            torch.cuda.is_current_stream_capturing()
            or getattr(attn_inputs, "is_cuda_graph", False)
        ):
            # RTP's BERT position buffer has max_seq_len * batch capacity and
            # is not refreshed for decode replay. The device sequence length
            # is refreshed every step, including when the graph is replayed.
            lengths = attn_inputs.sequence_lengths_plus_1_device
            if lengths is None or lengths.numel() != token_num:
                raise ValueError(
                    "Qwen4Exp Graph requires per-row device sequence lengths"
                )
            graph_buffers = getattr(self, "_cg_meta_bufs", None)
            if graph_buffers is not None:
                positions = graph_buffers["qwen4_positions_i32"][:token_num]
                torch.sub(lengths, 1, out=positions)
                return positions
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
        config.has_post_decoder_layernorm = True
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

    def support_cuda_graph(self) -> bool:
        return ATOMQwen35Moe.support_cuda_graph(self)

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
            load_model_in_plugin_mode(
                model=atom_model,
                config=atom_model.atom_config,
                prefix="model.",
                weights_mapper=WeightsMapper(
                    orig_to_new_prefix={"model.language_model.": "model."}
                ),
            )
            named = dict(atom_model.named_parameters())
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
