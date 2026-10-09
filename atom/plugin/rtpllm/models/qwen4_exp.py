"""Qwen3.8-Flash-Next (Qwen4Exp) wrapper for rtp-llm external model loading.

RTP only owns scheduling and KV paging. Compute, quantization, GDN, QSA, PLE
and hyper-connections stay on ``atom.models.qwen4_exp``.
"""

from __future__ import annotations

import json
import logging
import os
import sys
from typing import Any

import torch
from rtp_llm.config.model_config import ModelConfig
from rtp_llm.model_loader.model_weight_info import ModelDeployWeightInfo, ModelWeights
from rtp_llm.models.base_model import BaseModel
from rtp_llm.models_py.model_desc.module_base import GptModelBase
from rtp_llm.ops import HybridAttentionType, ParallelismConfig
from rtp_llm.ops.compute_ops import PyModelInputs, PyModelOutputs
from rtp_llm.utils.model_weight import W

logger = logging.getLogger("atom.plugin.rtpllm.models")

_FULL_ATTENTION_TYPES = frozenset({"full_attention", "qwen_sparse_attention"})


class _NoopWeightManager:
    def update(self, req):
        return None


class _NoopModelWeightsLoader:
    _py_eplb = None

    def load_lora_weights(self, adapter_name, lora_path, device):
        logger.warning(
            "No-op model_weights_loader received load_lora_weights(%s, %s, %s); "
            "external plugin mode uses ATOM model weights path only.",
            adapter_name,
            lora_path,
            device,
        )


class _StubWeightInfo(ModelDeployWeightInfo):
    def _get_weight_info(self):
        return []


class _ATOMQwen4ExpAttnPyObj:
    """RTP CudaGraphRunner hook container for Flash QSA / GDN layers."""

    def __init__(self, runtime: "_ATOMQwen4ExpRuntime") -> None:
        self._runtime = runtime
        self.is_cuda_graph = False
        self._qsa_layers: list[Any] = []
        for module in runtime.model.modules():
            if getattr(module, "is_qsa_attention", False):
                self._qsa_layers.append(module)

    @property
    def fmha_params(self):
        return None

    def prepare_cuda_graph(self, attn_inputs) -> None:
        del attn_inputs
        # QSA metadata is rebuilt from RTP attn_inputs inside bind(); the
        # captured graph already writes into the prewarmed buffers.
        return None


class _ATOMQwen4ExpRuntime(GptModelBase):
    """rtp-llm runtime adapter backed by ATOM Qwen4Exp."""

    def __init__(
        self,
        model_config: ModelConfig,
        parallelism_config: ParallelismConfig,
        weights: ModelWeights,
        max_generate_batch_size: int,
        atom_model: Any,
        fmha_config=None,
        py_hw_kernel_config=None,
        device_resource_config=None,
    ) -> None:
        super().__init__(
            model_config,
            parallelism_config,
            weights,
            max_generate_batch_size=max_generate_batch_size,
            fmha_config=fmha_config,
            py_hw_kernel_config=py_hw_kernel_config,
            device_resource_config=device_resource_config,
        )
        self.model = atom_model
        first_param = next(self.model.parameters(), None)
        if first_param is None:
            raise RuntimeError(
                "ATOM Qwen4Exp model has no parameters; cannot determine device/dtype."
            )
        self._model_device = first_param.device
        self._model_dtype = first_param.dtype
        from atom.plugin.rtpllm.utils import RTPForwardQwen4ExpHybridContext

        self._rtp_forward_context_cls = RTPForwardQwen4ExpHybridContext
        self._rtp_layer_maps = self._rtp_forward_context_cls.collect_layer_maps(
            model=self.model
        )
        self._rtp_kv_cache_data: dict | None = None
        self._rtp_kv_cache_signature: tuple | None = None
        self._rtp_layer_group_map: dict[int, int] | None = None
        self._rtp_layer_group_map_signature: tuple | None = None
        self._atom_attn_pyobj: _ATOMQwen4ExpAttnPyObj | None = None
        self._cg_layers_prewarmed: bool = False
        decode_caps = getattr(py_hw_kernel_config, "decode_capture_batch_sizes", None)
        if decode_caps:
            self._cg_max_num_tokens: int = min(
                int(max(decode_caps)), int(max_generate_batch_size)
            )
        else:
            self._cg_max_num_tokens: int = int(max_generate_batch_size)
        self._cg_max_seq_len: int = int(
            getattr(model_config, "max_seq_len", 0)
            or getattr(model_config, "max_position_embeddings", 0)
            or 32768
        )

    def load_weights(self):
        return None

    def _get_model_device(self) -> torch.device:
        return self._model_device

    def _get_model_dtype(self) -> torch.dtype:
        return self._model_dtype

    def _get_token_num(
        self, inputs: PyModelInputs, input_ids: torch.Tensor | None
    ) -> int:
        if input_ids is not None and input_ids.numel() > 0:
            return int(input_ids.numel())
        if inputs.input_hiddens is not None and inputs.input_hiddens.numel() > 0:
            return int(inputs.input_hiddens.shape[0])
        return 0

    @staticmethod
    def _build_token_positions(
        input_lengths: torch.Tensor,
        starts: torch.Tensor,
    ) -> torch.Tensor | None:
        token_starts = torch.repeat_interleave(starts, input_lengths)
        if token_starts.numel() == 0:
            return None
        per_seq_base = input_lengths.cumsum(dim=0) - input_lengths
        token_ordinal = (
            torch.cumsum(
                torch.repeat_interleave(torch.ones_like(input_lengths), input_lengths),
                dim=0,
            )
            - 1
        )
        token_ordinal = token_ordinal - torch.repeat_interleave(
            per_seq_base, input_lengths
        )
        return (token_starts + token_ordinal).to(dtype=torch.int32).contiguous()

    def _build_positions_from_attention_inputs(
        self, attn_inputs: Any, model_device: torch.device
    ) -> torch.Tensor | None:
        if attn_inputs is None:
            return None

        input_lengths = getattr(attn_inputs, "input_lengths", None)
        if input_lengths is None or input_lengths.numel() == 0:
            return None
        input_lengths_i32 = input_lengths.to(
            device=model_device, dtype=torch.int32, non_blocking=True
        ).contiguous()

        is_prefill = bool(getattr(attn_inputs, "is_prefill", False))
        if is_prefill:
            prefix_lengths = getattr(attn_inputs, "prefix_lengths", None)
            if prefix_lengths is None or prefix_lengths.numel() == 0:
                return None
            prefix_lengths_i32 = prefix_lengths.to(
                device=model_device, dtype=torch.int32, non_blocking=True
            ).contiguous()
            if int(prefix_lengths_i32.numel()) < int(input_lengths_i32.numel()):
                return None
            starts = prefix_lengths_i32[: int(input_lengths_i32.numel())]
            return self._build_token_positions(input_lengths_i32, starts)

        sequence_lengths = getattr(attn_inputs, "sequence_lengths", None)
        if sequence_lengths is None or sequence_lengths.numel() == 0:
            return None
        sequence_lengths_i32 = sequence_lengths.to(
            device=model_device, dtype=torch.int32, non_blocking=True
        ).contiguous()
        if int(sequence_lengths_i32.numel()) < int(input_lengths_i32.numel()):
            return None
        starts = (
            sequence_lengths_i32[: int(input_lengths_i32.numel())]
            - input_lengths_i32
            + 1
        )
        return self._build_token_positions(input_lengths_i32, starts)

    def _extract_combo_positions(
        self, inputs: PyModelInputs, model_device: torch.device
    ) -> torch.Tensor | None:
        bert_inputs = getattr(inputs, "bert_embedding_inputs", None)
        if bert_inputs is None:
            return None
        combo_position_ids = getattr(bert_inputs, "combo_position_ids", None)
        if combo_position_ids is None or combo_position_ids.numel() == 0:
            return None
        return combo_position_ids.to(
            device=model_device, dtype=torch.int32, non_blocking=True
        ).contiguous()

    def _extract_positions(
        self, inputs: PyModelInputs, model_device: torch.device, token_num: int
    ) -> torch.Tensor:
        attn_inputs = getattr(inputs, "attention_inputs", None)
        if attn_inputs is None:
            raise ValueError(
                "Qwen4Exp RTP plugin requires inputs.attention_inputs "
                "to provide position metadata."
            )
        positions = getattr(attn_inputs, "combo_position_ids", None)
        if positions is None or positions.numel() == 0:
            positions = self._extract_combo_positions(
                inputs=inputs, model_device=model_device
            )
        if positions is None or positions.numel() == 0:
            positions = self._build_positions_from_attention_inputs(
                attn_inputs=attn_inputs,
                model_device=model_device,
            )
        if positions is None or positions.numel() == 0:
            raise ValueError(
                "Qwen4Exp RTP plugin requires real position metadata from "
                "attention_inputs."
            )
        positions = positions.to(
            device=model_device, dtype=torch.int32, non_blocking=True
        ).contiguous()
        if not torch.cuda.is_current_stream_capturing():
            pos_tokens = (
                int(positions.shape[-1])
                if positions.dim() > 0
                else int(positions.numel())
            )
            if token_num > 0 and pos_tokens != token_num:
                rebuilt_positions = self._build_positions_from_attention_inputs(
                    attn_inputs=attn_inputs,
                    model_device=model_device,
                )
                rebuilt_tokens = (
                    int(rebuilt_positions.shape[-1])
                    if rebuilt_positions is not None and rebuilt_positions.dim() > 0
                    else (
                        int(rebuilt_positions.numel())
                        if rebuilt_positions is not None
                        else -1
                    )
                )
                if rebuilt_positions is not None and rebuilt_tokens == token_num:
                    positions = rebuilt_positions.to(
                        device=model_device, dtype=torch.int32, non_blocking=True
                    ).contiguous()
                elif pos_tokens > token_num:
                    positions = positions[..., -token_num:].contiguous()
                else:
                    raise ValueError(
                        "Qwen4Exp RTP plugin combo_position_ids/token_num mismatch "
                        f"(combo_position_ids_tokens={pos_tokens}, token_num={token_num})."
                    )
        return positions

    def prepare_fmha_impl(
        self, inputs: PyModelInputs, is_cuda_graph: bool = False
    ) -> Any:
        if self._atom_attn_pyobj is None:
            self._atom_attn_pyobj = _ATOMQwen4ExpAttnPyObj(self)
        self._atom_attn_pyobj.is_cuda_graph = bool(is_cuda_graph)
        if bool(is_cuda_graph):
            inputs.attention_inputs.is_cuda_graph = True
            self._ensure_cuda_graph_prewarmed()
        return self._atom_attn_pyobj

    def _qsa_layers(self) -> list[Any]:
        if self._atom_attn_pyobj is not None:
            return self._atom_attn_pyobj._qsa_layers
        return [
            module
            for module in self.model.modules()
            if getattr(module, "is_qsa_attention", False)
        ]

    def _infer_num_blocks(self, seq_size_per_block: int) -> int:
        kv_cache = getattr(self, "kv_cache", None)
        for layer in self._qsa_layers():
            if kv_cache is None:
                break
            try:
                layer_cache = kv_cache.get_layer_cache(int(layer.layer_num))
            except Exception:
                continue
            base = getattr(layer_cache, "kv_cache_base", None)
            if base is not None and base.dim() >= 1 and int(base.shape[0]) > 0:
                return int(base.shape[0])
        max_seq_len = int(self._cg_max_seq_len)
        block = max(int(seq_size_per_block), 1)
        return (max_seq_len + block - 1) // block + 1

    def _ensure_qsa_and_ple_states(self) -> None:
        """Allocate Flash side caches that RTP's hybrid KV pool does not own."""
        if getattr(self, "_qsa_ple_ready", False):
            return
        from atom.config import get_current_atom_config

        atom_config = get_current_atom_config()
        hf = getattr(atom_config, "hf_config", None)
        if hf is None:
            return
        device = self._get_model_device()
        dtype = self._get_model_dtype()
        kv_cache = getattr(self, "kv_cache", None)
        _kv_tags = list(getattr(kv_cache, "group_tags", None) or []) if kv_cache else []
        _kv_tag = _kv_tags[0] if _kv_tags else "full"
        if kv_cache is not None and hasattr(kv_cache, "get_seq_size_per_block"):
            seq_size_per_block = int(kv_cache.get_seq_size_per_block(_kv_tag)) or 16
        else:
            seq_size_per_block = (
                int(getattr(kv_cache, "seq_size_per_block", 0)) if kv_cache else 0
            ) or int(getattr(atom_config, "kv_cache_block_size", 16) or 16)
        num_blocks = self._infer_num_blocks(seq_size_per_block)
        compress_ratio = int(getattr(hf, "indexer_compress_ratio", 4) or 4)
        index_head_dim = int(getattr(hf, "indexer_head_dim", 128) or 128)

        for layer in self._qsa_layers():
            if getattr(layer, "k_cache", None) is not None:
                continue
            num_kv_heads = int(layer.num_kv_heads)
            head_dim = int(layer.head_dim)
            layer.bind_caches(
                torch.empty(
                    num_blocks,
                    seq_size_per_block,
                    num_kv_heads,
                    head_dim,
                    device=device,
                    dtype=torch.bfloat16,
                ),
                torch.empty(
                    num_blocks,
                    seq_size_per_block,
                    num_kv_heads,
                    head_dim,
                    device=device,
                    dtype=torch.bfloat16,
                ),
                torch.empty(
                    num_blocks,
                    seq_size_per_block,
                    1,
                    index_head_dim,
                    device=device,
                    dtype=torch.bfloat16,
                ),
                torch.empty(
                    num_blocks,
                    seq_size_per_block // compress_ratio,
                    1,
                    index_head_dim,
                    device=device,
                    dtype=torch.bfloat16,
                ),
                None,
            )

        gdn_slots = 0
        gdn_map = self._rtp_layer_maps[0] if self._rtp_layer_maps else {}
        if kv_cache is not None:
            for layer_num in gdn_map:
                try:
                    layer_cache = kv_cache.get_layer_cache(int(layer_num))
                except Exception:
                    continue
                base = getattr(layer_cache, "kv_cache_base", None)
                if base is not None and base.dim() >= 1:
                    gdn_slots = max(gdn_slots, int(base.shape[0]))
                    break
        if gdn_slots <= 0:
            gdn_slots = max(int(self._cg_max_num_tokens), 1)
        ngram = max(int(getattr(hf, "ngram_size", 3) or 3), 1)
        if getattr(self, "_ple_conv_state", None) is None and getattr(
            hf, "ple_layer_ids", None
        ):
            state_len = (int(getattr(hf, "ple_conv_kernel_size", 4)) - 1) * ngram
            channels = int(hf.hidden_size) * int(getattr(hf, "hc_count", 4))
            self._ple_conv_state = torch.zeros(
                gdn_slots, channels, state_len, device=device, dtype=dtype
            )
            eos = getattr(hf, "eos_token_id", 0)
            eos_id = int(eos[0] if isinstance(eos, (list, tuple)) else eos)
            self._ple_ngram_state = torch.full(
                (gdn_slots, max(ngram - 1, 1)),
                eos_id,
                device=device,
                dtype=torch.int64,
            )
        self._qsa_ple_ready = True
        logger.info(
            "ATOM Qwen4Exp allocated QSA/PLE side state "
            "(qsa_layers=%d, num_blocks=%d, block=%d, gdn_slots=%d)",
            len(self._qsa_layers()),
            num_blocks,
            seq_size_per_block,
            gdn_slots,
        )

    def _ensure_cuda_graph_prewarmed(self) -> None:
        if self._cg_layers_prewarmed:
            return
        max_num_tokens = int(self._cg_max_num_tokens)
        max_seq_len = int(self._cg_max_seq_len)
        if max_num_tokens <= 0 or max_seq_len <= 0:
            logger.warning(
                "ATOM Qwen4Exp cuda-graph prewarm skipped: invalid budget "
                "(max_num_tokens=%d, max_seq_len=%d)",
                max_num_tokens,
                max_seq_len,
            )
            return
        device = self._get_model_device()
        self._ensure_qsa_and_ple_states()
        kv_cache = getattr(self, "kv_cache", None)
        _kv_tags = list(getattr(kv_cache, "group_tags", None) or []) if kv_cache else []
        _kv_tag = _kv_tags[0] if _kv_tags else "full"
        if kv_cache is not None and hasattr(kv_cache, "get_seq_size_per_block"):
            _ks = int(kv_cache.get_kernel_seq_size_per_block(_kv_tag)) or int(
                kv_cache.get_seq_size_per_block(_kv_tag)
            )
        else:
            _ks = int(getattr(kv_cache, "kernel_seq_size_per_block", 0)) or int(
                getattr(kv_cache, "seq_size_per_block", 0)
            )
        kernel_seq_size_per_block = _ks or 1
        max_blocks = (
            int(max_seq_len) + kernel_seq_size_per_block - 1
        ) // kernel_seq_size_per_block + 1
        max_bs = max_num_tokens
        self._cg_meta_bufs: dict = {
            "query_start_loc": torch.arange(
                0, max_bs + 1, device=device, dtype=torch.int32
            ),
            "seq_id": torch.arange(0, max_bs, device=device, dtype=torch.int64),
            "seq_id_i32": torch.arange(0, max_bs, device=device, dtype=torch.int32),
            "block_col": torch.empty(max_bs, device=device, dtype=torch.int32),
            "block_col_i64": torch.empty(max_bs, device=device, dtype=torch.int64),
            "slot_base": torch.empty(max_bs, device=device, dtype=torch.int32),
            "token_offset": torch.empty(max_bs, device=device, dtype=torch.int32),
            "slot_mapping": torch.empty(max_bs, device=device, dtype=torch.int64),
            "seq_lens_i32": torch.empty(max_bs, device=device, dtype=torch.int32),
            "block_table_i32": torch.empty(
                max_bs, max_blocks, device=device, dtype=torch.int32
            ),
            "qsa_token_to_req": torch.empty(max_bs, device=device, dtype=torch.int32),
            "qsa_logical_positions": torch.empty(
                max_bs, device=device, dtype=torch.int64
            ),
            "qsa_compressed_slots": torch.empty(
                max_bs, device=device, dtype=torch.int64
            ),
            "ple_has_initial_state": torch.empty(
                max_bs, device=device, dtype=torch.bool
            ),
        }
        self._cg_layers_prewarmed = True
        logger.info(
            "ATOM Qwen4Exp cuda-graph prewarmed "
            "(max_num_tokens=%d, max_seq_len=%d, qsa_layers=%d)",
            max_num_tokens,
            max_seq_len,
            len(self._qsa_layers()),
        )

    def forward(self, inputs: PyModelInputs, fmha_impl: Any = None) -> PyModelOutputs:
        if bool(getattr(fmha_impl, "is_cuda_graph", False)):
            inputs.attention_inputs.is_cuda_graph = True
        self._ensure_qsa_and_ple_states()
        model_device = self._get_model_device()
        model_dtype = self._get_model_dtype()
        input_ids = inputs.input_ids
        inputs_embeds = None

        if (
            input_ids is not None
            and input_ids.numel() > 0
            and input_ids.device != model_device
        ):
            input_ids = input_ids.to(device=model_device, non_blocking=True)
        token_num = self._get_token_num(inputs=inputs, input_ids=input_ids)
        positions = self._extract_positions(
            inputs=inputs, model_device=model_device, token_num=token_num
        )
        if input_ids is None or input_ids.numel() == 0:
            inputs_embeds = inputs.input_hiddens
            if (
                inputs_embeds is not None
                and inputs_embeds.numel() > 0
                and inputs_embeds.device != model_device
            ):
                inputs_embeds = inputs_embeds.to(device=model_device, non_blocking=True)
            if (
                inputs_embeds is not None
                and inputs_embeds.numel() > 0
                and inputs_embeds.dtype != model_dtype
            ):
                inputs_embeds = inputs_embeds.to(dtype=model_dtype)

        with self._rtp_forward_context_cls.bind(
            model=self.model,
            runtime=self,
            inputs=inputs,
            positions=positions,
            layer_maps=self._rtp_layer_maps,
            cg_max_seq_len=int(self._cg_max_seq_len),
            cg_bufs=getattr(self, "_cg_meta_bufs", None),
        ):
            hidden_states = self.model(
                input_ids=input_ids,
                positions=positions,
                intermediate_tensors=None,
                inputs_embeds=inputs_embeds,
            )
        return PyModelOutputs(hidden_states)


class ATOMQwen4Exp(BaseModel):
    """Qwen3.8-Flash-Next model class that starts ATOM runtime in rtp-llm."""

    @staticmethod
    def _get_external_packages_from_args() -> list[str]:
        option_name = "--external_model_packages"
        argv = sys.argv[1:]
        raw_value = ""

        for idx, token in enumerate(argv):
            if token == option_name:
                if idx + 1 < len(argv) and not argv[idx + 1].startswith("-"):
                    raw_value = argv[idx + 1]
                break
            if token.startswith(f"{option_name}="):
                raw_value = token.split("=", 1)[1]
                break

        if not raw_value:
            return []
        return [item.strip() for item in raw_value.split(",") if item.strip()]

    @staticmethod
    def _is_external_plugin_mode() -> bool:
        target = "atom.plugin.rtpllm.models"
        return target in ATOMQwen4Exp._get_external_packages_from_args()

    @staticmethod
    def get_weight_cls():
        return _StubWeightInfo

    @classmethod
    def _hybrid_attention_types(
        cls, config_json: dict, num_layers: int
    ) -> list[HybridAttentionType]:
        layer_types = config_json.get("layer_types")
        if layer_types:
            if len(layer_types) != num_layers:
                raise ValueError(
                    "Qwen4Exp layer_types length "
                    f"{len(layer_types)} != num_hidden_layers {num_layers}"
                )
            return [
                (
                    HybridAttentionType.NONE
                    if str(layer_type) in _FULL_ATTENTION_TYPES
                    else HybridAttentionType.LINEAR
                )
                for layer_type in layer_types
            ]
        attention_step = int(config_json["full_attention_interval"])
        return [
            (
                HybridAttentionType.NONE
                if (idx + 1) % attention_step == 0
                else HybridAttentionType.LINEAR
            )
            for idx in range(num_layers)
        ]

    @classmethod
    def _create_config(cls, ckpt_path: str) -> ModelConfig:
        config_path = os.path.join(ckpt_path, "config.json")
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"config.json not found in {ckpt_path}")

        with open(config_path) as reader:
            config_json = json.loads(reader.read())
        config_json = config_json["text_config"]

        config = ModelConfig()
        config.ckpt_path = ckpt_path
        config.attn_config.head_num = config_json["num_attention_heads"]
        config.attn_config.kv_head_num = config_json["num_key_value_heads"]
        config.attn_config.size_per_head = config_json["head_dim"]
        config.num_layers = config_json["num_hidden_layers"]
        config.hidden_size = config_json["hidden_size"]
        config.vocab_size = config_json["vocab_size"]
        config.max_seq_len = config_json["max_position_embeddings"]
        config.tie_word_embeddings = config_json.get("tie_word_embeddings", False)

        rope_parameters = config_json["rope_parameters"]
        config.attn_config.rope_config.style = 1
        config.attn_config.rope_config.base = rope_parameters["rope_theta"]
        config.partial_rotary_factor = rope_parameters["partial_rotary_factor"]
        config.attn_config.rope_config.dim = int(
            config.attn_config.size_per_head * config.partial_rotary_factor
        )

        config.layernorm_eps = config_json["rms_norm_eps"]
        config.norm_type = "rmsnorm"
        config.has_pre_decoder_layernorm = False
        # Flash folds the final residual streams through hyper-connection, not
        # a standalone decoder RMSNorm.
        config.has_post_decoder_layernorm = False
        config.qk_norm = True
        config.activation_type = "SiGLU"

        config.moe_k = config_json["num_experts_per_tok"]
        config.expert_num = config_json["num_experts"]
        config.moe_inter_size = config_json["moe_intermediate_size"]
        config.inter_size = config_json["shared_expert_intermediate_size"]
        config.has_moe_norm = config_json.get("norm_topk_prob", True)
        config.moe_style = 2

        moe_step = config_json.get("decoder_sparse_step", 1)
        config.moe_layer_index = [
            idx for idx in range(config.num_layers) if (idx + 1) % moe_step == 0
        ]

        config.hybrid_attention_config.enable_hybrid_attention = True
        config.hybrid_attention_config.hybrid_attention_types = (
            cls._hybrid_attention_types(config_json, config.num_layers)
        )

        config.linear_attention_config.linear_conv_kernel_dim = config_json[
            "linear_conv_kernel_dim"
        ]
        config.linear_attention_config.linear_key_head_dim = config_json[
            "linear_key_head_dim"
        ]
        config.linear_attention_config.linear_num_key_heads = config_json[
            "linear_num_key_heads"
        ]
        config.linear_attention_config.linear_num_value_heads = config_json[
            "linear_num_value_heads"
        ]
        config.linear_attention_config.linear_value_head_dim = config_json[
            "linear_value_head_dim"
        ]
        return config

    def support_cuda_graph(self) -> bool:
        if os.getenv("ENABLE_CUDA_GRAPH", "1") == "0":
            logger.info("ENABLE_CUDA_GRAPH=0 — ATOMQwen4Exp forces eager forward.")
            return False
        return True

    def load(self, skip_python_model: bool = False):
        if self._is_external_plugin_mode():
            self.device = self._get_device_str()
            self.weight = ModelWeights(
                num_layers=self.model_config.num_layers,
                device=self.device,
                dtype=self.model_config.compute_dtype,
            )
            self.model_weights_loader = _NoopModelWeightsLoader()
            self.py_eplb = self.model_weights_loader._py_eplb
            self.weight_manager = _NoopWeightManager()
            if skip_python_model:
                logger.info(
                    "External plugin mode: skip ATOM Qwen4Exp python model creation"
                )
                return
            self._create_python_model()
            logger.info(
                "External plugin mode: use ATOM Qwen4Exp loading path and skip "
                "rtp-llm native load"
            )
            return

        raise RuntimeError("ATOMQwen4Exp is only supported as an RTP external plugin.")

    def _create_python_model(self):
        if not self._is_external_plugin_mode():
            raise RuntimeError(
                "ATOMQwen4Exp is only supported as an RTP external plugin."
            )

        from atom.model_loader.loader import load_model_in_plugin_mode
        from atom.plugin.prepare import _set_framework_backbone, prepare_model

        target_device = torch.device(
            self.device if getattr(self, "device", None) else "cuda"
        )
        target_dtype = self.model_config.compute_dtype
        old_default_dtype = torch.get_default_dtype()
        try:
            old_default_device = torch.get_default_device()
        except AttributeError:
            old_default_device = None

        torch.set_default_device(target_device)
        if target_dtype in {
            torch.float16,
            torch.bfloat16,
            torch.float32,
            torch.float64,
        }:
            torch.set_default_dtype(target_dtype)

        def _get_first_param_tensor(module: Any, name: str) -> torch.Tensor | None:
            if module is None:
                return None
            for p_name, p in module.named_parameters(recurse=True):
                if p_name == name and p is not None:
                    return p
            return None

        def _inject_rtp_projection_weights(atom_model_obj: Any) -> None:
            lm_head_w = _get_first_param_tensor(atom_model_obj, "lm_head.weight")
            if lm_head_w is None:
                lm_head_w = _get_first_param_tensor(
                    atom_model_obj, "language_model.lm_head.weight"
                )
            if lm_head_w is not None:
                self.weight.set_global_weight(W.lm_head, lm_head_w.detach())
                logger.info(
                    "Injected Qwen4Exp runtime lm_head weight for RTP: %s",
                    tuple(lm_head_w.shape),
                )
            else:
                raise ValueError(
                    "Cannot locate Qwen4Exp lm_head.weight for RTP runtime projection."
                )

            emb_w = _get_first_param_tensor(
                atom_model_obj, "model.embed_tokens.weight"
            )
            if emb_w is None:
                emb_w = _get_first_param_tensor(
                    atom_model_obj, "language_model.model.embed_tokens.weight"
                )
            if emb_w is not None:
                self.weight.set_global_weight(W.embedding, emb_w.detach())
                logger.info(
                    "Injected Qwen4Exp runtime embedding weight for RTP: %s",
                    tuple(emb_w.shape),
                )

        def _assert_weights_loaded(atom_model_obj: Any) -> None:
            # Flash has no input_layernorm; hyper-connection carries the norms.
            candidates = [
                "model.embed_tokens.weight",
                "language_model.model.embed_tokens.weight",
                "model.layers.0.linear_attn.A_log",
                "lm_head.weight",
            ]
            weight = None
            used = None
            for name in candidates:
                weight = _get_first_param_tensor(atom_model_obj, name)
                if weight is not None:
                    used = name
                    break
            if weight is None:
                raise ValueError(
                    "Cannot locate Qwen4Exp embed/GDN weights after ATOM load "
                    "in RTP plugin mode."
                )
            weight_cpu = weight.detach().float().reshape(-1).cpu()
            if weight_cpu.numel() == 0 or bool(torch.all(weight_cpu == 0)):
                raise ValueError(
                    f"Loaded Qwen4Exp {used} is all zeros; refusing to run "
                    "with default values."
                )

        try:
            _set_framework_backbone("rtpllm")
            from atom.plugin.rtpllm.attention_backend import (
                apply_attention_gdn_rtpllm_patch,
            )

            apply_attention_gdn_rtpllm_patch()
            atom_model = prepare_model(config=self, engine="rtpllm")
            if atom_model is None:
                raise ValueError(
                    "ATOM failed to create Qwen4Exp model for rtp-llm plugin"
                )

            atom_model = atom_model.to(target_device)

            atom_config = getattr(atom_model, "atom_config", None)
            if atom_config is None:
                atom_config = getattr(
                    getattr(atom_model, "model", None), "atom_config", None
                )
            if atom_config is None:
                raise ValueError(
                    "Cannot get atom_config from prepared Qwen4Exp model in "
                    "rtp-llm plugin mode"
                )

            load_model_in_plugin_mode(
                model=atom_model,
                config=atom_config,
                prefix="model.",
            )
            _assert_weights_loaded(atom_model)
            _inject_rtp_projection_weights(atom_model)
        finally:
            torch.set_default_dtype(old_default_dtype)
            if old_default_device is not None:
                torch.set_default_device(old_default_device)
            else:
                torch.set_default_device("cpu")

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
        logger.info("Created ATOM Qwen4Exp runtime for rtp-llm plugin mode")
        return self.py_model
