import json
import logging
import os
from contextlib import contextmanager
from typing import Any

import torch
from rtp_llm.config.model_config import ModelConfig
from rtp_llm.model_loader.model_weight_info import ModelDeployWeightInfo, ModelWeights
from rtp_llm.models.base_model import BaseModel
from rtp_llm.models_py.model_desc.module_base import GptModelBase
from rtp_llm.ops import HybridAttentionType, ParallelismConfig
from rtp_llm.ops.compute_ops import PyModelInputs, PyModelOutputs
from rtp_llm.utils.model_weight import W

from atom.model_loader.loader import WeightsMapper
from atom.models.qwen3_5 import (
    detect_fused_expert_format,
    get_fused_expert_mapping,
    load_fused_expert_weights,
)
from atom.plugin.rtpllm.attention_backend import (
    apply_attention_gdn_rtpllm_patch,
    apply_attention_mha_rtpllm_patch,
)
from atom.plugin.rtpllm.models.qwen3_next import apply_qwen3_next_rtpllm_patch
from atom.plugin.rtpllm.utils import RTPForwardContext

logger = logging.getLogger("atom.plugin.rtpllm.models")


class _NoopWeightManager:
    def update(self, req):  # noqa: ANN001
        return None


class _NoopModelWeightsLoader:
    _py_eplb = None

    def load_lora_weights(self, adapter_name, lora_path, device):  # noqa: ANN001
        logger.warning(
            "No-op model_weights_loader received load_lora_weights(%s, %s, %s); "
            "external plugin mode uses ATOM model weights path only.",
            adapter_name,
            lora_path,
            device,
        )
        return None


class _StubWeightInfo(ModelDeployWeightInfo):
    def _get_weight_info(self):
        return []


class _ATOMAttnPyObj:
    """Container returned by _ATOMQwen35MoeRuntime.prepare_fmha_impl.

    RTP CudaGraphRunner caches this object once at initCapture
    (CudaGraphRunner.cc:480) and calls .prepare_cuda_graph(attn_inputs) on it
    before each replay (CudaGraphRunner.cc:122). We delegate to every ATOM
    RTPFullAttention layer so each layer can refresh its capture-time state.

    Also exposes a .fmha_params attribute because RTP qwen3_next reference path
    constructs PyModelOutputs(hidden_states, fmha_impl.fmha_params); ATOM's own
    forward returns PyModelOutputs(hidden_states) so this is just a stub for
    type-compat with downstream code that may peek at the attribute.
    """

    def __init__(self, runtime: "_ATOMQwen35MoeRuntime") -> None:
        self._runtime = runtime
        self.is_cuda_graph = False
        self._rtp_full_attn_layers: list = []
        try:
            from atom.plugin.rtpllm.attention_backend import (
                AttentionForRTPLLM as _RTPAttn,
            )

            self._rtp_attention_cls = _RTPAttn
        except (ImportError, ModuleNotFoundError):
            self._rtp_attention_cls = None
        if self._rtp_attention_cls is not None:
            for module in runtime.model.modules():
                if isinstance(module, self._rtp_attention_cls):
                    self._rtp_full_attn_layers.append(module)

    @property
    def fmha_params(self):
        return None

    def prepare_cuda_graph(self, attn_inputs) -> None:
        # Replay enters here without re-running prepare_fmha_impl, so forward
        # the latest block mapping to each layer's fused-KV params cache.
        for layer in self._rtp_full_attn_layers:
            layer.prepare_cuda_graph(attn_inputs)


class _ATOMQwen35MoeRuntime(GptModelBase):
    """rtp-llm runtime adapter backed by ATOM model."""

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
                "ATOM model has no parameters; cannot determine device/dtype."
            )
        self._model_device = first_param.device
        self._model_dtype = first_param.dtype
        # Cache module layer maps once to avoid per-forward model.modules() traversal.
        self._rtp_layer_maps = RTPForwardContext.collect_layer_maps(model=self.model)
        # Lazy-built in forward_context; invalidated by kv buffer signature change.
        self._rtp_kv_cache_data: dict | None = None
        self._rtp_kv_cache_signature: tuple | None = None
        self._rtp_layer_group_map: dict[int, int] | None = None
        self._rtp_layer_group_map_signature: tuple | None = None
        # cuda-graph attn_pyobj cache (see _ATOMAttnPyObj).
        self._atom_attn_pyobj: _ATOMAttnPyObj | None = None
        self._cg_layers_prewarmed: bool = False
        # Prewarm only for buckets RTP will capture; using the full concurrency
        # limit can over-allocate graph static buffers enough to break capture.
        decode_caps = getattr(py_hw_kernel_config, "decode_capture_batch_sizes", None)
        if decode_caps:
            self._cg_max_num_tokens: int = min(
                int(max(decode_caps)), int(max_generate_batch_size)
            )
        else:
            self._cg_max_num_tokens: int = int(max_generate_batch_size)
        # max_seq_len comes from model_config; for Qwen3.5-MoE it is the model
        # context length.
        self._cg_max_seq_len: int = int(
            getattr(model_config, "max_seq_len", 0)
            or getattr(model_config, "max_position_embeddings", 0)
            or 32768
        )

    def load_weights(self):
        # ATOM weights should be loaded exactly once from ATOMQwen35Moe._create_python_model.
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
                "RTP plugin requires inputs.attention_inputs to provide position_ids."
            )
        # Keep plugin semantics aligned with RTP native path:
        # first use attention_inputs.position_ids, then fallback to combo_position_ids.
        positions = getattr(attn_inputs, "position_ids", None)
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
                "RTP plugin requires real position metadata from attention_inputs "
                "(position_ids or input/prefix/sequence lengths); fallback positions are disabled."
            )
        positions = positions.to(
            device=model_device, dtype=torch.int32, non_blocking=True
        ).contiguous()
        # Eager-only: shape-based fallback rebuild. In cuda-graph capture mode
        # this Python-level branch on tensor shape is unsafe (and unnecessary
        # because RTP guarantees position_ids has the same length as the
        # capture-time max_num_token). See rtp+atom_graph.md §4.3.
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
                        "RTP plugin position_ids/token_num mismatch "
                        f"(position_ids_tokens={pos_tokens}, token_num={token_num})."
                    )
        return positions

    def prepare_fmha_impl(
        self, inputs: PyModelInputs, is_cuda_graph: bool = False
    ) -> Any:
        """Return ATOM-aware attention container for RTP CUDA graph hooks."""
        if self._atom_attn_pyobj is None:
            self._atom_attn_pyobj = _ATOMAttnPyObj(self)
        self._atom_attn_pyobj.is_cuda_graph = bool(is_cuda_graph)
        # Keep eager/non-graph path untouched: only prewarm when graph path
        # explicitly asks for fmha_impl in cuda-graph mode.
        if bool(is_cuda_graph):
            inputs.attention_inputs.is_cuda_graph = True
            self._ensure_cuda_graph_prewarmed()
        return self._atom_attn_pyobj

    def _ensure_cuda_graph_prewarmed(self) -> None:
        if self._cg_layers_prewarmed:
            return
        if self._atom_attn_pyobj is None:
            return
        max_num_tokens = int(self._cg_max_num_tokens)
        max_seq_len = int(self._cg_max_seq_len)
        if max_num_tokens <= 0 or max_seq_len <= 0:
            logger.warning(
                "ATOM cuda-graph prewarm skipped: invalid budget "
                "(max_num_tokens=%d, max_seq_len=%d)",
                max_num_tokens,
                max_seq_len,
            )
            return
        device = self._get_model_device()
        dtype = self._get_model_dtype()

        # RTP C++ KVCache.num_kv_heads is the POST-TP-copy value — it stays at
        # the global total when kv_head_num < tp_size (no division is done).
        # e.g. Qwen3.5-MoE: global=2, tp=4 → KVCache.num_kv_heads=2, but
        # ATOM layer's self.num_kv_heads=max(1, 2//4)=1.
        # _align_kv_heads_for_cache() will repeat k/v from 1→2 heads before
        # writing to the kv cache, so the fused-QKV buffer must be sized for
        # the larger (post-alignment) count.
        kv_cache = getattr(self, "kv_cache", None)
        rtp_kv_heads: int | None = (
            int(kv_cache.num_kv_heads)
            if kv_cache is not None and int(kv_cache.num_kv_heads) > 0
            else None
        )

        for layer in self._atom_attn_pyobj._rtp_full_attn_layers:
            layer.prewarm_for_cuda_graph(
                max_num_tokens=max_num_tokens,
                max_seq_len=max_seq_len,
                query_dtype=dtype,
                device=device,
                effective_num_kv_heads=rtp_kv_heads,
            )

        # Pre-allocate metadata tensors consumed by _build_plugin_attention_metadata
        # during decode capture.  RTP captures via cudaStreamBeginCapture (not
        # torch.cuda.graph()), so PyTorch's caching allocator never switches to
        # graph-pool mode — any tensor allocated during capture is in the regular
        # pool and may be freed + reused after capture ends, causing replay faults.
        # By pre-allocating here (before capture) and holding a model-level
        # reference, the GPU addresses stay valid for the entire capture/replay
        # lifetime.  decode path: 1 token per seq, so max_num_tokens == max_bs.
        max_bs = max_num_tokens
        # block_table columns are indexed in kernel block granularity
        # (rtp_kernel_seq_size_per_block), not seq_size_per_block.
        # Qwen3.5 config example: max_seq_len=262144, kernel_block=16 -> 16384 columns.
        kernel_seq_size_per_block = (
            int(getattr(kv_cache, "kernel_seq_size_per_block", 0))
            or int(getattr(kv_cache, "seq_size_per_block", 0))
            or 1
        )
        max_blocks = (
            int(max_seq_len) + kernel_seq_size_per_block - 1
        ) // kernel_seq_size_per_block + 1
        # query_start_loc for decode: always [0, 1, 2, ..., bs], i.e. arange(bs+1).
        # seq_id for decode slot_mapping: seq_id[i] == i, i.e. arange(bs).
        self._cg_meta_bufs: dict = {
            "query_start_loc": torch.arange(
                0, max_bs + 1, device=device, dtype=torch.int32
            ),
            "seq_id": torch.arange(0, max_bs, device=device, dtype=torch.int64),
            "block_col": torch.empty(max_bs, device=device, dtype=torch.int32),
            "block_col_i64": torch.empty(max_bs, device=device, dtype=torch.int64),
            "slot_base": torch.empty(max_bs, device=device, dtype=torch.int32),
            "token_offset": torch.empty(max_bs, device=device, dtype=torch.int32),
            "slot_mapping": torch.empty(max_bs, device=device, dtype=torch.int64),
            "seq_lens_i32": torch.empty(max_bs, device=device, dtype=torch.int32),
            "block_table_i32": torch.empty(
                max_bs, max_blocks, device=device, dtype=torch.int32
            ),
        }
        self._cg_layers_prewarmed = True
        logger.info(
            "ATOM RTPFullAttention cuda-graph prewarmed for %d layers "
            "(max_num_tokens=%d, max_seq_len=%d, rtp_kv_heads=%s, "
            "meta_bufs: query_start_loc[%d], slot_mapping[%d], block_table_i32[%dx%d])",
            len(self._atom_attn_pyobj._rtp_full_attn_layers),
            max_num_tokens,
            max_seq_len,
            rtp_kv_heads,
            max_bs + 1,
            max_bs,
            max_bs,
            max_blocks,
        )

    def forward(self, inputs: PyModelInputs, fmha_impl: Any = None) -> PyModelOutputs:
        if bool(getattr(fmha_impl, "is_cuda_graph", False)):
            inputs.attention_inputs.is_cuda_graph = True
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

        with RTPForwardContext.bind(
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


class ATOMQwen35Moe(BaseModel):
    """Qwen3.5-MoE model class that starts ATOM runtime in rtp-llm."""

    @staticmethod
    def _is_external_plugin_mode() -> bool:
        modules = os.getenv("RTP_LLM_EXTERNAL_MODEL_PACKAGES", "")
        return "atom.plugin.rtpllm.models" in modules

    @staticmethod
    def get_weight_cls():
        return _StubWeightInfo

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
        config.has_post_decoder_layernorm = True
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

        attention_step = config_json["full_attention_interval"]
        config.hybrid_attention_config.enable_hybrid_attention = True
        config.hybrid_attention_config.hybrid_attention_types = [
            (
                HybridAttentionType.NONE
                if (idx + 1) % attention_step == 0
                else HybridAttentionType.LINEAR
            )
            for idx in range(config.num_layers)
        ]

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
        """Tell RTP PyWrappedModel.h:160 whether to construct CudaGraphRunner.

        Keep ATOM and RTP on the same switch: ENABLE_CUDA_GRAPH.
        Default: enabled (missing/other values behave as enabled).
        """
        if os.getenv("ENABLE_CUDA_GRAPH", "1") == "0":
            logger.info("ENABLE_CUDA_GRAPH=0 — ATOMQwen35Moe forces eager forward.")
            return False
        return True

    @staticmethod
    def _make_qwen35_hf_mapper() -> WeightsMapper:
        # Keep loading on outer text-only wrapper so packed_modules_mapping works.
        # Normalize checkpoint prefixes to match wrapper's weights_mapping rules.
        return WeightsMapper(
            orig_to_new_substr={"attn.qkv.": "attn.qkv_proj."},
            orig_to_new_prefix={
                # model.language_model.model.* -> model.language_model.*
                # then wrapper mapping turns it into language_model.model.*
                "model.language_model.model.": "model.language_model.",
                # model.language_model.lm_head.* -> lm_head.* -> language_model.lm_head.*
                "model.language_model.lm_head.": "lm_head.",
            },
        )

    @staticmethod
    @contextmanager
    def _maybe_disable_shared_expert_fusion_for_load(atom_model: Any):
        has_standalone_shared_expert = any(
            ".shared_expert." in name for name, _ in atom_model.named_parameters()
        )
        if not has_standalone_shared_expert:
            yield
            return

        import atom.model_loader.loader as atom_loader

        origin_fn = atom_loader.is_rocm_aiter_fusion_shared_expert_enabled
        atom_loader.is_rocm_aiter_fusion_shared_expert_enabled = lambda: False
        try:
            yield
        finally:
            atom_loader.is_rocm_aiter_fusion_shared_expert_enabled = origin_fn

    def load(self, skip_python_model: bool = False):
        # External plugin mode: bypass rtp-llm native weight loading path and
        # use ATOM model loading only.
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
                    "External plugin mode: skip ATOM python model creation as requested"
                )
                return
            self._create_python_model()
            logger.info(
                "External plugin mode: use ATOM loading path and skip rtp-llm native load"
            )
            return

        raise RuntimeError("ATOMQwen35Moe is only supported as an RTP external plugin.")

    def _create_python_model(self):
        if not self._is_external_plugin_mode():
            raise RuntimeError(
                "ATOMQwen35Moe is only supported as an RTP external plugin."
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
        except Exception:
            old_default_device = None

        # rtp-llm plugin mode bypasses ATOM ModelRunner, so we need to align
        # default dtype/device during ATOM model construction.
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
            lm_head_w = _get_first_param_tensor(
                atom_model_obj, "language_model.lm_head.weight"
            )
            if lm_head_w is None:
                lm_head_w = _get_first_param_tensor(atom_model_obj, "lm_head.weight")
            if lm_head_w is not None:
                self.weight.set_global_weight(W.lm_head, lm_head_w.detach())
                logger.info(
                    "Injected runtime lm_head weight for RTP: %s",
                    tuple(lm_head_w.shape),
                )
            else:
                logger.warning(
                    "Failed to find ATOM lm_head.weight for RTP runtime projection."
                )

            emb_w = _get_first_param_tensor(
                atom_model_obj, "language_model.model.embed_tokens.weight"
            )
            if emb_w is None:
                emb_w = _get_first_param_tensor(
                    atom_model_obj, "model.embed_tokens.weight"
                )
            if emb_w is not None:
                self.weight.set_global_weight(W.embedding, emb_w.detach())
                logger.info(
                    "Injected runtime embedding weight for RTP: %s", tuple(emb_w.shape)
                )

            final_ln = _get_first_param_tensor(
                atom_model_obj, "language_model.model.norm.weight"
            )
            if final_ln is None:
                final_ln = _get_first_param_tensor(atom_model_obj, "model.norm.weight")
            if final_ln is not None:
                self.weight.set_global_weight(W.final_ln_gamma, final_ln.detach())
                logger.info(
                    "Injected runtime final_ln_gamma for RTP: %s", tuple(final_ln.shape)
                )

        def _assert_norm_weights_loaded(atom_model_obj: Any) -> None:
            # Guard against silently using default-initialized GemmaRMSNorm weights.
            candidates = [
                "language_model.model.layers.0.input_layernorm.weight",
                "model.layers.0.input_layernorm.weight",
            ]
            norm_w = None
            for name in candidates:
                norm_w = _get_first_param_tensor(atom_model_obj, name)
                if norm_w is not None:
                    break
            if norm_w is None:
                raise ValueError(
                    "Cannot locate layer-0 input_layernorm.weight after ATOM load in RTP plugin mode."
                )
            norm_w_cpu = norm_w.detach().float().reshape(-1).cpu()
            if norm_w_cpu.numel() == 0 or bool(torch.all(norm_w_cpu == 0)):
                raise ValueError(
                    "Loaded layer-0 input_layernorm.weight is all zeros. "
                    "This indicates checkpoint mapping/load mismatch, refusing to run with default values."
                )

        def _load_fused_expert_weights_for_qwen35(
            original_name: str,
            name: str,
            params_dict: dict,
            loaded_weight: torch.Tensor,
            shard_id: str,
            num_experts: int,
        ) -> bool:
            if not detect_fused_expert_format(original_name):
                return False
            mapping = get_fused_expert_mapping()
            if not any(weight_name in original_name for _, weight_name, _ in mapping):
                return False
            return load_fused_expert_weights(
                original_name=original_name,
                name=name,
                params_dict=params_dict,
                loaded_weight=loaded_weight,
                shard_id=shard_id,
                num_experts=num_experts,
            )

        try:
            # Keep RTP-specific behavior in plugin path only.
            _set_framework_backbone("rtpllm")
            apply_attention_gdn_rtpllm_patch()
            apply_attention_mha_rtpllm_patch()
            apply_qwen3_next_rtpllm_patch()
            atom_model = prepare_model(config=self, engine="rtpllm")
            if atom_model is None:
                raise ValueError(
                    "ATOM failed to create qwen3.5-moe model for rtp-llm plugin"
                )

            # In rtp-llm plugin mode, ensure ATOM model parameters are on target GPU.
            atom_model = atom_model.to(target_device)

            atom_config = getattr(atom_model, "atom_config", None)
            if atom_config is None:
                atom_config = getattr(
                    getattr(atom_model, "language_model", None), "atom_config", None
                )
            if atom_config is None:
                raise ValueError(
                    "Cannot get atom_config from prepared ATOM model in rtp-llm plugin mode"
                )

            # External plugin mode: load checkpoint once through ATOM loader.
            # Keep Qwen3.5 MoE weight semantics aligned with #532 plugin path.
            with self._maybe_disable_shared_expert_fusion_for_load(atom_model):
                load_model_in_plugin_mode(
                    model=atom_model,
                    config=atom_config,
                    prefix="model.",
                    weights_mapper=self._make_qwen35_hf_mapper(),
                    load_fused_expert_weights_fn=_load_fused_expert_weights_for_qwen35,
                )
            _assert_norm_weights_loaded(atom_model)
            _inject_rtp_projection_weights(atom_model)
        finally:
            torch.set_default_dtype(old_default_dtype)
            if old_default_device is not None:
                torch.set_default_device(old_default_device)
            else:
                torch.set_default_device("cpu")

        self.py_model = _ATOMQwen35MoeRuntime(
            model_config=self.model_config,
            parallelism_config=self.parallelism_config,
            weights=self.weight,
            max_generate_batch_size=self.max_generate_batch_size,
            fmha_config=self.fmha_config,
            py_hw_kernel_config=self.hw_kernel_config,
            device_resource_config=self.device_resource_config,
            atom_model=atom_model,
        )
        logger.info("Created ATOM qwen3.5-moe runtime for rtp-llm plugin mode")
        return self.py_model
