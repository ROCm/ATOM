"""GLM5 wrapper for rtp-llm external model loading."""

from __future__ import annotations

import logging
import os
from typing import Any

import torch
from rtp_llm.config.model_config import ModelConfig
from rtp_llm.model_loader.model_weight_info import ModelWeights
from rtp_llm.models.deepseek_v2 import DeepSeekV2
from rtp_llm.models_py.model_desc.module_base import GptModelBase
from rtp_llm.ops import ParallelismConfig
from rtp_llm.ops.compute_ops import PyModelInputs, PyModelOutputs
from rtp_llm.utils.model_weight import W

logger = logging.getLogger("atom.plugin.rtpllm.models")

# Patched in tests; lazily imported in runtime to keep module import lightweight.
RTPForwardContext = None


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


class _ATOMGlm5AttnPyObj:
    """Container returned to RTP CudaGraphRunner for replay-time hooks."""

    def __init__(self, runtime: "_ATOMGlm5MoeRuntime") -> None:
        self._runtime = runtime
        self.is_cuda_graph = False
        self._rtp_mla_layers: list[Any] = []
        self._rtp_sparse_mla_backends: list[Any] = []
        self._collect_mla_layers()

    @staticmethod
    def _append_unique(items: list[Any], value: Any) -> None:
        if value is not None and all(value is not item for item in items):
            items.append(value)

    def _collect_mla_layers(self) -> None:
        try:
            from atom.plugin.rtpllm.attention_backend import (
                RTPMLAAttention,
                RTPSparseMlaBackend,
            )
        except (ImportError, ModuleNotFoundError):
            RTPMLAAttention = None
            RTPSparseMlaBackend = None

        candidates: list[Any] = []
        _, _, mla_layer_map = self._runtime._rtp_layer_maps
        candidates.extend(mla_layer_map.values())
        for module in self._runtime.model.modules():
            candidates.append(module)
            mla_attn = getattr(module, "mla_attn", None)
            if mla_attn is not None:
                candidates.append(mla_attn)

        for candidate in candidates:
            if RTPMLAAttention is not None and isinstance(candidate, RTPMLAAttention):
                self._append_unique(self._rtp_mla_layers, candidate)
                backend = getattr(candidate, "sparse_backend", None)
            else:
                backend = getattr(candidate, "sparse_backend", None)
                if (
                    backend is None
                    and RTPSparseMlaBackend is not None
                    and isinstance(candidate, RTPSparseMlaBackend)
                ):
                    backend = candidate

            if RTPSparseMlaBackend is not None and isinstance(
                backend, RTPSparseMlaBackend
            ):
                self._append_unique(self._rtp_sparse_mla_backends, backend)

    @property
    def fmha_params(self):
        return None

    def prepare_cuda_graph(self, attn_inputs) -> None:  # noqa: ANN001
        for layer in self._rtp_mla_layers:
            prepare = getattr(layer, "prepare_cuda_graph", None)
            if callable(prepare):
                prepare(attn_inputs)
        for backend in self._rtp_sparse_mla_backends:
            prepare = getattr(backend, "prepare_cuda_graph", None)
            if callable(prepare):
                prepare(attn_inputs)


class _ATOMGlm5MoeRuntime(GptModelBase):
    """rtp-llm runtime adapter backed by an ATOM GLM5 model."""

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
        first_param = next(iter(self.model.parameters()), None)
        if first_param is not None:
            self._model_device = first_param.device
            self._model_dtype = first_param.dtype
        else:
            self._model_device = torch.device("cpu")
            self._model_dtype = torch.get_default_dtype()
        forward_context_cls = self._get_forward_context_cls()
        self._rtp_layer_maps = forward_context_cls.collect_layer_maps(model=self.model)
        self._rtp_kv_cache_data: dict | None = None
        self._rtp_kv_cache_signature: tuple | None = None
        self._rtp_layer_group_map: dict[int, int] | None = None
        self._rtp_layer_group_map_signature: tuple | None = None
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
        self._atom_attn_pyobj: _ATOMGlm5AttnPyObj | None = None
        self._cg_layers_prewarmed: bool = False

    def load_weights(self):
        return None

    def prepare_fmha_impl(
        self, inputs: PyModelInputs, is_cuda_graph: bool = False
    ) -> _ATOMGlm5AttnPyObj:
        if self._atom_attn_pyobj is None:
            self._atom_attn_pyobj = _ATOMGlm5AttnPyObj(self)
        self._atom_attn_pyobj.is_cuda_graph = bool(is_cuda_graph)
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
                "ATOM GLM5 cuda-graph prewarm skipped: invalid budget "
                "(max_num_tokens=%d, max_seq_len=%d)",
                max_num_tokens,
                max_seq_len,
            )
            return

        device = self._get_model_device()
        dtype = self._get_model_dtype()
        kv_cache = getattr(self, "kv_cache", None)
        seq_size_per_block = (
            int(getattr(kv_cache, "seq_size_per_block", 0))
            or int(os.getenv("SEQ_SIZE_PER_BLOCK", "0") or 0)
            or 1
        )
        kernel_seq_size_per_block = (
            int(getattr(kv_cache, "kernel_seq_size_per_block", 0))
            or int(os.getenv("KERNEL_SEQ_SIZE_PER_BLOCK", "0") or 0)
            or seq_size_per_block
        )
        physical_max_blocks = (
            int(max_seq_len) + seq_size_per_block - 1
        ) // seq_size_per_block + 1
        recovered_physical_max_blocks = (
            int(max_seq_len) + seq_size_per_block - 1
        ) // seq_size_per_block
        indexer_max_blocks = (
            int(max_seq_len) + kernel_seq_size_per_block - 1
        ) // kernel_seq_size_per_block + 1
        block_table_max_blocks = max(physical_max_blocks, indexer_max_blocks)

        for backend in self._atom_attn_pyobj._rtp_sparse_mla_backends:
            prewarm = getattr(backend, "prewarm_for_cuda_graph", None)
            if callable(prewarm):
                prewarm(
                    max_num_tokens=max_num_tokens,
                    max_seq_len=max_seq_len,
                    query_dtype=dtype,
                    device=device,
                )
        self._cg_meta_bufs: dict[str, torch.Tensor] = {
            "query_start_loc": torch.arange(
                0, max_num_tokens + 1, device=device, dtype=torch.int32
            ),
            "seq_id": torch.arange(0, max_num_tokens, device=device, dtype=torch.int64),
            "seq_id_i32": torch.arange(
                0, max_num_tokens, device=device, dtype=torch.int32
            ),
            "positions_i32": torch.empty(
                max_num_tokens, device=device, dtype=torch.int32
            ),
            "positions_i64": torch.empty(
                max_num_tokens, device=device, dtype=torch.int64
            ),
            "block_col": torch.empty(max_num_tokens, device=device, dtype=torch.int32),
            "block_col_i64": torch.empty(
                max_num_tokens, device=device, dtype=torch.int64
            ),
            "slot_base": torch.empty(max_num_tokens, device=device, dtype=torch.int32),
            "token_offset": torch.empty(
                max_num_tokens, device=device, dtype=torch.int32
            ),
            "slot_mapping": torch.empty(
                max_num_tokens, device=device, dtype=torch.int64
            ),
            "seq_lens_i32": torch.empty(
                max_num_tokens, device=device, dtype=torch.int32
            ),
            "physical_block_table_i32": torch.empty(
                max_num_tokens,
                recovered_physical_max_blocks,
                device=device,
                dtype=torch.int32,
            ),
            "block_table_i32": torch.empty(
                max_num_tokens, block_table_max_blocks, device=device, dtype=torch.int32
            ),
            "indexer_block_table_i32": torch.empty(
                max_num_tokens, indexer_max_blocks, device=device, dtype=torch.int32
            ),
        }
        self._cg_layers_prewarmed = True
        logger.info(
            "ATOM GLM5 cuda-graph prewarmed "
            "(max_num_tokens=%d, max_seq_len=%d, sparse_layers=%d, "
            "physical_block_table_i32[%dx%d], block_table_i32[%dx%d], "
            "indexer_block_table_i32[%dx%d])",
            max_num_tokens,
            max_seq_len,
            len(self._atom_attn_pyobj._rtp_sparse_mla_backends),
            max_num_tokens,
            recovered_physical_max_blocks,
            max_num_tokens,
            block_table_max_blocks,
            max_num_tokens,
            indexer_max_blocks,
        )

    @staticmethod
    def _get_forward_context_cls():
        global RTPForwardContext
        if RTPForwardContext is None:
            from atom.plugin.rtpllm.utils import (
                RTPForwardMLAContext as _RTPForwardContext,
            )

            RTPForwardContext = _RTPForwardContext
        return RTPForwardContext

    def _get_model_device(self) -> torch.device:
        return self._model_device

    def _get_model_dtype(self) -> torch.dtype:
        return self._model_dtype

    def _get_token_num(
        self, inputs: PyModelInputs, input_ids: torch.Tensor | None
    ) -> int:
        if input_ids is not None and input_ids.numel() > 0:
            return int(input_ids.numel())
        input_hiddens = getattr(inputs, "input_hiddens", None)
        if input_hiddens is not None and input_hiddens.numel() > 0:
            return int(input_hiddens.shape[0])
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

        sequence_lengths_plus_1 = getattr(
            attn_inputs, "sequence_lengths_plus_1_d", None
        )
        if sequence_lengths_plus_1 is not None and sequence_lengths_plus_1.numel() > 0:
            seq_plus_one_i32 = sequence_lengths_plus_1.to(
                device=model_device, dtype=torch.int32, non_blocking=True
            ).contiguous()
            if int(seq_plus_one_i32.numel()) < int(input_lengths_i32.numel()):
                return None
            starts = (
                seq_plus_one_i32[: int(input_lengths_i32.numel())] - input_lengths_i32
            )
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

    def _build_graph_decode_positions(
        self, attn_inputs: Any, model_device: torch.device
    ) -> torch.Tensor | None:
        sequence_lengths_plus_1 = getattr(
            attn_inputs, "sequence_lengths_plus_1_d", None
        )
        if sequence_lengths_plus_1 is None or sequence_lengths_plus_1.numel() == 0:
            return None
        input_lengths = getattr(attn_inputs, "input_lengths", None)
        if input_lengths is None or input_lengths.numel() == 0:
            return None
        num_tokens = int(input_lengths.numel())
        seq_plus_one_i32 = sequence_lengths_plus_1.to(
            device=model_device, dtype=torch.int32, non_blocking=True
        )
        if int(seq_plus_one_i32.numel()) < num_tokens:
            return None
        cg_bufs = getattr(self, "_cg_meta_bufs", None)
        if isinstance(cg_bufs, dict):
            positions_buf = cg_bufs.get("positions_i32")
            if (
                isinstance(positions_buf, torch.Tensor)
                and int(positions_buf.numel()) >= num_tokens
            ):
                positions_i32 = positions_buf[:num_tokens]
                torch.sub(seq_plus_one_i32[:num_tokens], 1, out=positions_i32)
                positions_i64_buf = cg_bufs.get("positions_i64")
                if (
                    isinstance(positions_i64_buf, torch.Tensor)
                    and int(positions_i64_buf.numel()) >= num_tokens
                ):
                    positions_i64 = positions_i64_buf[:num_tokens]
                    positions_i64.copy_(positions_i32)
                    return positions_i64
                return positions_i32
        return (seq_plus_one_i32[:num_tokens] - 1).to(dtype=torch.long).contiguous()

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
            device=model_device, dtype=torch.long, non_blocking=True
        ).contiguous()

    def _extract_positions(
        self, inputs: PyModelInputs, model_device: torch.device, token_num: int
    ) -> torch.Tensor:
        attn_inputs = getattr(inputs, "attention_inputs", None)
        if attn_inputs is None:
            raise ValueError(
                "GLM5 RTP plugin requires inputs.attention_inputs to provide position metadata."
            )
        positions = None
        graph_decode = bool(getattr(attn_inputs, "is_cuda_graph", False)) and not bool(
            getattr(attn_inputs, "is_prefill", False)
        )
        if graph_decode:
            # RTP CudaGraphRunner refreshes sequence_lengths_plus_1_d before
            # replay, but not position_ids. Build decode positions from the
            # refreshed RTP length tensors so RoPE advances on every replay.
            positions = self._build_graph_decode_positions(
                attn_inputs=attn_inputs,
                model_device=model_device,
            )
        if positions is None or positions.numel() == 0:
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
                "GLM5 RTP plugin requires real position metadata from attention_inputs."
            )
        if torch.cuda.is_current_stream_capturing():
            if positions.device != model_device:
                raise RuntimeError(
                    "GLM5 RTP cuda-graph capture requires positions on model device."
                )
            positions = positions.contiguous()
        else:
            positions = positions.to(
                device=model_device, dtype=torch.long, non_blocking=True
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
                        device=model_device, dtype=torch.long, non_blocking=True
                    ).contiguous()
                elif pos_tokens > token_num:
                    positions = positions[..., -token_num:].contiguous()
                else:
                    raise ValueError(
                        "GLM5 RTP plugin position_ids/token_num mismatch "
                        f"(position_ids_tokens={pos_tokens}, token_num={token_num})."
                    )
        return positions

    def forward(
        self, inputs: PyModelInputs, fmha_impl=None
    ) -> PyModelOutputs:  # noqa: ANN001
        is_cuda_graph = bool(getattr(fmha_impl, "is_cuda_graph", False))
        if is_cuda_graph:
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
        if is_cuda_graph and token_num > 0:
            positions = positions[:token_num]
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

        forward_context_cls = self._get_forward_context_cls()
        with forward_context_cls.bind(
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


class ATOMGlm5Moe(DeepSeekV2):
    """GLM5 model class that starts ATOM runtime in rtp-llm plugin mode."""

    @staticmethod
    def _is_external_plugin_mode() -> bool:
        modules = os.getenv("RTP_LLM_EXTERNAL_MODEL_PACKAGES", "")
        return "atom.plugin.rtpllm.models" in modules

    @classmethod
    def _create_config(cls, ckpt_path: str):
        config = super()._create_config(ckpt_path)
        # ATOM sparse MLA reads the FP8 KV cache through aiter's 576-token layout.
        config.attn_config.mla_use_aiter_fp8_layout = True
        return config

    def support_cuda_graph(self) -> bool:
        if os.getenv("ENABLE_CUDA_GRAPH", "1") == "0":
            logger.info("ENABLE_CUDA_GRAPH=0 - ATOMGlm5Moe forces eager forward.")
            return False
        return True

    @staticmethod
    def _make_glm5_hf_mapper():
        from atom.model_loader.loader import WeightsMapper

        return WeightsMapper(
            orig_to_new_prefix={},
            orig_to_new_substr={
                "indexers_proj.": "indexer.weights_proj.",
            },
        )

    @staticmethod
    def _get_named_parameters(atom_model: Any) -> dict[str, torch.Tensor]:
        if atom_model is None or not hasattr(atom_model, "named_parameters"):
            return {}
        return {
            name: param
            for name, param in atom_model.named_parameters(recurse=True)
            if param is not None
        }

    @staticmethod
    def _first_param(
        params: dict[str, torch.Tensor], candidates: tuple[str, ...]
    ) -> torch.Tensor | None:
        for name in candidates:
            param = params.get(name)
            if param is not None:
                return param
        return None

    def _inject_rtp_projection_weights(self, atom_model: Any) -> None:
        params = self._get_named_parameters(atom_model)
        if not params:
            logger.warning(
                "Skip GLM5 RTP projection weight injection because atom_model has no named parameters."
            )
            return

        required = {
            W.lm_head: (
                "language_model.lm_head.weight",
                "lm_head.weight",
            ),
            W.embedding: (
                "language_model.model.embed_tokens.weight",
                "model.embed_tokens.weight",
            ),
            W.final_ln_gamma: (
                "language_model.model.norm.weight",
                "model.norm.weight",
            ),
        }
        missing = []
        for weight_name, candidates in required.items():
            param = self._first_param(params, candidates)
            if param is None:
                missing.append((weight_name, candidates))
                continue
            self.weight.set_global_weight(weight_name, param.detach())
            logger.info(
                "Injected GLM5 runtime %s for RTP: %s",
                weight_name,
                tuple(param.shape),
            )
        if missing:
            details = ", ".join(
                f"{weight_name} candidates={candidates}"
                for weight_name, candidates in missing
            )
            raise ValueError(
                f"Cannot locate GLM5 RTP runtime projection weights: {details}"
            )

    def _assert_norm_weights_loaded(self, atom_model: Any) -> None:
        params = self._get_named_parameters(atom_model)
        if not params:
            logger.warning(
                "Skip GLM5 norm weight validation because atom_model has no named parameters."
            )
            return
        norm_w = self._first_param(
            params,
            (
                "language_model.model.layers.0.input_layernorm.weight",
                "model.layers.0.input_layernorm.weight",
            ),
        )
        if norm_w is None:
            raise ValueError(
                "Cannot locate GLM5 layer-0 input_layernorm.weight after ATOM load in RTP plugin mode."
            )
        norm_w_cpu = norm_w.detach().float().reshape(-1).cpu()
        if norm_w_cpu.numel() == 0 or bool(torch.all(norm_w_cpu == 0)):
            raise ValueError(
                "Loaded GLM5 layer-0 input_layernorm.weight is all zeros; "
                "refusing to run with default values."
            )

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
                    "External plugin mode: skip ATOM GLM5 python model creation as requested"
                )
                return
            self._create_python_model()
            logger.info(
                "External plugin mode: use ATOM GLM5 loading path and skip native load"
            )
            return

        super().load(skip_python_model=skip_python_model)

    def _create_python_model(self):
        if not self._is_external_plugin_mode():
            return super()._create_python_model()

        import atom
        from atom.model_loader.loader import load_model_in_plugin_mode

        prepare_model = getattr(atom, "prepare_model", None)
        if prepare_model is None:
            from atom.plugin.prepare import prepare_model

        target_device = torch.device(
            self.device if getattr(self, "device", None) else "cuda"
        )
        target_dtype = self.model_config.compute_dtype
        old_default_dtype = torch.get_default_dtype()
        try:
            old_default_device = torch.get_default_device()
        except Exception:
            old_default_device = None

        torch.set_default_device(target_device)
        if target_dtype in {
            torch.float16,
            torch.bfloat16,
            torch.float32,
            torch.float64,
        }:
            torch.set_default_dtype(target_dtype)

        try:
            atom_model = prepare_model(config=self, engine="rtpllm")
            if atom_model is None:
                raise ValueError("ATOM failed to create GLM5 model for rtp-llm plugin")

            if hasattr(atom_model, "to"):
                atom_model = atom_model.to(target_device)

            atom_config = getattr(atom_model, "atom_config", None)
            if atom_config is None:
                atom_config = getattr(
                    getattr(atom_model, "model", None), "atom_config", None
                )
            if atom_config is None:
                # Unit tests may use mocked ATOM models; real loading must expose atom_config.
                atom_config = getattr(self, "atom_config", None)

            load_model_in_plugin_mode(
                model=atom_model,
                config=atom_config,
                prefix="model.",
                weights_mapper=self._make_glm5_hf_mapper(),
            )
            self._assert_norm_weights_loaded(atom_model)
            self._inject_rtp_projection_weights(atom_model)
        finally:
            torch.set_default_dtype(old_default_dtype)
            if old_default_device is not None:
                torch.set_default_device(old_default_device)
            else:
                torch.set_default_device("cpu")

        self.py_model = _ATOMGlm5MoeRuntime(
            model_config=self.model_config,
            parallelism_config=self.parallelism_config,
            weights=self.weight,
            max_generate_batch_size=self.max_generate_batch_size,
            fmha_config=self.fmha_config,
            py_hw_kernel_config=self.hw_kernel_config,
            device_resource_config=self.device_resource_config,
            atom_model=atom_model,
        )
        logger.info("Created ATOM GLM5 runtime for rtp-llm plugin mode")
        return self.py_model
