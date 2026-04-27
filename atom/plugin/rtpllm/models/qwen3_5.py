import logging
import os
from typing import Any

import torch
from rtp_llm.config.model_config import ModelConfig
from rtp_llm.model_loader.model_weight_info import ModelWeights
from rtp_llm.models.qwen3_next.qwen3_next import Qwen35Moe
from rtp_llm.models_py.model_desc.module_base import GptModelBase
from rtp_llm.ops import ParallelismConfig
from rtp_llm.ops.compute_ops import PyModelInputs, PyModelOutputs
from rtp_llm.utils.model_weight import W

from atom.plugin.rtpllm.model_ops.attention_gdn import apply_attention_gdn_rtpllm_patch
from atom.plugin.rtpllm.models.qwen3_next import apply_qwen3_next_rtpllm_patch

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
        self._warned_once_keys: set[str] = set()

    def load_weights(self):
        # ATOM weights should be loaded exactly once from ATOMQwen35Moe._create_python_model.
        return None

    def _get_model_device(self) -> torch.device:
        first_param = next(self.model.parameters(), None)
        if first_param is None:
            return torch.device("cuda")
        return first_param.device

    def _get_model_dtype(self) -> torch.dtype:
        first_param = next(self.model.parameters(), None)
        if first_param is None:
            return torch.bfloat16
        return first_param.dtype

    def _warn_once(self, key: str, msg: str, *args: Any) -> None:
        if key in self._warned_once_keys:
            return
        self._warned_once_keys.add(key)
        logger.warning(msg, *args)

    def _build_positions_fallback(
        self, inputs: PyModelInputs, model_device: torch.device
    ) -> torch.Tensor | None:
        attn_inputs = getattr(inputs, "attention_inputs", None)
        if attn_inputs is None:
            return None

        input_lengths = getattr(attn_inputs, "input_lengths", None)
        if input_lengths is None or input_lengths.numel() == 0:
            return None

        prefix_lengths = getattr(attn_inputs, "prefix_lengths", None)
        sequence_lengths = getattr(attn_inputs, "sequence_lengths", None)

        context_batch = (
            int(prefix_lengths.numel())
            if prefix_lengths is not None and prefix_lengths.numel() > 0
            else 0
        )
        decode_batch = (
            int(sequence_lengths.numel())
            if sequence_lengths is not None and sequence_lengths.numel() > 0
            else 0
        )

        parts: list[torch.Tensor] = []

        for i in range(context_batch):
            token_num = int(input_lengths[i].item())
            prefix = int(prefix_lengths[i].item())
            if token_num > 0:
                parts.append(
                    torch.arange(
                        prefix,
                        prefix + token_num,
                        dtype=torch.int32,
                        device=model_device,
                    )
                )

        for j in range(decode_batch):
            idx = context_batch + j
            token_num = int(input_lengths[idx].item()) if idx < input_lengths.numel() else 1
            seq_len = int(sequence_lengths[j].item())
            if token_num <= 0:
                continue
            if token_num == 1:
                parts.append(torch.tensor([seq_len], dtype=torch.int32, device=model_device))
            else:
                parts.append(
                    torch.arange(
                        seq_len - token_num + 1,
                        seq_len + 1,
                        dtype=torch.int32,
                        device=model_device,
                    )
                )

        if not parts:
            return None
        return torch.cat(parts, dim=0)

    def forward(self, inputs: PyModelInputs, fmha_impl: Any = None) -> PyModelOutputs:
        model_device = self._get_model_device()
        model_dtype = self._get_model_dtype()
        input_ids = inputs.input_ids
        positions = inputs.attention_inputs.position_ids
        if positions is None:
            # In some RTP plugin paths, position ids are populated in
            # bert_embedding_inputs instead of attention_inputs.
            positions = getattr(inputs.bert_embedding_inputs, "combo_position_ids", None)
        if positions is None:
            # Fallback for plugin paths that don't populate either field.
            positions = self._build_positions_fallback(inputs, model_device)
        inputs_embeds = None

        if input_ids is not None and input_ids.numel() > 0 and input_ids.device != model_device:
            input_ids = input_ids.to(device=model_device, non_blocking=True)
        if positions is not None and positions.numel() > 0 and positions.device != model_device:
            positions = positions.to(device=model_device, non_blocking=True)
        if positions is None and input_ids is not None and input_ids.numel() > 0:
            # Last resort: keep generation alive for simple text-only cases.
            positions = torch.arange(
                input_ids.numel(), dtype=torch.int32, device=model_device
            )
            self._warn_once(
                "positions_missing_local_arange",
                "RTP plugin did not provide position ids; using local arange fallback."
            )
        # Rotary embedding requires `positions.numel() == token_num`.
        # In some plugin paths RTP passes full-context positions while input_ids contains only decode tokens.
        token_num = 0
        if input_ids is not None and input_ids.numel() > 0:
            token_num = int(input_ids.numel())
        if token_num == 0 and inputs.input_hiddens is not None and inputs.input_hiddens.numel() > 0:
            token_num = int(inputs.input_hiddens.shape[0])
        if positions is not None and positions.numel() > 0 and token_num > 0:
            pos_num = int(positions.shape[-1])
            if pos_num != token_num:
                attn_inputs = getattr(inputs, "attention_inputs", None)
                sequence_lengths = (
                    getattr(attn_inputs, "sequence_lengths", None) if attn_inputs is not None else None
                )
                if sequence_lengths is not None and int(sequence_lengths.numel()) == token_num:
                    positions = sequence_lengths.to(device=model_device, dtype=torch.int32, non_blocking=True)
                    self._warn_once(
                        "position_token_mismatch_sequence_lengths",
                        "Position/token mismatch fixed via sequence_lengths: pos=%d token=%d",
                        pos_num,
                        token_num,
                    )
                elif pos_num > token_num:
                    positions = positions[..., -token_num:].contiguous()
                    self._warn_once(
                        "position_token_mismatch_tail_slice",
                        "Position/token mismatch fixed via tail slice: pos=%d token=%d",
                        pos_num,
                        token_num,
                    )
                else:
                    positions = torch.arange(token_num, dtype=torch.int32, device=model_device)
                    self._warn_once(
                        "position_token_mismatch_local_arange",
                        "Position/token mismatch fixed via local arange: pos=%d token=%d",
                        pos_num,
                        token_num,
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

        hidden_states = self.model(
            input_ids=input_ids,
            positions=positions,
            intermediate_tensors=None,
            inputs_embeds=inputs_embeds,
        )
        return PyModelOutputs(hidden_states)


class ATOMQwen35Moe(Qwen35Moe):
    """Qwen3.5-MoE model class that starts ATOM runtime in rtp-llm."""

    @staticmethod
    def _is_external_plugin_mode() -> bool:
        modules = os.getenv("RTP_LLM_EXTERNAL_MODEL_PACKAGES", "")
        return "atom.plugin.rtpllm.models" in modules

    def load(self):
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
            self._create_python_model()
            logger.info(
                "External plugin mode: use ATOM loading path and skip rtp-llm native load"
            )
            return

        # Non-plugin mode keeps native behavior.
        super().load()

    def _create_python_model(self):
        # Non-external mode should keep native rtp-llm Python model path.
        if not self._is_external_plugin_mode():
            return super()._create_python_model()

        import atom
        from atom.model_loader.loader import load_model_in_plugin_mode

        target_device = torch.device(self.device if getattr(self, "device", None) else "cuda")
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
            lm_head_w = _get_first_param_tensor(atom_model_obj, "language_model.lm_head.weight")
            if lm_head_w is None:
                lm_head_w = _get_first_param_tensor(atom_model_obj, "lm_head.weight")
            if lm_head_w is not None:
                self.weight.set_global_weight(W.lm_head, lm_head_w.detach())
                logger.info("Injected runtime lm_head weight for RTP: %s", tuple(lm_head_w.shape))
            else:
                logger.warning("Failed to find ATOM lm_head.weight for RTP runtime projection.")

            emb_w = _get_first_param_tensor(atom_model_obj, "language_model.model.embed_tokens.weight")
            if emb_w is None:
                emb_w = _get_first_param_tensor(atom_model_obj, "model.embed_tokens.weight")
            if emb_w is not None:
                self.weight.set_global_weight(W.embedding, emb_w.detach())
                logger.info("Injected runtime embedding weight for RTP: %s", tuple(emb_w.shape))

            final_ln = _get_first_param_tensor(atom_model_obj, "language_model.model.norm.weight")
            if final_ln is None:
                final_ln = _get_first_param_tensor(atom_model_obj, "model.norm.weight")
            if final_ln is not None:
                self.weight.set_global_weight(W.final_ln_gamma, final_ln.detach())
                logger.info("Injected runtime final_ln_gamma for RTP: %s", tuple(final_ln.shape))

        try:
            # Keep RTP-specific behavior in plugin path only.
            apply_attention_gdn_rtpllm_patch()
            apply_qwen3_next_rtpllm_patch()
            atom_model = atom.prepare_model(config=self, engine="rtpllm")
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
            load_model_in_plugin_mode(
                model=atom_model,
                config=atom_config,
                prefix="model.",
            )
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
