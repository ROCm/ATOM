# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

"""verl plugin entry-point.

Called automatically by verl when this package is declared under
[project.entry-points."verl.plugins"] in pyproject.toml.

Registers:
  - ATOM rollout backend ("atom") in verl's _ROLLOUT_REGISTRY
  - AtomRL FSDP2 training engine ("atomrl_fsdp2") in verl's EngineRegistry
  - AtomRL Megatron training engine ("atomrl_megatron") in verl's EngineRegistry
"""

from __future__ import annotations

import logging
from collections.abc import Generator
from typing import Any

logger = logging.getLogger(__name__)

_rollout_registered = False
_replica_registered = False
_fsdp_engines_registered = False
_megatron_engine_registered = False


# ---------------------------------------------------------------------------
# ATOMRolloutAdapter — verl BaseRollout implementation backed by ATOM
# ---------------------------------------------------------------------------


class ATOMRolloutAdapter:
    """Thin verl ``BaseRollout`` adapter around ``ATOMReplicaManager``.

    Weight sync is NOT implemented here — ATOM manages its own weight
    lifecycle via ``sleep`` / ``wake_up`` / ``load_weights``.
    ``update_weights`` is intentionally a no-op.

    The adapter is instantiated by verl's engine worker via the FQDN string
    stored in ``_ROLLOUT_REGISTRY``. verl resolves it with ``importlib`` and
    calls ``cls(config, model_config, device_mesh)``.
    """

    def __init__(self, config: Any, model_config: Any, device_mesh: Any, **kw: Any):
        try:
            from verl.workers.rollout.base import BaseRollout

            BaseRollout.__init__(self, config, model_config, device_mesh)
        except ImportError:
            pass
        self._manager = None

    def set_manager(self, manager: Any) -> None:
        """Inject the ``ATOMReplicaManager`` created by the trainer."""
        self._manager = manager

    async def resume(self, tags: list[str]) -> None:
        if self._manager is not None:
            self._manager.wake_all(tags=tags)

    async def update_weights(
        self,
        weights: Generator[tuple[str, Any], None, None],
        wire_format: str = "named_tensors",
        **kw: Any,
    ) -> None:
        logger.debug("ATOMRolloutAdapter.update_weights: no-op (ATOM-native sync).")

    async def release(self) -> None:
        if self._manager is not None:
            self._manager.sleep_all()

    def generate_sequences(self, prompts: Any) -> Any:
        raise NotImplementedError(
            "ATOMRolloutAdapter.generate_sequences is not supported; "
            "use async mode with verl."
        )


# ---------------------------------------------------------------------------
# AtomRLFSDP2Engine — verl BaseEngine shim for FSDP2 training
# ---------------------------------------------------------------------------


def _make_engine_classes():
    """Lazily create engine adapter classes that inherit from verl's BaseEngine.

    Deferred to avoid importing verl at module-level (it may not be installed).
    """
    from verl.workers.engine.base import BaseEngine as VerlBaseEngine

    # Collect all methods from BaseEngine that raise NotImplementedError
    # and generate forwarding methods to self._inner.
    _delegate_methods = []
    import inspect as _inspect

    for _name, _method in _inspect.getmembers(
        VerlBaseEngine, predicate=_inspect.isfunction
    ):
        if _name.startswith("_"):
            continue
        try:
            src = _inspect.getsource(_method)
            if "raise NotImplementedError" in src:
                _delegate_methods.append(_name)
        except (OSError, TypeError):
            pass

    class _AtomRLFSDP2Engine(VerlBaseEngine):
        """Delegates to ``atomrl.engine.training.fsdp_engine.FSDP2EngineWithLMHead``."""

        @staticmethod
        def _convert_verl_config(**kwargs: Any) -> dict:
            """Convert verl config dicts to AtomRL-compatible dicts."""
            import dataclasses

            from atomrl.core.config import (
                FSDPEngineConfig,
                HFModelConfig,
                LoRAConfig,
                OptimizerConfig,
            )

            def _instantiate(src: Any, dc_cls: type) -> Any:
                """Instantiate a AtomRL dataclass from a verl config dict/object."""
                if isinstance(src, dc_cls):
                    return src
                raw = dict(src) if hasattr(src, "items") else {}
                valid = {f.name for f in dataclasses.fields(dc_cls)}
                filtered = {}
                for k, v in raw.items():
                    if k in valid:
                        f_type = {
                            f.name: f.type for f in dataclasses.fields(dc_cls)
                        }.get(k)
                        if dataclasses.is_dataclass(f_type) and hasattr(v, "items"):
                            filtered[k] = _instantiate(v, f_type)
                        else:
                            filtered[k] = v
                return dc_cls(**filtered)

            result: dict = {}

            if "model_config" in kwargs:
                mc = (
                    dict(kwargs["model_config"])
                    if hasattr(kwargs["model_config"], "items")
                    else {}
                )
                if "path" in mc and "local_path" not in mc:
                    mc["local_path"] = mc.pop("path")
                if "lora" in mc and not isinstance(mc["lora"], LoRAConfig):
                    mc["lora"] = _instantiate(mc["lora"], LoRAConfig)
                result["model_config"] = _instantiate(mc, HFModelConfig)

            if "engine_config" in kwargs:
                result["engine_config"] = _instantiate(
                    kwargs["engine_config"], FSDPEngineConfig
                )

            if "optimizer_config" in kwargs:
                result["optimizer_config"] = _instantiate(
                    kwargs["optimizer_config"], OptimizerConfig
                )

            for k in ("model_name", "quant_config"):
                if k in kwargs:
                    result[k] = kwargs[k]

            return result

        def __init__(self, *args: Any, **kwargs: Any):
            from atomrl.engine.training.fsdp_engine import FSDP2EngineWithLMHead

            converted = self._convert_verl_config(**kwargs)
            self._inner = FSDP2EngineWithLMHead(*args, **converted)

        def __getattr__(self, name: str) -> Any:
            return getattr(self._inner, name)

        @property
        def is_param_offload_enabled(self) -> bool:
            return self._inner.is_param_offload_enabled

        @property
        def is_optimizer_offload_enabled(self) -> bool:
            return self._inner.is_optimizer_offload_enabled

        def get_per_tensor_param(self, **kwargs):
            """Convert DTensor → plain Tensor for verl weight sync."""

            items, meta = self._inner.get_per_tensor_param(**kwargs)

            def _to_plain(it):
                for name, param in it:
                    if hasattr(param, "full_tensor"):
                        yield name, param.full_tensor()
                    elif hasattr(param, "_local_tensor"):
                        yield name, param._local_tensor
                    else:
                        yield name, param

            return _to_plain(items), meta

        @staticmethod
        def _sanitize_verl_data(data):
            """Convert verl data formats to plain tensors AtomRL expects.

            - NestedTensor → padded dense tensor
            - Wraps data in a plain dict so AtomRL engine can freely
              read/write without TensorDict type restrictions
            """
            import torch

            # Convert TensorDict to plain dict for AtomRL compatibility
            # Convert to plain dict — AtomRL engine expects dict, and
            # we avoid TensorDict's NonTensorData/boolean conversion issues
            plain = {}
            has_nested = False
            for k in list(data.keys()):
                v = data[k]
                if isinstance(v, torch.Tensor):
                    if v.is_nested:
                        has_nested = True
                        plain[k] = torch.nested.to_padded_tensor(v, 0)
                    else:
                        plain[k] = v
                elif hasattr(v, "data"):
                    plain[k] = v.data
                else:
                    plain[k] = v

            plain["use_packed_forward"] = False

            # Build meta dict for AtomRL engine
            meta = {}
            for flag in (
                "calculate_entropy",
                "calculate_sum_pi_squared",
                "compute_loss",
                "use_fused_kernels",
                "max_token_len_per_gpu",
                "micro_batch_size_per_gpu",
            ):
                if flag in plain:
                    meta[flag] = plain[flag]
            if meta:
                plain["meta"] = meta

            # Ensure ppo_loss required fields
            import torch.distributed as dist

            if "dp_size" not in plain:
                plain["dp_size"] = dist.get_world_size() if dist.is_initialized() else 1
            if "batch_num_tokens" not in plain:
                am = plain.get("attention_mask")
                if am is not None and isinstance(am, torch.Tensor):
                    plain["batch_num_tokens"] = int(am.sum().item())

            # Reconstruct input_ids from prompts+responses
            if has_nested:
                prompts = plain.get("prompts")
                responses = plain.get("responses")
                if prompts is not None and responses is not None:
                    new_ids = torch.cat([prompts, responses], dim=1)
                    plain["input_ids"] = new_ids
                    seq_len = new_ids.shape[1]
                    plain["position_ids"] = (
                        torch.arange(seq_len, device=new_ids.device)
                        .unsqueeze(0)
                        .expand(new_ids.shape[0], -1)
                    )

            return plain

        @staticmethod
        def _restore_standard_attention():
            """Undo AtomRL's varlen attention monkey-patch if installed."""
            try:
                from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

                from atomrl.engine.training.packing import _original_attn_fn

                if _original_attn_fn is not None:
                    ALL_ATTENTION_FUNCTIONS["sdpa"] = _original_attn_fn
            except (ImportError, AttributeError):
                pass

        @staticmethod
        def _flatten_output(output, original_data):
            """Convert padded [B, S-1] outputs to flat [total_nnz] for verl.

            Uses the original TensorDict (with NestedTensor prompts/responses)
            to compute per-sequence lengths that match verl's expectations.
            """
            import torch

            if not isinstance(output, dict):
                return output
            mo = output.get("model_output", {})
            if not isinstance(mo, dict):
                return output

            def require_original_field(name: str):
                if name not in original_data:
                    raise KeyError(
                        f"verl original_data is missing required field {name!r}"
                    )
                return original_data[name]

            # Compute per-sequence actual lengths from original data
            import sys

            require_original_field("prompts")
            require_original_field("responses")
            # Get total_nnz from original NestedTensor input_ids
            input_ids_orig = original_data.get("input_ids")
            if input_ids_orig is not None and input_ids_orig.is_nested:
                total_nnz = input_ids_orig.values().shape[0]
                seq_lens = input_ids_orig.offsets().diff()
            else:
                am = require_original_field("attention_mask")
                seq_lens = am.sum(dim=1)
                total_nnz = int(seq_lens.sum().item())

            for key in ("log_probs", "entropy", "sum_pi_squared"):
                val = mo.get(key)
                if val is None or not isinstance(val, torch.Tensor):
                    continue
                if val.dim() < 2:
                    continue
                # val is [B, S-1] padded; pad to [B, S] then extract per-seq
                padded = torch.nn.functional.pad(val, (0, 1), value=0.0)
                parts = []
                for i in range(val.shape[0]):
                    slen = int(seq_lens[i])
                    parts.append(padded[i, :slen])
                result = torch.cat(parts, dim=0)

                print(
                    f"[FLATTEN] {key}: padded={val.shape} → flat={result.shape}, seq_lens={seq_lens.tolist()}, total_nnz={total_nnz}",
                    file=sys.stderr,
                    flush=True,
                )
                mo[key] = result

            return output

        def _get_seq_lens(self, original_data):
            """Get per-sequence actual token lengths from original TensorDict."""

            input_ids = original_data.get("input_ids")
            if (
                input_ids is not None
                and hasattr(input_ids, "is_nested")
                and input_ids.is_nested
            ):
                return input_ids.offsets().diff()
            am = original_data.get("attention_mask")
            if am is not None:
                return am.sum(dim=1)
            return None

        @staticmethod
        def _pad_to_flat(tensor, seq_lens):
            """Convert padded [B, S] tensor to flat [total_nnz] using seq_lens."""
            import torch

            if (
                tensor is None
                or not isinstance(tensor, torch.Tensor)
                or tensor.dim() < 2
            ):
                return tensor
            if tensor.shape[0] != len(seq_lens):
                return tensor
            parts = []
            for i in range(tensor.shape[0]):
                slen = min(int(seq_lens[i]), tensor.shape[1])
                parts.append(tensor[i, :slen])
            return torch.cat(parts, dim=0)

        def _wrap_loss_function(self, loss_function, full_seq_lens):
            """Wrap loss_function so ppo_loss sees flat [total_nnz] tensors."""
            import torch

            if loss_function is None:
                return loss_function
            pad_to_flat = self._pad_to_flat

            def _wrapped(model_output, data, **kwargs):
                # Compute seq_lens for this micro-batch from attention_mask
                am = data.get("attention_mask")
                if am is not None and isinstance(am, torch.Tensor) and am.dim() == 2:
                    mb_seq_lens = am.sum(dim=1)
                else:
                    return loss_function(model_output=model_output, data=data, **kwargs)

                # Flatten model_output tensors (log_probs, entropy)
                if isinstance(model_output, dict):
                    for key in list(model_output.keys()):
                        model_output[key] = pad_to_flat(model_output[key], mb_seq_lens)

                # Convert plain dict back to TensorDict for ppo_loss
                # (ppo_loss calls data.select().to_padded_tensor())
                if isinstance(data, dict) and not hasattr(data, "select"):
                    from tensordict import NonTensorData as NTD
                    from tensordict import TensorDict as TD

                    td_items = {}
                    B = int(mb_seq_lens.shape[0])
                    # Use CUDA device — model_output tensors are on GPU
                    device = torch.device("cuda")
                    for k, v in data.items():
                        if isinstance(v, torch.Tensor):
                            td_items[k] = v.to(device)
                        else:
                            td_items[k] = NTD(v)
                    data = TD(td_items, batch_size=[B], device=device)

                return loss_function(model_output=model_output, data=data, **kwargs)

            return _wrapped

        @staticmethod
        def _sanitize_metrics(output):
            """Convert Metric objects to lists of floats for verl compatibility.

            verl's _postprocess_output expects each metric value to be a list
            (one element per micro-batch) and uses chain.from_iterable to flatten.
            """
            if not isinstance(output, dict):
                return output
            # Wrap loss values: verl expects list-of-lists for flattening
            loss = output.get("loss")
            if isinstance(loss, list) and loss and not isinstance(loss[0], list):
                output["loss"] = [
                    [x] if isinstance(x, (int, float)) else x for x in loss
                ]

            # Fix all list values in output to be list-of-lists
            # (verl's train_mini_batch uses chain.from_iterable to flatten)
            for top_key in list(output.keys()):
                val = output[top_key]
                if isinstance(val, list) and val and isinstance(val[0], (int, float)):
                    output[top_key] = [[x] for x in val]

            metrics = output.get("metrics")
            if isinstance(metrics, dict):
                for k, v in list(metrics.items()):
                    if isinstance(v, list):
                        # Ensure list-of-lists and convert Metric objects
                        fixed = []
                        for item in v:
                            if isinstance(item, list):
                                fixed.append(item)
                            elif isinstance(item, (int, float)):
                                fixed.append([item])
                            elif hasattr(item, "value"):
                                fixed.append([float(item.value)])
                            elif hasattr(item, "item"):
                                fixed.append([item.item()])
                            else:
                                try:
                                    fixed.append([float(item)])
                                except (TypeError, ValueError):
                                    fixed.append([0.0])
                        metrics[k] = fixed
                    elif hasattr(v, "value"):
                        metrics[k] = [[float(v.value)]]
                    elif hasattr(v, "item"):
                        metrics[k] = [[v.item()]]
                    elif isinstance(v, (int, float)):
                        metrics[k] = [[v]]
                    else:
                        try:
                            metrics[k] = [[float(v)]]
                        except (TypeError, ValueError):
                            metrics[k] = [[0.0]]
            return output

        def train_batch(self, data, *args, **kwargs):
            seq_lens = self._get_seq_lens(data)
            plain = self._sanitize_verl_data(data)
            self._restore_standard_attention()

            # Wrap loss_function to flatten padded → flat for ppo_loss
            if args:
                args = list(args)
                args[0] = self._wrap_loss_function(args[0], seq_lens)
            elif "loss_function" in kwargs:
                kwargs["loss_function"] = self._wrap_loss_function(
                    kwargs["loss_function"], seq_lens
                )

            output = self._inner.train_batch(plain, *args, **kwargs)
            output = self._sanitize_metrics(output)
            return self._flatten_output(output, data)

        def infer_batch(self, data, *args, **kwargs):
            plain = self._sanitize_verl_data(data)
            self._restore_standard_attention()
            output = self._inner.infer_batch(plain, *args, **kwargs)
            return self._flatten_output(output, data)

        def forward_backward_batch(self, data, *args, **kwargs):
            plain = self._sanitize_verl_data(data)
            self._restore_standard_attention()
            output = self._inner.forward_backward_batch(plain, *args, **kwargs)
            return self._flatten_output(output, data)

    # Dynamically add forwarding methods for all BaseEngine abstract methods
    for _mname in _delegate_methods:
        if _mname not in _AtomRLFSDP2Engine.__dict__:

            def _make_forwarder(name):
                def _forwarder(self, *a, **kw):
                    return getattr(self._inner, name)(*a, **kw)

                _forwarder.__name__ = name
                return _forwarder

            setattr(_AtomRLFSDP2Engine, _mname, _make_forwarder(_mname))

    class _AtomRLMegatronEngine(VerlBaseEngine):
        """Delegates to ``atomrl.engine.training.megatron_engine.MegatronEngine``."""

        def __init__(self, *args: Any, **kwargs: Any):
            from atomrl.engine.training.megatron_engine import MegatronEngineWithLMHead

            self._inner = MegatronEngineWithLMHead(*args, **kwargs)

        def __getattr__(self, name: str) -> Any:
            return getattr(self._inner, name)

        @property
        def is_param_offload_enabled(self) -> bool:
            return self._inner.is_param_offload_enabled

        @property
        def is_optimizer_offload_enabled(self) -> bool:
            return self._inner.is_optimizer_offload_enabled

    for _mname in _delegate_methods:
        if _mname not in _AtomRLMegatronEngine.__dict__:

            def _make_forwarder(name):
                def _forwarder(self, *a, **kw):
                    return getattr(self._inner, name)(*a, **kw)

                _forwarder.__name__ = name
                return _forwarder

            setattr(_AtomRLMegatronEngine, _mname, _make_forwarder(_mname))

    return _AtomRLFSDP2Engine, _AtomRLMegatronEngine


# ---------------------------------------------------------------------------
# Entry-point callable
# ---------------------------------------------------------------------------


def register() -> None:
    """Register AtomRL backends in verl's registries.

    Called by verl's plugin auto-loader at ``import verl`` time.
    """
    global _rollout_registered
    global _replica_registered
    global _fsdp_engines_registered
    global _megatron_engine_registered

    if not _rollout_registered:
        try:
            from verl.workers.rollout.base import _ROLLOUT_REGISTRY

            _ROLLOUT_REGISTRY[("atom", "async")] = (
                "atomrl.plugin.verl.register.ATOMRolloutAdapter"
            )
            _rollout_registered = True
            logger.info("atomrl: registered ATOM rollout in verl._ROLLOUT_REGISTRY")
        except ImportError:
            logger.warning(
                "atomrl: verl.workers.rollout.base not importable; "
                "skipped rollout registration"
            )

    # Also register in RolloutReplicaRegistry (used by v0 legacy trainer)
    if not _replica_registered:
        try:
            from verl.workers.rollout.replica import RolloutReplicaRegistry

            def _load_atom():
                from atomrl.plugin.verl.atom_replica import ATOMRolloutReplica

                return ATOMRolloutReplica

            if "atom" not in RolloutReplicaRegistry._registry:
                RolloutReplicaRegistry.register("atom", _load_atom)
                logger.info(
                    "atomrl: registered ATOM in RolloutReplicaRegistry (v0 compat)"
                )
            _replica_registered = True
        except (ImportError, KeyError, RuntimeError, ValueError) as e:
            logger.warning("atomrl: replica registration failed: %s", e)

    if _fsdp_engines_registered and _megatron_engine_registered:
        return

    try:
        from verl.workers.engine.base import EngineRegistry as VerlEngineRegistry

        FSDP2Cls, MegatronCls = _make_engine_classes()
    except (AttributeError, ImportError, TypeError) as e:
        logger.warning("atomrl: engine adapter initialization failed: %s", e)
        return

    if not _fsdp_engines_registered:
        try:
            VerlEngineRegistry.register(
                model_type="language_model",
                backend=["atomrl_fsdp", "atomrl_fsdp2"],
                device=["cuda"],
            )(FSDP2Cls)
            _fsdp_engines_registered = True
            logger.info(
                "atomrl: registered atomrl_fsdp/atomrl_fsdp2 in verl EngineRegistry"
            )
        except (ImportError, KeyError, RuntimeError, ValueError) as e:
            logger.warning("atomrl: FSDP engine registration failed: %s", e)

    if not _megatron_engine_registered:
        try:
            VerlEngineRegistry.register(
                model_type="language_model",
                backend="atomrl_megatron",
                device=["cuda"],
            )(MegatronCls)
            _megatron_engine_registered = True
            logger.info("atomrl: registered atomrl_megatron in verl EngineRegistry")
        except (ImportError, KeyError, RuntimeError, ValueError) as e:
            logger.warning("atomrl: Megatron engine registration failed: %s", e)


# Auto-register when this module is imported (verl's plugin loader calls
# _ep.load() which imports this module but doesn't call register()).
register()
