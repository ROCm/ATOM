# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

"""DSV4 on-disk patches for ROCm/Megatron-LM (PatchPhase.SOURCE).

Adds TransformerConfig fields and TransformerBlock/Layer hooks needed by
Lumen's ``get_dsv4_spec()``.  Individual functions are registered via
:func:`atomrl.lumen.patches.register_patch` and applied through
:func:`atomrl.lumen.patches.apply_megatron_source_patches`.
"""

from __future__ import annotations

import os
import re

from atomrl.lumen.patches.registry import PatchPhase, register_patch


def patch_file(path: str, replacements: list[tuple[str, str]]) -> bool:
    with open(path) as f:
        content = f.read()
    original = content
    for old, new in replacements:
        content = content.replace(old, new)
    if content != original:
        with open(path, "w") as f:
            f.write(content)
        return True
    return False


def patch_transformer_config(megatron_root: str) -> bool:
    path = os.path.join(
        megatron_root, "megatron", "core", "transformer", "transformer_config.py"
    )

    with open(path) as f:
        content = f.read()
    original = content

    # 1. Add 'dsv4' to the Literal type
    content = content.replace(
        "Literal['gated_delta_net', 'dsa']",
        "Literal['gated_delta_net', 'dsa', 'dsv4']",
    )

    # 2. Add DSV4 fields after the DSA section
    dsa_marker = "    ####################\n    # DSA\n    ####################"
    if dsa_marker in content and "dsv4_mode: bool" not in content:
        last_dsa_match = None
        for m in re.finditer(r'    dsa_\w+:.*\n(?:    """.*?"""\n)?', content):
            last_dsa_match = m
        if last_dsa_match:
            insert_pos = last_dsa_match.end()
            dsv4_fields = """
    ####################
    # DSV4
    ####################
    dsv4_mode: bool = False
    dsv4_hc_mult: Optional[int] = None
    dsv4_hc_sinkhorn_iters: int = 20
    dsv4_hc_eps: float = 1e-6
    dsv4_compress_ratios: Optional[List[int]] = None
    dsv4_compress_rope_theta: float = 160000.0
    dsv4_o_groups: Optional[int] = None
    dsv4_o_lora_rank: Optional[int] = None
    dsv4_n_hash_layers: int = 0
    dsv4_window_size: int = 128

"""
            content = content[:insert_pos] + dsv4_fields + content[insert_pos:]

    # 3. Add dsv4_mode = True in __post_init__ when variant == "dsv4"
    dsa_post_init = '        if self.experimental_attention_variant == "dsa":'
    if (
        dsa_post_init in content
        and 'experimental_attention_variant == "dsv4"' not in content
    ):
        dsv4_post_init = """        if self.experimental_attention_variant == "dsv4":
            self.dsv4_mode = True

"""
        # Insert before the dsa check
        content = content.replace(dsa_post_init, dsv4_post_init + dsa_post_init)

    if content != original:
        with open(path, "w") as f:
            f.write(content)
        return True
    return False


def patch_moe_sqrtsoftplus(megatron_root: str) -> bool:
    """Add sqrtsoftplus MoE router score function (DeepSeek V4 default)."""
    changed = False

    cfg_path = os.path.join(
        megatron_root, "megatron", "core", "transformer", "transformer_config.py"
    )
    with open(cfg_path) as f:
        cfg = f.read()
    cfg_orig = cfg

    cfg = cfg.replace(
        "moe_router_score_function: Literal['softmax', 'sigmoid'] = \"softmax\"",
        "moe_router_score_function: Literal['softmax', 'sigmoid', 'sqrtsoftplus'] = \"softmax\"",
    )
    cfg = cfg.replace(
        '"""Score function for MoE routing. Can be "softmax" or "sigmoid"."""',
        '"""Score function for MoE routing. Can be "softmax", "sigmoid", or "sqrtsoftplus"."""',
    )
    cfg = cfg.replace(
        """        if self.moe_router_enable_expert_bias and self.moe_router_score_function != "sigmoid":
            raise ValueError(
                "Expert bias for aux-loss-free routing only supports sigmoid score function."
                "Please set --moe-router-score-function sigmoid for sigmoid score function."
            )""",
        """        if self.moe_router_enable_expert_bias and self.moe_router_score_function not in (
            "sigmoid",
            "sqrtsoftplus",
        ):
            raise ValueError(
                "Expert bias for aux-loss-free routing only supports sigmoid or sqrtsoftplus score function. "
                "Please set --moe-router-score-function to sigmoid or sqrtsoftplus."
            )""",
    )
    if cfg != cfg_orig:
        with open(cfg_path, "w") as f:
            f.write(cfg)
        changed = True

    moe_path = os.path.join(
        megatron_root, "megatron", "core", "transformer", "moe", "moe_utils.py"
    )
    with open(moe_path) as f:
        moe = f.read()
    moe_orig = moe

    sqrtsoftplus_branch = """    elif score_function == "sqrtsoftplus":
        assert num_groups is None
        assert group_topk is None
        scores = torch.nn.functional.softplus(logits.float()).sqrt().type_as(logits)
        if expert_bias is not None:
            scores_for_routing = scores + expert_bias
            _, top_indices = compute_topk(scores_for_routing, topk, num_groups, group_topk)
            scores = torch.gather(scores, dim=1, index=top_indices).type_as(logits)
        else:
            scores, top_indices = compute_topk(scores, topk, num_groups, group_topk)
        probs = scores / (scores.sum(dim=-1, keepdim=True) + 1e-20) if topk > 1 else scores
"""
    if 'elif score_function == "sqrtsoftplus"' not in moe:
        moe = moe.replace(
            """        probs = scores / (scores.sum(dim=-1, keepdim=True) + 1e-20) if topk > 1 else scores
    else:
        raise ValueError(f"Invalid score_function: {score_function}")""",
            """        probs = scores / (scores.sum(dim=-1, keepdim=True) + 1e-20) if topk > 1 else scores
"""
            + sqrtsoftplus_branch
            + """    else:
        raise ValueError(f"Invalid score_function: {score_function}")""",
        )
    if moe != moe_orig:
        with open(moe_path, "w") as f:
            f.write(moe)
        changed = True

    return changed


def patch_dsv4_training_config(megatron_root: str) -> bool:
    """DSV4 finetune flags + typed compress ratios (ROCm Megatron gaps vs miles fork)."""
    cfg_path = os.path.join(
        megatron_root, "megatron", "core", "transformer", "transformer_config.py"
    )
    with open(cfg_path) as f:
        cfg = f.read()
    cfg_orig = cfg

    cfg = cfg.replace(
        "dsv4_compress_ratios: Optional[list] = None",
        "dsv4_compress_ratios: Optional[List[int]] = None",
    )
    if "activation_func_clamp_shared_expert:" not in cfg:
        cfg = cfg.replace(
            "    activation_func_clamp_value: Optional[float] = None",
            """    activation_func_clamp_value: Optional[float] = None
    activation_func_clamp_shared_expert: bool = True""",
            1,
        )
    if "freeze_e_score_correction_bias:" not in cfg:
        anchor = "    moe_router_enable_expert_bias: bool = False"
        cfg = cfg.replace(
            anchor,
            anchor + """
    freeze_e_score_correction_bias: bool = False
    moe_router_freeze_gate: bool = False""",
            1,
        )

    if cfg != cfg_orig:
        with open(cfg_path, "w") as f:
            f.write(cfg)
        return True
    return False


def patch_moe_router_freeze(megatron_root: str) -> bool:
    path = os.path.join(
        megatron_root, "megatron", "core", "transformer", "moe", "router.py"
    )
    with open(path) as f:
        content = f.read()
    orig = content

    if "moe_router_freeze_gate" not in content:
        content = content.replace(
            "        self.reset_parameters()\n",
            """        self.reset_parameters()

        if self.config.moe_router_freeze_gate:
            self.weight.requires_grad = False
            if self.bias is not None:
                self.bias.requires_grad = False
""",
            1,
        )
        content = content.replace(
            """        if self.bias is not None and self.bias.device.type == 'cpu':
            self.bias.data = self.bias.data.to(device=torch.cuda.current_device())

        # Convert to specified datatype""",
            """        if self.bias is not None and self.bias.device.type == 'cpu':
            self.bias.data = self.bias.data.to(device=torch.cuda.current_device())

        if self.config.moe_router_freeze_gate:
            assert not self.weight.requires_grad
            if self.bias is not None:
                assert not self.bias.requires_grad

        # Convert to specified datatype""",
            1,
        )

    if content != orig:
        with open(path, "w") as f:
            f.write(content)
        return True
    return False


def patch_skip_none_router_expert_bias(megatron_root: str) -> bool:
    """Hash-routing layers keep expert_bias=None; skip them in bias updates."""
    path = os.path.join(
        megatron_root, "megatron", "core", "distributed", "finalize_model_grads.py"
    )
    with open(path) as f:
        content = f.read()
    orig = content
    content = content.replace(
        "            if config.moe_router_enable_expert_bias and hasattr(module, 'expert_bias'):\n"
        "                module.local_tokens_per_expert.zero_()",
        "            if config.moe_router_enable_expert_bias and getattr(\n"
        "                module, 'local_tokens_per_expert', None\n"
        "            ) is not None:\n"
        "                module.local_tokens_per_expert.zero_()",
        1,
    )
    content = content.replace(
        "            if hasattr(module, 'expert_bias'):\n"
        "                tokens_per_expert_list.append(module.local_tokens_per_expert)\n"
        "                expert_bias_list.append(module.expert_bias)",
        "            if getattr(module, 'expert_bias', None) is not None and getattr(\n"
        "                module, 'local_tokens_per_expert', None\n"
        "            ) is not None:\n"
        "                tokens_per_expert_list.append(module.local_tokens_per_expert)\n"
        "                expert_bias_list.append(module.expert_bias)",
        1,
    )
    if content != orig:
        with open(path, "w") as f:
            f.write(content)
        return True
    return False


def patch_dsv4_hash_routing(megatron_root: str) -> bool:
    """Route the first DSV4 MoE layers through ``tid2eid[input_ids]``.

    The converted checkpoint contains ``mlp.router.tid2eid`` for layers
    ``1..dsv4_n_hash_layers`` (Megatron layer numbers are one-based).  Thread
    token IDs from GPTModel down to TopKRouter and use the table for expert
    selection while retaining sqrt-softplus logits as combine weights.
    """
    changed = False

    def update(relpath: str, transform) -> None:
        nonlocal changed
        path = os.path.join(megatron_root, relpath)
        with open(path) as f:
            content = f.read()
        patched = transform(content)
        if patched != content:
            with open(path, "w") as f:
                f.write(patched)
            changed = True

    def patch_gpt(content: str) -> str:
        if (
            "            input_ids=input_ids,\n            **(extra_block_kwargs or {}),"
            in content
        ):
            return content
        return content.replace(
            """            padding_mask=padding_mask,
            **(extra_block_kwargs or {}),""",
            """            padding_mask=padding_mask,
            input_ids=input_ids,
            **(extra_block_kwargs or {}),""",
            1,
        )

    update("megatron/core/models/gpt/gpt_model.py", patch_gpt)

    def patch_block(content: str) -> str:
        if "        input_ids: Optional[Tensor] = None,\n        *," not in content:
            content = content.replace(
                """        sequence_len_offset: Optional[Tensor] = None,
        padding_mask: Optional[Tensor] = None,
        *,""",
                """        sequence_len_offset: Optional[Tensor] = None,
        padding_mask: Optional[Tensor] = None,
        input_ids: Optional[Tensor] = None,
        *,""",
                1,
            )
        if (
            '        input_ids: Optional[Tensor] = None,\n    ):\n        """Forward method with activation checkpointing.'
            not in content
        ):
            content = content.replace(
                """        use_inner_quantization_context: bool,
        padding_mask: Optional[Tensor] = None,
    ):
        \"\"\"Forward method with activation checkpointing.""",
                """        use_inner_quantization_context: bool,
        padding_mask: Optional[Tensor] = None,
        input_ids: Optional[Tensor] = None,
    ):
        \"\"\"Forward method with activation checkpointing.""",
                1,
            )
        content = content.replace(
            """                rotary_pos_emb,
                padding_mask=None,
            ):
                for index in range(start, end):""",
            """                rotary_pos_emb,
                padding_mask=None,
                input_ids=None,
            ):
                for index in range(start, end):""",
            1,
        )
        content = content.replace(
            """                            packed_seq_params=packed_seq_params,
                            padding_mask=padding_mask,
                        )
                return hidden_states, context""",
            """                            packed_seq_params=packed_seq_params,
                            padding_mask=padding_mask,
                            input_ids=input_ids,
                        )
                return hidden_states, context""",
            1,
        )
        # Both TE and tensor-parallel checkpoint calls pass the same final pair.
        content = content.replace(
            """                    rotary_pos_emb,
                    padding_mask,
                )""",
            """                    rotary_pos_emb,
                    padding_mask,
                    input_ids,
                )""",
            2,
        )
        content = content.replace(
            """                    use_inner_quantization_context=use_inner_quantization_context,
                    padding_mask=padding_mask,
                )""",
            """                    use_inner_quantization_context=use_inner_quantization_context,
                    padding_mask=padding_mask,
                    input_ids=input_ids,
                )""",
            1,
        )
        content = content.replace(
            """                            sequence_len_offset=sequence_len_offset,
                            padding_mask=padding_mask,
                        )""",
            """                            sequence_len_offset=sequence_len_offset,
                            padding_mask=padding_mask,
                            input_ids=input_ids,
                        )""",
            1,
        )
        return content

    update("megatron/core/transformer/transformer_block.py", patch_block)

    def patch_transformer_layer_hash(content: str) -> str:
        if '        input_ids = kwargs.pop("input_ids", None)' not in content:
            content = content.replace(
                """        kwargs.pop("dynamic_inference_decode_only", None)
        hidden_states, context = self._forward_attention(*args, **kwargs)""",
                """        kwargs.pop("dynamic_inference_decode_only", None)
        input_ids = kwargs.pop("input_ids", None)
        hidden_states, context = self._forward_attention(*args, **kwargs)""",
                1,
            )
        content = content.replace(
            """            kwargs.get("inference_context", None),
            padding_mask=kwargs.get("padding_mask", None),
        )
        return output, context""",
            """            kwargs.get("inference_context", None),
            padding_mask=kwargs.get("padding_mask", None),
            input_ids=input_ids,
        )
        return output, context""",
            1,
        )
        content = content.replace(
            "    def _forward_mlp(self, hidden_states, inference_context=None, padding_mask=None):",
            "    def _forward_mlp(self, hidden_states, inference_context=None, padding_mask=None, input_ids=None):",
        )
        content = content.replace(
            """            mlp_output_with_bias = self.mlp(pre_mlp_layernorm_output, padding_mask=padding_mask)

        nvtx_range_pop(suffix="mlp")""",
            """            mlp_output_with_bias = self.mlp(
                pre_mlp_layernorm_output,
                padding_mask=padding_mask,
                **(dict(input_ids=input_ids) if self.is_moe_layer else {}),
            )

        nvtx_range_pop(suffix="mlp")""",
            1,
        )
        content = content.replace(
            """        else:
            return super()._forward_mlp(hidden_states, padding_mask=padding_mask)""",
            """        else:
            return super()._forward_mlp(
                hidden_states, padding_mask=padding_mask, input_ids=input_ids
            )""",
            1,
        )
        return content

    update(
        "megatron/core/transformer/transformer_layer.py",
        patch_transformer_layer_hash,
    )

    def patch_moe_layer(content: str) -> str:
        content = content.replace(
            """        if input_ids is not None and self.config.sequence_parallel:
            from megatron.core.tensor_parallel.mappings import split_along_nth_dim

            input_ids = split_along_nth_dim(
                input_ids,
                dim=1,
                group=parallel_state.get_tensor_model_parallel_group(),
            )""",
            """        if input_ids is not None and self.config.sequence_parallel:
            from megatron.core.tensor_parallel.mappings import (
                scatter_to_sequence_parallel_region,
            )

            input_ids = scatter_to_sequence_parallel_region(
                input_ids.transpose(0, 1).contiguous(),
                group=parallel_state.get_tensor_model_parallel_group(),
            ).transpose(0, 1).contiguous()""",
            1,
        )
        content = content.replace(
            """        self.router = submodules.router(
            config=self.config, pg_collection=pg_collection, is_mtp_layer=is_mtp_layer
        )""",
            """        self.router = submodules.router(
            config=self.config,
            pg_collection=pg_collection,
            is_mtp_layer=is_mtp_layer,
            layer_number=layer_number,
        )""",
            1,
        )
        content = content.replace(
            """    def route(self, hidden_states: torch.Tensor, padding_mask: Optional[torch.Tensor] = None):
        \"\"\"Compute token routing for preprocessing.""",
            """    def route(
        self,
        hidden_states: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
        input_ids: Optional[torch.Tensor] = None,
    ):
        \"\"\"Compute token routing for preprocessing.""",
            1,
        )
        content = content.replace(
            """        probs, routing_map = apply_module(self.router)(hidden_states, padding_mask)
        return probs, routing_map""",
            """        if input_ids is not None and self.config.sequence_parallel:
            from megatron.core.tensor_parallel.mappings import (
                scatter_to_sequence_parallel_region,
            )

            input_ids = scatter_to_sequence_parallel_region(
                input_ids.transpose(0, 1).contiguous(),
                group=parallel_state.get_tensor_model_parallel_group(),
            ).transpose(0, 1).contiguous()
        probs, routing_map = apply_module(self.router)(
            hidden_states, padding_mask, input_ids=input_ids
        )
        return probs, routing_map""",
            1,
        )
        content = content.replace(
            """        intermediate_tensors=None,
        padding_mask: Optional[torch.Tensor] = None,
    ):
        \"\"\"Forward pass for the MoE layer.""",
            """        intermediate_tensors=None,
        padding_mask: Optional[torch.Tensor] = None,
        input_ids: Optional[torch.Tensor] = None,
    ):
        \"\"\"Forward pass for the MoE layer.""",
            1,
        )
        content = content.replace(
            "        def custom_forward(hidden_states, intermediate_tensors=None, padding_mask=None):",
            "        def custom_forward(hidden_states, intermediate_tensors=None, padding_mask=None, input_ids=None):",
            1,
        )
        content = content.replace(
            "                    probs, routing_map = self.route(hidden_states, padding_mask)",
            "                    probs, routing_map = self.route(hidden_states, padding_mask, input_ids=input_ids)",
            1,
        )
        content = content.replace(
            """                    hidden_states,
                    intermediate_tensors,
                    padding_mask,
                )""",
            """                    hidden_states,
                    intermediate_tensors,
                    padding_mask,
                    input_ids,
                )""",
            1,
        )
        content = content.replace(
            """                    custom_forward, False, hidden_states, intermediate_tensors, padding_mask
                )""",
            """                    custom_forward,
                    False,
                    hidden_states,
                    intermediate_tensors,
                    padding_mask,
                    input_ids,
                )""",
            1,
        )
        content = content.replace(
            "            outputs = custom_forward(hidden_states, intermediate_tensors, padding_mask)",
            "            outputs = custom_forward(hidden_states, intermediate_tensors, padding_mask, input_ids)",
            1,
        )
        return content

    update("megatron/core/transformer/moe/moe_layer.py", patch_moe_layer)

    def patch_router(content: str) -> str:
        topk_marker = "class TopKRouter(Router):"
        prefix, topk = content.split(topk_marker, 1)
        # Repair trees patched by an earlier version that accidentally added
        # layer_number to the abstract Router constructor.
        prefix = prefix.replace(
            """        pg_collection: Optional[ProcessGroupCollection] = None,
        is_mtp_layer: bool = False,
        layer_number: Optional[int] = None,
    ) -> None:""",
            """        pg_collection: Optional[ProcessGroupCollection] = None,
        is_mtp_layer: bool = False,
    ) -> None:""",
            1,
        )
        topk = topk.replace(
            """        pg_collection: Optional[ProcessGroupCollection] = None,
        is_mtp_layer: bool = False,
    ) -> None:""",
            """        pg_collection: Optional[ProcessGroupCollection] = None,
        is_mtp_layer: bool = False,
        layer_number: Optional[int] = None,
    ) -> None:""",
            1,
        )
        content = prefix + topk_marker + topk
        old_init = """        self.input_jitter = None

        self.enable_expert_bias = self.config.moe_router_enable_expert_bias
        if self.enable_expert_bias:"""
        new_init = """        self.input_jitter = None

        self._routing_mode_initialized = False
        self.enable_expert_bias = False
        self.tid2eid = None
        if layer_number is not None:
            self._init_routing_mode(layer_number)

        if self.enable_expert_bias:"""
        content = content.replace(old_init, new_init, 1)
        # Layer number is often assigned after construction. Do not create or
        # clear expert_bias in __init__ based on the still-False flag.
        leftover_init = """        if layer_number is not None:
            self._init_routing_mode(layer_number)

        if self.enable_expert_bias:
            self.register_buffer(
                'local_tokens_per_expert',
                torch.zeros(
                    self.config.num_moe_experts,
                    dtype=torch.float32,
                    device=torch.cuda.current_device(),
                ),
                persistent=False,
            )
            self.register_buffer(
                'expert_bias',
                torch.zeros(
                    self.config.num_moe_experts,
                    dtype=torch.float32,
                    device=torch.cuda.current_device(),
                ),
            )
        else:
            self.local_tokens_per_expert = None
            self.expert_bias = None
"""
        leftover_new = """        if layer_number is not None:
            self._init_routing_mode(layer_number)
"""
        content = content.replace(leftover_init, leftover_new, 1)
        routing_mode_body = """    def _init_routing_mode(self, layer_number: int):
        if self._routing_mode_initialized:
            return
        self._routing_mode_initialized = True
        self.layer_number = layer_number
        mode_hash = (
            getattr(self.config, "dsv4_mode", False)
            and layer_number <= self.config.dsv4_n_hash_layers
        )

        self.enable_expert_bias = (
            self.config.moe_router_enable_expert_bias and not mode_hash
        )
        if self.enable_expert_bias:
            alloc_kwargs = {}
            if torch.cuda.is_available():
                alloc_kwargs["device"] = torch.device(
                    "cuda", torch.cuda.current_device()
                )
            self.register_buffer(
                "local_tokens_per_expert",
                torch.zeros(
                    self.config.num_moe_experts,
                    dtype=torch.float32,
                    **alloc_kwargs,
                ),
                persistent=False,
            )
            self.register_buffer(
                "expert_bias",
                torch.zeros(
                    self.config.num_moe_experts,
                    dtype=torch.float32,
                    **alloc_kwargs,
                ),
            )
        else:
            self.local_tokens_per_expert = None
            self.expert_bias = None
        if mode_hash:
            alloc_kwargs = {}
            if torch.cuda.is_available():
                alloc_kwargs["device"] = torch.device(
                    "cuda", torch.cuda.current_device()
                )
            self.tid2eid = torch.nn.Parameter(
                torch.full(
                    (int(self.config.vocab_size), int(self.topk)),
                    -1,
                    dtype=torch.int32,
                    **alloc_kwargs,
                ),
                requires_grad=False,
            )

    def set_layer_number(self, layer_number: int):
        self.layer_number = layer_number
        self._init_routing_mode(layer_number)

"""
        if "    def _init_routing_mode(self, layer_number: int):" not in content:
            anchor = "    def _maintain_float32_expert_bias(self):"
            content = content.replace(anchor, routing_mode_body + anchor, 1)
        else:
            # Re-patch trees that created tid2eid but never allocated expert_bias.
            old_mode = """    def _init_routing_mode(self, layer_number: int):
        if self._routing_mode_initialized:
            return
        self._routing_mode_initialized = True
        self.layer_number = layer_number
        mode_hash = (
            getattr(self.config, "dsv4_mode", False)
            and layer_number <= self.config.dsv4_n_hash_layers
        )

        self.enable_expert_bias = (
            self.config.moe_router_enable_expert_bias and not mode_hash
        )
        if mode_hash:"""
            new_mode = """    def _init_routing_mode(self, layer_number: int):
        if self._routing_mode_initialized:
            return
        self._routing_mode_initialized = True
        self.layer_number = layer_number
        mode_hash = (
            getattr(self.config, "dsv4_mode", False)
            and layer_number <= self.config.dsv4_n_hash_layers
        )

        self.enable_expert_bias = (
            self.config.moe_router_enable_expert_bias and not mode_hash
        )
        if self.enable_expert_bias:
            alloc_kwargs = {}
            if torch.cuda.is_available():
                alloc_kwargs["device"] = torch.device(
                    "cuda", torch.cuda.current_device()
                )
            self.register_buffer(
                "local_tokens_per_expert",
                torch.zeros(
                    self.config.num_moe_experts,
                    dtype=torch.float32,
                    **alloc_kwargs,
                ),
                persistent=False,
            )
            self.register_buffer(
                "expert_bias",
                torch.zeros(
                    self.config.num_moe_experts,
                    dtype=torch.float32,
                    **alloc_kwargs,
                ),
            )
        else:
            self.local_tokens_per_expert = None
            self.expert_bias = None
        if mode_hash:"""
            content = content.replace(old_mode, new_mode, 1)
        content = content.replace(
            "    def routing(self, logits: torch.Tensor, padding_mask: Optional[torch.Tensor] = None):",
            """    def routing(
        self,
        logits: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
        input_ids: Optional[torch.Tensor] = None,
    ):""",
            1,
        )
        content = content.replace(
            """                fused=self.config.moe_router_fusion,
                router_replay=self.router_replay,
            )""",
            """                fused=self.config.moe_router_fusion,
                router_replay=self.router_replay,
                tid2eid=self.tid2eid,
                input_ids=(
                    input_ids.reshape(-1)
                    if self.tid2eid is not None and input_ids is not None
                    else None
                ),
            )""",
            1,
        )
        content = content.replace(
            "    def forward(self, input: torch.Tensor, padding_mask: Optional[torch.Tensor] = None):",
            """    def forward(
        self,
        input: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
        input_ids: Optional[torch.Tensor] = None,
    ):""",
            1,
        )
        content = content.replace(
            "        probs, routing_map = self.routing(logits, padding_mask=padding_mask)",
            """        probs, routing_map = self.routing(
            logits, padding_mask=padding_mask, input_ids=input_ids
        )""",
            1,
        )
        return content

    update("megatron/core/transformer/moe/router.py", patch_router)

    def patch_moe_utils_hash(content: str) -> str:
        if "    tid2eid: Optional[torch.Tensor] = None," not in content:
            content = content.replace(
                """    fused: bool = False,
    router_replay: Optional['RouterReplay'] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:""",
                """    fused: bool = False,
    router_replay: Optional['RouterReplay'] = None,
    tid2eid: Optional[torch.Tensor] = None,
    input_ids: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:""",
                1,
            )
        content = content.replace(
            "        if router_replay is None:\n            return _compute_topk(",
            "        if router_replay is None or tid2eid is not None:\n            return _compute_topk(",
            1,
        )
        old_sqrt = """    elif score_function == "sqrtsoftplus":
        assert num_groups is None
        assert group_topk is None
        scores = torch.nn.functional.softplus(logits.float()).sqrt().type_as(logits)
        if expert_bias is not None:
            scores_for_routing = scores + expert_bias
            _, top_indices = compute_topk(scores_for_routing, topk, num_groups, group_topk)
            scores = torch.gather(scores, dim=1, index=top_indices).type_as(logits)
        else:
            scores, top_indices = compute_topk(scores, topk, num_groups, group_topk)
        probs = scores / (scores.sum(dim=-1, keepdim=True) + 1e-20) if topk > 1 else scores
"""
        new_sqrt = """    elif score_function == "sqrtsoftplus":
        assert num_groups is None
        assert group_topk is None
        scores = torch.nn.functional.softplus(logits.float()).sqrt().type_as(logits)
        if tid2eid is not None:
            assert not tid2eid.requires_grad
            assert input_ids is not None and not input_ids.requires_grad
            top_indices = tid2eid[input_ids].long()
            assert torch.all(top_indices >= 0)
        elif expert_bias is not None:
            scores_for_routing = scores + expert_bias
            _, top_indices = compute_topk(scores_for_routing, topk, num_groups, group_topk)
        else:
            _, top_indices = compute_topk(scores, topk, num_groups, group_topk)
        scores = torch.gather(scores, dim=1, index=top_indices).type_as(logits)
        probs = scores / (scores.sum(dim=-1, keepdim=True) + 1e-20)
"""
        content = content.replace(old_sqrt, new_sqrt, 1)
        content = content.replace(
            """    if scaling_factor:
        probs = probs * scaling_factor""",
            """    if scaling_factor and tid2eid is None:
        probs = probs * scaling_factor""",
            1,
        )
        return content

    update("megatron/core/transformer/moe/moe_utils.py", patch_moe_utils_hash)
    return changed


def patch_dist_ckpt_skip_optional_dsv4_norms(megatron_root: str) -> bool:
    """Skip missing optional shards when converted ckpt omits them.

    - q_norm/kv_norm on late layers (flash ckpt)
    - mlp.router.expert_bias (4-layer torch_dist from Miles convert)

    Enable with LUMEN_DSV4_SKIP_OPTIONAL_NORMS=1 (default). Set to 0 to fail on missing keys.
    """
    if os.environ.get("LUMEN_DSV4_SKIP_OPTIONAL_NORMS", "1") != "1":
        return False
    path = os.path.join(
        megatron_root,
        "megatron",
        "core",
        "dist_checkpointing",
        "strategies",
        "torch.py",
    )
    with open(path) as f:
        content = f.read()
    orig = content
    needle = """            if sh_ten.key not in metadata.state_dict_metadata:
                raise KeyError(
                    f"{sh_ten.key} from model not in state dict:"
                    f" {sorted(metadata.state_dict_metadata.keys())}"
                )"""
    replacement = """            if sh_ten.key not in metadata.state_dict_metadata:
                if sh_ten.key.endswith(_LUMEN_OPTIONAL_MISSING_SUFFIXES):
                    logger.warning(
                        f"{sh_ten.key} from model not in state dict, will skip"
                    )
                    continue
                raise KeyError(
                    f"{sh_ten.key} from model not in state dict:"
                    f" {sorted(metadata.state_dict_metadata.keys())}"
                )"""
    if needle in content:
        content = content.replace(needle, replacement, 1)

    # _validate_global_shapes only guards Megatron's own shape check; the keys are
    # still handed to the PyT DCP planner, which raises on any missing shard.
    plan_needle = """    def create_local_plan(self) -> LoadPlan:
        \"\"\"Runs additional shapes validation.\"\"\"
        self._validate_global_shapes(self.metadata, self.shapes_validation_sharded_tensors)

        with self._temporarily_bypass_shape_validation():"""
    plan_replacement = """    def _drop_optional_missing_keys(self) -> None:
        ckpt_keys = self.metadata.state_dict_metadata
        for key in [
            key
            for key in self.state_dict
            if key not in ckpt_keys and key.endswith(_LUMEN_OPTIONAL_MISSING_SUFFIXES)
        ]:
            logger.warning(f"{key} not in checkpoint, keeping initialized value")
            del self.state_dict[key]

    def create_local_plan(self) -> LoadPlan:
        \"\"\"Runs additional shapes validation.\"\"\"
        self._validate_global_shapes(self.metadata, self.shapes_validation_sharded_tensors)
        self._drop_optional_missing_keys()

        with self._temporarily_bypass_shape_validation():"""
    if plan_needle in content:
        content = content.replace(plan_needle, plan_replacement, 1)

    suffix_anchor = "logger = getLogger(__name__)"
    suffix_decl = """logger = getLogger(__name__)

# DSV4 converted checkpoints omit these model-side buffers: hash-routed layers have
# no router expert_bias, and flash layers have no q_norm/kv_norm.
_LUMEN_OPTIONAL_MISSING_SUFFIXES = (
    ".self_attention.q_norm.weight",
    ".self_attention.kv_norm.weight",
    ".self_attention.q_norm._norm.weight",
    ".self_attention.kv_norm._norm.weight",
    ".mlp.router.expert_bias",
)"""
    if (
        "_LUMEN_OPTIONAL_MISSING_SUFFIXES = (" not in content
        and suffix_anchor in content
    ):
        content = content.replace(suffix_anchor, suffix_decl, 1)

    if content != orig:
        with open(path, "w") as f:
            f.write(content)
        return True
    return False


def patch_shared_expert_clamp(megatron_root: str) -> bool:
    path = os.path.join(
        megatron_root, "megatron", "core", "transformer", "moe", "shared_experts.py"
    )
    with open(path) as f:
        content = f.read()
    orig = content

    needle = '        assert config.add_bias_linear == False, "bias is not supported in the shared experts, "'
    if "activation_func_clamp_shared_expert" not in content and needle in content:
        content = content.replace(
            needle,
            """        if not config.activation_func_clamp_shared_expert:
            config.activation_func_clamp_value = None

""" + needle,
            1,
        )

    if content != orig:
        with open(path, "w") as f:
            f.write(content)
        return True
    return False


def patch_transformer_block(megatron_root: str) -> bool:
    path = os.path.join(
        megatron_root, "megatron", "core", "transformer", "transformer_block.py"
    )

    with open(path) as f:
        content = f.read()
    original = content

    # Add HC utility and head params after _build_layers() call.
    build_layers_call = "        self._build_layers()"
    old_hc_block = """
        # DSV4 Hyper-Connection head params (last PP rank only)
        if getattr(self.config, 'dsv4_mode', False):
            from atomrl.lumen.models.dsv4.ops.hyper_connection import HCHeadParams
            from megatron.core import parallel_state as mpu
            if mpu.is_pipeline_last_stage():
                self.hc_head_params = HCHeadParams(self.config)
"""
    hc_block = """
        # DSV4 Hyper-Connection state and learned output contraction.
        if getattr(self.config, 'dsv4_mode', False):
            from atomrl.lumen.models.dsv4.ops.hyper_connection import (
                DeepSeekV4HyperConnectionUtil,
                HCHeadParams,
            )
            self.hc_util = DeepSeekV4HyperConnectionUtil(self.config)
            if self.post_process:
                self.hc_head_params = HCHeadParams(self.config)
"""
    if old_hc_block in content:
        content = content.replace(old_hc_block, hc_block, 1)
    elif build_layers_call in content and "self.hc_util =" not in content:
        content = content.replace(
            build_layers_call,
            build_layers_call + hc_block,
            1,
        )

    expand_anchor = (
        "        hidden_states = make_viewless_tensor("
        "inp=hidden_states, requires_grad=True, keep_graph=True)"
    )
    expand_block = """

        # DSV4 mHC keeps four residual streams through every transformer layer.
        if getattr(self.config, 'dsv4_mode', False) and self.pre_process:
            hidden_states = self.hc_util.block_expand(hidden_states)
"""
    if (
        expand_anchor in content
        and "self.hc_util.block_expand(hidden_states)" not in content
    ):
        content = content.replace(expand_anchor, expand_anchor + expand_block, 1)

    head_anchor = "        # Final layer norm."
    head_block = """        # Collapse DSV4 mHC streams before the final norm and LM head.
        if (
            getattr(self.config, 'dsv4_mode', False)
            and self.post_process
            and hasattr(self, "hc_head_params")
        ):
            hidden_states = self.hc_util.block_head(
                hidden_states,
                self.hc_head_params.hc_head_fn,
                self.hc_head_params.hc_head_scale,
                self.hc_head_params.hc_head_base,
            )

"""
    if head_anchor in content and "self.hc_util.block_head(" not in content:
        content = content.replace(head_anchor, head_block + head_anchor, 1)

    if content != original:
        with open(path, "w") as f:
            f.write(content)
        return True
    return False


def patch_transformer_layer(megatron_root: str) -> bool:
    path = os.path.join(
        megatron_root, "megatron", "core", "transformer", "transformer_layer.py"
    )

    with open(path) as f:
        content = f.read()
    original = content

    # Add per-layer HC params after self.mlp assignment in __init__
    # Find a stable anchor point — after the mlp is built
    if "dsv4_mode" not in content and "self.mlp = build_module" in content:
        # Find the class __init__ and add HC params
        # Look for the end of __init__ where self.mlp is assigned
        anchor = "        self.bias_dropout_add_exec_handler = torch.enable_grad"
        if anchor in content:
            hc_layer = """
        # DSV4 Hyper-Connection per-layer params
        if getattr(self.config, 'dsv4_mode', False):
            import torch.nn as nn
            hc_mult = self.config.dsv4_hc_mult or 4
            hc_dim = hc_mult * self.config.hidden_size
            mix_size = (2 + hc_mult) * hc_mult
            self.hc_attn_fn = nn.Parameter(torch.zeros(mix_size, hc_dim, dtype=torch.float32))
            self.hc_attn_base = nn.Parameter(torch.zeros(mix_size, dtype=torch.float32))
            self.hc_attn_scale = nn.Parameter(torch.zeros(3, dtype=torch.float32))
            self.hc_ffn_fn = nn.Parameter(torch.zeros(mix_size, hc_dim, dtype=torch.float32))
            self.hc_ffn_base = nn.Parameter(torch.zeros(mix_size, dtype=torch.float32))
            self.hc_ffn_scale = nn.Parameter(torch.zeros(3, dtype=torch.float32))
            for p in [self.hc_attn_fn, self.hc_attn_base, self.hc_attn_scale,
                       self.hc_ffn_fn, self.hc_ffn_base, self.hc_ffn_scale]:
                p._keep_fp32 = True

"""
            content = content.replace(anchor, hc_layer + anchor)

    # Lumen's DSV4 attention returns a tensor, while Megatron BDA expects
    # the standard (output, bias) pair.
    attention_anchor = '        nvtx_range_pop(suffix="self_attention")'
    attention_compat = """

        if isinstance(attention_output_with_bias, torch.Tensor):
            attention_output_with_bias = (attention_output_with_bias, None)
"""
    if (
        attention_anchor in content
        and "isinstance(attention_output_with_bias, torch.Tensor)" not in content
    ):
        content = content.replace(
            attention_anchor, attention_anchor + attention_compat, 1
        )

    attention_residual = """        # Residual connection.
        residual = hidden_states

        # Optional Input Layer norm"""
    attention_pre = """        # Residual connection.
        residual = hidden_states

        # DSV4 mHC contracts the residual streams for the attention sublayer.
        if getattr(self.config, 'dsv4_mode', False):
            from atomrl.lumen.models.dsv4.ops.hyper_connection import (
                DeepSeekV4HyperConnectionUtil,
            )
            hc_util = DeepSeekV4HyperConnectionUtil(self.config)
            hidden_states, hc_attn_post, hc_attn_comb = hc_util.layer_pre(
                hidden_states,
                self.hc_attn_fn,
                self.hc_attn_scale,
                self.hc_attn_base,
            )

        # Optional Input Layer norm"""
    if (
        attention_residual in content
        and "hc_attn_post, hc_attn_comb = hc_util.layer_pre" not in content
    ):
        content = content.replace(attention_residual, attention_pre, 1)

    attention_bda = """        nvtx_range_push(suffix="self_attn_bda")
        if using_fused_tp_inference_kernel:"""
    attention_hc_post = """        nvtx_range_push(suffix="self_attn_bda")
        if getattr(self.config, 'dsv4_mode', False):
            hidden_states = hc_util.layer_post(
                attention_output_with_bias,
                residual,
                hc_attn_post,
                hc_attn_comb,
            )
        elif using_fused_tp_inference_kernel:"""
    if (
        attention_bda in content
        and "                hc_attn_post,\n                hc_attn_comb,"
        not in content
    ):
        content = content.replace(attention_bda, attention_hc_post, 1)

    mlp_pre_anchor = """        # Residual connection.
        residual = hidden_states

        # Optional Layer norm post the cross-attention."""
    mlp_pre = """        # Residual connection.
        residual = hidden_states

        # DSV4 mHC contracts the residual streams for the MLP sublayer.
        if getattr(self.config, 'dsv4_mode', False):
            from atomrl.lumen.models.dsv4.ops.hyper_connection import (
                DeepSeekV4HyperConnectionUtil,
            )
            hc_util = DeepSeekV4HyperConnectionUtil(self.config)
            hidden_states, hc_ffn_post, hc_ffn_comb = hc_util.layer_pre(
                hidden_states,
                self.hc_ffn_fn,
                self.hc_ffn_scale,
                self.hc_ffn_base,
            )

        # Optional Layer norm post the cross-attention."""
    if (
        mlp_pre_anchor in content
        and "hc_ffn_post, hc_ffn_comb = hc_util.layer_pre" not in content
    ):
        content = content.replace(mlp_pre_anchor, mlp_pre, 1)

    post_call = "return self._forward_post_mlp(mlp_output_with_bias, residual)"
    post_call_hc = """return self._forward_post_mlp(
                mlp_output_with_bias,
                residual,
                hc_ffn_post=hc_ffn_post if getattr(self.config, 'dsv4_mode', False) else None,
                hc_ffn_comb=hc_ffn_comb if getattr(self.config, 'dsv4_mode', False) else None,
            )"""
    if post_call in content and "hc_ffn_post=hc_ffn_post" not in content:
        content = content.replace(post_call, post_call_hc, 1)

    post_signature = "    def _forward_post_mlp(self, mlp_output_with_bias, residual):"
    post_signature_hc = """    def _forward_post_mlp(
        self,
        mlp_output_with_bias,
        residual,
        *,
        hc_ffn_post=None,
        hc_ffn_comb=None,
    ):"""
    if post_signature in content:
        content = content.replace(post_signature, post_signature_hc, 1)

    mlp_bda = """        nvtx_range_push(suffix="mlp_bda")
        if using_fused_tp_inference_kernel:"""
    mlp_hc_post = """        nvtx_range_push(suffix="mlp_bda")
        if getattr(self.config, 'dsv4_mode', False):
            from atomrl.lumen.models.dsv4.ops.hyper_connection import (
                DeepSeekV4HyperConnectionUtil,
            )
            hc_util = DeepSeekV4HyperConnectionUtil(self.config)
            hidden_states = hc_util.layer_post(
                mlp_output_with_bias,
                residual,
                hc_ffn_post,
                hc_ffn_comb,
            )
        elif using_fused_tp_inference_kernel:"""
    old_mlp_hc_post = """        nvtx_range_push(suffix="mlp_bda")
        if getattr(self.config, 'dsv4_mode', False):
            hidden_states = hc_util.layer_post(
                mlp_output_with_bias,
                residual,
                hc_ffn_post,
                hc_ffn_comb,
            )
        elif using_fused_tp_inference_kernel:"""
    if old_mlp_hc_post in content:
        content = content.replace(old_mlp_hc_post, mlp_hc_post, 1)
    if (
        mlp_bda in content
        and "                hc_ffn_post,\n                hc_ffn_comb," not in content
    ):
        content = content.replace(mlp_bda, mlp_hc_post, 1)

    if content != original:
        with open(path, "w") as f:
            f.write(content)
        return True
    return False


def patch_eav_specs(megatron_root: str) -> bool:
    """Add dsv4 branch to get_experimental_attention_variant_module_spec."""
    path = os.path.join(
        megatron_root,
        "megatron",
        "core",
        "models",
        "gpt",
        "experimental_attention_variant_module_specs.py",
    )
    with open(path) as f:
        content = f.read()
    original = content

    # Replace the else branch to handle dsv4 before raising.
    # Lumen's get_dsv4_spec monkey-patches this at runtime, but the Literal
    # type needs to accept 'dsv4' without erroring.
    old_else = """    else:
        raise ValueError(
            f"Invalid experimental attention variant: {config.experimental_attention_variant}"
        )"""
    new_else = """    elif config.experimental_attention_variant == "dsv4":
        # DSV4 spec is injected by Lumen's get_dsv4_spec() monkey-patch at runtime
        raise ValueError(
            "DSV4 attention variant requires Lumen's get_dsv4_spec() — "
            "call it before get_experimental_attention_variant_module_spec()"
        )
    else:
        raise ValueError(
            f"Invalid experimental attention variant: {config.experimental_attention_variant}"
        )"""
    if old_else in content and "dsv4" not in content:
        content = content.replace(old_else, new_else)

    if content != original:
        with open(path, "w") as f:
            f.write(content)
        return True
    return False


def patch_tp_layers(megatron_root: str) -> bool:
    """Add condition_init_method to tensor_parallel/layers.py (needed by Lumen linears)."""
    path = os.path.join(
        megatron_root, "megatron", "core", "tensor_parallel", "layers.py"
    )
    with open(path) as f:
        content = f.read()
    if "def condition_init_method" in content:
        return False
    stub = '''

def condition_init_method(config, init_method):
    """Condition weight initialization on config (Lumen compatibility shim).

    Returns the init_method unchanged — Lumen's LumenColumnParallelLinear calls
    this during CPU initialization. Xavier-uniform override is not used for DSV4.
    """
    if getattr(config, "init_method_xavier_uniform", False):
        import torch.nn.init as init
        return init.xavier_uniform_
    return init_method

'''
    content += stub
    with open(path, "w") as f:
        f.write(content)
    return True


def _env_flag(name: str, default: str = "0"):
    return lambda: os.environ.get(name, default) == "1"


register_patch(
    "dsv4_transformer_config",
    PatchPhase.SOURCE,
    description="Add dsv4 variant and dsv4_* fields to TransformerConfig",
    tags=frozenset({"dsv4", "config", "megatron", "rocm"}),
    config_fields=(
        "dsv4_mode",
        "dsv4_hc_mult",
        "dsv4_hc_sinkhorn_iters",
        "dsv4_hc_eps",
        "dsv4_compress_ratios",
        "dsv4_compress_rope_theta",
        "dsv4_o_groups",
        "dsv4_o_lora_rank",
        "dsv4_n_hash_layers",
        "dsv4_window_size",
    ),
)(patch_transformer_config)

register_patch(
    "moe_sqrtsoftplus",
    PatchPhase.SOURCE,
    description="Add sqrtsoftplus MoE router score function",
    depends_on=("dsv4_transformer_config",),
    tags=frozenset({"dsv4", "moe", "megatron", "rocm"}),
)(patch_moe_sqrtsoftplus)

register_patch(
    "dsv4_training_config",
    PatchPhase.SOURCE,
    description="DSV4 finetune flags and typed compress ratios",
    depends_on=("dsv4_transformer_config",),
    tags=frozenset({"dsv4", "config", "megatron", "rocm"}),
    config_fields=(
        "activation_func_clamp_shared_expert",
        "freeze_e_score_correction_bias",
        "moe_router_freeze_gate",
    ),
)(patch_dsv4_training_config)

register_patch(
    "moe_router_freeze",
    PatchPhase.SOURCE,
    description="Honor moe_router_freeze_gate in TopKRouter",
    depends_on=("dsv4_training_config",),
    tags=frozenset({"dsv4", "moe", "megatron", "rocm"}),
)(patch_moe_router_freeze)

register_patch(
    "dsv4_hash_routing",
    PatchPhase.SOURCE,
    description="Hash routing via tid2eid[input_ids] for early MoE layers",
    depends_on=("moe_sqrtsoftplus",),
    tags=frozenset({"dsv4", "moe", "megatron", "rocm"}),
)(patch_dsv4_hash_routing)

register_patch(
    "skip_none_router_expert_bias",
    PatchPhase.SOURCE,
    description="Skip expert_bias updates when expert_bias is None",
    tags=frozenset({"dsv4", "moe", "megatron", "rocm"}),
)(patch_skip_none_router_expert_bias)

register_patch(
    "dist_ckpt_skip_dsv4_norms",
    PatchPhase.SOURCE,
    description="Skip missing optional DSV4 norm/router ckpt keys on load",
    enabled=_env_flag("LUMEN_DSV4_SKIP_OPTIONAL_NORMS", "1"),
    tags=frozenset({"dsv4", "checkpoint", "megatron", "rocm"}),
)(patch_dist_ckpt_skip_optional_dsv4_norms)

register_patch(
    "shared_expert_clamp",
    PatchPhase.SOURCE,
    description="Honor activation_func_clamp_shared_expert on shared experts",
    depends_on=("dsv4_training_config",),
    tags=frozenset({"dsv4", "moe", "megatron", "rocm"}),
)(patch_shared_expert_clamp)

register_patch(
    "dsv4_transformer_block",
    PatchPhase.SOURCE,
    description="mHC expand/collapse hooks in TransformerBlock",
    depends_on=("dsv4_transformer_config",),
    tags=frozenset({"dsv4", "hc", "megatron", "rocm"}),
)(patch_transformer_block)

register_patch(
    "dsv4_transformer_layer",
    PatchPhase.SOURCE,
    description="Per-layer HC params and mHC pre/post in TransformerLayer",
    depends_on=("dsv4_transformer_config",),
    tags=frozenset({"dsv4", "hc", "megatron", "rocm"}),
)(patch_transformer_layer)

register_patch(
    "dsv4_eav_specs",
    PatchPhase.SOURCE,
    description="Add dsv4 branch to experimental attention variant specs",
    depends_on=("dsv4_transformer_config",),
    tags=frozenset({"dsv4", "attention", "megatron", "rocm"}),
)(patch_eav_specs)

register_patch(
    "tp_layers_condition_init",
    PatchPhase.SOURCE,
    description="condition_init_method shim for Lumen parallel linears",
    tags=frozenset({"dsv4", "tp", "megatron", "rocm"}),
)(patch_tp_layers)
