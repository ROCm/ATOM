#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

"""Add DSV4 support to ROCm/Megatron-LM (rocm_dev branch).

Adds the minimal TransformerConfig fields and TransformerBlock/Layer hooks
needed by Lumen's ``get_dsv4_spec()`` to construct the DSV4 model.  The actual
DSV4 attention/compressor/indexer/HC layers come from the Lumen library — this
patch only makes Megatron's config and block infrastructure accept them.

Changes:
  1. TransformerConfig: add 'dsv4' to experimental_attention_variant Literal,
     add dsv4_* fields (hc_mult, compress_ratios, window_size, etc.),
     add dsv4_mode post_init flag.
  2. TransformerBlock: add HCHeadParams on last PP rank when dsv4_mode.
  3. TransformerLayer: add per-layer HC params (hc_attn_fn/base/scale,
     hc_ffn_fn/base/scale) when dsv4_mode.

Usage (inside container):
    python3 examples/GRPO/dsv4/patch_rocm_megatron_dsv4.py /workspace/Megatron-LM
"""

from __future__ import annotations

import ast
import os
import re
import sys

CHECKPOINT_REPLAY_CAPABILITY = "ATOMRL_R3_CAPABILITY_CHECKPOINT_REPLAY_BACKWARD"
ROUTER_REPLAY_FIFO_CAPABILITY = "ATOMRL_R3_CAPABILITY_ROUTER_REPLAY_FIFO"
REPLAY_DIAGNOSTICS_CAPABILITY = "ATOMRL_R3_CAPABILITY_REPLAY_DIAGNOSTICS"
STREAMED_ADAM_CAPABILITY = "ATOMRL_DSV4_CAPABILITY_STREAMED_ADAM"


def _stamp_capabilities(content: str, markers: tuple[str, ...]) -> str:
    """Add import-visible capability flags without duplicating existing flags."""
    missing = []
    for marker in markers:
        true_assignment = re.compile(rf"(?m)^{re.escape(marker)}\s*=\s*True\s*$")
        if true_assignment.search(content):
            continue
        existing_assignment = re.compile(rf"(?m)^{re.escape(marker)}\s*=.*$")
        if existing_assignment.search(content):
            content = existing_assignment.sub(f"{marker} = True", content, count=1)
        else:
            missing.append(marker)
    if not missing:
        return content
    separator = "" if content.endswith("\n") else "\n"
    assignments = "\n".join(f"{marker} = True" for marker in missing)
    return f"{content}{separator}\n{assignments}\n"


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


def _top_level_source_extent(content: str, start: int) -> tuple[int, int]:
    """Return the extent of one top-level def/class starting at ``start``."""
    line_end = content.find("\n", start)
    search_start = len(content) if line_end < 0 else line_end + 1
    boundary = re.search(
        r"(?m)^(?:def|class)\s+[A-Za-z_]\w*",
        content[search_start:],
    )
    end = len(content) if boundary is None else search_start + boundary.start()
    return start, end


def _find_router_definition(
    megatron_root: str,
) -> tuple[str, str, int, int] | None:
    """Find the preferred file that actually defines the routing function."""
    moe_dir = os.path.join(megatron_root, "megatron", "core", "transformer", "moe")
    for filename in ("moe_utils.py", "router.py"):
        candidate = os.path.join(moe_dir, filename)
        if not os.path.isfile(candidate):
            continue
        with open(candidate) as f:
            content = f.read()
        definition = re.search(
            r"(?m)^def topk_routing_with_score_function\(",
            content,
        )
        if definition is not None:
            start, end = _top_level_source_extent(content, definition.start())
            return candidate, content, start, end
    return None


def _literal_tokens(values: str) -> set[str]:
    return set(re.findall(r"""["']([^"']+)["']""", values))


def _config_has_exact_sqrtsoftplus(content: str) -> bool:
    score_literal = re.search(
        r"moe_router_score_function:\s*Literal\[(?P<values>[^\]]+)\]",
        content,
    )
    return score_literal is not None and "sqrtsoftplus" in _literal_tokens(
        score_literal.group("values")
    )


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

    # 2. Admit the DSV4 checkpoint's score function without changing defaults.
    score_literal = re.compile(
        r"(moe_router_score_function:\s*Literal\[)(?P<values>[^\]]+)(\])"
    )

    def add_sqrtsoftplus(match: re.Match[str]) -> str:
        values = match.group("values")
        if "sqrtsoftplus" in _literal_tokens(values):
            return match.group(0)
        quote = '"' if '"' in values else "'"
        return (
            f"{match.group(1)}{values}, {quote}sqrtsoftplus{quote}" f"{match.group(3)}"
        )

    content = score_literal.sub(add_sqrtsoftplus, content, count=1)

    # 3. DSV4 uses its checkpoint expert bias with sqrtsoftplus. Megatron's
    # upstream validation only admits expert bias with sigmoid.
    expert_bias_guard = re.compile(
        r"(self\.moe_router_enable_expert_bias\s+and\s+)"
        r"self\.moe_router_score_function\s*!=\s*[\"']sigmoid[\"']"
    )
    content = expert_bias_guard.sub(
        r'\1self.moe_router_score_function not in ("sigmoid", "sqrtsoftplus")',
        content,
        count=1,
    )

    # 4. Add DSV4 fields after the DSA section
    # Find the right place — after the last DSA field
    dsa_marker = "    ####################\n    # DSA\n    ####################"
    if dsa_marker in content and "dsv4_mode: bool = False" not in content:
        # Find end of DSA fields section (next ## or blank-line-then-field)
        # Insert DSV4 fields after all dsa_ fields
        # Find last dsa_ field
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
    dsv4_compress_ratios: Optional[list] = None
    dsv4_compress_rope_theta: float = 160000.0
    dsv4_o_groups: Optional[int] = None
    dsv4_o_lora_rank: Optional[int] = None
    dsv4_n_hash_layers: int = 0
    dsv4_window_size: int = 128

"""
            content = content[:insert_pos] + dsv4_fields + content[insert_pos:]

    # 5. Add dsv4_mode = True in __post_init__ when variant == "dsv4"
    dsa_post_init = '        if self.experimental_attention_variant == "dsa":'
    if (
        dsa_post_init in content
        and 'if self.experimental_attention_variant == "dsv4":' not in content
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


def patch_moe_router_score_function(megatron_root: str) -> bool:
    """Add sqrt-softplus routing while keeping expert bias selection-only."""
    definition = _find_router_definition(megatron_root)
    if definition is None:
        return False
    path, content, function_start, function_end = definition
    function_source = content[function_start:function_end]
    if 'elif score_function in ("sigmoid", "sqrtsoftplus"):' in function_source:
        return False

    branch = re.search(
        r'(?m)^(?P<indent>[ \t]*)elif score_function == ["\']sigmoid["\']:\s*$',
        function_source,
    )
    if branch is None:
        return False
    branch_start = function_start + branch.start()
    branch_body_start = function_start + branch.end()
    indent = branch.group("indent")
    error_else = re.search(
        rf"(?m)^{re.escape(indent)}else:\s*\n"
        rf"{re.escape(indent)}    raise ValueError\(",
        content[branch_body_start:function_end],
    )
    if error_else is None:
        return False
    branch_end = branch_body_start + error_else.start()
    replacement = (
        f'{indent}elif score_function in ("sigmoid", "sqrtsoftplus"):\n'
        f'{indent}    if score_function == "sigmoid":\n'
        f"{indent}        scores = torch.sigmoid(logits.float()).type_as(logits)\n"
        f"{indent}    else:\n"
        f"{indent}        scores = torch.nn.functional.softplus("
        "logits.float()).sqrt().type_as(logits)\n"
        f"{indent}    if expert_bias is not None:\n"
        f"{indent}        scores_for_routing = scores + expert_bias\n"
        f"{indent}        _, top_indices = compute_topk("
        "scores_for_routing, topk, num_groups, group_topk)\n"
        f"{indent}        scores = torch.gather("
        "scores, dim=1, index=top_indices).type_as(logits)\n"
        f"{indent}    else:\n"
        f"{indent}        scores, top_indices = compute_topk("
        "scores, topk, num_groups, group_topk)\n"
        f"{indent}    probs = scores / "
        "scores.sum(dim=-1, keepdim=True).clamp(min=1e-20) "
        "if topk > 1 else scores\n"
    )
    content = content[:branch_start] + replacement + content[branch_end:]
    with open(path, "w") as f:
        f.write(content)
    return True


def patch_checkpoint_router_replay(megatron_root: str) -> bool:
    """Scope backward routing replay to activation-checkpoint recomputation."""
    path = os.path.join(
        megatron_root, "megatron", "core", "tensor_parallel", "random.py"
    )
    with open(path) as f:
        content = f.read()
    original = content
    class_definition = re.search(r"(?m)^class CheckpointFunction\(", content)
    if class_definition is None:
        return False
    class_start, class_end = _top_level_source_extent(content, class_definition.start())
    marker = "router_replay_actions = ["
    if marker not in content[class_start:class_end]:
        anchor = re.search(
            r"(?m)^(?P<indent>[ \t]*)with torch\.enable_grad\(\):\s*\n"
            r"(?P=indent)    outputs = ctx\.run_function\(\*detached_inputs\)\s*$",
            content[class_start:class_end],
        )
        if anchor is None:
            return False
        anchor_start = class_start + anchor.start()
        anchor_end = class_start + anchor.end()
        indent = anchor.group("indent")
        replacement = (
            f"{indent}from megatron.core.transformer.moe.router_replay import (\n"
            f"{indent}    RouterReplay,\n"
            f"{indent}    RouterReplayAction,\n"
            f"{indent})\n"
            f"{indent}router_replays = list(RouterReplay.global_router_replay_instances)\n"
            f"{indent}router_replay_actions = [\n"
            f"{indent}    instance.router_replay_action for instance in router_replays\n"
            f"{indent}]\n"
            f"{indent}try:\n"
            f"{indent}    for instance in router_replays:\n"
            f"{indent}        instance.set_router_replay_action(\n"
            f"{indent}            RouterReplayAction.REPLAY_BACKWARD\n"
            f"{indent}        )\n"
            f"{indent}    with torch.enable_grad():\n"
            f"{indent}        outputs = ctx.run_function(*detached_inputs)\n"
            f"{indent}finally:\n"
            f"{indent}    for instance, prior_action in zip(\n"
            f"{indent}        router_replays, router_replay_actions\n"
            f"{indent}    ):\n"
            f"{indent}        if prior_action is None:\n"
            f"{indent}            instance.clear_router_replay_action()\n"
            f"{indent}        else:\n"
            f"{indent}            instance.set_router_replay_action(prior_action)"
        )
        content = content[:anchor_start] + replacement + content[anchor_end:]

    class_definition = re.search(r"(?m)^class CheckpointFunction\(", content)
    class_start, class_end = _top_level_source_extent(content, class_definition.start())
    checkpoint_source = content[class_start:class_end]
    if not (
        "router_replay_actions = [" in checkpoint_source
        and "RouterReplayAction.REPLAY_BACKWARD" in checkpoint_source
        and "finally:" in checkpoint_source
        and "instance.set_router_replay_action(prior_action)" in checkpoint_source
    ):
        raise RuntimeError(
            "CheckpointFunction replay patch failed structural postconditions; "
            "refusing to stamp its runtime capability"
        )
    content = _stamp_capabilities(
        content,
        (CHECKPOINT_REPLAY_CAPABILITY,),
    )
    if content == original:
        return False
    with open(path, "w") as f:
        f.write(content)
    return True


def _router_replay_diagnostics_state(content: str) -> str:
    """Return ``absent``, ``complete``, or ``partial`` for the diagnostics."""
    diagnostic_names = {
        "recompute_forward_indices",
        "recompute_compared_ids",
        "recompute_flips",
        "reset_recompute_diagnostics",
        "get_recompute_diagnostics",
    }
    has_diagnostic_marker = any(name in content for name in diagnostic_names)
    try:
        tree = ast.parse(content)
    except SyntaxError:
        return "partial" if has_diagnostic_marker else "absent"

    classes = [
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "RouterReplay"
    ]
    if len(classes) != 1:
        return "partial" if has_diagnostic_marker else "absent"
    methods = {
        node.name: node
        for node in classes[0].body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    required_methods = {
        "__init__",
        "reset_recompute_diagnostics",
        "get_recompute_diagnostics",
        "get_replay_topk",
    }
    if not required_methods <= methods.keys():
        return "partial" if has_diagnostic_marker else "absent"

    def self_attr(node, name: str) -> bool:
        return (
            isinstance(node, ast.Attribute)
            and isinstance(node.value, ast.Name)
            and node.value.id == "self"
            and node.attr == name
        )

    def assigned_attrs(method) -> dict[str, ast.AST]:
        assigned = {}
        for node in ast.walk(method):
            if isinstance(node, (ast.Assign, ast.AnnAssign)):
                targets = (
                    node.targets if isinstance(node, ast.Assign) else [node.target]
                )
                for target in targets:
                    if (
                        isinstance(target, ast.Attribute)
                        and isinstance(target.value, ast.Name)
                        and target.value.id == "self"
                    ):
                        assigned[target.attr] = node.value
        return assigned

    init_assignments = assigned_attrs(methods["__init__"])
    init_ok = (
        isinstance(
            init_assignments.get("recompute_forward_indices"),
            ast.List,
        )
        and not init_assignments["recompute_forward_indices"].elts
        and all(
            isinstance(init_assignments.get(name), ast.Constant)
            and init_assignments[name].value == 0
            for name in ("recompute_compared_ids", "recompute_flips")
        )
    )

    reset = methods["reset_recompute_diagnostics"]
    reset_assignments = assigned_attrs(reset)
    reset_clears = any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "clear"
        and self_attr(node.func.value, "recompute_forward_indices")
        for node in ast.walk(reset)
    )
    reset_ok = reset_clears and all(
        isinstance(reset_assignments.get(name), ast.Constant)
        and reset_assignments[name].value == 0
        for name in ("recompute_compared_ids", "recompute_flips")
    )

    getter = methods["get_recompute_diagnostics"]
    getter_ok = any(
        isinstance(node, ast.Return)
        and isinstance(node.value, ast.Tuple)
        and len(node.value.elts) == 2
        and self_attr(node.value.elts[0], "recompute_compared_ids")
        and self_attr(node.value.elts[1], "recompute_flips")
        for node in ast.walk(getter)
    )

    replay = methods["get_replay_topk"]
    append_calls = [
        node
        for node in ast.walk(replay)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "append"
        and self_attr(node.func.value, "recompute_forward_indices")
        and len(node.args) == 1
        and isinstance(node.args[0], ast.Name)
        and node.args[0].id == "top_indices"
    ]
    pop_assignments = [
        node
        for node in ast.walk(replay)
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "forward_indices"
            for target in node.targets
        )
        and isinstance(node.value, ast.Call)
        and isinstance(node.value.func, ast.Attribute)
        and node.value.func.attr == "pop"
        and self_attr(node.value.func.value, "recompute_forward_indices")
        and len(node.value.args) == 1
        and isinstance(node.value.args[0], ast.Constant)
        and node.value.args[0].value == 0
    ]
    missing_forward_guard = any(
        isinstance(node, ast.If)
        and isinstance(node.test, ast.UnaryOp)
        and isinstance(node.test.op, ast.Not)
        and self_attr(node.test.operand, "recompute_forward_indices")
        and any(isinstance(child, ast.Raise) for child in ast.walk(node))
        for node in ast.walk(replay)
    )
    augmented = {
        node.target.attr
        for node in ast.walk(replay)
        if isinstance(node, ast.AugAssign)
        and isinstance(node.op, ast.Add)
        and isinstance(node.target, ast.Attribute)
        and isinstance(node.target.value, ast.Name)
        and node.target.value.id == "self"
    }
    replay_ok = (
        len(append_calls) == 1
        and len(pop_assignments) == 1
        and missing_forward_guard
        and {"recompute_compared_ids", "recompute_flips"} <= augmented
    )
    return (
        "complete"
        if init_ok and reset_ok and getter_ok and replay_ok
        else ("partial" if has_diagnostic_marker else "absent")
    )


def _router_replay_has_fifo_semantics(content: str) -> bool:
    """Verify RouterReplay records and consumes backward routes in FIFO order."""
    try:
        tree = ast.parse(content)
    except SyntaxError:
        return False
    replay_class = next(
        (
            node
            for node in tree.body
            if isinstance(node, ast.ClassDef) and node.name == "RouterReplay"
        ),
        None,
    )
    if replay_class is None:
        return False
    methods = {
        node.name: node
        for node in replay_class.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    record = methods.get("set_target_indices")
    replay = methods.get("get_replay_topk")
    if record is None or replay is None:
        return False

    def replay_list(node: ast.AST) -> bool:
        return (
            isinstance(node, ast.Attribute)
            and isinstance(node.value, ast.Name)
            and node.value.id == "self"
            and node.attr == "replay_backward_list"
        )

    appends = any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "append"
        and replay_list(node.func.value)
        for node in ast.walk(record)
    )
    pops_front = any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "pop"
        and replay_list(node.func.value)
        and len(node.args) == 1
        and isinstance(node.args[0], ast.Constant)
        and node.args[0].value == 0
        for node in ast.walk(replay)
    )
    return appends and pops_front


def patch_router_replay_diagnostics(megatron_root: str) -> bool:
    """Track exact forward/recompute replay agreement without tensor clones."""
    path = os.path.join(
        megatron_root,
        "megatron",
        "core",
        "transformer",
        "moe",
        "router_replay.py",
    )
    with open(path) as f:
        content = f.read()
    original = content
    diagnostic_state = _router_replay_diagnostics_state(content)
    if diagnostic_state == "complete":
        if not _router_replay_has_fifo_semantics(content):
            raise RuntimeError(
                "RouterReplay lacks FIFO record/replay semantics; "
                "refusing to stamp runtime capabilities"
            )
        content = _stamp_capabilities(
            content,
            (
                ROUTER_REPLAY_FIFO_CAPABILITY,
                REPLAY_DIAGNOSTICS_CAPABILITY,
            ),
        )
        if content == original:
            return False
        with open(path, "w") as f:
            f.write(content)
        return True
    if diagnostic_state == "partial":
        raise RuntimeError(
            "RouterReplay has a partial or malformed diagnostic patch; "
            "refusing to modify it"
        )

    class_definition = re.search(r"(?m)^class RouterReplay(?:\([^)]*\))?:", content)
    if class_definition is None:
        return False
    class_start, class_end = _top_level_source_extent(content, class_definition.start())
    class_source = content[class_start:class_end]

    init_anchor = re.search(
        r"(?m)^(?P<indent>[ \t]*)RouterReplay"
        r"\.global_router_replay_instances\.append\(self\)\s*$",
        class_source,
    )
    methods_anchor = re.search(
        r"(?m)^(?P<indent>[ \t]*)def set_target_indices"
        r"\(self,\s*topk_indices(?::[^)]*)?\):\s*$",
        class_source,
    )
    forward_anchor = re.search(
        r"(?m)^(?P<indent>[ \t]*)top_indices = self\.target_topk_idx\s*\n"
        r"(?:(?P=indent)#[^\n]*\n)?"
        r"(?P=indent)top_indices = top_indices\.to\(scores\.device\)\s*$",
        class_source,
    )
    backward_anchor = re.search(
        r"(?m)^(?P<indent>[ \t]*)top_indices = "
        r"self\.replay_backward_list\.pop\(0\)\s*\n"
        r"(?:(?P=indent)#[^\n]*\n)?"
        r"(?P=indent)top_indices = top_indices\.to\(scores\.device\)\s*$",
        class_source,
    )
    if (
        init_anchor is None
        or methods_anchor is None
        or forward_anchor is None
        or backward_anchor is None
    ):
        return False

    init_indent = init_anchor.group("indent")
    init_insert = (
        f"{init_indent}self.recompute_forward_indices: List[torch.Tensor] = []\n"
        f"{init_indent}self.recompute_compared_ids = 0\n"
        f"{init_indent}self.recompute_flips = 0\n"
    )
    init_pos = class_start + init_anchor.start()
    content = content[:init_pos] + init_insert + content[init_pos:]

    # Recompute offsets after the first insertion.
    class_definition = re.search(r"(?m)^class RouterReplay(?:\([^)]*\))?:", content)
    class_start, class_end = _top_level_source_extent(content, class_definition.start())
    class_source = content[class_start:class_end]
    methods_anchor = re.search(
        r"(?m)^(?P<indent>[ \t]*)def set_target_indices"
        r"\(self,\s*topk_indices(?::[^)]*)?\):\s*$",
        class_source,
    )
    method_indent = methods_anchor.group("indent")
    body_indent = method_indent + "    "
    diagnostics_methods = (
        f"{method_indent}def reset_recompute_diagnostics(self):\n"
        f'{body_indent}"""Reset counters and references without touching backward replay."""\n'
        f"{body_indent}self.recompute_forward_indices.clear()\n"
        f"{body_indent}self.recompute_compared_ids = 0\n"
        f"{body_indent}self.recompute_flips = 0\n\n"
        f"{method_indent}def get_recompute_diagnostics(self):\n"
        f'{body_indent}"""Return ``(compared expert ids, changed expert ids)``."""\n'
        f"{body_indent}return self.recompute_compared_ids, self.recompute_flips\n\n"
    )
    methods_pos = class_start + methods_anchor.start()
    content = content[:methods_pos] + diagnostics_methods + content[methods_pos:]

    class_definition = re.search(r"(?m)^class RouterReplay(?:\([^)]*\))?:", content)
    class_start, class_end = _top_level_source_extent(content, class_definition.start())
    class_source = content[class_start:class_end]
    forward_anchor = re.search(
        r"(?m)^(?P<indent>[ \t]*)top_indices = self\.target_topk_idx\s*\n"
        r"(?:(?P=indent)#[^\n]*\n)?"
        r"(?P=indent)top_indices = top_indices\.to\(scores\.device\)\s*$",
        class_source,
    )
    forward_pos = class_start + forward_anchor.end()
    forward_indent = forward_anchor.group("indent")
    content = (
        content[:forward_pos]
        + f"\n{forward_indent}self.recompute_forward_indices.append(top_indices)"
        + content[forward_pos:]
    )

    class_definition = re.search(r"(?m)^class RouterReplay(?:\([^)]*\))?:", content)
    class_start, class_end = _top_level_source_extent(content, class_definition.start())
    class_source = content[class_start:class_end]
    backward_anchor = re.search(
        r"(?m)^(?P<indent>[ \t]*)top_indices = "
        r"self\.replay_backward_list\.pop\(0\)\s*\n"
        r"(?:(?P=indent)#[^\n]*\n)?"
        r"(?P=indent)top_indices = top_indices\.to\(scores\.device\)\s*$",
        class_source,
    )
    backward_pos = class_start + backward_anchor.end()
    backward_indent = backward_anchor.group("indent")
    nested_indent = backward_indent + "    "
    compare = (
        f"\n{backward_indent}if not self.recompute_forward_indices:\n"
        f"{nested_indent}raise RuntimeError(\n"
        f'{nested_indent}    "RouterReplay recompute has no matching forward indices"\n'
        f"{nested_indent})\n"
        f"{backward_indent}forward_indices = self.recompute_forward_indices.pop(0)\n"
        f"{backward_indent}forward_flat = forward_indices.reshape(-1)\n"
        f"{backward_indent}backward_flat = top_indices.reshape(-1)\n"
        f"{backward_indent}overlap = min(forward_flat.numel(), backward_flat.numel())\n"
        f"{backward_indent}self.recompute_compared_ids += max(\n"
        f"{nested_indent}forward_flat.numel(), backward_flat.numel()\n"
        f"{backward_indent})\n"
        f"{backward_indent}self.recompute_flips += abs(\n"
        f"{nested_indent}forward_flat.numel() - backward_flat.numel()\n"
        f"{backward_indent})\n"
        f"{backward_indent}if overlap:\n"
        f"{nested_indent}self.recompute_flips += int(\n"
        f"{nested_indent}    (forward_flat[:overlap] != backward_flat[:overlap])\n"
        f"{nested_indent}    .sum().item()\n"
        f"{nested_indent})"
    )
    content = content[:backward_pos] + compare + content[backward_pos:]
    if _router_replay_diagnostics_state(
        content
    ) != "complete" or not _router_replay_has_fifo_semantics(content):
        raise RuntimeError(
            "RouterReplay diagnostic/FIFO patch failed structural postconditions; "
            "refusing to stamp runtime capabilities"
        )
    content = _stamp_capabilities(
        content,
        (
            ROUTER_REPLAY_FIFO_CAPABILITY,
            REPLAY_DIAGNOSTICS_CAPABILITY,
        ),
    )
    with open(path, "w") as f:
        f.write(content)
    return True


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


def patch_distrib_optimizer_fp32_detach(megatron_root: str) -> bool:
    """Fix missing .detach() on FP32 param shard views in DistributedOptimizer.

    The BF16 path uses model_param.detach().view(-1)[...] but the FP32 path
    uses model_param.view(-1)[...] without .detach(). When FP32 params have
    requires_grad=True (e.g. DSV4 HC params, attn_sink, compressor APE), the
    view creates a non-leaf tensor, causing 'can't optimize a non-leaf Tensor'
    when HybridDeviceOptimizer (CPU offload) validates param groups.
    """
    path = os.path.join(
        megatron_root, "megatron", "core", "optimizer", "distrib_optimizer.py"
    )
    with open(path) as f:
        content = f.read()
    old = "shard_model_param = model_param.view(-1)["
    if old not in content:
        return False
    content = content.replace(old, "shard_model_param = model_param.detach().view(-1)[")
    with open(path, "w") as f:
        f.write(content)
    return True


def patch_distrib_optimizer_grad_copy(megatron_root: str) -> bool:
    """Route fully offloaded grads to HybridDeviceOptimizer without an FP32 GPU copy."""
    path = os.path.join(
        megatron_root, "megatron", "core", "optimizer", "distrib_optimizer.py"
    )
    with open(path) as f:
        content = f.read()
    marker = "shard_main_param in self.optimizer.gpu_params_map_cpu_copy"
    if marker in content:
        return False
    old = next(
        (
            candidate
            for candidate in (
                "shard_main_param.grad = shard_model_grad.float()",
                "shard_main_param.grad = shard_model_grad.to(shard_main_param)",
            )
            if candidate in content
        ),
        None,
    )
    if old is None:
        return False
    new = (
        "if (\n"
        "                            isinstance(self.optimizer, HybridDeviceOptimizer)\n"
        "                            and shard_main_param in self.optimizer.gpu_params_map_cpu_copy\n"
        "                        ):\n"
        "                            shard_main_param.decoupled_grad = shard_model_grad\n"
        "                        else:\n"
        "                            shard_main_param.grad = shard_model_grad.float()"
    )
    content = content.replace(old, new)
    with open(path, "w") as f:
        f.write(content)
    return True


def patch_distrib_optimizer_hdo_checkpoint(megatron_root: str) -> bool:
    """Keep mixed-dtype HDO parameter/state ordering stable for checkpoints.

    DistributedOptimizer first records model parameters in grad-buffer order,
    then rebuilds each optimizer group with native-FP32 shards before BF16
    shards.  The old ``model_param_group_index_map`` therefore points at the
    wrong HDO state whenever a group contains both dtypes. For DP=1, load
    coalesced HDO state directly into its existing tensors so the checkpoint
    does not temporarily allocate padded send and receive buffers.
    """
    path = os.path.join(
        megatron_root, "megatron", "core", "optimizer", "distrib_optimizer.py"
    )
    with open(path) as f:
        content = f.read()
    original = content
    marker = "checkpoint order must match optimizer.param_groups"
    build_anchor = (
        "        ) = self._build_model_and_main_param_groups(\n"
        "            self.gbuf_ranges, self.model_param_gbuf_map, "
        "self.opt_group_ranges, config\n"
        "        )"
    )
    if marker not in content:
        if build_anchor not in content:
            raise RuntimeError(
                "Unable to patch HDO checkpoint order: model/main group anchor missing"
            )
        remap = build_anchor + (
            "\n\n"
            "        # HDO checkpoint order must match optimizer.param_groups, which\n"
            "        # groups native-FP32 shards before BF16 shards.\n"
            "        for group_index, model_groups in enumerate(zip(\n"
            "            self.model_fp32_groups, self.model_float16_groups\n"
            "        )):\n"
            "            ordered_model_params = (*model_groups[0], *model_groups[1])\n"
            "            for group_order, model_param in enumerate(ordered_model_params):\n"
            "                self.model_param_group_index_map[model_param] = "
            "(group_index, group_order)"
        )
        content = content.replace(build_anchor, remap, 1)

    torch_hdo_step_marker = "Torch-backed HDO checkpoint uses a canonical Adam step"
    if torch_hdo_step_marker not in content:
        torch_step_block = (
            '            steps = list(set([s["step"].item() '
            'for s in inner_state_dict["state"].values()]))\n'
            "            assert len(steps) == 1\n"
            "            step = steps[0]"
        )
        if torch_step_block not in content:
            raise RuntimeError(
                "Unable to patch Torch-backed HDO checkpoint step: "
                "extraction anchor missing"
            )
        canonical_torch_step_block = (
            "            # Torch-backed HDO checkpoint uses a canonical Adam step.\n"
            + torch_step_block.splitlines()[0]
            + "\n"
            "            if len(steps) > 1 and isinstance(\n"
            "                self.optimizer, HybridDeviceOptimizer\n"
            "            ):\n"
            "                logger.warning(\n"
            '                    "canonicalizing divergent HDO Adam steps: "\n'
            '                    "minimum=%s maximum=%s unique_count=%s",\n'
            "                    min(steps),\n"
            "                    max(steps),\n"
            "                    len(steps),\n"
            "                )\n"
            "            else:\n"
            "                assert len(steps) == 1\n"
            "            step = max(steps)"
        )
        content = content.replace(
            torch_step_block,
            canonical_torch_step_block,
            1,
        )

    step_marker = "canonicalizing divergent HDO Adam steps"
    if step_marker not in content:
        step_prefix = (
            '                    steps = list(set([s["step"].item() '
            "for s in optimizer.state.values()]))\n"
        )
        step_suffixes = (
            (
                "                    assert len(steps) == 1\n"
                "                    step = steps[0]"
            ),
            (
                '                    assert len(steps) == 1, f"steps: '
                '{optimizer.state}"\n'
                "                    step = steps[0]"
            ),
        )
        old_step_block = next(
            (
                step_prefix + suffix
                for suffix in step_suffixes
                if step_prefix + suffix in content
            ),
            None,
        )
        if old_step_block is None:
            raise RuntimeError(
                "Unable to patch HDO checkpoint step: extraction anchor missing"
            )
        canonical_step_block = step_prefix + (
            "                    if len(steps) > 1:\n"
            "                        logger.warning(\n"
            '                            "canonicalizing divergent HDO Adam steps: "\n'
            '                            "minimum=%s maximum=%s unique_count=%s",\n'
            "                            min(steps),\n"
            "                            max(steps),\n"
            "                            len(steps),\n"
            "                        )\n"
            "                    step = max(steps)"
        )
        content = content.replace(old_step_block, canonical_step_block, 1)

    fast_path_marker = "DP=1 HDO checkpoint state in place"
    if fast_path_marker not in content:
        split_anchor = "            self.split_state_dict_if_needed(state_dict)"
        if split_anchor not in content:
            raise RuntimeError(
                "Unable to patch HDO checkpoint load: state split anchor missing"
            )
        fast_path = split_anchor + """

        if (
            data_parallel_world_size == 1
            and isinstance(self.optimizer, HybridDeviceOptimizer)
        ):
            def validation_message(
                reason,
                *,
                gbuf,
                dtype="n/a",
                bucket="all",
                key="metadata",
                param="n/a",
            ):
                return (
                    "DP=1 HDO checkpoint validation failed: "
                    f"gbuf={gbuf} dtype={dtype} bucket={bucket} "
                    f"key={key} param={param}: {reason}"
                )

            moment_dtype = getattr(
                self.optimizer, "_lumen_streamed_adam_moment_dtype", None
            )
            if moment_dtype not in (None, torch.float32, torch.bfloat16):
                raise ValueError(
                    validation_message(
                        "streamed Adam moment dtype must be float32 or bfloat16",
                        gbuf="all",
                        key="moment_dtype",
                    )
                )

            # Validate the complete layout before mutating any optimizer state.
            for gbuf_idx, gbuf_range_maps in enumerate(self.gbuf_ranges):
                if gbuf_idx not in state_dict:
                    raise ValueError(
                        validation_message(
                            f"missing checkpoint grad buffer {gbuf_idx}",
                            gbuf=gbuf_idx,
                        )
                    )
                for dtype, gbuf_range_map_for_all_buckets in gbuf_range_maps.items():
                    if dtype not in state_dict[gbuf_idx]:
                        raise ValueError(
                            validation_message(
                                f"missing checkpoint dtype {dtype}",
                                gbuf=gbuf_idx,
                                dtype=dtype,
                            )
                        )
                    checkpoint_state = state_dict[gbuf_idx][dtype]
                    buffer_numel_unpadded = self.buffers[gbuf_idx].numel_unpadded
                    if "numel_unpadded" not in checkpoint_state:
                        raise ValueError(
                            validation_message(
                                "missing checkpoint numel_unpadded",
                                gbuf=gbuf_idx,
                                dtype=dtype,
                                key="numel_unpadded",
                            )
                        )
                    checkpoint_numel_unpadded = checkpoint_state["numel_unpadded"]
                    if buffer_numel_unpadded != checkpoint_numel_unpadded:
                        raise RuntimeError(
                            validation_message(
                                "Number of unpadded elements must be same in "
                                f"current run ({buffer_numel_unpadded}) and "
                                f"checkpoint ({checkpoint_numel_unpadded})",
                                gbuf=gbuf_idx,
                                dtype=dtype,
                                key="numel_unpadded",
                            )
                        )
                    buckets = self.buffers[gbuf_idx].buckets
                    if len(gbuf_range_map_for_all_buckets) != len(buckets):
                        raise RuntimeError(
                            validation_message(
                                f"{len(gbuf_range_map_for_all_buckets)} range "
                                f"maps do not match {len(buckets)} buckets",
                                gbuf=gbuf_idx,
                                dtype=dtype,
                            )
                        )
                    bucket_numel_total = sum(
                        bucket.numel_unpadded for bucket in buckets
                    )
                    if bucket_numel_total != buffer_numel_unpadded:
                        raise RuntimeError(
                            validation_message(
                                f"bucket unpadded aggregate size "
                                f"{bucket_numel_total} does not match "
                                f"{buffer_numel_unpadded}",
                                gbuf=gbuf_idx,
                                dtype=dtype,
                            )
                        )
                    for key in ("param", "exp_avg", "exp_avg_sq"):
                        if key not in checkpoint_state:
                            raise ValueError(
                                validation_message(
                                    f"missing checkpoint state {key}",
                                    gbuf=gbuf_idx,
                                    dtype=dtype,
                                    key=key,
                                )
                            )
                        source = checkpoint_state[key]
                        if not torch.is_tensor(source):
                            raise ValueError(
                                validation_message(
                                    f"checkpoint state {key} is not a tensor",
                                    gbuf=gbuf_idx,
                                    dtype=dtype,
                                    key=key,
                                )
                            )
                        if source.numel() != buffer_numel_unpadded:
                            raise ValueError(
                                validation_message(
                                    f"checkpoint state {key} has "
                                    f"{source.numel()} elements; expected "
                                    f"{buffer_numel_unpadded}",
                                    gbuf=gbuf_idx,
                                    dtype=dtype,
                                    key=key,
                                )
                            )
                        if source.shape != (buffer_numel_unpadded,):
                            raise ValueError(
                                validation_message(
                                    f"checkpoint state {key} shape "
                                    f"{tuple(source.shape)} must be "
                                    f"({buffer_numel_unpadded},)",
                                    gbuf=gbuf_idx,
                                    dtype=dtype,
                                    key=key,
                                )
                            )
                        bucket_offset = 0
                        for bucket_idx, gbuf_range_map in enumerate(
                            gbuf_range_map_for_all_buckets
                        ):
                            bucket_numel_unpadded = buckets[
                                bucket_idx
                            ].numel_unpadded
                            if not (
                                0
                                <= bucket_numel_unpadded
                                <= buckets[bucket_idx].grad_data.numel()
                            ):
                                raise ValueError(
                                    validation_message(
                                        "invalid bucket unpadded size "
                                        f"{bucket_numel_unpadded}",
                                        gbuf=gbuf_idx,
                                        dtype=dtype,
                                        bucket=bucket_idx,
                                        key=key,
                                    )
                                )
                            if "param_map" not in gbuf_range_map:
                                raise ValueError(
                                    validation_message(
                                        "missing param_map",
                                        gbuf=gbuf_idx,
                                        dtype=dtype,
                                        bucket=bucket_idx,
                                        key=key,
                                    )
                                )
                            for model_param, param_range_map in gbuf_range_map[
                                "param_map"
                            ].items():
                                range_name = (
                                    "gbuf_world_in_bucket"
                                    if "gbuf_world_in_bucket" in param_range_map
                                    else "gbuf_world"
                                )
                                if range_name not in param_range_map:
                                    raise ValueError(
                                        validation_message(
                                            f"missing {range_name} range",
                                            gbuf=gbuf_idx,
                                            dtype=dtype,
                                            bucket=bucket_idx,
                                            key=key,
                                            param=model_param,
                                        )
                                    )
                                source_range = param_range_map[range_name]
                                if not (
                                    0
                                    <= source_range.start
                                    <= source_range.end
                                    <= bucket_numel_unpadded
                                ):
                                    raise ValueError(
                                        validation_message(
                                            f"invalid {range_name} range "
                                            f"[{source_range.start}, "
                                            f"{source_range.end}) for "
                                            f"{bucket_numel_unpadded} "
                                            "unpadded elements",
                                            gbuf=gbuf_idx,
                                            dtype=dtype,
                                            bucket=bucket_idx,
                                            key=key,
                                            param=model_param,
                                        )
                                    )
                                try:
                                    destination_states = (
                                        self._get_main_param_and_optimizer_states(
                                            model_param
                                        )
                                    )
                                    destination = destination_states[key]
                                except (KeyError, TypeError) as exc:
                                    raise RuntimeError(
                                        validation_message(
                                            "destination state is unavailable",
                                            gbuf=gbuf_idx,
                                            dtype=dtype,
                                            bucket=bucket_idx,
                                            key=key,
                                            param=model_param,
                                        )
                                    ) from exc
                                if not torch.is_tensor(destination):
                                    raise RuntimeError(
                                        validation_message(
                                            "destination is not a tensor",
                                            gbuf=gbuf_idx,
                                            dtype=dtype,
                                            bucket=bucket_idx,
                                            key=key,
                                            param=model_param,
                                        )
                                    )
                                range_numel = (
                                    source_range.end - source_range.start
                                )
                                if destination.numel() != range_numel:
                                    raise RuntimeError(
                                        validation_message(
                                            f"destination {key} has "
                                            f"{destination.numel()} elements; "
                                            f"expected {range_numel}",
                                            gbuf=gbuf_idx,
                                            dtype=dtype,
                                            bucket=bucket_idx,
                                            key=key,
                                            param=model_param,
                                        )
                                    )
                                if destination.shape != (range_numel,):
                                    raise RuntimeError(
                                        validation_message(
                                            f"destination {key} shape "
                                            f"{tuple(destination.shape)} must "
                                            f"be ({range_numel},)",
                                            gbuf=gbuf_idx,
                                            dtype=dtype,
                                            bucket=bucket_idx,
                                            key=key,
                                            param=model_param,
                                        )
                                    )
                            bucket_offset += bucket_numel_unpadded
                        if bucket_offset != source.numel():
                            raise RuntimeError(
                                validation_message(
                                    f"final offset {bucket_offset} does not "
                                    f"match source length {source.numel()}",
                                    gbuf=gbuf_idx,
                                    dtype=dtype,
                                    key=key,
                                )
                            )

            print(
                "[AtomRL] loading DP=1 HDO checkpoint state in place",
                flush=True,
            )
            for gbuf_idx, gbuf_range_maps in enumerate(self.gbuf_ranges):
                for dtype, gbuf_range_map_for_all_buckets in gbuf_range_maps.items():
                    for key in ("param", "exp_avg", "exp_avg_sq"):
                        source = state_dict[gbuf_idx][dtype][key]
                        bucket_offset = 0
                        for bucket_idx, gbuf_range_map in enumerate(
                            gbuf_range_map_for_all_buckets
                        ):
                            bucket_numel_unpadded = self.buffers[gbuf_idx].buckets[
                                bucket_idx
                            ].numel_unpadded
                            bucket_source = source[
                                bucket_offset : bucket_offset + bucket_numel_unpadded
                            ]
                            for model_param, param_range_map in gbuf_range_map[
                                "param_map"
                            ].items():
                                destination = (
                                    self._get_main_param_and_optimizer_states(
                                        model_param
                                    )[key]
                                )
                                if (
                                    key in ("exp_avg", "exp_avg_sq")
                                    and moment_dtype is not None
                                    and destination.dtype != moment_dtype
                                ):
                                    group_index, group_order = (
                                        self.model_param_group_index_map[model_param]
                                    )
                                    optimizer_param = self.optimizer.param_groups[
                                        group_index
                                    ]["params"][group_order]
                                    destination = torch.empty_like(
                                        destination,
                                        dtype=moment_dtype,
                                        device="cpu",
                                    )
                                    self.optimizer.state[optimizer_param][key] = (
                                        destination
                                    )
                                source_range = param_range_map.get(
                                    "gbuf_world_in_bucket",
                                    param_range_map["gbuf_world"],
                                )
                                destination.copy_(
                                    bucket_source[
                                        source_range.start : source_range.end
                                    ]
                                )
                            bucket_offset += bucket_numel_unpadded
                        if bucket_offset != source.numel():
                            raise RuntimeError(
                                validation_message(
                                    f"final offset {bucket_offset} changed "
                                    f"after preflight for source length "
                                    f"{source.numel()}",
                                    gbuf=gbuf_idx,
                                    dtype=dtype,
                                    key=key,
                                )
                            )
            self.optimizer._sync_hdo_state_to_sub_optimizers()
            return
"""
        content = content.replace(split_anchor, fast_path, 1)

    moment_setup = """
            moment_dtype = getattr(
                self.optimizer, "_lumen_streamed_adam_moment_dtype", None
            )
            if moment_dtype not in (None, torch.float32, torch.bfloat16):
                raise ValueError(
                    validation_message(
                        "streamed Adam moment dtype must be float32 or bfloat16",
                        gbuf="all",
                        key="moment_dtype",
                    )
                )
"""
    if moment_setup not in content:
        setup_anchor = """                return (
                    "DP=1 HDO checkpoint validation failed: "
                    f"gbuf={gbuf} dtype={dtype} bucket={bucket} "
                    f"key={key} param={param}: {reason}"
                )
"""
        if setup_anchor not in content:
            raise RuntimeError(
                "Unable to upgrade HDO checkpoint load: validation anchor missing"
            )
        content = content.replace(
            setup_anchor,
            setup_anchor + moment_setup,
            1,
        )

    moment_migration = """
                                if (
                                    key in ("exp_avg", "exp_avg_sq")
                                    and moment_dtype is not None
                                    and destination.dtype != moment_dtype
                                ):
                                    group_index, group_order = (
                                        self.model_param_group_index_map[model_param]
                                    )
                                    optimizer_param = self.optimizer.param_groups[
                                        group_index
                                    ]["params"][group_order]
                                    destination = torch.empty_like(
                                        destination,
                                        dtype=moment_dtype,
                                        device="cpu",
                                    )
                                    self.optimizer.state[optimizer_param][key] = (
                                        destination
                                    )
"""
    if moment_migration not in content:
        migration_anchor = """                                destination = (
                                    self._get_main_param_and_optimizer_states(
                                        model_param
                                    )[key]
                                )
                                source_range = param_range_map.get(
"""
        if content.count(migration_anchor) != 1:
            raise RuntimeError(
                "Unable to upgrade HDO checkpoint load: copy anchor missing"
            )
        content = content.replace(
            migration_anchor,
            migration_anchor.removesuffix(
                "                                source_range = param_range_map.get(\n"
            )
            + moment_migration
            + "                                source_range = param_range_map.get(\n",
            1,
        )

    load_anchor = (
        "                for model_param, tensors in recv_tensors.items():\n"
        "                    self._set_main_param_and_optimizer_states(model_param, tensors)\n"
        "\n"
        "    @torch.no_grad()\n"
        "    def load_parameter_state_from_fully_reshardable"
    )
    sync_marker = (
        "        if isinstance(self.optimizer, HybridDeviceOptimizer):\n"
        "            self.optimizer._sync_hdo_state_to_sub_optimizers()\n\n"
        "    @torch.no_grad()\n"
        "    def load_parameter_state_from_fully_reshardable"
    )
    if sync_marker not in content:
        if load_anchor not in content:
            raise RuntimeError(
                "Unable to patch HDO checkpoint load: DP-zero load anchor missing"
            )
        load_replacement = (
            "                for model_param, tensors in recv_tensors.items():\n"
            "                    self._set_main_param_and_optimizer_states(model_param, tensors)\n"
            "\n"
            "        if isinstance(self.optimizer, HybridDeviceOptimizer):\n"
            "            self.optimizer._sync_hdo_state_to_sub_optimizers()\n"
            "\n"
            "    @torch.no_grad()\n"
            "    def load_parameter_state_from_fully_reshardable"
        )
        content = content.replace(load_anchor, load_replacement, 1)

    mmap_load = "            state_dict = torch.load(filename)"
    mmap_load_patched = (
        "            state_dict = torch.load("
        "filename, mmap=True, weights_only=False)"
    )
    if mmap_load_patched not in content:
        if mmap_load not in content:
            raise RuntimeError(
                "Unable to patch HDO checkpoint load: torch.load anchor missing"
            )
        content = content.replace(mmap_load, mmap_load_patched, 1)

    if content == original:
        return False
    with open(path, "w") as f:
        f.write(content)
    return True


def patch_tp_copy_fp32_gradient_reduce(megatron_root: str) -> bool:
    """Support Lumen's FP32 tensor-parallel gradient reduction option."""
    path = os.path.join(
        megatron_root, "megatron", "core", "tensor_parallel", "mappings.py"
    )
    with open(path) as f:
        content = f.read()
    old_signature = "def copy_to_tensor_model_parallel_region(input_, group=None):"
    if old_signature not in content:
        return False

    content = content.replace(
        old_signature,
        "def copy_to_tensor_model_parallel_region("
        "input_, group=None, all_reduce_grad_fp32=False):",
        1,
    )
    content = content.replace(
        "return _CopyToModelParallelRegion.apply(input_, group)",
        "return _CopyToModelParallelRegion.apply("
        "input_, group, all_reduce_grad_fp32)",
        1,
    )
    content = content.replace(
        "def symbolic(graph, input_, group):",
        "def symbolic(graph, input_, group, all_reduce_grad_fp32=False):",
        1,
    )
    content = content.replace(
        "def forward(ctx, input_, group):",
        "def forward(ctx, input_, group, all_reduce_grad_fp32=False):",
        1,
    )
    content = content.replace(
        "        ctx.group = group\n        return input_",
        "        ctx.group = group\n"
        "        ctx.all_reduce_grad_fp32 = all_reduce_grad_fp32\n"
        "        return input_",
        1,
    )
    content = content.replace(
        "        return _reduce(grad_output, ctx.group), None",
        "        if ctx.all_reduce_grad_fp32:\n"
        "            grad_input = _reduce(grad_output.float(), ctx.group).to(grad_output.dtype)\n"
        "        else:\n"
        "            grad_input = _reduce(grad_output, ctx.group)\n"
        "        return grad_input, None, None",
        1,
    )
    with open(path, "w") as f:
        f.write(content)
    return True


def patch_hybrid_optimizer_disable_foreach(megatron_root: str) -> bool:
    """Avoid full-gradient foreach temporaries in torch CPU/GPU optimizers."""
    path = os.path.join(
        megatron_root,
        "megatron",
        "core",
        "optimizer",
        "cpu_offloading",
        "hybrid_optimizer.py",
    )
    with open(path) as f:
        content = f.read()
    replacements = {
        "self.cpu_optimizer_cls(self.cpu_param_groups)": "self.cpu_optimizer_cls(self.cpu_param_groups, foreach=False)",
        "self.gpu_optimizer_cls(self.gpu_param_groups)": "self.gpu_optimizer_cls(self.gpu_param_groups, foreach=False)",
        "cpu_optimizer_cls([_cpu_param_group])": "cpu_optimizer_cls([_cpu_param_group], foreach=False)",
    }
    original = content
    for old, new in replacements.items():
        content = content.replace(old, new)
    if content == original:
        return False
    with open(path, "w") as f:
        f.write(content)
    return True


def _streamed_adam_patch_state(content: str) -> str:
    """Return ``absent``, ``complete``, or ``partial`` for streamed Adam."""
    patch_markers = (
        STREAMED_ADAM_CAPABILITY,
        "atomrl.engine.training.streamed_adam",
        "_validate_full_offload_streaming_adam",
        "_stream_full_offload_adam_step",
        "_lumen_streamed_adam_chunk_numel",
    )
    has_marker = any(marker in content for marker in patch_markers)
    if not has_marker:
        return "absent"
    try:
        tree = ast.parse(content)
    except SyntaxError:
        return "partial"
    optimizer_class = next(
        (
            node
            for node in tree.body
            if isinstance(node, ast.ClassDef) and node.name == "HybridDeviceOptimizer"
        ),
        None,
    )
    if optimizer_class is None:
        return "partial"
    methods = {
        node.name: node
        for node in optimizer_class.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    required_methods = {
        "_validate_full_offload_streaming_adam",
        "_raise_streamed_adam_transfer_error",
        "_stream_full_offload_adam_step",
        "step",
    }
    imported_names = set()
    for node in tree.body:
        if (
            isinstance(node, ast.ImportFrom)
            and node.module == "atomrl.engine.training.streamed_adam"
        ):
            imported_names.update(alias.name for alias in node.names)
    required_imports = {
        "AdamChunkOptions",
        "adam_step_chunk_",
        "initialize_adam_state",
    }
    capability_is_true = any(
        isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == STREAMED_ADAM_CAPABILITY
            for target in node.targets
        )
        and isinstance(node.value, ast.Constant)
        and node.value.value is True
        for node in tree.body
    )
    source_lines = content.splitlines(keepends=True)

    def method_source(name: str) -> str:
        node = methods.get(name)
        if node is None or node.end_lineno is None:
            return ""
        return "".join(source_lines[node.lineno - 1 : node.end_lineno])

    validate_source = method_source("_validate_full_offload_streaming_adam")
    transfer_error_source = method_source("_raise_streamed_adam_transfer_error")
    stream_source = method_source("_stream_full_offload_adam_step")
    step_source = method_source("step")
    validate_tokens = (
        "_lumen_streamed_adam_chunk_numel",
        "_lumen_streamed_adam_moment_dtype",
        "AdamChunkOptions(",
        "torch.optim.Adam",
        "torch.optim.AdamW",
        "cpu_copys_map_gpu_param",
        "existing_state = optimizer.state.get(cpu_param)",
        "tensor=cpu_master",
        "not orig_param.is_contiguous()",
        "tensor=original",
        "not grad.is_contiguous()",
        "tensor=gradient",
    )
    transfer_error_tokens = (
        "failed_stream=None",
        '("_d2h_stream", "_h2d_stream")',
        "getattr(self, stream_name, None)",
        "stream is None or stream is failed_stream",
        "stream.synchronize()",
        "except Exception:",
        "raise RuntimeError(message) from exc",
    )
    stream_tokens = (
        "staging.shape != (staging_rows, chunk_numel)",
        "staging.dtype != torch.float32",
        "(staging_rows, chunk_numel)",
        "dtype=torch.float32",
        "pin_memory=True",
        "pending = [None, None]",
        "slot = sequence % 2",
        "def finish_slot(slot):",
        "event.synchronize()",
        "adam_step_chunk_(",
        "exp_avg_workspace=",
        "exp_avg_sq_workspace=",
        "orig_chunk.copy_(cpu_chunk, non_blocking=True)",
        "for chunk_start in range(0, cpu_param.numel(), chunk_numel):",
        "scratch.copy_(",
        "self._d2h_stream.record_event()",
        "last_context = (group_index, chunk_index)",
        "self._lumen_streamed_adam_last_h2d_event = None",
        (
            "self._lumen_streamed_adam_last_h2d_event = "
            "self._h2d_stream.record_event()"
        ),
        "orig_param.decoupled_grad = None",
        "orig_param.grad = None",
        "self._sync_sub_optimizers_state_to_hdo()",
        "streamed Adam D2H copy failed at ",
        "streamed Adam D2H event recording failed at ",
        "streamed Adam D2H synchronize failed at ",
        "streamed Adam update failed at ",
        "streamed Adam H2D scheduling failed at ",
        "streamed Adam H2D completion event recording failed at ",
        "streamed Adam final H2D synchronize failed at ",
        "failed_stream=self._h2d_stream",
    )
    sync_index = stream_source.find("self._sync_hdo_param_groups_to_sub_optimizers()")
    validate_index = stream_source.find(
        "self._validate_full_offload_streaming_adam(closure)"
    )
    initialize_index = stream_source.find(
        "initialize_adam_state(cpu_param, moment_dtype=moment_dtype)"
    )
    finish_start = stream_source.find("def finish_slot(slot):")
    finish_end = stream_source.find("\n        consumed =", finish_start)
    finish_source = stream_source[finish_start:finish_end]
    d2h_copy_index = stream_source.find("scratch.copy_(")
    record_event_index = stream_source.find("self._d2h_stream.record_event()")
    h2d_copy_index = stream_source.find(
        "orig_chunk.copy_(cpu_chunk, non_blocking=True)"
    )
    h2d_event_index = stream_source.find(
        "self._lumen_streamed_adam_last_h2d_event = " "self._h2d_stream.record_event()"
    )
    final_sync_index = stream_source.find(
        "self._h2d_stream.synchronize()", h2d_event_index
    )
    final_context_index = stream_source.find("group_index, chunk_index = last_context")
    final_error_index = stream_source.find(
        "streamed Adam final H2D synchronize failed at "
    )
    clear_grad_index = stream_source.find("orig_param.grad = None")
    stream_structure_is_complete = (
        all(token in stream_source for token in stream_tokens)
        and stream_source.count("(staging_rows, chunk_numel)") == 2
        and stream_source.count("pending = [None, None]") == 1
        and stream_source.count("event.synchronize()") == 1
        and stream_source.count("orig_chunk.copy_(cpu_chunk, non_blocking=True)") == 1
        and stream_source.count("pin_memory=True") == 1
        and stream_source.count("dtype=torch.float32") == 1
        and stream_source.count("staging.dtype != torch.float32") == 1
        and stream_source.count("slot = sequence % 2") == 1
        and stream_source.count("scratch.copy_(") == 1
        and stream_source.count("self._d2h_stream.record_event()") == 1
        and stream_source.count(
            "self._lumen_streamed_adam_last_h2d_event = "
            "self._h2d_stream.record_event()"
        )
        == 1
        and stream_source.count("self._h2d_stream.synchronize()") == 1
        and stream_source.count("self._raise_streamed_adam_transfer_error(") == 7
        and 0 <= sync_index < validate_index < initialize_index
        and 0 <= d2h_copy_index < record_event_index
        and 0 <= h2d_copy_index < h2d_event_index < final_sync_index
        and 0 <= finish_start < finish_end
        and finish_source.find("event.synchronize()")
        < finish_source.find("adam_step_chunk_(")
        < finish_source.find("orig_chunk.copy_(cpu_chunk, non_blocking=True)")
        and 0 <= final_context_index < final_error_index < clear_grad_index
        and "cpu_copy_map_grad" not in stream_source
        and "_set_sub_optimizer_grads" not in stream_source
    )
    complete = (
        required_methods <= methods.keys()
        and required_imports <= imported_names
        and capability_is_true
        and all(token in validate_source for token in validate_tokens)
        and '"fused",' not in validate_source
        and all(token in transfer_error_source for token in transfer_error_tokens)
        and stream_structure_is_complete
        and 'streamed_optimizer_mode == "adam"' in step_source
    )
    return "complete" if complete else "partial"


def _upgrade_legacy_streamed_adam_patch(content: str) -> str | None:
    """Upgrade the exact FP32-only streamed-Adam patch emitted previously."""
    legacy_tokens = (
        "ATOMRL_DSV4_CAPABILITY_STREAMED_ADAM = True",
        "initialize_adam_state(cpu_param)",
        "staging.shape != (2, chunk_numel)",
        "torch.empty(\n                (2, chunk_numel),",
        "or tensor.dtype != torch.float32",
    )
    if (
        not all(token in content for token in legacy_tokens)
        or "_lumen_streamed_adam_moment_dtype" in content
        or "exp_avg_workspace=" in content
    ):
        return None

    chunk_validation = """        if chunk_numel <= 0:
            raise RuntimeError(
                "_lumen_streamed_adam_chunk_numel must be positive"
            )
"""
    moment_validation = chunk_validation + """        moment_dtype = getattr(
            self, "_lumen_streamed_adam_moment_dtype", torch.float32
        )
        if moment_dtype not in (torch.float32, torch.bfloat16):
            raise RuntimeError(
                "_lumen_streamed_adam_moment_dtype must be float32 or bfloat16"
            )
"""
    if content.count(chunk_validation) != 1:
        return None
    upgraded = content.replace(chunk_validation, moment_validation, 1)
    upgraded = upgraded.replace(
        "or tensor.dtype != torch.float32",
        "or tensor.dtype != moment_dtype",
        1,
    )
    upgraded = upgraded.replace(
        "initialize_adam_state(cpu_param)",
        "initialize_adam_state(cpu_param, moment_dtype=moment_dtype)",
        1,
    )
    validated_marker = """        if not validated:
            self._sync_sub_optimizers_state_to_hdo()
            return None
"""
    moment_setup = validated_marker + """        moment_dtype = getattr(
            self, "_lumen_streamed_adam_moment_dtype", torch.float32
        )
"""
    if upgraded.count(validated_marker) != 1:
        return None
    upgraded = upgraded.replace(validated_marker, moment_setup, 1)
    staging_marker = """        staging = getattr(
            self, "_lumen_streamed_adam_buffers", None)
"""
    if staging_marker not in upgraded:
        staging_marker = """        staging = getattr(self, "_lumen_streamed_adam_buffers", None)
"""
    if staging_marker not in upgraded:
        return None
    staging_setup = """        staging_rows = 4 if moment_dtype == torch.bfloat16 else 2
"""
    upgraded = upgraded.replace(staging_marker, staging_setup + staging_marker, 1)
    upgraded = upgraded.replace("(2, chunk_numel)", "(staging_rows, chunk_numel)")
    call_marker = """                    exp_avg_sq_chunk,
                    step=step,
"""
    workspace_block = """                    exp_avg_sq_chunk,
                    exp_avg_workspace=(
                        staging[2, : exp_avg_chunk.numel()]
                        if moment_dtype == torch.bfloat16
                        else None
                    ),
                    exp_avg_sq_workspace=(
                        staging[3, : exp_avg_sq_chunk.numel()]
                        if moment_dtype == torch.bfloat16
                        else None
                    ),
                    step=step,
"""
    if upgraded.count(call_marker) != 1:
        return None
    return upgraded.replace(call_marker, workspace_block, 1)


def _repair_streamed_adam_moment_ordering(content: str) -> str | None:
    """Move moment dtype lookup before first-state initialization."""
    method_start = content.find(
        "    def _stream_full_offload_adam_step(self, closure=None):"
    )
    method_end = content.find("\n    def step(self, closure=None):", method_start)
    if method_start < 0 or method_end < 0:
        return None
    method = content[method_start:method_end]
    moment_setup = """        moment_dtype = getattr(
            self, "_lumen_streamed_adam_moment_dtype", torch.float32
        )
"""
    initialize_index = method.find(
        "initialize_adam_state(cpu_param, moment_dtype=moment_dtype)"
    )
    moment_index = method.find(moment_setup)
    if moment_index < 0 or initialize_index < 0 or moment_index < initialize_index:
        return None
    validated_marker = """        if not validated:
            self._sync_sub_optimizers_state_to_hdo()
            return None
"""
    if method.count(moment_setup) != 1 or method.count(validated_marker) != 1:
        return None
    method = method.replace(moment_setup, "", 1)
    method = method.replace(
        validated_marker,
        validated_marker + moment_setup,
        1,
    )
    return content[:method_start] + method + content[method_end:]


def patch_hybrid_optimizer_streaming_adam(megatron_root: str) -> bool:
    """Add bounded-memory full-offload Adam/AdamW chunk streaming."""
    path = os.path.join(
        megatron_root,
        "megatron",
        "core",
        "optimizer",
        "cpu_offloading",
        "hybrid_optimizer.py",
    )
    with open(path) as f:
        content = f.read()
    state = _streamed_adam_patch_state(content)
    if state == "complete":
        repaired = _repair_streamed_adam_moment_ordering(content)
        if repaired is not None:
            with open(path, "w") as f:
                f.write(repaired)
            return True
        return False
    if state == "partial":
        upgraded = _upgrade_legacy_streamed_adam_patch(content)
        if upgraded is None or _streamed_adam_patch_state(upgraded) != "complete":
            raise RuntimeError(
                "HybridDeviceOptimizer has a partial streamed Adam patch; "
                "refusing to modify it"
            )
        with open(path, "w") as f:
            f.write(upgraded)
        return True

    class_marker = "class HybridDeviceOptimizer"
    step_signature = "    def step(self, closure=None):\n"
    if class_marker not in content or step_signature not in content:
        return False
    imports = """import math

from atomrl.engine.training.streamed_adam import (
    AdamChunkOptions,
    adam_step_chunk_,
    initialize_adam_state,
)


"""
    content = content.replace(class_marker, imports + class_marker, 1)
    methods = '''    def _validate_full_offload_streaming_adam(self, closure=None):
        """Validate the complete streamed Adam step before mutating any state."""
        if closure is not None:
            raise RuntimeError(
                "streamed full-offload Adam does not support closures"
            )
        if self.offload_fraction != 1.0:
            raise RuntimeError(
                "streamed full-offload Adam requires offload_fraction exactly 1.0"
            )
        if self.gpu_optimizer is not None:
            raise RuntimeError(
                "streamed full-offload Adam requires no GPU optimizer"
            )
        if not self.cpu_optimizers:
            raise RuntimeError(
                "streamed full-offload Adam requires a CPU optimizer"
            )
        chunk_numel = int(
            getattr(self, "_lumen_streamed_adam_chunk_numel", 4 * 1024 * 1024)
        )
        if chunk_numel <= 0:
            raise RuntimeError(
                "_lumen_streamed_adam_chunk_numel must be positive"
            )
        moment_dtype = getattr(
            self, "_lumen_streamed_adam_moment_dtype", torch.float32
        )
        if moment_dtype not in (torch.float32, torch.bfloat16):
            raise RuntimeError(
                "_lumen_streamed_adam_moment_dtype must be float32 or bfloat16"
            )

        validated = []
        for optimizer_index, optimizer in enumerate(self.cpu_optimizers):
            if not isinstance(optimizer, (torch.optim.Adam, torch.optim.AdamW)):
                raise RuntimeError(
                    "streamed full-offload Adam requires torch.optim.Adam or "
                    f"AdamW; optimizer={optimizer_index}"
                )
            for group_index, group in enumerate(optimizer.param_groups):
                for flag in (
                    "amsgrad",
                    "differentiable",
                    "capturable",
                    "foreach",
                ):
                    if bool(group.get(flag, False)):
                        raise RuntimeError(
                            "streamed full-offload Adam has an incompatible flag; "
                            f"parameter_group={group_index} flag={flag}"
                        )
                betas = group.get("betas", (0.9, 0.999))
                if not (
                    isinstance(betas, (tuple, list))
                    and len(betas) == 2
                ):
                    raise RuntimeError(
                        "streamed full-offload Adam has invalid betas; "
                        f"parameter_group={group_index}"
                    )
                try:
                    options = AdamChunkOptions(
                        lr=float(group["lr"]),
                        beta1=float(betas[0]),
                        beta2=float(betas[1]),
                        eps=float(group.get("eps", 1e-8)),
                        weight_decay=float(group.get("weight_decay", 0.0)),
                        maximize=bool(group.get("maximize", False)),
                        decoupled_weight_decay=isinstance(
                            optimizer, torch.optim.AdamW
                        ),
                    )
                except (KeyError, TypeError, ValueError) as exc:
                    raise RuntimeError(
                        "streamed full-offload Adam options are invalid; "
                        f"parameter_group={group_index}"
                    ) from exc
                if (
                    not math.isfinite(options.lr)
                    or options.lr < 0
                    or not math.isfinite(options.eps)
                    or options.eps < 0
                    or not math.isfinite(options.weight_decay)
                    or options.weight_decay < 0
                    or not math.isfinite(options.beta1)
                    or not 0 <= options.beta1 < 1
                    or not math.isfinite(options.beta2)
                    or not 0 <= options.beta2 < 1
                ):
                    raise RuntimeError(
                        "streamed full-offload Adam options are out of range; "
                        f"parameter_group={group_index}"
                    )
                for cpu_param in group["params"]:
                    if (
                        cpu_param.device.type != "cpu"
                        or cpu_param.dtype != torch.float32
                        or not cpu_param.is_contiguous()
                    ):
                        raise RuntimeError(
                            "streamed full-offload Adam requires contiguous CPU "
                            "FP32 masters; "
                            f"parameter_group={group_index} tensor=cpu_master"
                        )
                    if cpu_param not in self.cpu_copys_map_gpu_param:
                        raise RuntimeError(
                            "streamed full-offload Adam has no mapped CUDA "
                            f"parameter; parameter_group={group_index}"
                        )
                    orig_param = self.cpu_copys_map_gpu_param[cpu_param]
                    if not orig_param.is_cuda:
                        raise RuntimeError(
                            "streamed full-offload Adam requires a mapped CUDA "
                            f"parameter; parameter_group={group_index}"
                        )
                    if not orig_param.is_contiguous():
                        raise RuntimeError(
                            "streamed full-offload Adam requires a contiguous "
                            "mapped parameter; "
                            f"parameter_group={group_index} tensor=original"
                        )
                    grad = getattr(orig_param, "decoupled_grad", None)
                    if grad is None:
                        grad = orig_param.grad
                    if grad is None:
                        continue
                    if not grad.is_cuda or grad.is_sparse:
                        raise RuntimeError(
                            "streamed full-offload Adam requires mapped CUDA "
                            f"dense gradients; parameter_group={group_index}"
                        )
                    if not grad.is_contiguous():
                        raise RuntimeError(
                            "streamed full-offload Adam requires a contiguous "
                            "gradient; "
                            f"parameter_group={group_index} tensor=gradient"
                        )
                    if grad.numel() != cpu_param.numel():
                        raise RuntimeError(
                            "streamed full-offload Adam gradient shape mismatch; "
                            f"parameter_group={group_index}"
                        )
                    existing_state = optimizer.state.get(cpu_param)
                    if existing_state:
                        required = ("step", "exp_avg", "exp_avg_sq")
                        if any(name not in existing_state for name in required):
                            raise RuntimeError(
                                "streamed full-offload Adam state is incomplete; "
                                f"parameter_group={group_index}"
                            )
                        step = existing_state["step"]
                        if (
                            not torch.is_tensor(step)
                            or step.numel() != 1
                            or step.device.type != "cpu"
                            or not math.isfinite(float(step.item()))
                            or float(step.item()) < 0
                            or not float(step.item()).is_integer()
                        ):
                            raise RuntimeError(
                                "streamed full-offload Adam state step is invalid; "
                                f"parameter_group={group_index}"
                            )
                        for name in ("exp_avg", "exp_avg_sq"):
                            tensor = existing_state[name]
                            if (
                                tensor.device.type != "cpu"
                                or tensor.dtype != moment_dtype
                                or tensor.shape != cpu_param.shape
                                or not tensor.is_contiguous()
                            ):
                                raise RuntimeError(
                                    "streamed full-offload Adam state tensor is "
                                    f"invalid; parameter_group={group_index} "
                                    f"state={name}"
                                )
                    validated.append(
                        (
                            optimizer,
                            group_index,
                            group,
                            cpu_param,
                            orig_param,
                            grad,
                            options,
                        )
                    )
        return validated

    def _lumen_streamed_adam_rss(self, phase):
        """Emit one RSS diagnostic for each allocation-sensitive phase."""
        announced = getattr(self, "_lumen_streamed_adam_rss_announced", set())
        if phase in announced:
            return
        rss_kib = "unavailable"
        try:
            with open("/proc/self/status") as status_file:
                for line in status_file:
                    if line.startswith("VmRSS:"):
                        rss_kib = line.split(":", 1)[1].strip()
                        break
        except OSError:
            pass
        print(
            f"[AtomRL] streamed Adam {phase}; rss={rss_kib}",
            flush=True,
        )
        announced.add(phase)
        self._lumen_streamed_adam_rss_announced = announced

    def _raise_streamed_adam_transfer_error(
        self, message, exc, failed_stream=None
    ):
        """Quiesce both transfer streams best-effort, then retain the cause."""
        for stream_name in ("_d2h_stream", "_h2d_stream"):
            stream = getattr(self, stream_name, None)
            if stream is None or stream is failed_stream:
                continue
            try:
                stream.synchronize()
            except Exception:
                pass
        raise RuntimeError(message) from exc

    def _stream_full_offload_adam_step(self, closure=None):
        """Stream Adam updates through two pinned FP32 chunk buffers."""
        self._lumen_streamed_adam_last_h2d_event = None
        self._sync_hdo_param_groups_to_sub_optimizers()
        validated = self._validate_full_offload_streaming_adam(closure)
        if not validated:
            self._sync_sub_optimizers_state_to_hdo()
            return None
        moment_dtype = getattr(
            self, "_lumen_streamed_adam_moment_dtype", torch.float32
        )

        initialized_state = False
        prepared = []
        for (
            optimizer,
            group_index,
            _group,
            cpu_param,
            orig_param,
            grad,
            options,
        ) in validated:
            state = optimizer.state.get(cpu_param)
            if not state:
                state = initialize_adam_state(cpu_param, moment_dtype=moment_dtype)
                optimizer.state[cpu_param] = state
                initialized_state = True
            state["step"].add_(1)
            step = int(state["step"].item())
            prepared.append(
                (
                    group_index,
                    cpu_param,
                    orig_param,
                    grad,
                    state,
                    step,
                    options,
                )
            )
        if initialized_state:
            self._lumen_streamed_adam_rss("after state allocation")

        chunk_numel = int(
            getattr(self, "_lumen_streamed_adam_chunk_numel", 4 * 1024 * 1024)
        )
        staging_rows = 4 if moment_dtype == torch.bfloat16 else 2
        staging = getattr(self, "_lumen_streamed_adam_buffers", None)
        if (
            staging is None
            or staging.shape != (staging_rows, chunk_numel)
            or staging.dtype != torch.float32
            or not staging.is_pinned()
        ):
            staging = torch.empty(
                (staging_rows, chunk_numel),
                dtype=torch.float32,
                device="cpu",
                pin_memory=True,
            )
            self._lumen_streamed_adam_buffers = staging

        self._d2h_stream.wait_stream(torch.cuda.current_stream())
        pending = [None, None]

        def finish_slot(slot):
            item = pending[slot]
            if item is None:
                return
            (
                event,
                group_index,
                chunk_index,
                cpu_chunk,
                orig_chunk,
                scratch,
                exp_avg_chunk,
                exp_avg_sq_chunk,
                step,
                options,
            ) = item
            try:
                event.synchronize()
            except Exception as exc:
                self._raise_streamed_adam_transfer_error(
                    "streamed Adam D2H synchronize failed at "
                    f"parameter_group={group_index} chunk={chunk_index}",
                    exc,
                )
            try:
                adam_step_chunk_(
                    cpu_chunk,
                    scratch,
                    exp_avg_chunk,
                    exp_avg_sq_chunk,
                    exp_avg_workspace=(
                        staging[2, : exp_avg_chunk.numel()]
                        if moment_dtype == torch.bfloat16
                        else None
                    ),
                    exp_avg_sq_workspace=(
                        staging[3, : exp_avg_sq_chunk.numel()]
                        if moment_dtype == torch.bfloat16
                        else None
                    ),
                    step=step,
                    options=options,
                )
            except Exception as exc:
                self._raise_streamed_adam_transfer_error(
                    "streamed Adam update failed at "
                    f"parameter_group={group_index} chunk={chunk_index}",
                    exc,
                )
            try:
                with torch.cuda.stream(self._h2d_stream):
                    orig_chunk.copy_(cpu_chunk, non_blocking=True)
            except Exception as exc:
                self._raise_streamed_adam_transfer_error(
                    "streamed Adam H2D scheduling failed at "
                    f"parameter_group={group_index} chunk={chunk_index}",
                    exc,
                )
            pending[slot] = None

        consumed = []
        sequence = 0
        last_context = (-1, -1)
        with torch.no_grad():
            for (
                group_index,
                cpu_param,
                orig_param,
                grad,
                state,
                step,
                options,
            ) in prepared:
                cpu_flat = cpu_param.view(-1)
                orig_flat = orig_param.view(-1)
                grad_flat = grad.view(-1)
                exp_avg_flat = state["exp_avg"].view(-1)
                exp_avg_sq_flat = state["exp_avg_sq"].view(-1)
                for chunk_start in range(0, cpu_param.numel(), chunk_numel):
                    chunk_index = chunk_start // chunk_numel
                    chunk_end = min(chunk_start + chunk_numel, cpu_param.numel())
                    slot = sequence % 2
                    finish_slot(slot)
                    scratch = staging[slot, : chunk_end - chunk_start]
                    try:
                        with torch.cuda.stream(self._d2h_stream):
                            scratch.copy_(
                                grad_flat[chunk_start:chunk_end],
                                non_blocking=True,
                            )
                    except Exception as exc:
                        self._raise_streamed_adam_transfer_error(
                            "streamed Adam D2H copy failed at "
                            f"parameter_group={group_index} chunk={chunk_index}",
                            exc,
                        )
                    try:
                        with torch.cuda.stream(self._d2h_stream):
                            event = self._d2h_stream.record_event()
                    except Exception as exc:
                        self._raise_streamed_adam_transfer_error(
                            "streamed Adam D2H event recording failed at "
                            f"parameter_group={group_index} chunk={chunk_index}",
                            exc,
                        )
                    pending[slot] = (
                        event,
                        group_index,
                        chunk_index,
                        cpu_flat[chunk_start:chunk_end],
                        orig_flat[chunk_start:chunk_end],
                        scratch,
                        exp_avg_flat[chunk_start:chunk_end],
                        exp_avg_sq_flat[chunk_start:chunk_end],
                        step,
                        options,
                    )
                    last_context = (group_index, chunk_index)
                    sequence += 1
                consumed.append(orig_param)
            for index in range(max(0, sequence - 2), sequence):
                finish_slot(index % 2)
        group_index, chunk_index = last_context
        try:
            self._lumen_streamed_adam_last_h2d_event = self._h2d_stream.record_event()
        except Exception as exc:
            self._raise_streamed_adam_transfer_error(
                "streamed Adam H2D completion event recording failed at "
                f"parameter_group={group_index} chunk={chunk_index}",
                exc,
            )
        try:
            self._h2d_stream.synchronize()
        except Exception as exc:
            self._raise_streamed_adam_transfer_error(
                "streamed Adam final H2D synchronize failed at "
                f"parameter_group={group_index} chunk={chunk_index}",
                exc,
                failed_stream=self._h2d_stream,
            )
        for orig_param in consumed:
            if hasattr(orig_param, "decoupled_grad"):
                orig_param.decoupled_grad = None
            orig_param.grad = None
        self._sync_sub_optimizers_state_to_hdo()
        self._lumen_streamed_adam_rss("after first step")
        return None

'''
    content = content.replace(step_signature, methods + step_signature, 1)
    adam_dispatch = (
        "        streamed_optimizer_mode = getattr(\n"
        '            self, "_lumen_streamed_optimizer_mode", None\n'
        "        )\n"
        '        if streamed_optimizer_mode == "adam":\n'
        "            return self._stream_full_offload_adam_step(closure)\n\n"
    )
    content = content.replace(
        step_signature,
        step_signature + adam_dispatch,
        1,
    )
    legacy_sgd_dispatch = (
        "        if self._can_stream_full_offload_sgd():\n"
        "            return self._stream_full_offload_sgd_step(closure)\n\n"
    )
    explicit_sgd_dispatch = (
        '        if streamed_optimizer_mode in (None, "sgd"):\n'
        "            if self._can_stream_full_offload_sgd():\n"
        "                return self._stream_full_offload_sgd_step(closure)\n\n"
    )
    content = content.replace(
        legacy_sgd_dispatch,
        explicit_sgd_dispatch,
        1,
    )
    content = _stamp_capabilities(content, (STREAMED_ADAM_CAPABILITY,))
    if _streamed_adam_patch_state(content) != "complete":
        raise RuntimeError(
            "HybridDeviceOptimizer streamed Adam patch failed structural "
            "postconditions; refusing to write it"
        )
    with open(path, "w") as f:
        f.write(content)
    return True


def patch_hybrid_optimizer_streaming_sgd(megatron_root: str) -> bool:
    """Stream fully offloaded zero-momentum SGD grads through one CPU buffer."""
    path = os.path.join(
        megatron_root,
        "megatron",
        "core",
        "optimizer",
        "cpu_offloading",
        "hybrid_optimizer.py",
    )
    with open(path) as f:
        content = f.read()
    marker = "def _can_stream_full_offload_sgd(self):"
    if marker in content:
        return False
    step_signature = "    def step(self, closure=None):\n"
    if step_signature not in content:
        return False
    methods = '''    def _can_stream_full_offload_sgd(self):
        """Return whether the bounded-memory SGD fast path is semantically safe."""
        if self.offload_fraction != 1.0 or self.gpu_optimizer is not None:
            return False
        if not self.cpu_optimizers:
            return False
        for optimizer in self.cpu_optimizers:
            if not isinstance(optimizer, torch.optim.SGD):
                return False
            for group in optimizer.param_groups:
                if float(group.get("momentum", 0.0)) != 0.0:
                    return False
                if float(group.get("dampening", 0.0)) != 0.0:
                    return False
                if bool(group.get("nesterov", False)) or bool(group.get("maximize", False)):
                    return False
        return True

    def _stream_full_offload_sgd_step(self, closure=None):
        """Update CPU masters with one reusable FP32 gradient staging tensor."""
        if closure is not None:
            raise RuntimeError("streaming full-offload SGD does not support closures")
        self._sync_hdo_param_groups_to_sub_optimizers()
        cpu_params = [
            cpu_param
            for optimizer in self.cpu_optimizers
            for group in optimizer.param_groups
            for cpu_param in group["params"]
        ]
        max_numel = max((cpu_param.numel() for cpu_param in cpu_params), default=0)
        if max_numel == 0:
            self._sync_sub_optimizers_state_to_hdo()
            return None
        staging = torch.empty(
            (2, max_numel),
            dtype=torch.float32,
            device="cpu",
            pin_memory=True,
        )
        if not getattr(self, "_streaming_sgd_announced", False):
            print(
                "[AtomRL] enabled double-buffered streaming CPU SGD; "
                f"staging={staging.numel() * staging.element_size() / 2**20:.1f} MiB",
                flush=True,
            )
            self._streaming_sgd_announced = True
        self._d2h_stream.wait_stream(torch.cuda.current_stream())
        pending = [None, None]

        def finish_slot(slot):
            item = pending[slot]
            if item is None:
                return
            event, cpu_param, orig_param, staging_view, lr, weight_decay = item
            event.synchronize()
            if weight_decay:
                cpu_param.mul_(1.0 - lr * weight_decay)
            cpu_param.add_(staging_view, alpha=-lr)
            with torch.cuda.stream(self._h2d_stream):
                orig_param.copy_(cpu_param, non_blocking=True)
            pending[slot] = None

        with torch.no_grad():
            sequence = 0
            for optimizer in self.cpu_optimizers:
                for group in optimizer.param_groups:
                    lr = float(group["lr"])
                    weight_decay = float(group.get("weight_decay", 0.0))
                    for cpu_param in group["params"]:
                        orig_param = self.cpu_copys_map_gpu_param[cpu_param]
                        grad = getattr(orig_param, "decoupled_grad", orig_param.grad)
                        if grad is None:
                            continue
                        if grad.is_sparse:
                            raise RuntimeError(
                                "streaming full-offload SGD requires dense gradients"
                            )
                        slot = sequence % 2
                        finish_slot(slot)
                        staging_view = staging[slot, : cpu_param.numel()].view_as(cpu_param)
                        with torch.cuda.stream(self._d2h_stream):
                            staging_view.copy_(grad, non_blocking=True)
                            d2h_event = self._d2h_stream.record_event()
                        pending[slot] = (
                            d2h_event,
                            cpu_param,
                            orig_param,
                            staging_view,
                            lr,
                            weight_decay,
                        )
                        sequence += 1
            for index in range(max(0, sequence - 2), sequence):
                finish_slot(index % 2)
        self._h2d_stream.synchronize()
        self._sync_sub_optimizers_state_to_hdo()
        return None

'''
    content = content.replace(step_signature, methods + step_signature, 1)
    adam_dispatch = (
        "        streamed_optimizer_mode = getattr(\n"
        '            self, "_lumen_streamed_optimizer_mode", None\n'
        "        )\n"
        '        if streamed_optimizer_mode == "adam":\n'
        "            return self._stream_full_offload_adam_step(closure)\n\n"
    )
    if adam_dispatch in content:
        step_preamble = (
            '        if streamed_optimizer_mode in (None, "sgd"):\n'
            "            if self._can_stream_full_offload_sgd():\n"
            "                return self._stream_full_offload_sgd_step(closure)\n\n"
        )
        content = content.replace(
            adam_dispatch,
            adam_dispatch + step_preamble,
            1,
        )
    else:
        step_preamble = (
            "        streamed_optimizer_mode = getattr(\n"
            '            self, "_lumen_streamed_optimizer_mode", None\n'
            "        )\n"
            '        if streamed_optimizer_mode in (None, "sgd"):\n'
            "            if self._can_stream_full_offload_sgd():\n"
            "                return self._stream_full_offload_sgd_step(closure)\n\n"
        )
        content = content.replace(
            step_signature,
            step_signature + step_preamble,
            1,
        )
    with open(path, "w") as f:
        f.write(content)
    return True


def _validate_required_patches(megatron_root: str) -> str:
    """Validate required DSV4 patches and return the router source path."""
    problems = []
    config_path = os.path.join(
        megatron_root,
        "megatron",
        "core",
        "transformer",
        "transformer_config.py",
    )
    try:
        with open(config_path) as f:
            config_content = f.read()
    except OSError:
        config_content = ""
    if not _config_has_exact_sqrtsoftplus(config_content):
        problems.append(
            "transformer_config.py moe_router_score_function lacks exact "
            "'sqrtsoftplus' Literal token"
        )

    router_definition = _find_router_definition(megatron_root)
    router_path = ""
    if router_definition is None:
        problems.append("topk_routing_with_score_function definition not found")
    else:
        router_path, router_content, router_start, router_end = router_definition
        router_source = router_content[router_start:router_end]
        required_router_markers = (
            'elif score_function in ("sigmoid", "sqrtsoftplus"):',
            "torch.nn.functional.softplus(logits.float()).sqrt().type_as(logits)",
            "scores_for_routing = scores + expert_bias",
            "scores.sum(dim=-1, keepdim=True).clamp(min=1e-20)",
        )
        if not all(marker in router_source for marker in required_router_markers):
            problems.append(
                "topk_routing_with_score_function has missing or incompatible "
                "sqrtsoftplus patch anchors"
            )

    random_path = os.path.join(
        megatron_root, "megatron", "core", "tensor_parallel", "random.py"
    )
    try:
        with open(random_path) as f:
            random_content = f.read()
    except OSError:
        random_content = ""
    checkpoint_definition = re.search(
        r"(?m)^class CheckpointFunction\(", random_content
    )
    checkpoint_source = ""
    if checkpoint_definition is not None:
        checkpoint_start, checkpoint_end = _top_level_source_extent(
            random_content, checkpoint_definition.start()
        )
        checkpoint_source = random_content[checkpoint_start:checkpoint_end]
    if not (
        "router_replay_actions = [" in checkpoint_source
        and "RouterReplayAction.REPLAY_BACKWARD" in checkpoint_source
        and "finally:" in checkpoint_source
    ):
        problems.append(
            "CheckpointFunction.backward lacks checkpoint router replay marker"
        )
    if f"{CHECKPOINT_REPLAY_CAPABILITY} = True" not in random_content:
        problems.append(f"tensor_parallel.random lacks {CHECKPOINT_REPLAY_CAPABILITY}")

    replay_path = os.path.join(
        megatron_root,
        "megatron",
        "core",
        "transformer",
        "moe",
        "router_replay.py",
    )
    try:
        with open(replay_path) as f:
            replay_content = f.read()
    except OSError:
        replay_content = ""
    if _router_replay_diagnostics_state(replay_content) != "complete":
        problems.append(
            "RouterReplay lacks structurally complete forward/recompute "
            "ID diagnostics"
        )
    if not _router_replay_has_fifo_semantics(replay_content):
        problems.append("RouterReplay lacks FIFO record/replay semantics")
    for capability in (
        ROUTER_REPLAY_FIFO_CAPABILITY,
        REPLAY_DIAGNOSTICS_CAPABILITY,
    ):
        if f"{capability} = True" not in replay_content:
            problems.append(f"router_replay.py lacks {capability}")

    if problems:
        raise RuntimeError(
            "Required ROCm Megatron DSV4 patch validation failed: "
            + "; ".join(problems)
        )
    return router_path


def main(megatron_root: str) -> None:
    try:
        replay_path = os.path.join(
            megatron_root,
            "megatron",
            "core",
            "transformer",
            "moe",
            "router_replay.py",
        )
        with open(replay_path) as replay_file:
            replay_content = replay_file.read()
        if _router_replay_diagnostics_state(replay_content) == "partial":
            raise RuntimeError(
                "RouterReplay has a partial or malformed diagnostic patch; "
                "refusing all writes"
            )
        transformer_config_changed = patch_transformer_config(megatron_root)
        router_changed = patch_moe_router_score_function(megatron_root)
        checkpoint_changed = patch_checkpoint_router_replay(megatron_root)
        replay_diagnostics_changed = patch_router_replay_diagnostics(megatron_root)
    except OSError as exc:
        raise RuntimeError(
            f"Required ROCm Megatron patch inputs are unavailable: {exc}"
        ) from exc
    router_path = _validate_required_patches(megatron_root)
    router_label = os.path.relpath(
        router_path,
        os.path.join(megatron_root, "megatron", "core"),
    ).replace(os.sep, "/")
    results = {
        "transformer_config.py": transformer_config_changed,
        router_label: router_changed,
        "tensor_parallel/random.py": checkpoint_changed,
        "transformer/moe/router_replay.py diagnostics": replay_diagnostics_changed,
        "transformer_block.py": patch_transformer_block(megatron_root),
        "transformer_layer.py": patch_transformer_layer(megatron_root),
        "experimental_attention_variant_module_specs.py": patch_eav_specs(
            megatron_root
        ),
        "tensor_parallel/layers.py": patch_tp_layers(megatron_root),
        "tensor_parallel/mappings.py": patch_tp_copy_fp32_gradient_reduce(
            megatron_root
        ),
        "optimizer/hybrid_optimizer.py": patch_hybrid_optimizer_disable_foreach(
            megatron_root
        ),
        "optimizer/hybrid_optimizer.py streaming Adam": patch_hybrid_optimizer_streaming_adam(
            megatron_root
        ),
        "optimizer/hybrid_optimizer.py streaming SGD": patch_hybrid_optimizer_streaming_sgd(
            megatron_root
        ),
        "optimizer/distrib_optimizer.py": patch_distrib_optimizer_fp32_detach(
            megatron_root
        ),
        "optimizer/distrib_optimizer.py grad copy": patch_distrib_optimizer_grad_copy(
            megatron_root
        ),
        "optimizer/distrib_optimizer.py HDO checkpoint": patch_distrib_optimizer_hdo_checkpoint(
            megatron_root
        ),
    }
    print(f"Patched ROCm Megatron at {megatron_root}:")
    for name, ok in results.items():
        print(f"  {'PATCHED' if ok else 'skipped'}: {name}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <megatron-root>")
        sys.exit(1)
    main(sys.argv[1])
