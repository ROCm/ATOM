import ast
from pathlib import Path
from types import SimpleNamespace

KIMI_PATH = Path(__file__).resolve().parents[1] / "atom" / "models" / "kimi_k3.py"


def _tree() -> ast.Module:
    return ast.parse(KIMI_PATH.read_text())


def _class(name: str) -> ast.ClassDef:
    return next(
        node
        for node in _tree().body
        if isinstance(node, ast.ClassDef) and node.name == name
    )


def _method(class_name: str, method_name: str) -> ast.FunctionDef:
    return next(
        node
        for node in _class(class_name).body
        if isinstance(node, ast.FunctionDef) and node.name == method_name
    )


def _method_source(class_name: str, method_name: str) -> str:
    return ast.get_source_segment(
        KIMI_PATH.read_text(), _method(class_name, method_name)
    )


def _load_effective_config_helpers():
    helper_names = {
        "_layer_effectively_consumes_per_token_fp8",
        "_linears_effectively_consume_per_token_fp8",
    }
    helpers = [
        node
        for node in _tree().body
        if isinstance(node, ast.FunctionDef) and node.name in helper_names
    ]
    per_token = object()
    no_quant = object()
    fp8 = object()
    namespace = {
        "QuantizationConfig": object,
        "QuantType": SimpleNamespace(per_Token=per_token, No=no_quant),
        "dtypes": SimpleNamespace(fp8=fp8),
        "nn": SimpleNamespace(Module=object),
        "should_skip_online_quant": lambda current_type, current_dtype, online: (
            online.quant_type is no_quant
            or (
                current_type is online.quant_type
                and current_dtype is online.quant_dtype
            )
        ),
    }
    module = ast.Module(body=helpers, type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), str(KIMI_PATH), "exec"), namespace)
    return namespace, per_token, no_quant, fp8


def _load_packed_modules_mapping():
    helper = next(
        node
        for node in _tree().body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_kimi_packed_modules_mapping"
    )
    namespace = {}
    module = ast.Module(body=[helper], type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), str(KIMI_PATH), "exec"), namespace)
    return namespace["_kimi_packed_modules_mapping"]


class _FakeQuantConfig:
    def __init__(self, source, online=None):
        self.source = source
        self.online = online or {}
        self.online_quant = bool(online)

    def get_layer_quant_config(self, prefix, use_online_quant=False):
        configs = self.online if use_online_quant else self.source
        return configs[prefix]


def test_full_attention_fuses_q_and_kv_input_projections():
    init_source = _method_source("KimiFullAttention", "__init__")
    forward_source = _method_source("KimiFullAttention", "forward")

    assert "self.fused_qkv_a_proj = MergedReplicatedLinear(" in init_source
    assert (
        "[self.q_lora_rank, self.kv_lora_rank + self.qk_rope_head_dim]" in init_source
    )
    assert "self.q_a_proj = " not in init_source
    assert "self.kv_a_proj_with_mqa = " not in init_source
    assert "qkv_lora = self.fused_qkv_a_proj(" in forward_source
    assert "q, compressed_kv = torch.split(" in forward_source


def test_full_attention_fuses_latent_norm_quant_and_forwards_scales():
    init_source = _method_source("KimiFullAttention", "__init__")
    forward_source = _method_source("KimiFullAttention", "forward")

    assert 'prefix=f"{prefix}.q_b_proj"' in init_source
    assert 'prefix=f"{prefix}.kv_b_proj"' in init_source
    assert "fused_quant=self.fuse_q_b_norm_quant" in init_source
    assert "fused_quant=self.fuse_kv_b_norm_quant" in init_source
    assert "if isinstance(q, tuple):" in forward_source
    assert "if isinstance(kv, tuple):" in forward_source
    assert "if isinstance(hidden_states, tuple):" in forward_source
    assert "self.q_b_proj(q, x_scale=q_scale)" in forward_source
    assert "self.kv_b_proj(kv, x_scale=kv_scale)" in forward_source
    assert "self.g_proj(hidden_states, x_scale=hidden_states_scale)" in forward_source


def test_decoder_fuses_attention_input_and_dense_mlp_norms_only():
    init_source = _method_source("KimiDecoderLayer", "__init__")
    forward_source = _method_source("KimiDecoderLayer", "forward")

    assert "fused_quant=self.self_attn.fuse_input_norm_quant" in init_source
    assert "prefix=self.self_attn.input_quant_prefix" in init_source
    assert "self.ffn_quant_linear" in init_source
    assert "self.fuse_ffn_norm_quant" in init_source
    assert "fused_quant=self.fuse_ffn_norm_quant" in init_source
    assert "quant_config if self.fuse_ffn_norm_quant else None" in init_source
    assert "hidden_states_scale" not in forward_source


def test_decoder_input_fusion_requires_every_attention_consumer_to_be_ptpc():
    init_source = _method_source("KimiFullAttention", "__init__")

    assert "self.input_quant_linears = (" in init_source
    assert "self.fused_qkv_a_proj" in init_source
    assert "self.g_proj" in init_source
    assert "_linears_effectively_consume_per_token_fp8(" in init_source


def test_effective_ptpc_helper_checks_online_target_without_leaking_other_schemes():
    source = KIMI_PATH.read_text()

    assert "def _layer_effectively_consumes_per_token_fp8(" in source
    assert "use_online_quant=True" in source
    assert "should_skip_online_quant(" in source
    assert "effective_cfg.quant_type == QuantType.per_Token" in source
    assert "effective_cfg.quant_dtype == dtypes.fp8" in source


def test_effective_ptpc_helper_resolves_source_and_online_configs():
    namespace, per_token, no_quant, fp8 = _load_effective_config_helpers()
    consumes_ptpc = namespace["_layer_effectively_consumes_per_token_fp8"]
    bf16 = object()
    block_fp8 = object()
    mxfp4 = object()
    fp4 = object()

    def cfg(quant_type, quant_dtype):
        return SimpleNamespace(quant_type=quant_type, quant_dtype=quant_dtype)

    source = {
        "online_ptpc": cfg(no_quant, bf16),
        "excluded_block": cfg(block_fp8, fp8),
        "source_mxfp4": cfg(mxfp4, fp4),
        "source_ptpc": cfg(per_token, fp8),
    }
    online = {
        "online_ptpc": cfg(per_token, fp8),
        "excluded_block": cfg(no_quant, bf16),
        "source_mxfp4": cfg(no_quant, bf16),
        "source_ptpc": cfg(per_token, fp8),
    }
    quant_config = _FakeQuantConfig(source, online)

    assert consumes_ptpc(quant_config, "online_ptpc")
    assert not consumes_ptpc(quant_config, "excluded_block")
    assert not consumes_ptpc(quant_config, "source_mxfp4")
    assert consumes_ptpc(quant_config, "source_ptpc")


def test_shared_input_fusion_requires_all_consumers_to_be_ptpc():
    namespace, per_token, no_quant, fp8 = _load_effective_config_helpers()
    all_consume_ptpc = namespace["_linears_effectively_consume_per_token_fp8"]
    bf16 = object()

    def cfg(quant_type, quant_dtype):
        return SimpleNamespace(quant_type=quant_type, quant_dtype=quant_dtype)

    linears = tuple(SimpleNamespace(prefix=name) for name in ("q", "kv", "g"))
    source = {name: cfg(no_quant, bf16) for name in ("q", "kv", "g")}
    online = {name: cfg(per_token, fp8) for name in ("q", "kv", "g")}
    assert all_consume_ptpc(_FakeQuantConfig(source, online), linears)

    online["g"] = cfg(no_quant, bf16)
    assert not all_consume_ptpc(_FakeQuantConfig(source, online), linears)


def test_kda_custom_op_carries_scale_and_preserves_bf16_output():
    fake_source = ast.get_source_segment(
        KIMI_PATH.read_text(),
        next(
            node
            for node in _tree().body
            if isinstance(node, ast.FunctionDef)
            and node.name == "_kda_attention_with_output_fake"
        ),
    )
    op_source = ast.get_source_segment(
        KIMI_PATH.read_text(),
        next(
            node
            for node in _tree().body
            if isinstance(node, ast.FunctionDef)
            and node.name == "kda_attention_with_output"
        ),
    )
    forward_source = _method_source("KimiKDAAttention", "forward")
    impl_source = _method_source("KimiKDAAttention", "_forward_impl")

    assert "hidden_states_scale" in fake_source
    assert "dtype=torch.bfloat16" in fake_source
    assert "hidden_states_scale" in op_source
    assert "if isinstance(hidden_states, tuple):" in forward_source
    assert "hidden_states, hidden_states_scale = hidden_states" in forward_source
    assert (
        "hidden_states_scale = hidden_states_scale[:num_actual_tokens]" in impl_source
    )
    assert "self.in_proj(hidden_states, x_scale=hidden_states_scale)" in impl_source
    assert "out = fused_in.new_empty" in impl_source


def test_dense_mlp_accepts_prequantized_activation_scale():
    source = _method_source("KimiMLP", "forward")

    assert "if isinstance(x, tuple):" in source
    assert "x, x_scale = x" in source
    assert "self.gate_up_proj(x, x_scale=x_scale)" in source


def test_full_attention_checkpoint_projections_map_into_fused_qkv_a():
    mapping = _load_packed_modules_mapping()([0])

    assert mapping[".q_a_proj"] == (".fused_qkv_a_proj", 0)
    assert mapping[".kv_a_proj_with_mqa"] == (".fused_qkv_a_proj", 1)
    assert mapping[".layers.0.self_attn.q_proj"] == (
        ".layers.0.self_attn.in_proj",
        0,
    )


def test_root_config_rebuild_reapplies_packed_quant_name_mapping():
    source = _method_source("KimiK3ForCausalLM", "__init__")

    rebuild_at = source.index("atom_config.quant_config = QuantizationConfig(")
    remap_at = source.index("atom_config.quant_config.remap_layer_name(")
    assert remap_at > rebuild_at
    assert "packed_modules_mapping=self.packed_modules_mapping" in source
