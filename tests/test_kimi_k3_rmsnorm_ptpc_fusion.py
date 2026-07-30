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


def _module_func(name: str) -> ast.FunctionDef:
    return next(
        node
        for node in _tree().body
        if isinstance(node, ast.FunctionDef) and node.name == name
    )


def _load_effective_layer_quant():
    """Exec ``_effective_layer_quant`` in isolation with faked dependencies.

    The helper resolves ``(quant_type, quant_dtype)`` a Linear runs with: the
    static checkpoint scheme, overridden by the online-quant target only when
    that override actually applies (``should_skip_online_quant``).
    """
    no_quant = object()
    per_token = object()

    class _DType:  # supports ``_DType | None`` in the return annotation
        pass

    namespace = {
        "QuantizationConfig": object,
        "QuantType": SimpleNamespace(No=no_quant, per_Token=per_token),
        "torch": SimpleNamespace(dtype=_DType),
        "should_skip_online_quant": lambda current_type, current_dtype, online: (
            online.quant_type is no_quant
            or (
                current_type is online.quant_type
                and current_dtype is online.quant_dtype
            )
        ),
    }
    module = ast.Module(body=[_module_func("_effective_layer_quant")], type_ignores=[])
    code = compile(ast.fix_missing_locations(module), str(KIMI_PATH), "exec")
    exec(code, namespace)  # noqa: S102
    return namespace["_effective_layer_quant"], no_quant, per_token


def _load_packed_modules_mapping():
    module = ast.Module(
        body=[_module_func("_kda_packed_modules_mapping")], type_ignores=[]
    )
    namespace = {}
    code = compile(ast.fix_missing_locations(module), str(KIMI_PATH), "exec")
    exec(code, namespace)  # noqa: S102
    return namespace["_kda_packed_modules_mapping"]


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
    assert "self.fused_qkv_a_proj(hidden_states, hidden_states_scale)" in forward_source
    assert "q_c, kv_c, k_rope = torch.split(" in forward_source


def test_full_attention_fuses_latent_norm_quant_and_forwards_scales():
    init_source = _method_source("KimiFullAttention", "__init__")
    forward_source = _method_source("KimiFullAttention", "forward")

    # The q/kv latent norms (+ optional q-activation quant) collapse into one
    # _fuse_rmsnorm_quant launch; the quant scheme is q_b_proj's (the consumer of
    # the normed q). kv_c stays bf16 (written to the KV cache, re-projected in MLA).
    assert "qknorm_type, qknorm_dtype = _effective_layer_quant(" in init_source
    assert 'f"{prefix}.q_b_proj"' in init_source
    assert "self.fuse_qknorm_quant = qknorm_dtype in (dtypes.fp8, dtypes.fp4x2)" in (
        init_source
    )
    assert "from atom.models.deepseek_v2 import _fuse_rmsnorm_quant" in forward_source
    assert "if isinstance(hidden_states, tuple):" in forward_source
    assert "hidden_states, hidden_states_scale = hidden_states" in forward_source
    assert "(q, q_scale), _, kv, _ = _fuse_rmsnorm_quant(" in forward_source
    assert "dtype_quant=self.qknorm_dtype" in forward_source
    assert "quant_type=self.qknorm_quant_type_value" in forward_source
    assert "self.attn(q, kv, k_rope, positions, q_scale=q_scale)" in forward_source
    assert "self.g_proj(hidden_states, hidden_states_scale)" in forward_source


def test_full_attention_input_fusion_requires_both_consumers_same_scheme():
    # input_layernorm may fuse its activation quant only when BOTH consumers of
    # the normed hidden state -- fused_qkv_a_proj and g_proj -- run the same
    # fusable RMSNorm quant scheme (else a mismatched consumer mis-GEMMs).
    init_source = _method_source("KimiFullAttention", "__init__")

    assert (
        'a_scheme = _effective_layer_quant(quant_config, f"{prefix}.fused_qkv_a_proj")'
        in init_source
    )
    assert 'g_scheme = _effective_layer_quant(quant_config, f"{prefix}.g_proj")' in (
        init_source
    )
    assert "a_scheme[0] in _RMS_FUSABLE_QUANT_TYPES and a_scheme == g_scheme" in (
        init_source
    )
    assert 'self.input_quant_prefix = f"{prefix}.fused_qkv_a_proj"' in init_source


def test_kda_input_fusion_gated_on_in_proj_scheme():
    # KDA has a single hidden-state consumer (the fused in_proj), so its input
    # fusion gates on that one scheme rather than a cross-consumer agreement.
    init_source = _method_source("KimiKDAAttention", "__init__")

    assert (
        'in_proj_type, _ = _effective_layer_quant(quant_config, f"{prefix}.in_proj")'
        in init_source
    )
    assert "self.fuse_input_norm_quant = in_proj_type in _RMS_FUSABLE_QUANT_TYPES" in (
        init_source
    )
    assert 'self.input_quant_prefix = f"{prefix}.in_proj"' in init_source


def test_decoder_fuses_attention_input_and_dense_mlp_norms_only():
    init_source = _method_source("KimiDecoderLayer", "__init__")
    forward_source = _method_source("KimiDecoderLayer", "forward")

    # input_layernorm delegates its fusion decision to self_attn.
    assert "fused_quant=self.self_attn.fuse_input_norm_quant" in init_source
    assert "prefix=self.self_attn.input_quant_prefix" in init_source
    # post_attention_layernorm fuses into the dense-MLP gate_up_proj, and only for
    # dense layers -- MoE layers (no self.mlp) have mixed-precision consumers.
    assert 'if hasattr(self, "mlp"):' in init_source
    assert "self.fuse_ffn_norm_quant = ffn_type in _RMS_FUSABLE_QUANT_TYPES" in (
        init_source
    )
    assert "fused_quant=self.fuse_ffn_norm_quant" in init_source
    assert "quant_config if self.fuse_ffn_norm_quant else None" in init_source
    assert 'prefix=f"{prefix}.mlp.gate_up_proj"' in init_source
    # The (fp8, scale) tuple is confined to the norm->consumer hop; the decoder's
    # own dataflow never names a scale.
    assert "hidden_states_scale" not in forward_source


def test_effective_layer_quant_source_matches_linear_resolution():
    source = KIMI_PATH.read_text()

    assert "def _effective_layer_quant(" in source
    assert "use_online_quant=True" in source
    assert "should_skip_online_quant(" in source
    assert "return cfg.quant_type, cfg.quant_dtype" in source


def test_effective_layer_quant_resolves_source_and_online_configs():
    effective, no_quant, per_token = _load_effective_layer_quant()
    bf16 = object()
    fp8 = object()
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

    # Online override applies -> resolves to the online ptpc target.
    assert effective(quant_config, "online_ptpc") == (per_token, fp8)
    # Online says No -> the static checkpoint scheme is kept.
    assert effective(quant_config, "excluded_block") == (block_fp8, fp8)
    assert effective(quant_config, "source_mxfp4") == (mxfp4, fp4)
    # Static already ptpc and online matches -> kept (no double-apply).
    assert effective(quant_config, "source_ptpc") == (per_token, fp8)
    # No quant_config at all -> No/None.
    assert effective(None, "anything") == (no_quant, None)


def test_kda_custom_op_carries_scale_and_preserves_bf16_output():
    fake_source = ast.get_source_segment(
        KIMI_PATH.read_text(), _module_func("_kda_attention_with_output_fake")
    )
    op_source = ast.get_source_segment(
        KIMI_PATH.read_text(), _module_func("kda_attention_with_output")
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
    assert "self.gate_up_proj(x, x_scale)" in source


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
    remap_at = source.index("self.quant_config.remap_layer_name(")
    assert remap_at > rebuild_at
    assert "packed_modules_mapping=self.packed_modules_mapping" in source
