import ast
from pathlib import Path
from types import SimpleNamespace

LINEAR_PATH = Path(__file__).resolve().parents[1] / "atom" / "model_ops" / "linear.py"


def _linear_source_tree() -> ast.Module:
    return ast.parse(LINEAR_PATH.read_text())


def _load_alignment_helper():
    tree = _linear_source_tree()
    helper = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "_can_use_a8w8_preshuffle"
    )
    namespace = {}
    module = ast.Module(body=[helper], type_ignores=[])
    exec(
        compile(ast.fix_missing_locations(module), str(LINEAR_PATH), "exec"), namespace
    )
    return namespace["_can_use_a8w8_preshuffle"]


def _load_output_padding_helper():
    tree = _linear_source_tree()
    helper = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_a8w8_preshuffle_output_padding"
    )
    namespace = {}
    module = ast.Module(body=[helper], type_ignores=[])
    exec(
        compile(ast.fix_missing_locations(module), str(LINEAR_PATH), "exec"), namespace
    )
    return namespace["_a8w8_preshuffle_output_padding"]


def _load_linear_method(method_name: str):
    tree = _linear_source_tree()
    linear_base = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "LinearBase"
    )
    method = next(
        node
        for node in linear_base.body
        if isinstance(node, ast.FunctionDef) and node.name == method_name
    )
    namespace = {}
    module = ast.Module(body=[method], type_ignores=[])
    exec(
        compile(ast.fix_missing_locations(module), str(LINEAR_PATH), "exec"), namespace
    )
    return namespace[method_name]


def test_a8w8_preshuffle_requires_aligned_output_and_input_dims():
    can_preshuffle = _load_alignment_helper()

    assert can_preshuffle(6288, 7168)
    assert not can_preshuffle(6284, 7168)
    assert not can_preshuffle(6288, 7169)


def test_unaligned_fp8_output_rows_pad_to_ck_tile_boundary():
    output_padding = _load_output_padding_helper()

    assert output_padding(6284) == 116
    assert output_padding(6400) == 0


def test_small_k_uses_tuned_aiter_preshuffle_path():
    source = LINEAR_PATH.read_text()

    assert "_requires_triton_a8w8_fallback" not in source
    assert "_use_triton_a8w8_fallback" not in source
    assert "elif use_triton_gemm() and gemm_a8w8_triton is not None:" in source


def test_padded_fp8_outputs_are_sliced_back_to_logical_width():
    source = LINEAR_PATH.read_text()

    assert "_output_size_before_padding" in source
    assert "self.weight_scale.data = torch.cat" in source
    assert "y = y[..., : self._output_size_before_padding]" in source


def test_empty_fused_linear_shell_skips_duplicate_online_quantization():
    process_weights = _load_linear_method("process_weights_after_loading")
    shell = SimpleNamespace(
        weight=SimpleNamespace(numel=lambda: 0),
        quant_config=SimpleNamespace(online_quant=True),
        online_quantize_weight=lambda: (_ for _ in ()).throw(
            AssertionError("empty fused shell must not be quantized")
        ),
    )

    process_weights(shell)
