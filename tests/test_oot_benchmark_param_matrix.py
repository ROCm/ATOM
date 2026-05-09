import importlib.util
from pathlib import Path

import pytest


ATOM_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ATOM_ROOT / ".github/scripts/build_oot_benchmark_param_matrix.py"


def _load_param_matrix_module():
    assert SCRIPT_PATH.exists(), "Expected benchmark param matrix helper to exist."
    spec = importlib.util.spec_from_file_location(
        "build_oot_benchmark_param_matrix", SCRIPT_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_build_param_matrix_generates_cartesian_product():
    module = _load_param_matrix_module()

    params = module.build_param_matrix(
        isl_osl_pairs_text="1024,1024;2048,1024",
        concurrency_values_text="4,8",
        random_range_ratios_text="0.8,1.0",
    )

    assert params == [
        {
            "input_length": 1024,
            "output_length": 1024,
            "concurrency": 4,
            "random_range_ratio": "0.8",
        },
        {
            "input_length": 1024,
            "output_length": 1024,
            "concurrency": 4,
            "random_range_ratio": "1.0",
        },
        {
            "input_length": 1024,
            "output_length": 1024,
            "concurrency": 8,
            "random_range_ratio": "0.8",
        },
        {
            "input_length": 1024,
            "output_length": 1024,
            "concurrency": 8,
            "random_range_ratio": "1.0",
        },
        {
            "input_length": 2048,
            "output_length": 1024,
            "concurrency": 4,
            "random_range_ratio": "0.8",
        },
        {
            "input_length": 2048,
            "output_length": 1024,
            "concurrency": 4,
            "random_range_ratio": "1.0",
        },
        {
            "input_length": 2048,
            "output_length": 1024,
            "concurrency": 8,
            "random_range_ratio": "0.8",
        },
        {
            "input_length": 2048,
            "output_length": 1024,
            "concurrency": 8,
            "random_range_ratio": "1.0",
        },
    ]


def test_build_param_matrix_defaults_random_range_ratio_to_point_eight():
    module = _load_param_matrix_module()

    params = module.build_param_matrix(
        isl_osl_pairs_text="1024,1024",
        concurrency_values_text="64",
        random_range_ratios_text="",
    )

    assert params == [
        {
            "input_length": 1024,
            "output_length": 1024,
            "concurrency": 64,
            "random_range_ratio": "0.8",
        }
    ]


def test_build_param_matrix_supports_512_concurrency_without_implicit_expansion():
    module = _load_param_matrix_module()

    params = module.build_param_matrix(
        isl_osl_pairs_text="8192,1024",
        concurrency_values_text="512",
        random_range_ratios_text="0.8",
    )

    assert [param["concurrency"] for param in params] == [512]


def test_build_param_matrix_rejects_unsupported_concurrency():
    module = _load_param_matrix_module()

    with pytest.raises(
        ValueError, match="Unsupported concurrency: 12. Allowed values: 4,8,16,32,64,128,256,512"
    ):
        module.build_param_matrix(
            isl_osl_pairs_text="1024,1024",
            concurrency_values_text="12",
            random_range_ratios_text="0.8",
        )
