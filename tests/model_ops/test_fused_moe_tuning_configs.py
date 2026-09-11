import csv
from pathlib import Path

from atom.model_ops.fused_moe.configs import DSV4_RCCL_FMOE_CONFIG


def test_dsv4_rccl_fmoe_config_is_packaged_and_targets_validated_shapes():
    config_path = Path(DSV4_RCCL_FMOE_CONFIG)
    assert config_path.is_file()

    with config_path.open(newline="") as config_file:
        rows = list(csv.DictReader(config_file))

    assert [(row["token"], row["expert"], row["topk"]) for row in rows] == [
        ("32768", "48", "6"),
        ("131072", "48", "6"),
    ]
    assert {row["gfx"] for row in rows} == {"gfx950"}
    assert {row["model_dim"] for row in rows} == {"7168"}
    assert {row["inter_dim"] for row in rows} == {"3072"}
    assert {row["kernelName1"] for row in rows} == {
        "flydsl_moe1_afp8_wfp4_bf16_t128x256x256_w2_bnt0_gui_fp8"
    }
    assert {row["kernelName2"] for row in rows} == {
        "flydsl_moe2_afp8_wfp4_bf16_t64x128x256_atomic"
    }
