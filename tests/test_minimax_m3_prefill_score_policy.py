# SPDX-License-Identifier: MIT

import pytest

from atom.model_ops.minimax_m3.prefill_score_policy import prefill_score_config


@pytest.mark.parametrize(
    "arch,is_fp8,q_len,blocks,batch,heads,want",
    [
        ("gfx950", True, 2517, 823, 1, 1, (128, None, None, None, 1, False, False)),
        (
            "gfx1250",
            False,
            2517,
            823,
            1,
            1,
            (128, None, None, None, 1, False, False),
        ),
        ("gfx1250", True, 96, 640, 1, 4, (128, None, None, None, 1, False, False)),
        ("gfx1250", True, 512, 640, 1, 4, (128, None, None, None, 1, False, False)),
        (
            "gfx1250",
            True,
            2048,
            256,
            1,
            1,
            (128, None, None, None, 1, False, False),
        ),
        (
            "gfx1250",
            True,
            2048,
            640,
            1,
            1,
            (128, None, None, None, 1, False, False),
        ),
        ("gfx1250", True, 2517, 823, 1, 1, (64, 22, 2, 3, 4, True, True)),
        ("gfx1250", True, 2655, 796, 1, 1, (64, 22, 2, 3, 4, True, True)),
        (
            "gfx1250",
            True,
            2159,
            974,
            1,
            1,
            (128, None, None, None, 1, False, False),
        ),
        (
            "gfx1250",
            True,
            3072,
            640,
            1,
            1,
            (128, None, None, None, 1, False, False),
        ),
        (
            "gfx1250",
            True,
            2858,
            3942,
            1,
            1,
            (128, None, None, None, 1, False, False),
        ),
        (
            "gfx1250",
            True,
            6259,
            663,
            1,
            1,
            (256, 34, 4, 1, 1, False, True),
        ),
        ("gfx1250", True, 4104, 1895, 1, 1, (256, 64, 4, 1, 1, False, True)),
        (
            "gfx1250",
            True,
            4096,
            320,
            1,
            1,
            (128, None, None, None, 1, False, False),
        ),
        ("gfx1250", True, 6144, 320, 1, 1, (256, 16, 4, 1, 1, False, True)),
        ("gfx1250", True, 5459, 1682, 1, 1, (256, 74, 4, 1, 1, False, True)),
        ("gfx1250", True, 2331, 2706, 1, 1, (256, 54, 4, 1, 1, False, True)),
        (
            "gfx1250",
            True,
            3072,
            4096,
            1,
            1,
            (128, None, None, None, 1, False, False),
        ),
        ("gfx1250", True, 640, 257, 1, 4, (128, None, None, None, 1, False, False)),
        ("gfx1250", True, 640, 257, 4, 1, (128, None, None, None, 1, False, False)),
        ("gfx1250", True, 4096, 640, 4, 1, (128, None, None, None, 1, False, False)),
        ("gfx1250", True, 2048, 640, 1, 4, (128, None, None, None, 1, False, False)),
        ("gfx1250", True, 2517, 640, 1, 4, (128, None, None, None, 1, False, False)),
        ("gfx1250", True, 4096, 320, 1, 4, (128, None, None, None, 1, False, False)),
        ("gfx1250", True, 8192, 320, 1, 4, (128, 28, 2, 1, 1, False, True)),
        ("gfx1250", True, 4096, 640, 1, 4, (128, 28, 2, 1, 1, False, True)),
        ("gfx1250", True, 32768, 640, 1, 4, (128, 28, 2, 1, 1, False, True)),
        ("gfx1250", True, 32769, 640, 1, 4, (128, 28, 2, 1, 1, False, True)),
        (
            "gfx1250",
            True,
            65536,
            640,
            1,
            1,
            (128, None, None, None, 1, False, False),
        ),
        ("gfx1250", True, 65536, 640, 1, 4, (128, 28, 2, 1, 1, False, True)),
        ("gfx1250", True, 32768, 2048, 1, 4, (128, 28, 2, 1, 1, False, True)),
        (
            "gfx1250",
            True,
            65536,
            2048,
            1,
            4,
            (128, None, None, None, 1, False, False),
        ),
        ("gfx1250", True, 131072, 2048, 1, 1, (256, 2048, 4, 1, 1, False, True)),
    ],
)
def test_prefill_score_config(arch, is_fp8, q_len, blocks, batch, heads, want):
    assert prefill_score_config(arch, is_fp8, q_len, blocks, batch, heads, 256) == want


def test_prefill_score_config_cache_is_bounded():
    assert prefill_score_config.cache_parameters() == {
        "maxsize": 128,
        "typed": False,
    }
