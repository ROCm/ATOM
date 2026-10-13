"""The alignment we register must let vLLM's own validity test pass.

`is_mamba_prefill_checkpoint_valid` ends with
`(checkpoint_position - query_start) % checkpoint_alignment == 0`, and the
position it checks is the last hash-block boundary of the prompt. Register
the tier's state interval there and the test fails on every step of every
run -- measured as `image_write_col_calls=4, image_write_col_null=4`, no
image ever written, while every other counter looked healthy.

This asserts against vLLM's function, not against a copy of its rule.
"""

import pytest

from vllm.v1.kv_cache_interface import (
    get_mamba_prefill_checkpoint_position,
    is_mamba_prefill_checkpoint_valid,
)

BLOCK = 256


def _valid(seq_len, alignment, query_start=0):
    position = get_mamba_prefill_checkpoint_position(
        seq_len, BLOCK, drop_eagle_block=False
    )
    return is_mamba_prefill_checkpoint_valid(
        query_start=query_start,
        query_end=seq_len,
        checkpoint_position=position,
        hash_block_size=BLOCK,
        mamba_block_size=BLOCK,
        checkpoint_alignment=alignment,
    )


@pytest.mark.parametrize("seq_len", [1024, 2816, 6912, 9472, 12345])
def test_the_block_size_lets_a_checkpoint_through(seq_len):
    assert _valid(seq_len, BLOCK) is True


@pytest.mark.parametrize("seq_len", [6912, 9472])
def test_the_state_interval_usually_does_not(seq_len):
    """The value this branch shipped first. Kept as the negative control:
    without it, "images are written" and "alignment is anything" are the
    same test."""
    assert _valid(seq_len, 4096) is False


def test_the_state_interval_fails_intermittently_rather_than_always():
    """Worse than always failing, and worth stating: a prompt whose last
    block boundary happens to be a multiple of 4096 does get a checkpoint.
    So the broken configuration writes images for some prompt lengths and
    not others, and a run that happened to use the lucky lengths would have
    looked like it worked."""
    assert _valid(12345, 4096) is True   # boundary 12288 = 3 * 4096
    assert _valid(12000, 4096) is False  # boundary 11776


def test_what_we_actually_register_is_the_block_size():
    import os

    from atom.plugin.vllm.deepseek_v41_bridge import ATOM_DEEPSEEK_V41_BLOCK_SIZE

    registered = int(
        os.environ.get("ATOM_V41_IMAGE_ALIGNMENT", "")
        or ATOM_DEEPSEEK_V41_BLOCK_SIZE
    )
    assert _valid(6912, registered) is True
