"""Numeric parity: ATOM's Qwen3.8-Flash-Next HyperConnection vs the vLLM reference."""

import pytest
import torch

from tests.qwen3_8_flash_next.parity_harness import init_single_rank, load_reference

H, HC, LOWRANK, EPS = 256, 4, 32, 1e-6


@pytest.fixture(scope="module")
def modules():
    init_single_rank()
    ref_mod = load_reference("common/hyperconnection.py", "wef_hyperconnection")
    from atom.models.qwen3_8_flash_next import Qwen3_8FlashNextHyperConnection

    cfg = ref_mod.HyperConnectionConfig(
        hc_count=HC,
        hidden_size=H,
        params_dtype=torch.float32,
        hc_lowrank=LOWRANK,
        rms_norm_eps=EPS,
        hc_per_branch_norm=True,
    )
    # Both sides on CUDA in fp32: ATOM's Linear builds bf16 params by default,
    # so cast to a common dtype to make the comparison exact rather than
    # measuring bf16 rounding.
    ref = (
        ref_mod.GatedResidualSimple(cfg, layer_idx=0, role="attn").eval().cuda().float()
    )
    mine = Qwen3_8FlashNextHyperConnection(H, HC, LOWRANK, eps=EPS).eval().cuda().float()

    torch.manual_seed(0)
    with torch.no_grad():
        for name, shape in (
            ("hc_norm", (HC * H,)),
            ("input_mix_weight_down", (LOWRANK, HC * H)),
            ("input_mix_weight_up", (HC * H, LOWRANK)),
            ("block_inject_weight", (HC, HC * H)),
        ):
            w = (torch.randn(*shape) * 0.05).cuda()
            getattr(ref, name).weight.copy_(w)
            getattr(mine, name).weight.copy_(w)
    return ref, mine


def _run_pair(ref, mine, x, block_out, fused: bool):
    mix = mine.mix if fused else mine.mix_native
    combine = mine.combine if fused else mine.combine_native
    with torch.no_grad():
        mixed_ref, res_ref = ref.mix(x)
        out_ref = ref.combine(block_out, res_ref)
        mixed_mine, res_mine = mix(x)
        out_mine = combine(block_out, res_mine)
    return (mixed_ref, out_ref), (mixed_mine, out_mine)


def test_mix_and_combine_match_reference(modules):
    """The MATH is bitwise identical to the reference.

    Pinned to the eager path on purpose: `mix_native` / `combine_native` are
    the definition this equality is about, and holding it exact is what makes
    the fused kernels' tolerance below meaningful rather than circular.
    """
    ref, mine = modules
    torch.manual_seed(1)
    x = torch.randn(7, HC * H).cuda()
    block_out = torch.randn(7, H).cuda()

    (mixed_ref, out_ref), (mixed_mine, out_mine) = _run_pair(
        ref, mine, x, block_out, fused=False
    )
    assert mixed_ref.shape == (7, H)
    assert out_ref.shape == (7, HC * H)
    torch.testing.assert_close(mixed_mine, mixed_ref, rtol=0, atol=0)
    torch.testing.assert_close(out_mine, out_ref, rtol=0, atol=0)


def test_fused_path_matches_reference_to_rounding(modules):
    """The fused kernels reach the same answer up to fp32 reduction order.

    They sum the group and the streams in a different order than torch, so
    equality can only be to rounding; anything larger would be an algorithmic
    difference, which is what this bound is set to catch.
    """
    ref, mine = modules
    torch.manual_seed(1)
    x = torch.randn(7, HC * H).cuda()
    block_out = torch.randn(7, H).cuda()

    (mixed_ref, out_ref), (mixed_mine, out_mine) = _run_pair(
        ref, mine, x, block_out, fused=True
    )
    torch.testing.assert_close(mixed_mine, mixed_ref, rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(out_mine, out_ref, rtol=1e-6, atol=1e-6)


def test_final_mixer_has_no_inject_weight(modules):
    """The model's final hyper_connection_mixer only ever calls mix()."""
    from atom.models.qwen3_8_flash_next import Qwen3_8FlashNextHyperConnection

    final = Qwen3_8FlashNextHyperConnection(H, HC, LOWRANK, has_block_inject=False, eps=EPS)
    assert final.block_inject_weight is None
    with pytest.raises(RuntimeError, match="combine was disabled"):
        final.combine(torch.zeros(1, H), (torch.zeros(1, HC * H),) * 2)
