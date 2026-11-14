# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

"""Wan2.1 VAE: parity with Diffusers' AutoencoderKLWan, layout, conv3d dispatch.

CPU, except the fused norm checks at the end, which skip without a GPU. The
AITER kernels are stood in for by torch convs that take and return the same
layouts the real ones do, so what is checked here is the plumbing around them
-- the GPU numerics of the kernels themselves are what
``atom.diffusion.examples.wan_vae_bench`` measures.
"""

import json
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from atom.diffusion.models.wan import conv_dispatch
from atom.diffusion.models.wan import vae as wan_vae
from atom.diffusion.models.wan.conv_dispatch import (
    conv3d,
    conv3d_backend_override,
    conv3d_call_counts,
    reset_conv3d_call_counts,
    select_conv3d_backend,
)
from atom.diffusion.models.wan.vae import (
    WanVAE,
    WanVAEConfig,
    convert_original_wan_vae_state_dict,
    load_wan_vae,
)

Z_DIM = 4


def tiny_config() -> WanVAEConfig:
    gen = torch.Generator().manual_seed(7)
    return WanVAEConfig(
        base_dim=8,
        z_dim=Z_DIM,
        latents_mean=tuple(torch.randn(Z_DIM, generator=gen).tolist()),
        latents_std=tuple((torch.rand(Z_DIM, generator=gen) + 0.5).tolist()),
    )


def _randomize_gammas(model: torch.nn.Module) -> None:
    gen = torch.Generator().manual_seed(11)
    with torch.no_grad():
        for name, param in model.named_parameters():
            if name.endswith("gamma"):
                param.copy_(torch.rand(param.shape, generator=gen) + 0.5)


def tiny_vae(channels_last: bool = True) -> WanVAE:
    torch.manual_seed(0)
    vae = WanVAE(tiny_config(), channels_last=channels_last).eval()
    _randomize_gammas(vae)
    return vae


def sample_video(batch=1, frames=9, height=32, width=48) -> torch.Tensor:
    gen = torch.Generator().manual_seed(3)
    return torch.rand(batch, 3, frames, height, width, generator=gen) * 2 - 1


@pytest.fixture
def diffusers_pair():
    """A tiny Diffusers AutoencoderKLWan and a WanVAE holding its weights."""
    diffusers = pytest.importorskip("diffusers")
    cfg = tiny_config()
    torch.manual_seed(0)
    ref = diffusers.AutoencoderKLWan(
        base_dim=cfg.base_dim,
        z_dim=cfg.z_dim,
        dim_mult=list(cfg.dim_mult),
        num_res_blocks=cfg.num_res_blocks,
        temperal_downsample=list(cfg.temporal_downsample),
        latents_mean=list(cfg.latents_mean),
        latents_std=list(cfg.latents_std),
    ).eval()
    _randomize_gammas(ref)

    def ours(channels_last: bool) -> WanVAE:
        vae = WanVAE(cfg, channels_last=channels_last).eval()
        vae.load_state_dict(ref.state_dict(), strict=True)
        return vae

    return ref, ours


@pytest.mark.parametrize("channels_last", [True, False])
@pytest.mark.parametrize("batch", [1, 2])
def test_encode_matches_diffusers(diffusers_pair, channels_last, batch):
    ref, ours = diffusers_pair
    video = sample_video(batch=batch)
    with torch.no_grad():
        expected = ref.encode(video).latent_dist
    moments = ours(channels_last).encode_moments(video)
    assert moments.shape == (batch, 2 * Z_DIM, 3, 4, 6)
    torch.testing.assert_close(moments, expected.parameters, rtol=1e-4, atol=1e-5)
    torch.testing.assert_close(
        ours(channels_last).encode(video), expected.mode(), rtol=1e-4, atol=1e-5
    )


@pytest.mark.parametrize("channels_last", [True, False])
@pytest.mark.parametrize("batch", [1, 2])
def test_decode_matches_diffusers(diffusers_pair, channels_last, batch):
    ref, ours = diffusers_pair
    latents = torch.randn(
        batch, Z_DIM, 3, 4, 6, generator=torch.Generator().manual_seed(5)
    )
    with torch.no_grad():
        expected = ref.decode(latents).sample
    video = ours(channels_last).decode(latents)
    assert video.shape == (batch, 3, 9, 32, 48)
    assert video.is_contiguous()
    torch.testing.assert_close(video, expected, rtol=1e-4, atol=1e-5)


def test_bf16_decode_tracks_diffusers(diffusers_pair):
    """bf16 is the dtype the AITER kernels run in: decoding in it must be as
    accurate as Diffusers decoding in it, both measured against fp32.

    Not a max-diff bound between the two bf16 runs: each sits ~0.045 from
    fp32 on its own, so their mutual max diff has no stable bound. A 10%
    error in one norm's gamma costs ~4 dB here; the bar allows ~1.8.

    NCDHW: torch's CPU bf16 conv3d on channels_last_3d input is not
    deterministic (the same call repeated differs), which would make this
    flaky without saying anything about the VAE.
    """
    ref, ours = diffusers_pair
    latents = torch.randn(1, Z_DIM, 2, 4, 6, generator=torch.Generator().manual_seed(9))
    with torch.no_grad():
        exact = ref.decode(latents).sample
        ref_bf16 = ref.to(torch.bfloat16).decode(latents.bfloat16()).sample
    video = ours(False).to(torch.bfloat16).decode(latents)
    assert video.dtype == torch.bfloat16
    ours_mse = torch.mean((video.float() - exact) ** 2)
    ref_mse = torch.mean((ref_bf16.float() - exact) ** 2)
    assert ours_mse <= 1.5 * ref_mse, (ours_mse.item(), ref_mse.item())


def test_normalization_round_trips_and_uses_config_stats():
    vae = tiny_vae()
    z = torch.randn(1, Z_DIM, 2, 3, 3)
    mean = torch.tensor(vae.config.latents_mean).view(1, -1, 1, 1, 1)
    std = torch.tensor(vae.config.latents_std).view(1, -1, 1, 1, 1)
    torch.testing.assert_close(vae.normalize_latents(z), (z - mean) / std)
    torch.testing.assert_close(vae.denormalize_latents(vae.normalize_latents(z)), z)


def test_channels_last_is_kept_end_to_end():
    """Every conv of an NDHWC VAE sees NDHWC: no op on the way silently
    falls back to NCDHW, which on GPU would cost FlyDSL a transpose."""
    vae = tiny_vae(channels_last=True)
    reset_conv3d_call_counts()
    vae.decode(vae.encode(sample_video()))
    counts = conv3d_call_counts()
    assert counts and set(counts) == {("torch", "NDHWC")}


def test_ncdhw_vae_feeds_ncdhw():
    vae = tiny_vae(channels_last=False)
    reset_conv3d_call_counts()
    vae.decode(vae.encode(sample_video(frames=5)))
    assert set(conv3d_call_counts()) == {("torch", "NCDHW")}


def test_rejects_frame_counts_off_the_causal_lattice():
    with pytest.raises(ValueError, match="1 \\+ 4k"):
        tiny_vae().encode(sample_video(frames=8))


def test_rejects_non_wan21_configs():
    with pytest.raises(ValueError, match="Wan2.2"):
        WanVAEConfig.from_diffusers({"is_residual": True})
    cfg = WanVAEConfig.from_diffusers(
        {"base_dim": 96, "temperal_downsample": [False, True, True], "dropout": 0.0}
    )
    assert cfg.temporal_downsample == (False, True, True)
    assert (cfg.spatial_compression, cfg.temporal_compression) == (8, 4)


# --- conv3d dispatch -------------------------------------------------------


def _fake_gpu_tensor(dtype):
    return SimpleNamespace(device=SimpleNamespace(type="cuda", index=0), dtype=dtype)


@pytest.fixture
def fake_gpu(monkeypatch):
    """Pretend tensors live on a GPU of ``arch`` with both AITER kernels."""
    state = {"arch": "gfx950", "kernels": {"flydsl", "triton"}}
    monkeypatch.setattr(conv_dispatch, "_gfx_arch", lambda index: state["arch"])
    monkeypatch.setattr(
        conv_dispatch,
        "_aiter_kernel",
        lambda name: (lambda *a, **k: None) if name in state["kernels"] else None,
    )
    monkeypatch.delenv("ATOM_DIFFUSION_CONV3D_BACKEND", raising=False)
    return state


@pytest.mark.parametrize(
    ("arch", "dtype", "env", "kernels", "expected"),
    [
        ("gfx950", torch.bfloat16, None, {"flydsl", "triton"}, "flydsl"),
        ("gfx1250", torch.bfloat16, None, {"flydsl", "triton"}, "flydsl"),
        # FlyDSL is gfx950 / gfx1250 only; Triton runs elsewhere.
        ("gfx942", torch.bfloat16, None, {"flydsl", "triton"}, "triton"),
        # FlyDSL is bf16 only; Triton takes fp16 too, neither takes fp32.
        ("gfx950", torch.float16, None, {"flydsl", "triton"}, "triton"),
        ("gfx950", torch.float32, None, {"flydsl", "triton"}, "torch"),
        # The env var is a ceiling.
        ("gfx950", torch.bfloat16, "triton", {"flydsl", "triton"}, "triton"),
        ("gfx950", torch.bfloat16, "torch", {"flydsl", "triton"}, "torch"),
        # An AITER build without the kernel falls a rung.
        ("gfx950", torch.bfloat16, None, {"triton"}, "triton"),
        ("gfx950", torch.bfloat16, None, set(), "torch"),
    ],
)
def test_backend_ladder(monkeypatch, fake_gpu, arch, dtype, env, kernels, expected):
    fake_gpu.update(arch=arch, kernels=kernels)
    if env is not None:
        monkeypatch.setenv("ATOM_DIFFUSION_CONV3D_BACKEND", env)
    x = _fake_gpu_tensor(dtype)
    assert select_conv3d_backend(x, _fake_gpu_tensor(dtype)) == expected


def test_cpu_tensors_fall_back_to_torch(monkeypatch):
    """Off-GPU every call is torch's, whatever the env asks for, and AITER is
    never imported to find that out."""
    monkeypatch.setenv("ATOM_DIFFUSION_CONV3D_BACKEND", "flydsl")
    imported = []
    monkeypatch.setattr(
        conv_dispatch, "_aiter_kernel", lambda name: imported.append(name)
    )
    x = torch.randn(1, 4, 3, 6, 6, dtype=torch.bfloat16)
    w = torch.randn(8, 4, 3, 3, 3, dtype=torch.bfloat16)
    assert select_conv3d_backend(x, w) == "torch"
    with conv3d_backend_override("triton"):
        assert select_conv3d_backend(x, w) == "torch"
    reset_conv3d_call_counts()
    y = conv3d(x, w, padding=1)
    torch.testing.assert_close(y, F.conv3d(x, w, padding=1))
    assert conv3d_call_counts() == {("torch", "NCDHW"): 1}
    assert imported == []


@pytest.mark.parametrize(
    ("value", "expected"),
    [(None, "flydsl"), ("", "flydsl"), (" Triton ", "triton"), ("torch", "torch")],
)
def test_backend_env_parsing(monkeypatch, value, expected):
    if value is None:
        monkeypatch.delenv("ATOM_DIFFUSION_CONV3D_BACKEND", raising=False)
    else:
        monkeypatch.setenv("ATOM_DIFFUSION_CONV3D_BACKEND", value)
    assert conv_dispatch.configured_conv3d_backend() == expected


def test_unknown_backend_is_rejected(monkeypatch):
    monkeypatch.setenv("ATOM_DIFFUSION_CONV3D_BACKEND", "cudnn")
    x = torch.randn(1, 2, 1, 2, 2)
    with pytest.raises(ValueError, match="flydsl"):
        select_conv3d_backend(x, x)
    with pytest.raises(ValueError, match="flydsl"), conv3d_backend_override("cudnn"):
        pass


def _fake_flydsl(calls):
    """FlyDSL's contract: NDHWC in/out when asked, else NCDHW; K-major weight."""

    def kernel(x, weight, bias, *, stride, padding, input_layout="NCDHW", **kw):
        output_layout = kw.get("output_layout", "NCDHW")
        calls.append((input_layout, output_layout, x.is_contiguous()))
        assert x.is_contiguous()
        ncdhw = x.permute(0, 4, 1, 2, 3) if input_layout == "NDHWC" else x
        y = F.conv3d(ncdhw, weight, bias, stride, padding)
        return y.permute(0, 2, 3, 4, 1).contiguous() if output_layout == "NDHWC" else y

    return kernel


def _fake_triton(calls):
    """Triton's contract: logical NCDHW, channels_last_3d strides for ndhwc."""

    def kernel(x, weight, bias, stride, padding, layout="ncdhw"):
        calls.append((layout, conv_dispatch.is_channels_last(x)))
        fmt = torch.channels_last_3d if layout == "ndhwc" else torch.contiguous_format
        x = x.contiguous(memory_format=fmt)
        return F.conv3d(x, weight, bias, stride, padding).contiguous(memory_format=fmt)

    return kernel


@pytest.mark.parametrize("backend", ["flydsl", "triton"])
@pytest.mark.parametrize("channels_last", [True, False])
def test_aiter_layout_plumbing_matches_torch(monkeypatch, backend, channels_last):
    """Route a whole VAE through stand-ins with the AITER kernels' layout
    contracts: the output must equal the torch path, and an NDHWC VAE must
    hand every conv over as NDHWC."""
    vae = tiny_vae(channels_last=channels_last)
    video = sample_video(frames=5)
    expected = vae.decode(vae.encode(video))

    calls = []
    fake = (_fake_flydsl if backend == "flydsl" else _fake_triton)(calls)
    monkeypatch.setattr(conv_dispatch, "select_conv3d_backend", lambda x, w: backend)
    monkeypatch.setattr(conv_dispatch, "_aiter_kernel", lambda name: fake)
    got = vae.decode(vae.encode(video))

    torch.testing.assert_close(got, expected, rtol=1e-5, atol=1e-6)
    if backend == "flydsl":
        layouts = {(i, o) for i, o, _ in calls}
        assert layouts == (
            {("NDHWC", "NDHWC")} if channels_last else {("NCDHW", "NCDHW")}
        )
    else:
        assert {c for c in calls} == {
            ("ndhwc" if channels_last else "ncdhw", channels_last)
        }


# --- loading -----------------------------------------------------------------


def test_load_diffusers_checkpoint_dir(tmp_path):
    pytest.importorskip("safetensors")
    from safetensors.torch import save_file

    src = tiny_vae()
    cfg = src.config
    vae_dir = tmp_path / "vae"
    vae_dir.mkdir()
    (vae_dir / "config.json").write_text(
        json.dumps(
            {
                "_class_name": "AutoencoderKLWan",
                "base_dim": cfg.base_dim,
                "z_dim": cfg.z_dim,
                "dim_mult": list(cfg.dim_mult),
                "num_res_blocks": cfg.num_res_blocks,
                "temperal_downsample": list(cfg.temporal_downsample),
                "latents_mean": list(cfg.latents_mean),
                "latents_std": list(cfg.latents_std),
                "attn_scales": [],
                "dropout": 0.0,
            }
        )
    )
    save_file(
        {k: v.contiguous() for k, v in src.state_dict().items()},
        str(vae_dir / "diffusion_pytorch_model.safetensors"),
    )
    # The checkpoint root resolves to its vae/ subfolder.
    vae = load_wan_vae(str(tmp_path), dtype=torch.float32)
    assert vae.config == cfg
    video = sample_video(frames=5)
    torch.testing.assert_close(vae.encode(video), src.encode(video))

    (vae_dir / "diffusion_pytorch_model.safetensors").unlink()
    save_file(
        {k: v for k, v in src.state_dict().items() if "gamma" not in k},
        str(vae_dir / "diffusion_pytorch_model.safetensors"),
    )
    with pytest.raises(RuntimeError, match="missing"):
        load_wan_vae(str(vae_dir))


# Literal entries of the key tables in Diffusers' convert_wan_to_diffusers.py.
ORIGINAL_TO_DIFFUSERS = {
    "conv1.weight": "quant_conv.weight",
    "conv2.bias": "post_quant_conv.bias",
    "encoder.conv1.weight": "encoder.conv_in.weight",
    "decoder.conv1.bias": "decoder.conv_in.bias",
    "encoder.head.0.gamma": "encoder.norm_out.gamma",
    "decoder.head.2.weight": "decoder.conv_out.weight",
    "encoder.middle.0.residual.3.gamma": "encoder.mid_block.resnets.0.norm2.gamma",
    "decoder.middle.2.residual.6.bias": "decoder.mid_block.resnets.1.conv2.bias",
    "encoder.middle.1.to_qkv.weight": "encoder.mid_block.attentions.0.to_qkv.weight",
    "decoder.middle.1.proj.bias": "decoder.mid_block.attentions.0.proj.bias",
    "encoder.downsamples.3.shortcut.weight": "encoder.down_blocks.3.conv_shortcut.weight",
    "encoder.downsamples.5.time_conv.weight": "encoder.down_blocks.5.time_conv.weight",
    "encoder.downsamples.8.resample.1.bias": "encoder.down_blocks.8.resample.1.bias",
    "decoder.upsamples.2.residual.2.weight": "decoder.up_blocks.0.resnets.2.conv1.weight",
    "decoder.upsamples.3.time_conv.weight": "decoder.up_blocks.0.upsamplers.0.time_conv.weight",
    "decoder.upsamples.4.shortcut.weight": "decoder.up_blocks.1.resnets.0.conv_shortcut.weight",
    "decoder.upsamples.11.resample.1.weight": "decoder.up_blocks.2.upsamplers.0.resample.1.weight",
    "decoder.upsamples.14.residual.0.gamma": "decoder.up_blocks.3.resnets.2.norm1.gamma",
}


def test_original_checkpoint_key_conversion():
    converted = convert_original_wan_vae_state_dict(
        {key: torch.zeros(1) for key in ORIGINAL_TO_DIFFUSERS}
    )
    assert set(converted) == set(ORIGINAL_TO_DIFFUSERS.values())
    with torch.device("meta"):
        expected_keys = set(WanVAE().state_dict())
    assert set(ORIGINAL_TO_DIFFUSERS.values()) <= expected_keys


def _original_name(key: str, num_res_blocks: int = 2) -> str:
    """A Diffusers key spelled the way Wan2.1_VAE.pth spells it."""
    for diffusers_name, original in (
        (".norm1.", ".residual.0."),
        (".conv1.", ".residual.2."),
        (".norm2.", ".residual.3."),
        (".conv2.", ".residual.6."),
        (".conv_shortcut.", ".shortcut."),
    ):
        key = key.replace(diffusers_name, original)
    parts = key.split(".")
    if key.startswith("decoder.up_blocks."):
        level, kind = int(parts[2]), parts[3]
        per_level = num_res_blocks + 2
        if kind == "resnets":
            index, rest = level * per_level + int(parts[4]), parts[5:]
        else:
            index, rest = level * per_level + num_res_blocks + 1, parts[5:]
        return ".".join(["decoder", "upsamples", str(index), *rest])
    for diffusers_name, original in (
        ("quant_conv.", "conv1."),
        ("post_quant_conv.", "conv2."),
        ("encoder.conv_in.", "encoder.conv1."),
        ("decoder.conv_in.", "decoder.conv1."),
        ("encoder.norm_out.", "encoder.head.0."),
        ("encoder.conv_out.", "encoder.head.2."),
        ("decoder.norm_out.", "decoder.head.0."),
        ("decoder.conv_out.", "decoder.head.2."),
        ("encoder.mid_block.resnets.0.", "encoder.middle.0."),
        ("encoder.mid_block.attentions.0.", "encoder.middle.1."),
        ("encoder.mid_block.resnets.1.", "encoder.middle.2."),
        ("decoder.mid_block.resnets.0.", "decoder.middle.0."),
        ("decoder.mid_block.attentions.0.", "decoder.middle.1."),
        ("decoder.mid_block.resnets.1.", "decoder.middle.2."),
        ("encoder.down_blocks.", "encoder.downsamples."),
    ):
        if key.startswith(diffusers_name):
            return original + key[len(diffusers_name) :]
    return key


def test_original_checkpoint_conversion_agrees_with_diffusers():
    """Every tensor of the full Wan2.1 VAE, renamed by Diffusers' own
    single-file converter and by ours: the two must agree key for key."""
    single_file_utils = pytest.importorskip("diffusers.loaders.single_file_utils")
    with torch.device("meta"):
        keys = list(WanVAE().state_dict())
    original = {_original_name(key): torch.zeros(1) for key in keys}
    assert len(original) == len(keys)
    theirs = single_file_utils.convert_wan_vae_to_diffusers(dict(original))
    ours = convert_original_wan_vae_state_dict(original)
    assert set(theirs) == set(ours) == set(keys)


# --- fused norm + SiLU (GPU) ---------------------------------------------------

needs_gpu = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")


@needs_gpu
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("channels_last", [True, False])
@pytest.mark.parametrize(("c", "t"), [(96, 4), (20, 1), (384, 2)])
def test_fused_norm_silu_matches_unfused(dtype, channels_last, c, t):
    """The kernel against ``F.silu(norm(x))``, writing into a strided view the
    way it writes into a conv's staging buffer. They differ only by the order
    of the fp32 sum of squares: about an ulp in fp32, and in bf16 only where
    that tips a rounding."""
    from atom.diffusion.models.wan.norm_kernels import rms_norm_silu

    gen = torch.Generator().manual_seed(c + t)
    fmt = torch.channels_last_3d if channels_last else torch.contiguous_format
    x = (torch.randn(2, c, t, 9, 13, generator=gen) * 3).to("cuda", dtype)
    x = x.contiguous(memory_format=fmt)
    norm = wan_vae.WanRMSNorm(c, images=False).to("cuda", dtype)
    with torch.no_grad():
        norm.gamma.copy_(torch.rand(c, 1, 1, 1, generator=gen) + 0.5)
        expected = F.silu(norm(x))
    buffer = torch.full(
        (2, c, t + 2, 11, 15), float("nan"), device="cuda", dtype=dtype
    ).contiguous(memory_format=fmt)
    out = buffer[:, :, 2:, 1:-1, 1:-1]
    rms_norm_silu(x, norm.gamma, norm.scale, out=out)

    assert torch.isnan(buffer[:, :, :2]).all() and torch.isnan(buffer[..., 0]).all()
    ulp = 2**-7 if dtype == torch.bfloat16 else 2**-20
    torch.testing.assert_close(out, expected, rtol=ulp, atol=1e-6)
    if dtype == torch.bfloat16:
        assert (out != expected).float().mean() < 1e-3


@needs_gpu
@pytest.mark.parametrize("channels_last", [True, False])
def test_fused_vae_matches_unfused(monkeypatch, channels_last):
    """The whole VAE, fused against unfused, on GPU in fp32: the cached tail
    frames read back from the staging buffer -- first chunk, one-frame and
    four-frame chunks -- must be the ones the unfused path caches."""
    vae = tiny_vae(channels_last=channels_last).cuda()
    video = sample_video(frames=9).cuda()
    latents = torch.randn(1, Z_DIM, 3, 4, 6, generator=torch.Generator().manual_seed(5))
    fused = vae.encode_moments(video), vae.decode(latents.cuda())

    def unfused(norm, conv, x, cache):
        return wan_vae._cached_conv(conv, F.silu(norm(x)), cache)

    monkeypatch.setattr(wan_vae, "_norm_silu_conv", unfused)
    expected = vae.encode_moments(video), vae.decode(latents.cuda())
    for got, want in zip(fused, expected):
        torch.testing.assert_close(got, want, rtol=1e-4, atol=1e-5)
