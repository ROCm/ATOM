"""Parity check: ATOM's GLM-5.3-Flash vision tower vs transformers, real weights.

Loads `model.visual.*` from the checkpoint into both `Glm5NextVisionTower` (ATOM)
and `Glm5NextVisionModel` (transformers) and compares the merged output — the
`[n_tokens, 4096]` tensor that gets scattered onto image placeholders — over a
few image grids.

The ATOM module is loaded by path so this does not drag in `atom.models` and its
engine dependencies.

  python vision_parity.py
"""

from __future__ import annotations

import importlib.util
import json

import torch
import torch.nn.functional as F
from safetensors import safe_open

MP = "/models/GLM-5.3-Flash"
PREFIX = "model.visual."
ATOM_VL = "/atom/atom/models/glm5_next_vl.py"


def load_atom_module():
    spec = importlib.util.spec_from_file_location("glm5_next_vl", ATOM_VL)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def load_vision_weights(device: str) -> dict[str, torch.Tensor]:
    with open(f"{MP}/model.safetensors.index.json") as f:
        wm = json.load(f)["weight_map"]
    by_shard: dict[str, list[str]] = {}
    for k in (k for k in wm if k.startswith(PREFIX)):
        by_shard.setdefault(wm[k], []).append(k)
    out: dict[str, torch.Tensor] = {}
    for shard, keys in by_shard.items():
        with safe_open(f"{MP}/{shard}", framework="pt", device=device) as f:
            for k in keys:
                out[k[len(PREFIX) :]] = f.get_tensor(k)
    return out


def _fuse_gate_up(weights: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Concatenate `gate_proj` + `up_proj` into `gate_up_proj`.

    ATOM fuses these (see Glm5NextVisionMLP); the checkpoint stores them apart,
    and at serving time the model's `packed_modules_mapping` does the same join.
    """
    out: dict[str, torch.Tensor] = {}
    for key, value in weights.items():
        if ".gate_proj." in key:
            up = weights[key.replace(".gate_proj.", ".up_proj.")]
            out[key.replace(".gate_proj.", ".gate_up_proj.")] = torch.cat(
                [value, up], 0
            )
        elif ".up_proj." in key:
            continue
        else:
            out[key] = value
    return out


def main() -> None:
    from transformers import AutoConfig
    from transformers.models.glm5_next.modeling_glm5_next import Glm5NextVisionModel

    dev = "cuda"
    torch.manual_seed(0)

    vcfg = AutoConfig.from_pretrained(MP).vision_config
    weights = load_vision_weights(dev)
    print(f"vision tensors: {len(weights)}")

    atom_mod = load_atom_module()

    def build(dtype):
        r = Glm5NextVisionModel(vcfg).to(dev).to(dtype).eval()
        miss, unexp = r.load_state_dict(
            {k: v.to(dtype) for k, v in weights.items()}, strict=False
        )
        miss = [m for m in miss if "inv_freq" not in m]
        a = atom_mod.Glm5NextVisionTower(vcfg).to(dev).to(dtype).eval()
        m2, u2 = a.load_state_dict(
            {k: v.to(dtype) for k, v in _fuse_gate_up(weights).items()}, strict=False
        )
        m2 = [m for m in m2 if "inv_freq" not in m]
        if miss or unexp or m2 or u2:
            print(f"reference  load: missing={miss} unexpected={list(unexp)}")
            print(f"atom tower load: missing={m2} unexpected={list(u2)}")
            raise SystemExit("parameter names do not match the checkpoint")
        return r, a

    print("parameter names match on both sides")

    patch_dim = vcfg.in_channels * vcfg.temporal_patch_size * vcfg.patch_size**2
    # (t, h, w); h and w must be multiples of spatial_merge_size.
    grids = [(1, 2, 2), (1, 4, 4), (1, 8, 12), (2, 4, 4)]
    ok = True

    # Two separate questions, so two separate runs.
    #
    # fp32 + SDPA: is the maths right? Same kernel as the reference and no BF16
    # rounding, so this is the correctness assertion.
    #
    # bf16 + aiter: what does the serving path cost? Reported, not asserted. Two
    # sources of rounding live here -- the packed-varlen kernel versus SDPA, and
    # ATOM's single fused `[gate|up]` GEMM versus the reference's two -- and both
    # compound over 24 blocks. The inputs below are random patches, far worse
    # conditioned than real normalised image input, so this is a pessimistic read.
    for dtype, attn, threshold in (
        (torch.float32, "torch", 0.999999),
        (torch.bfloat16, "aiter", None),
    ):
        ref, mine = build(dtype)
        atom_mod._USE_TORCH_ATTN = attn == "torch"
        gate = f"pass if cos >= {threshold}" if threshold else "informational"
        print(
            f"\n--- {str(dtype).replace('torch.', '')} + {attn} attention ({gate}) ---"
        )
        for t, h, w in grids:
            n_patches = t * h * w
            torch.manual_seed(0)
            pixel_values = (
                torch.randn(n_patches, patch_dim, device=dev, dtype=dtype) * 0.5
            )
            grid_thw = torch.tensor([[t, h, w]], device=dev)

            with torch.no_grad():
                expected = ref(pixel_values, grid_thw=grid_thw).pooler_output
                got = mine(pixel_values, grid_thw)

            if expected.shape != got.shape:
                print(
                    f"  ({t},{h},{w}): SHAPE {tuple(expected.shape)} vs {tuple(got.shape)}"
                )
                ok = False
                continue
            cos = F.cosine_similarity(
                got.float().flatten(), expected.float().flatten(), dim=0
            ).item()
            d = (got.float() - expected.float()).abs()
            good = threshold is None or cos >= threshold
            ok &= good
            print(
                f"  grid=({t},{h},{w})".ljust(20)
                + f"patches={n_patches:>4} out={tuple(got.shape)}  "
                f"cos={cos:.6f} max|d|={d.max():.5f} "
                + ("" if threshold is None else ("OK" if good else "MISMATCH"))
            )

    print("\nRESULT:", "ALL MATCH" if ok else "MISMATCH")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
