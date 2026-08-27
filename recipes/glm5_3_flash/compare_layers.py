"""Localise where ATOM's GLM-5.3-Flash forward diverges from the reference.

Runs the transformers reference over the same prompt with per-decoder-layer
hooks, then compares each layer's output residual against ATOM's dump from
`ATOM_FWD_DUMP_DIR`. The first layer whose cosine similarity drops tells you
which component is wrong: layers 0-2 are KDA + dense MLP, layer 3 is the first
NoPE MLA + MoE layer.

Produce the ATOM side first:

  ATOM_FWD_DUMP_DIR=/out/atomdump \\
  ATOM_FWD_DUMP_BLOCK_CLASS=Glm5NextDecoderLayer \\
  ATOM_FWD_DUMP_LAYER_ATTR=layer_num \\
  ATOM_FWD_DUMP_LAYERS=0,1,2,3,4,5 python -m recipes.glm5_3_flash.atom_run ...
"""

from __future__ import annotations

import glob
import json
import os

import fp8_aiter_backend
import torch
import torch.nn.functional as F
from transformers import AutoModelForImageTextToText, AutoTokenizer

MP = "/models/GLM-5.3-Flash"
DUMP = os.environ.get("ATOM_FWD_DUMP_DIR", "/out/atomdump")
ORACLE = "/out/ref_top10.json"


def load_atom_layer(
    idx: int, want_tokens: int
) -> tuple[torch.Tensor, torch.Tensor] | None:
    """Return (input, output) for the call that actually ran the prompt.

    Set ATOM_FWD_DUMP_ONE_SHOT=0 when producing the dumps: with one-shot on,
    the only file written is the *warmup* forward over dummy tokens, and
    comparing against that is meaningless (it looks like a catastrophic
    divergence in layer 0). The real prefill is the call whose row count equals
    the prompt length.
    """
    hits = sorted(glob.glob(f"{DUMP}/layer{idx:02d}_*_rank0_call*.pt")) or sorted(
        glob.glob(f"{DUMP}/layer{idx:02d}_*_rank0.pt")
    )
    for path in hits:
        obj = torch.load(path, map_location="cpu", weights_only=False)
        out = obj["hidden"]
        if out.shape[0] != want_tokens:
            continue
        inp = obj.get("input")
        return (None if inp is None else inp.float()), out.float()
    return None


def main() -> None:
    fp8_aiter_backend.install()

    with open(ORACLE) as f:
        prompt_ids = json.load(f)["input_ids"]

    AutoTokenizer.from_pretrained(MP)
    model = AutoModelForImageTextToText.from_pretrained(
        MP, dtype=torch.bfloat16, device_map="auto", attn_implementation="sdpa"
    )
    model.eval()
    _unfp8_stray_bf16_linears(model)

    captured: dict[int, torch.Tensor] = {}

    def hook(idx):
        def fn(_mod, _inp, out):
            t = out[0] if isinstance(out, tuple) else out
            captured[idx] = t.detach().float().cpu()

        return fn

    layers = model.model.language_model.layers
    handles = [
        layers[i].register_forward_hook(hook(i))
        for i in [0, 3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43, 44]
    ]

    ids = torch.tensor([prompt_ids], device=model.device)
    with torch.no_grad():
        model(ids, use_cache=False)
    for h in handles:
        h.remove()

    print(
        f"\n{'layer':>5}  {'kind':<18} {'ref shape':<22} {'atom shape':<22} "
        f"{'cosine':>9} {'max|d|':>9} {'rel':>8}"
    )
    for i in sorted(captured):
        ref = captured[i][0]  # drop batch
        loaded = load_atom_layer(i, ref.shape[0])
        if loaded is None:
            print(
                f"{i:>5}  no dump with {ref.shape[0]} rows "
                f"(re-run with ATOM_FWD_DUMP_ONE_SHOT=0)"
            )
            continue
        _atom_in, atom = loaded
        a, r = atom.reshape(-1), ref.reshape(-1)
        if a.numel() != r.numel():
            print(
                f"{i:>5}  {'SHAPE MISMATCH':<18} {tuple(ref.shape)!s:<22} "
                f"{tuple(atom.shape)!s:<22}"
            )
            continue
        cos = F.cosine_similarity(a, r, dim=0).item()
        d = (a - r).abs()
        kind = "KDA" if i in (0, 1, 2, 4, 5, 6) else "MLA"
        kind += " + dense" if i < 3 else " + MoE"
        print(
            f"{i:>5}  {kind:<18} {tuple(ref.shape)!s:<22} {tuple(atom.shape)!s:<22} "
            f"{cos:>9.6f} {d.max():>9.4f} {(d.max() / r.abs().max()).item():>8.4f}"
        )


def _unfp8_stray_bf16_linears(model) -> None:
    from torch import nn

    n = 0
    for name, mod in list(model.named_modules()):
        if type(mod).__name__ != "FP8Linear":
            continue
        w = getattr(mod, "weight", None)
        if w is None or w.dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
            continue
        repl = nn.Linear(
            w.shape[1], w.shape[0], bias=mod.bias is not None, device="meta"
        )
        repl.weight = nn.Parameter(w.data, requires_grad=False)
        if mod.bias is not None:
            repl.bias = nn.Parameter(mod.bias.data, requires_grad=False)
        parent = model.get_submodule(name.rsplit(".", 1)[0])
        setattr(parent, name.rsplit(".", 1)[1], repl)
        n += 1
    print(f"[fix] un-FP8'd {n} BF16 linears", flush=True)


if __name__ == "__main__":
    main()
