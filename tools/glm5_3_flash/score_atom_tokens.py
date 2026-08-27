"""Score ATOM's generated tokens with the transformers reference (teacher forcing).

Greedy text from two implementations forks the moment one near-tie resolves the
other way, so matching completions is a weak test and diverging completions are
weak evidence of a bug. This asks the sharper question: run the reference over
`prompt + ATOM's own tokens` in a single forward and check, at every position,

  * what probability the reference assigns to the token ATOM picked, and
  * whether the reference would itself have picked that token (argmax match).

A correct port produces tokens the reference also finds highly likely. Rank-1
agreement well below 100% with high mean probability means the two disagree only
on near-ties -- which is what different kernels and FP8 activation quant do.

  python score_atom_tokens.py            # reads /out/atom_gen.json
"""

from __future__ import annotations

import json
import os

import fp8_aiter_backend
import torch
from transformers import AutoModelForImageTextToText, AutoTokenizer

MP = "/models/GLM-5.3-Flash"
ATOM_OUT = os.environ.get("GLM53_SCORE_INPUT", "/out/atom_gen.json")


def main() -> None:
    fp8_aiter_backend.install()

    with open(ATOM_OUT) as f:
        gen = json.load(f)
    prompt_ids = gen["prompt_ids"]
    out_ids = gen["output_ids"]
    atom_lp = gen.get("logprobs") or []
    print(f"prompt {len(prompt_ids)} tokens, ATOM generated {len(out_ids)}", flush=True)

    tok = AutoTokenizer.from_pretrained(MP)
    model = AutoModelForImageTextToText.from_pretrained(
        MP, dtype=torch.bfloat16, device_map="auto", attn_implementation="sdpa"
    )
    model.eval()
    _unfp8_stray_bf16_linears(model)

    stats = _instrument_swiglu_clamp(model)

    ids = torch.tensor([prompt_ids + out_ids], device=model.device)
    with torch.no_grad():
        logits = model(ids, use_cache=False).logits[0].float()

    if stats["n"]:
        print(
            f"\n=== GLM's clamped SwiGLU in the MoE experts ===\n"
            f"  calls              : {stats['n']}\n"
            f"  gate elems clamped : {100.0 * stats['gate'] / stats['tot']:.3f}%\n"
            f"  up   elems clamped : {100.0 * stats['up'] / stats['tot']:.3f}%\n"
            f"  max |gate|         : {stats['gmax']:.1f}   max |up|: {stats['umax']:.1f}\n"
            f"  (limit = {stats['limit']}; ATOM's default 'standard'/CK MoE path does not "
            f"plumb swiglu_limit, so any nonzero figure here is dropped clamping)",
            flush=True,
        )

    # Position p predicts token p+1; ATOM's first generated token is at index
    # len(prompt_ids), so it is predicted by logits[len(prompt_ids) - 1].
    start = len(prompt_ids) - 1
    logprobs = torch.log_softmax(logits[start : start + len(out_ids)], dim=-1)
    picked = torch.tensor(out_ids, device=logprobs.device)
    ref_lp = logprobs[torch.arange(len(out_ids), device=logprobs.device), picked]
    ref_argmax = logprobs.argmax(-1)
    agree = ref_argmax == picked

    print(
        "\n  idx  token                    ref_logprob  ref_p    ref_top1  match",
        flush=True,
    )
    for i in range(min(len(out_ids), 24)):
        top1 = int(ref_argmax[i])
        print(
            f"  {i:>3}  {tok.decode([out_ids[i]])!r:<24} {ref_lp[i]:>10.4f} "
            f"{ref_lp[i].exp():>7.4f}  {tok.decode([top1])!r:<10} {'y' if agree[i] else 'n'}",
            flush=True,
        )

    p = ref_lp.exp()
    print(
        f"\n=== reference scoring ATOM's {len(out_ids)} tokens ===\n"
        f"  rank-1 agreement : {int(agree.sum())}/{len(out_ids)} "
        f"({100.0 * agree.float().mean():.1f}%)\n"
        f"  mean probability : {p.mean():.4f}\n"
        f"  median           : {p.median():.4f}\n"
        f"  min              : {p.min():.4f} (at index {int(p.argmin())})\n"
        f"  below 0.05       : {int((p < 0.05).sum())} tokens",
        flush=True,
    )
    if atom_lp:
        a = torch.tensor(atom_lp[: len(out_ids)]).exp()
        print(
            f"  ATOM's own mean p: {a.mean():.4f}  (same tokens, ATOM's numerics)",
            flush=True,
        )


def _instrument_swiglu_clamp(model) -> dict:
    """Measure how often GLM's expert SwiGLU clamp actually binds.

    The clamp is part of the trained function, so if it fires at all then a MoE
    kernel that ignores `swiglu_limit` is computing something else.
    """
    stats = {
        "n": 0,
        "gate": 0,
        "up": 0,
        "tot": 0,
        "gmax": 0.0,
        "umax": 0.0,
        "limit": None,
    }
    patched = 0
    for mod in model.modules():
        # `@use_experts_implementation` rewrites the class, so match on shape
        # rather than on the name in modeling_glm5_next.py.
        if not (hasattr(mod, "_apply_gate") and hasattr(mod, "swiglu_limit")):
            continue
        limit = float(mod.swiglu_limit)
        stats["limit"] = limit
        original = mod._apply_gate

        def wrapped(gate_up, _orig=original, _limit=limit):
            g, u = gate_up.chunk(2, dim=-1)
            stats["n"] += 1
            stats["tot"] += g.numel()
            stats["gate"] += int((g > _limit).sum())
            stats["up"] += int((u.abs() > _limit).sum())
            stats["gmax"] = max(stats["gmax"], float(g.abs().max()))
            stats["umax"] = max(stats["umax"], float(u.abs().max()))
            return _orig(gate_up)

        mod._apply_gate = wrapped
        patched += 1
    print(f"[instrument] wrapped {patched} expert blocks", flush=True)
    return stats


def _unfp8_stray_bf16_linears(model) -> None:
    """See ref_run.py: transformers wraps the BF16 KDA forget gate in FP8Linear."""
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
    if not os.path.exists(ATOM_OUT):
        raise SystemExit(f"{ATOM_OUT} not found -- run atom_run.py first")
    main()
