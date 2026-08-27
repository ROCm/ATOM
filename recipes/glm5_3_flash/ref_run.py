"""Reference GLM-5.3-Flash run under transformers on 4x MI355X.

Purpose is twofold:
  1. Prove the checkpoint loads and generates coherently on this machine.
  2. Dump logits for a fixed prompt so the ATOM port can be diffed against it.
"""

import json
import os
import time

import torch
from transformers import AutoConfig, AutoModelForImageTextToText, AutoTokenizer

MP = "/models/GLM-5.3-Flash"
OUT = "/out"
PROMPT = "Give three reasons why the sky appears blue."
# "aiter"  -> ATOM's gemm_a8w8_blockscale (fast, FP8 activations as trained)
# "torch"  -> pure-torch dequant (slow, BF16 activations; useful as a cross-check)
FP8_BACKEND = os.environ.get("GLM53_FP8_BACKEND", "aiter")
MAX_NEW_TOKENS = int(os.environ.get("GLM53_MAX_NEW_TOKENS", "32"))


def install_fp8_backend() -> None:
    """Replace the hub FP8 kernel, which does not compile on gfx950."""
    if FP8_BACKEND == "aiter":
        import fp8_aiter_backend

        fp8_aiter_backend.install(verify=os.environ.get("GLM53_FP8_VERIFY") == "1")
    else:
        import fp8_torch_fallback

        fp8_torch_fallback.install()


def unfp8_stray_bf16_linears(model) -> None:
    """Undo FP8 wrapping of modules whose checkpoint weights are actually BF16.

    The checkpoint's `modules_to_not_convert` names the KDA forget gate as
    `model.layers.N.self_attn.f_a_proj`, but transformers' `glm5_next` conversion
    mapping renames those tensors to `self_attn.forget_gate.f_a_proj` before the FP8
    quantizer runs. The exclusion therefore misses, and f_a_proj / f_b_proj get
    wrapped in FP8Linear while still holding BF16 weights plus a freshly-initialised
    `weight_scale_inv` -- which is both numerically wrong and what makes the Triton
    FP8 kernel assert on gfx950.

    Swap any FP8Linear whose weight is not actually FP8 back to a plain nn.Linear.
    """
    from torch import nn

    fixed = []
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
        fixed.append(name)
    print(f"[fix] un-FP8'd {len(fixed)} BF16 linears (e.g. {fixed[:2]})", flush=True)


def main() -> None:
    # The hub FP8 Triton kernel compiles to an LLVM assert on gfx950; swap in a
    # working backend before anything can trigger a lazy load of the real one.
    install_fp8_backend()

    t0 = time.time()
    cfg = AutoConfig.from_pretrained(MP)
    print(
        f"config: {type(cfg).__name__} layers={cfg.text_config.num_hidden_layers}",
        flush=True,
    )

    tok = AutoTokenizer.from_pretrained(MP)
    print(f"tokenizer: {type(tok).__name__} vocab={len(tok)}", flush=True)

    print("loading model (this reads ~306 GiB)...", flush=True)
    model = AutoModelForImageTextToText.from_pretrained(
        MP,
        dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="sdpa",
    )
    model.eval()
    print(f"loaded in {time.time() - t0:.1f}s", flush=True)
    print(
        "device map sample:",
        list(getattr(model, "hf_device_map", {}).items())[:5],
        flush=True,
    )

    unfp8_stray_bf16_linears(model)

    # The checkpoint's `modules_to_not_convert` is written with `model.layers.N.`
    # / `visual.` prefixes while the actual keys are `model.language_model.layers.N.`
    # / `model.visual.`, so confirm what the FP8 quantizer actually wrapped before
    # trusting any of these numbers.
    print(
        "\n=== quantization sanity (layer 0 = KDA, layer 3 = MLA/DSA) ===", flush=True
    )
    for name in [
        "model.language_model.layers.0.self_attn.q_proj",
        "model.language_model.layers.0.self_attn.forget_gate.f_a_proj",
        "model.language_model.layers.0.self_attn.forget_gate.f_b_proj",
        "model.language_model.layers.0.mlp.gate_proj",
        "model.language_model.layers.3.self_attn.q_a_proj",
        "model.language_model.layers.3.self_attn.kv_b_proj",
    ]:
        mod = model.get_submodule(name)
        w = getattr(mod, "weight", None)
        scale = getattr(mod, "weight_scale_inv", None)
        print(
            f"  {name.split('layers.')[-1]:<45} {type(mod).__name__:<12}"
            f" w={None if w is None else w.dtype}"
            f" scale={None if scale is None else tuple(scale.shape)}",
            flush=True,
        )

    msgs = [{"role": "user", "content": PROMPT}]
    enc = tok.apply_chat_template(
        msgs,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    )
    ids = enc["input_ids"].to(model.device)
    print(f"\nprompt tokens: {tuple(ids.shape)}", flush=True)

    # Single forward first: cheap, and it is what the ATOM port gets compared against.
    with torch.no_grad():
        out = model(ids, use_cache=False)
    logits = out.logits[0, -1].float().cpu()
    top = torch.topk(logits, 10)
    print("\n=== next-token top-10 ===", flush=True)
    for score, idx in zip(top.values.tolist(), top.indices.tolist()):
        print(f"  {idx:>7}  {score:9.4f}  {tok.decode([idx])!r}", flush=True)

    os.makedirs(OUT, exist_ok=True)
    torch.save(
        {"input_ids": ids.cpu(), "last_logits": logits, "prompt": PROMPT},
        f"{OUT}/ref_logits.pt",
    )
    with open(f"{OUT}/ref_top10.json", "w") as f:
        json.dump(
            {
                "prompt": PROMPT,
                "input_ids": ids[0].tolist(),
                "top10_ids": top.indices.tolist(),
                "top10_logits": top.values.tolist(),
            },
            f,
            indent=1,
        )

    print("\n=== generation ===", flush=True)
    t1 = time.time()
    with torch.no_grad():
        gen = model.generate(ids, max_new_tokens=MAX_NEW_TOKENS, do_sample=False)
    text = tok.decode(gen[0, ids.shape[1] :], skip_special_tokens=False)
    dt = time.time() - t1
    ntok = gen.shape[1] - ids.shape[1]
    print(text, flush=True)
    print(f"\n[{ntok} tokens in {dt:.1f}s = {ntok / dt:.2f} tok/s]", flush=True)


if __name__ == "__main__":
    main()
