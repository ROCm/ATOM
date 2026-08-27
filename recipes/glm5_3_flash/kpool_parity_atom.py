"""Parity check for ATOM's `atom/model_ops/kpool_indexer.py` vs transformers.

Loads the real GLM-5.3-Flash layer-3 indexer weights and requires the ATOM op to
select exactly the same token indices as `Glm5NextTextIndexer`, including under
left padding (which exercises the `first_key` anchoring).

The op module is loaded by path so this does not drag in `atom.model_ops.__init__`
and its aiter dependencies.
"""

from __future__ import annotations

import importlib.util
import json

import torch
import torch.nn.functional as F
from safetensors import safe_open

MP = "/models/GLM-5.3-Flash"
LAYER = 3
PREFIX = f"model.language_model.layers.{LAYER}.self_attn.indexer."
OP_PATH = "/atom/atom/model_ops/kpool_indexer.py"


def load_op():
    spec = importlib.util.spec_from_file_location("kpool_indexer", OP_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def load_weights(device: str) -> dict[str, torch.Tensor]:
    with open(f"{MP}/model.safetensors.index.json") as f:
        wm = json.load(f)["weight_map"]
    by_shard: dict[str, list[str]] = {}
    for k in (k for k in wm if k.startswith(PREFIX)):
        by_shard.setdefault(wm[k], []).append(k)
    out: dict[str, torch.Tensor] = {}
    for shard, ks in by_shard.items():
        with safe_open(f"{MP}/{shard}", framework="pt", device=device) as f:
            for k in ks:
                out[k[len(PREFIX) :]] = f.get_tensor(k)
    return out


def main() -> None:
    from transformers import AutoConfig
    from transformers.models.glm5_next.modeling_glm5_next import Glm5NextTextIndexer

    dev = "cuda"
    torch.manual_seed(0)
    op = load_op()
    cfg = AutoConfig.from_pretrained(MP).text_config
    w = load_weights(dev)

    ref = Glm5NextTextIndexer(cfg, LAYER).to(dev).to(torch.bfloat16).eval()
    ref.load_state_dict(w, strict=True)

    scale = cfg.index_head_dim**-0.5
    all_ok = True

    # (seq_len, n_left_pad) -- left padding exercises `first_key` anchoring.
    cases = [(7, 0), (64, 0), (300, 0), (2048, 0), (3000, 0), (64, 5), (300, 17)]
    for seq, pad in cases:
        hidden = (
            torch.randn(1, seq, cfg.hidden_size, device=dev, dtype=torch.bfloat16)
            * 0.05
        )
        q_resid = (
            torch.randn(1, seq, cfg.q_lora_rank, device=dev, dtype=torch.bfloat16)
            * 0.05
        )
        mask = torch.ones(1, seq, dtype=torch.bool, device=dev)
        if pad:
            mask[:, :pad] = False

        with torch.no_grad():
            expected = ref(
                hidden_states=hidden,
                q_resid=q_resid,
                attention_mask=mask,
                past_key_values=None,
            )

            q = F.linear(q_resid, w["wq_b.weight"]).view(1, seq, -1, cfg.index_head_dim)
            keys = F.layer_norm(
                F.linear(hidden, w["wk.weight"]),
                (cfg.index_head_dim,),
                w["k_norm.weight"],
                w["k_norm.bias"],
                eps=1e-6,
            )
            gate = F.linear(hidden, w["index_kpool_compress_gate"])
            head_w = F.linear(hidden, w["weights_proj.weight"])

            pos = torch.arange(seq, device=dev)
            visible = (pos[None, None, :] <= pos[None, :, None]) & mask[:, None, :]

            got = op.kpool_topk_indices(
                q=q,
                keys=keys,
                gate_scores=gate,
                head_weights=head_w,
                valid_keys=mask,
                visible=visible,
                ape=w["index_kpool_compress_ape"],
                index_topk=cfg.index_topk,
                kpool=cfg.index_kpool,
                softmax_scale=scale,
                always_select_tail=cfg.index_kpool_always_select_tail,
            )

        ok = True
        for t in range(seq):
            a = set(expected[0, t].tolist()) - {-1}
            b = set(got[0, t].tolist()) - {-1}
            if a != b:
                ok = False
                print(
                    f"  seq={seq} pad={pad} row={t}: ref={len(a)} atom={len(b)} "
                    f"missing={sorted(a - b)[:6]} extra={sorted(b - a)[:6]}"
                )
                break
        all_ok &= ok
        print(
            f"  seq={seq:>5} pad={pad:>3}: ref={tuple(expected.shape)} "
            f"atom={tuple(got.shape)} -> {'MATCH' if ok else 'MISMATCH'}"
        )

    print("\nRESULT:", "ALL MATCH" if all_ok else "MISMATCH")
    raise SystemExit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
