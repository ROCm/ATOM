# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.
"""Needle-in-a-haystack retrieval check for sparse-indexer models, with a CONTROL.

Motivation: a sparse indexer that selects the wrong keys still produces fluent
text, so "the answer looks right" is not a measurement. Two things make this one:

1. **A control.** Each depth is run twice with a different secret in an
   otherwise byte-identical prompt. A real retrieval tracks the control; a
   model that guessed, or that inferred the answer from the question, does not.
   Without the control a needle test passes for the wrong reason.
2. **A proof the sparse path ran at all.** Below ``index_topk`` candidates,
   top-k selects *every* pool, dense attention runs, and the indexer's
   selection is computed but never used -- so a short prompt says nothing about
   selection. This script reads ``index_topk`` from the model config and FAILS
   if any prompt came in under it, rather than reporting a green pass on a run
   that never exercised the path.

Usage (GLM-5.3-Flash, whose pooled indexer is what this was written for):

    python scripts/run_longctx_needle.py \
        --model /data/amd_int/models/GLM-5.3-Flash \
        --kv-cache-dtype fp8 -tp 8 --max-model-len 8192 \
        --gpu-memory-utilization 0.85 --no-enable_prefix_caching

Set ``ATOM_GLM5_KPOOL=0`` to run the same check against the token-granular
path; it refuses a context past ``index_topk``, which is the point of the flag.
"""

import argparse
import sys
from pathlib import Path

from transformers import AutoTokenizer

from atom import SamplingParams
from atom.config import get_hf_config
from atom.model_engine.arg_utils import EngineArgs
from atom.utils.arg_parser import FlexibleArgumentParser

# Digit-disjoint so a cross-contaminated answer is unambiguous rather than a
# partial-digit coincidence.
SECRET_A = "48213"
SECRET_B = "90577"

# Digit-free, so the only numerals anywhere in the prompt are the secret and the
# building number -- which is what makes the digits-only match below safe.
FILLER = "The garden was quiet and the hedges were neatly trimmed. "


def check_imported_atom_is_this_checkout() -> str | None:
    """Refuse to measure a checkout other than the one this script lives in.

    ``python3 scripts/run_longctx_needle.py`` puts the SCRIPT's directory on
    sys.path, not the cwd, so `import atom` silently resolves to whatever is
    pip-installed. On a machine with more than one ATOM tree that is how you
    spend an hour measuring code you did not change -- and the run looks
    perfectly healthy while doing it. Returns an error message, or None if the
    imported package is the one next to this script.
    """
    import atom

    repo_pkg = (Path(__file__).resolve().parents[1] / "atom" / "__init__.py").resolve()
    used = Path(atom.__file__).resolve()
    if not repo_pkg.exists() or used == repo_pkg:
        return None
    return (
        f"FAIL: `import atom` resolved to {used}, but this script lives beside "
        f"{repo_pkg}. Those are different checkouts, so the run would measure "
        f"code other than this one. Re-run as:\n"
        f"    PYTHONPATH={repo_pkg.parents[1]} python3 {Path(__file__).name} ..."
    )


def build_prompt(secret: str, depth: float, n_units: int) -> str:
    at = int(n_units * depth)
    return (
        "Read the passage and answer the question at the end.\n\n"
        + FILLER * at
        + f"The secret access code for building seven is {secret}. "
        + FILLER * (n_units - at)
        + "\n\nQuestion: what is the secret access code for building seven?\nAnswer:"
    )


def found(secret: str, text: str) -> bool:
    """Match on digits alone, so "48,213" and "48 213" still count."""
    return secret in "".join(c for c in text if c.isdigit())


def main() -> int:
    parser = FlexibleArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    EngineArgs.add_cli_args(parser)
    parser.add_argument(
        "--filler-tokens",
        type=int,
        default=6000,
        help="approximate prompt length; must exceed the model's index_topk",
    )
    parser.add_argument(
        "--depths",
        type=str,
        default="0.1,0.5,0.9",
        help="comma-separated needle positions in the passage, 0..1",
    )
    parser.add_argument("--max-tokens", type=int, default=48)
    args = parser.parse_args()

    wrong_tree = check_imported_atom_is_this_checkout()
    if wrong_tree:
        print(wrong_tree)
        return 1

    depths = [float(d) for d in args.depths.split(",") if d.strip()]
    if not depths:
        print("FAIL: --depths is empty, so nothing would be compared")
        return 1

    # The engine's own config loader, not AutoConfig: transformers has no
    # config class for architectures like glm5_next and raises outright. Going
    # through the same function the engine does also means the threshold here
    # cannot drift from the one the indexer actually applies -- and a wrong
    # threshold would turn the context check below into a rubber stamp.
    hf = get_hf_config(args.model, trust_remote_code=True)
    index_topk = int(getattr(hf, "index_topk", 0) or 0)
    index_kpool = int(getattr(hf, "index_kpool", 1) or 1)
    if index_topk <= 0:
        print(
            f"FAIL: {args.model} has no index_topk in its config, so this "
            "script cannot tell whether sparse selection ran"
        )
        return 1

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    per_unit = len(tok(FILLER)["input_ids"])
    n_units = max(1, args.filler_tokens // per_unit)

    prompts, cases = [], []
    for depth in depths:
        cases.append((depth, len(prompts)))
        prompts.append(build_prompt(SECRET_A, depth, n_units))
        prompts.append(build_prompt(SECRET_B, depth, n_units))

    import atom

    print(
        f"[longctx-needle] model={args.model} index_topk={index_topk} "
        f"index_kpool={index_kpool} depths={depths} cases={len(cases)} "
        f"atom={atom.__file__}"
    )

    llm = EngineArgs.from_cli_args(args).create_engine()
    outs = llm.generate(
        prompts, SamplingParams(temperature=0.0, max_tokens=args.max_tokens)
    )

    # The engine's own count, not the tokenizer's estimate: the template and any
    # special tokens are the engine's business, and this number is the one the
    # indexer actually saw.
    ctx = [int(o["num_tokens_input"]) for o in outs]

    ok = True
    for depth, i in cases:
        a, b = outs[i]["text"], outs[i + 1]["text"]
        hit_a, hit_b = found(SECRET_A, a), found(SECRET_B, b)
        cross = found(SECRET_B, a) or found(SECRET_A, b)
        passed = hit_a and hit_b and not cross
        ok = ok and passed
        print(f"--- depth {depth} ctx={ctx[i]}/{ctx[i + 1]} ---")
        print(f"  A wants {SECRET_A}: {a[:200]!r}")
        print(f"  B wants {SECRET_B}: {b[:200]!r}")
        print(
            f"  hit_A={hit_a} hit_B={hit_b} cross_contaminated={cross} -> {'ok' if passed else 'FAILED'}"
        )

    # A run whose prompts never crossed index_topk proves nothing about sparse
    # selection, so it fails here instead of passing quietly on dense attention.
    min_ctx = min(ctx)
    exercised = min_ctx > index_topk
    print(f"compared {len(cases)} depths x 2 variants = {len(prompts)} generations")
    print(
        f"CONTEXT: min prompt {min_ctx} vs index_topk {index_topk} -> "
        + (
            "sparse selection exercised on every prompt"
            if exercised
            else "TOO SHORT, dense attention ran and nothing was measured; "
            "raise --filler-tokens"
        )
    )
    ok = ok and exercised
    print("LONGCTX-NEEDLE", "PASSED" if ok else "FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
