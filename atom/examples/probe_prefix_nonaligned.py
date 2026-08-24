# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.
"""Non-aligned prefix-cache correctness probe for DeepSeek-V4.

The concern (config.py per-request-SWA comment): on a prefix-cache hit the
SWA sliding window of a freshly-forwarded token can reach BACK into an already
-cached block. If the prompt length is NOT a multiple of block_size, the hit
boundary lands in the MIDDLE of a SWA window, so the window's prior portion is
read from the per-request SWA ring of a brand-new slot.

Concretely with block_size=128, SWA window=128:
  prompt length L = 128 + r  (r in (0,128), e.g. r=22 -> L=150)
  -> can_allocate hits block 0 (128 tokens), excludes the last block
  -> the new tokens [128..L) are re-forwarded
  -> the LAST new token at pos L-1 has SWA window [L-128 .. L-1]
     which spans (128-r) tokens INSIDE the cached block 0.

This probe runs the SAME prompt twice:
  COLD: fresh (no hit) - reference
  HIT : after seeding the cache - exercises the non-aligned hit path
and compares greedy output token-by-token. If the non-aligned SWA read is
wrong, the HIT continuation diverges from COLD.

We sweep several remainders r so the hit boundary lands at different offsets
within the SWA window.
"""

import argparse

from atom import SamplingParams
from atom.model_engine.arg_utils import EngineArgs
from transformers import AutoTokenizer

parser = argparse.ArgumentParser(
    formatter_class=argparse.RawTextHelpFormatter,
    description="Non-aligned prefix-cache correctness probe",
)
EngineArgs.add_cli_args(parser)
parser.add_argument("--max-tokens", type=int, default=48)


# Filler text; repeated and then trimmed to an exact token count.
FILLER = (
    "The mitochondrion is a double-membrane-bound organelle found in most "
    "eukaryotic cells. It generates most of the cell's supply of adenosine "
    "triphosphate, used as a source of chemical energy. Mitochondria have "
    "their own genetic material and machinery to manufacture RNAs and "
    "proteins. The number per cell varies widely across tissues and species. "
)


def make_prompt_of_len(tokenizer, target_len: int) -> list[int]:
    """Return an exact-length token id list (no chat template, raw tokens)."""
    ids = tokenizer(FILLER * 64, add_special_tokens=False)["input_ids"]
    assert len(ids) >= target_len, f"filler too short: {len(ids)} < {target_len}"
    return ids[:target_len]


def main():
    args = parser.parse_args()
    args.cudagraph_capture_sizes = "[1, 2, 4, 8]"
    engine_args = EngineArgs.from_cli_args(args)
    llm = engine_args.create_engine()
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    block_size = 128
    # Use a LONG prompt (many blocks) so that after the B' fix re-forwards the
    # last block, there is still NET cache hit (~base_blocks-1 blocks). This
    # tests BOTH: (a) compressed-KV hit is retained, (b) the non-aligned SWA
    # window at the tail reaches back into a cached block and must read correctly.
    # base_blocks tokens of cached prefix + r tail tokens.
    base_blocks = 10  # 10*128 = 1280 cached tokens
    # Remainders so the hit boundary lands at various offsets inside the
    # 128-token SWA window of the final new token.
    remainders = [4, 22, 64, 100, 127]
    sp = SamplingParams(temperature=0.0, max_tokens=args.max_tokens)

    print("\n" + "=" * 72)
    print(
        f"NON-ALIGNED PREFIX-CACHE PROBE (block_size=128, SWA window=128, "
        f"base_blocks={base_blocks})"
    )
    print("=" * 72)

    all_ok = True
    for idx, r in enumerate(remainders):
        L = block_size * base_blocks + r  # base_blocks cached + r tail tokens
        # CRITICAL: give each r a UNIQUE leading token sequence so its block
        # hashes never collide with another r's prompt (otherwise the "seed"
        # run could itself hit a prior r's cache and stop being a no-hit
        # reference). The marker is a few distinct words per idx; we then trim
        # to exactly L tokens so the hit boundary math holds.
        marker = f"UNIQUEPREFIX{idx}ALPHA{idx}BETA{idx}GAMMA{idx}. "
        base_ids = tokenizer(marker + FILLER * 256, add_special_tokens=False)[
            "input_ids"
        ]
        prompt_ids = base_ids[:L]
        prompt_text = tokenizer.decode(prompt_ids)

        # SEED (no-hit reference): first time this exact prompt is seen -> no
        # prior cache for its blocks -> pure cold forward. Then HIT: re-send the
        # SAME prompt -> block 0 hits, tokens [128..L) re-forward -> the tail
        # token's SWA window reaches back into the cached block (non-aligned).
        # seed and hit are the SAME prompt + greedy, so they MUST match unless
        # the non-aligned SWA read is wrong.
        seed_out = llm.generate([prompt_text], sp)[0]["text"]
        hit_out = llm.generate([prompt_text], sp)[0]["text"]

        match = seed_out == hit_out
        all_ok = all_ok and match
        status = "OK " if match else "DIVERGE"
        print(f"\n[r={r:>3} L={L}] hit-boundary offset-in-window={block_size - r}")
        print(f"  {status}  seed==hit: {match}")
        if not match:
            print(f"  seed: {seed_out[:160]!r}")
            print(f"  hit : {hit_out[:160]!r}")
        else:
            print(f"  out : {hit_out[:120]!r}")

    print("\n" + "=" * 72)
    print(
        f"RESULT: {'ALL MATCH (non-aligned SWA correct)' if all_ok else 'DIVERGENCE FOUND'}"
    )
    print("=" * 72)


if __name__ == "__main__":
    main()
