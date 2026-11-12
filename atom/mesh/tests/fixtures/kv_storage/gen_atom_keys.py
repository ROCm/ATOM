"""Generate atom_keys.jsonl, the vectors ATOMesh's `AtomChunk` key rule is checked against.

    python gen_atom_keys.py [--atom-root DIR] [--out FILE]

Every case is computed with `atom/kv_transfer/offload/mooncake_store/keys.py` of the
ATOM checkout at `--atom-root` (default: this one). Token `i` of a case is
`(mul * i + add) % vocab`. `keys` are rank `rank`'s keys of every full chunk, and
`chain_check` is the digest `/kv_event_sources` reports for the chunk size.
"""

import argparse
import json
import sys
from pathlib import Path

NAMESPACE = "atomkv1-b3fd64e0ee0549784107510e"
# name, (len, mul, add, vocab), chunk_tokens, world, rank
CASES = [
    ("two_chunks_and_a_tail", (40, 1, 0, 151936), 16, 1, 0),
    ("shorter_than_a_chunk", (15, 1, 0, 151936), 16, 1, 0),
    ("rank_5_of_8", (1000, 2654435761, 7, 151936), 256, 8, 5),
    ("rank_0_of_4", (4096, 40503, 11, 129280), 64, 4, 0),
    ("rank_3_of_4", (4096, 40503, 11, 129280), 64, 4, 3),
]


def main() -> None:
    script = Path(__file__).resolve()
    parser = argparse.ArgumentParser()
    parser.add_argument("--atom-root", type=Path, default=script.parents[5])
    parser.add_argument("--out", type=Path, default=script.with_name("atom_keys.jsonl"))
    args = parser.parse_args()
    sys.path.insert(0, str(args.atom_root))
    from atom.kv_transfer.offload.mooncake_store import keys

    lines = []
    for name, (length, mul, add, vocab), chunk, world, rank in CASES:
        tokens = [(mul * i + add) % vocab for i in range(length)]
        chain = keys.chunk_hash_chain(tokens, chunk)
        count = len(chain) // keys.DIGEST_BYTES
        case = {"name": name, "len": length, "mul": mul, "add": add, "vocab": vocab}
        case |= {"chunk_tokens": chunk, "namespace": NAMESPACE}
        case |= {"world": world, "rank": rank}
        case["keys"] = keys.chunk_keys(NAMESPACE, rank, world, chain, 0, count)
        case["chain_check"] = keys.chunk_hash_chain(range(chunk), chunk).hex()
        lines.append(json.dumps(case) + "\n")
    args.out.write_text("".join(lines))
    print(f"{len(lines)} cases written to {args.out}")


if __name__ == "__main__":
    main()
