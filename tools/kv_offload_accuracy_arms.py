#!/usr/bin/env python3
"""Three-arm accuracy check for a KV-offload tier, with validity gates.

The sibling of `kv_offload_twopass_check.py`, for the case its own docstring
warns about. That one compares pass 2's text against pass 1's, and notes that
a prefix-cache hit is not bit-reproducible even with no connector attached --
measured here, the no-cache arm alone differed on 4 of 8 answers, so text
identity cannot be the criterion. This asks instead whether the answer is
still RIGHT, and whether it could have been right for any reason other than
the restored bytes.

Three arms over one document of invented facts -- access codes that exist in
no training set -- each asking for one of them by index:

  blind  no document at all. Must COLLAPSE, or the questions did not need the
         prefix and neither of the other arms is about the prefix either.
  cold   the document, carrying a per-question, per-invocation nonce at its
         HEAD so every prefix hash is unique. Must score HIGH and must take
         ~0% of its prompt from the tier. The nonce goes first because
         anywhere later leaves the tier serving everything up to it, which is
         how a control quietly becomes a second copy of the measurement.
  warm   the shared document, already in the tier. The measurement, reported
         beside the share of prompt tokens the tier actually supplied.

Four gates, because "the warm arm scored well" is not by itself evidence. It
exits non-zero when any of them says the experiment did not happen.

Discriminating power, measured by zeroing restored PAGE blocks on
DeepSeek-V4.1 while the cold arm stayed at 32/32: 25% of blocks -> 15/16,
100% -> 1/16. It saturates correctly but is sublinear, so read it as a
SAMPLING detector -- it catches corruption that lands on the fact being
asked, not all corruption.

It does not cover STATE at all. A state image that fails to restore is
refused by the model's own cursor check before any answer is produced: the
request 500s and the engine stops, so this harness never sees a wrong answer
to score. PAGE has no such check, and that is the half this is for.

Usage:
  python3 tools/kv_offload_accuracy_arms.py [port] [questions] [facts]
                                            [max_tokens] [served_model_name]

`max_tokens` has to leave room for the model to think: at 24 a reasoning
model emitted chain-of-thought, was cut mid-sentence, and scored as a wrong
code -- which reads as corruption. The cold gate is what caught that.
"""

import json
import os
import random
import sys
import time
import urllib.request

PORT = sys.argv[1] if len(sys.argv) > 1 else "8021"
NQ = int(sys.argv[2]) if len(sys.argv) > 2 else 16
NFACTS = int(sys.argv[3]) if len(sys.argv) > 3 else 900
# Big enough for the model to think. At 24 it emitted chain-of-thought and
# was cut mid-sentence ("We need answer only access code. Need fi"), which
# scores as a wrong code and reads as corruption. The fresh arm caught it.
MAXTOK = int(sys.argv[4]) if len(sys.argv) > 4 else 640
MODEL = sys.argv[5] if len(sys.argv) > 5 else "DeepSeek-V4.1-Flash"

rng = random.Random(20261011)
# Salted per INVOCATION, not just per question. With a fixed salt the second
# run of this script reused the first run's nonces, so its "cold" arm found
# them already in the tier and took 93.8% of its prompt from there -- a
# control that had silently become a second measurement. The supply gate
# caught it, which is the only reason this is a comment and not a result.
SALT = f"{os.getpid():d}-{time.time_ns():d}"
ALPHA = "ABCDEFGHJKLMNPQRSTUVWXYZ"


def code():
    return (
        "".join(rng.choice(ALPHA) for _ in range(2))
        + "-"
        + "".join(rng.choice("0123456789") for _ in range(4))
    )


CODES = [code() for _ in range(NFACTS)]
DOC = "\n".join(
    f"Record {i:04d}: the access code for sector {i:04d} is {CODES[i]}."
    for i in range(NFACTS)
)
HEAD = (
    "You are reading a facility register. Answer only with the access code, "
    "nothing else.\n\n"
)


def ask(sector, *, with_doc, nonce=None):
    body = HEAD
    if with_doc:
        # The nonce goes at the HEAD, before a single shared byte. Anywhere
        # later and the tier would still serve everything up to it, which is
        # how the control quietly became a second copy of the measurement.
        if nonce is not None:
            body += f"Register revision {nonce}.\n\n"
        body += DOC + "\n\n"
    body += f"Question: what is the access code for sector {sector:04d}?"
    payload = json.dumps(
        {
            "model": MODEL,
            "messages": [{"role": "user", "content": body}],
            "temperature": 0.0,
            "max_tokens": MAXTOK,
            "seed": 0,
        }
    ).encode()
    req = urllib.request.Request(
        f"http://127.0.0.1:{PORT}/v1/chat/completions",
        payload,
        {"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=900) as h:
        d = json.load(h)
    return d["choices"][0]["message"]["content"], d["usage"]["prompt_tokens"]


def tier():
    try:
        with urllib.request.urlopen(
            f"http://127.0.0.1:{PORT}/metrics", timeout=10
        ) as h:
            text = h.read().decode()
    except Exception:
        return None
    out = {}
    for line in text.splitlines():
        for key in (
            "vllm:external_prefix_cache_queries_total",
            "vllm:external_prefix_cache_hits_total",
            "vllm:prefix_cache_queries_total",
            "vllm:prefix_cache_hits_total",
        ):
            if line.startswith(key):
                out[key.split(":")[1]] = float(line.rsplit(" ", 1)[1])
    return out


def run(name, sectors, *, with_doc, unique=False):
    before = tier()
    right = 0
    ntok = 0
    for i, s in enumerate(sectors):
        try:
            txt, ntok = ask(
                s, with_doc=with_doc, nonce=f"{name}-{i}-{SALT}" if unique else None
            )
        except Exception as exc:
            print(f"ARM {name} sector {s:04d} FAILED {type(exc).__name__}: {exc}")
            return None
        if CODES[s] in txt:
            right += 1
        elif with_doc:
            print(f"  {name} sector {s:04d} wanted {CODES[s]}: {txt.strip()[:40]!r}")
    after = tier()
    supply = ""
    share = None
    if before and after:
        dq = after["external_prefix_cache_queries_total"] - before[
            "external_prefix_cache_queries_total"
        ]
        dh = after["external_prefix_cache_hits_total"] - before[
            "external_prefix_cache_hits_total"
        ]
        lq = after["prefix_cache_queries_total"] - before["prefix_cache_queries_total"]
        lh = after["prefix_cache_hits_total"] - before["prefix_cache_hits_total"]
        share = 100 * dh / dq if dq else 0.0
        supply = (
            f" tier={dh:.0f}/{dq:.0f} ({share:.1f}%)"
            f" hbm={lh:.0f}/{lq:.0f} ({100 * lh / lq if lq else 0:.1f}%)"
        )
    print(f"ARM {name} correct {right}/{len(sectors)} prompt_tokens={ntok}{supply}")
    return right, share


pool = rng.sample(range(NFACTS), 3 * NQ)
blind, cold, warm = pool[:NQ], pool[NQ : 2 * NQ], pool[2 * NQ :]

print(f"document facts={NFACTS} questions_per_arm={NQ} max_tokens={MAXTOK}")
# The shared document is primed once, outside every arm, so the warm arm is
# measured on a tier that is already full rather than on one it is filling.
try:
    ask(pool[0], with_doc=True)
except Exception as exc:
    print(f"INVALID: priming request failed {type(exc).__name__}: {exc}")
    sys.exit(1)

b = run("blind", blind, with_doc=False)
c = run("cold ", cold, with_doc=True, unique=True)
w = run("warm ", warm, with_doc=True)

print("--- verdict ---")
if None in (b, c, w):
    print("INVALID: an arm failed to complete")
    sys.exit(1)
(b, _), (c, c_share), (w, w_share) = b, c, w
if b > NQ // 4:
    print(f"INVALID: blind scored {b}/{NQ}; the questions do not need the prefix,")
    print("         so neither of the other arms is about the prefix either")
    sys.exit(1)
if c < NQ * 3 // 4:
    print(f"INVALID: cold scored {c}/{NQ}; the task is not solvable, so a low")
    print("         warm score would be read as corruption when it is not")
    sys.exit(1)
if c_share is not None and c_share > 5:
    print(f"INVALID: cold arm took {c_share:.1f}% of its prompt from the tier;")
    print("         it is a second copy of the measurement, not a control")
    sys.exit(1)
if w_share is not None and w_share < 50:
    print(f"INVALID: warm arm took only {w_share:.1f}% of its prompt from the")
    print("         tier; the experiment did not happen, whatever it scored")
    sys.exit(1)
print(
    f"VALID test. blind={b}/{NQ} cold={c}/{NQ} warm={w}/{NQ} "
    f"(warm tier supply {w_share:.1f}%)"
)
sys.exit(0 if w >= c else 2)
