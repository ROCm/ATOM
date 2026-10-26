# Output sanity of an endpoints run: length stats, share of outputs that look like token salad
# (many scripts mixed in one output), and a few samples.   garbage.py <events.jsonl>
import json
import random
import sys
import unicodedata

outs = []
with open(sys.argv[1]) as f:
    lines = list(f)
for line in lines:
    if '"sample.complete"' in line[:40]:
        d = json.loads(line)
        data = d["data"]
        if not data:
            continue
        txt = "".join(x for x in (data[1] or []) if isinstance(x, str)) + "".join(
            x for x in (data[2] or []) if isinstance(x, str)
        )
        outs.append(txt)


def scripts(t):
    s = set()
    for ch in t[:2000]:
        if ch.isalpha():
            n = unicodedata.name(ch, "")
            s.add(n.split()[0] if n else "?")
    return s


bad = [t for t in outs if len(scripts(t) - {"LATIN"}) >= 3]
L = sorted(len(t) for t in outs)
print(
    f"{len(outs)} outputs, chars median {L[len(L)//2]} p90 {L[int(len(L)*.9)]}; salad-like (3+ non-Latin scripts in the first 2k chars): {len(bad)} ({100*len(bad)/max(1,len(outs)):.1f}%)"
)
random.seed(0)
for t in random.sample(outs, 2):
    print("   sample:", repr(t[:160]))
