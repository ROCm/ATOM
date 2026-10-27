# Context-length profile of the MLPerf agentic dataset under the Qwen3.5 tokenizer:
# prompt tokens at each assistant turn (system + tools + all earlier messages, no
# template overhead) and the reference output length of that turn.
import collections
import json
import sys

import numpy as np
from transformers import AutoTokenizer

path, model = sys.argv[1], sys.argv[2]
tok = AutoTokenizer.from_pretrained(model)
n = lambda s: (
    len(
        tok.encode(s if isinstance(s, str) else json.dumps(s), add_special_tokens=False)
    )
    if s
    else 0
)
convs = collections.defaultdict(list)
with open(path) as f:
    lines = list(f)
for l in lines:
    r = json.loads(l)
    if "conversation_id" in r:
        convs[r["conversation_id"]].append(r)
prompt_at_turn, out_len, conv_max, turns = [], [], [], []
for rows in convs.values():
    rows.sort(key=lambda r: int(r["turn"]))
    ctx = n(rows[0].get("system", "")) + n(rows[0].get("tools", ""))
    m = 0
    for r in rows:
        body = sum(
            n(r.get(k, ""))
            for k in ("content", "reasoning_content", "tool_calls", "tool_results")
        )
        if r["role"] == "assistant":
            prompt_at_turn.append(ctx)
            out_len.append(body)
            m = max(m, ctx + body)
        ctx += body
    conv_max.append(m)
    turns.append(sum(r["role"] == "assistant" for r in rows))
pct = lambda a: " ".join(f"p{p}={int(np.percentile(a, p))}" for p in (50, 90, 99, 100))
print(f"conversations {len(convs)}  assistant turns {len(prompt_at_turn)}")
print("assistant turns/conv:", pct(turns))
print("prompt tokens per assistant turn:", pct(prompt_at_turn))
print("reference output tokens per turn:", pct(out_len))
print("max context per conversation:", pct(conv_max))
for lim in (32768, 65536, 131072, 262144):
    print(
        f"  conversations fitting {lim}: {np.mean(np.array(conv_max) + 8192 <= lim) * 100:.1f}% (with 8192 new tokens)"
    )
