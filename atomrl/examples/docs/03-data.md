> [Examples README](../README.md) > Models and Data

# 3. Models and Data

## 3.1 Download

```bash
sudo docker exec "$CONTAINER" bash -lc '
python3 - <<PY
from huggingface_hub import snapshot_download
import os; D = os.environ["DATA_ROOT"]
snapshot_download("Qwen/Qwen3-8B-Base", local_dir=f"{D}/models/Qwen3-8B-Base",
                  allow_patterns=["*.json","*.txt","*.safetensors","*.model","tokenizer*"])
snapshot_download("BytedTsinghua-SIA/DAPO-Math-17k", repo_type="dataset",
                  local_dir=f"{D}/raw/DAPO-Math-17k")
snapshot_download("BytedTsinghua-SIA/AIME-2024", repo_type="dataset",
                  local_dir=f"{D}/raw/AIME-2024")
PY
'

# Additionally for examples 6 and 7 (~57G)
sudo docker exec "$CONTAINER" bash -lc '
hf download Qwen/Qwen3-30B-A3B-Base \
  --local-dir "$DATA_ROOT/models/Qwen3-30B-A3B-Base" --max-workers 8'
```

From a restricted network use ModelScope instead. The IDs are identical
(`Qwen/Qwen3-8B-Base`, `Qwen/Qwen3-30B-A3B-Base`, `BytedTsinghua-SIA/DAPO-Math-17k`,
`BytedTsinghua-SIA/AIME-2024`), everything lands in the same local paths, and no later
command changes:

```bash
sudo docker exec "$CONTAINER" bash -lc '
pip install modelscope
python3 - <<PY
from modelscope.hub.snapshot_download import snapshot_download
import os
D = os.environ["DATA_ROOT"]
snapshot_download("Qwen/Qwen3-8B-Base", local_dir=f"{D}/models/Qwen3-8B-Base",
    allow_patterns=["*.json","*.txt","*.safetensors","*.model","tokenizer*","*.py","*.tiktoken"],
    max_workers=8)
for rid, sub in (("BytedTsinghua-SIA/DAPO-Math-17k", "DAPO-Math-17k"),
                 ("BytedTsinghua-SIA/AIME-2024", "AIME-2024")):
    snapshot_download(repo_id=rid, repo_type="dataset", local_dir=f"{D}/raw/{sub}",
        allow_patterns=["*.parquet","*.json","*.jsonl","*.md","*.txt"], max_workers=4)
PY
'
```

> **The MoE example requires the Base model.** The instruct/thinking Qwen3-30B-A3B
> **never closes `</think>`** within `max_response_length` (measured: still unclosed at
> 3072 tokens, and no `\boxed` either), so every sample gets truncated, reward is
> permanently -1, `filter_groups` keeps 0 for 10 consecutive rounds, and the run dies with
> `RuntimeError: filter_groups collected no valid groups`. The Base model emits `Answer:`
> normally.

---

## 3.2 Filter prompts to <=1024

Without pre-filtering, startup enters a slow overlong-prompt scan.

```bash
cat > "$RL_ROOT/filter_prompts.py" <<'PYEOF'
import os, glob
import datasets
from transformers import AutoTokenizer

DATA = os.environ["DATA_ROOT"]
MODEL_PATH = f"{DATA}/models/Qwen3-8B-Base"
MAX_PROMPT_LENGTH = 1024
PROMPT_KEY = "prompt"
OUT_DIR = f"{DATA}/data_cached/qwen3-8b-maxprompt1024"

def first_parquet(*dir_globs):
    for g in dir_globs:
        hits = sorted(glob.glob(g, recursive=True))
        if hits:
            return hits[0]
    raise FileNotFoundError(f"no parquet under {dir_globs}")

JOBS = [
    (first_parquet(f"{DATA}/raw/DAPO-Math-17k/**/*.parquet"),
     os.path.join(OUT_DIR, "dapo-math-17k.filtered.parquet")),
    (first_parquet(f"{DATA}/raw/AIME-2024/**/*.parquet"),
     os.path.join(OUT_DIR, "aime-2024.filtered.parquet")),
]

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

def doc2len(doc) -> int:
    # transformers 4.x returns a flat list of token ids; 5.x returns a BatchEncoding,
    # where len() counts the dict keys (2) instead of the tokens and filters nothing.
    enc = tokenizer.apply_chat_template(doc[PROMPT_KEY], add_generation_prompt=True, tokenize=True)
    ids = enc["input_ids"] if hasattr(enc, "keys") else enc
    if ids and isinstance(ids[0], (list, tuple)):
        ids = ids[0]
    return len(ids)

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    nproc = max(1, min(64, (os.cpu_count() or 8) // 4))
    for src, dst in JOBS:
        ds = datasets.Dataset.from_parquet(src)
        before = len(ds)
        ds = ds.filter(lambda d: doc2len(d) <= MAX_PROMPT_LENGTH, num_proc=nproc,
                       desc=f"Filtering prompts > {MAX_PROMPT_LENGTH} tokens")
        ds.to_parquet(dst)
        print(f"[{src}] -> {dst}: {before} -> {len(ds)} (removed {before-len(ds)})")

if __name__ == "__main__":
    main()
PYEOF
sudo docker exec "$CONTAINER" bash -lc 'cd "$RL_ROOT" && python3 filter_prompts.py'
```

The outputs are exactly `run_dapo.sh`'s default `TRAIN_FILE` / `VAL_FILE`:

```text
$DATA_ROOT/data_cached/qwen3-8b-maxprompt1024/dapo-math-17k.filtered.parquet   # train
$DATA_ROOT/data_cached/qwen3-8b-maxprompt1024/aime-2024.filtered.parquet       # val
```

> **Filter once, share across all examples.** Qwen3-8B-Base and Qwen3-30B-A3B-Base have
> byte-identical `tokenizer.json` / `vocab.json` / `merges.txt` (vocab 151936), so the
> filtering done with the 8B tokenizer holds for the MoE model too.
