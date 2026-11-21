> [Examples README](../README_cn.md) > 模型与数据

# 3. 模型与数据

## 3.1 下载

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

# 例子 6、7 追加（约 57G）
sudo docker exec "$CONTAINER" bash -lc '
hf download Qwen/Qwen3-30B-A3B-Base \
  --local-dir "$DATA_ROOT/models/Qwen3-30B-A3B-Base" --max-workers 8'
```

国内网络改用 ModelScope，ID 同名（`Qwen/Qwen3-8B-Base`、`Qwen/Qwen3-30B-A3B-Base`、
`BytedTsinghua-SIA/DAPO-Math-17k`、`BytedTsinghua-SIA/AIME-2024`），落到同样的本地路径，
后续命令不用改：

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

> **MoE 必须用 Base 版**。instruct/thinking 版的 Qwen3-30B-A3B 在 `max_response_length` 内
> **永远不闭合 `</think>`**（实测给到 3072 token 仍不闭合、也不出 `\boxed`），于是每条样本都被
> 截断、reward 恒为 -1、`filter_groups` 连续 10 轮 kept 0，直接抛
> `RuntimeError: filter_groups collected no valid groups`。Base 版能正常输出 `Answer:`。

---

## 3.2 过滤 prompt <=1024

不预过滤的话启动会进入耗时的 overlong-prompt 扫描。

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
    # transformers 4.x 返回扁平的 token id 列表；5.x 返回 BatchEncoding，
    # 此时 len() 数的是 dict 的 2 个 key 而不是 token 数，过滤会完全失效。
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

产出即 `run_dapo.sh` 的默认 `TRAIN_FILE` / `VAL_FILE`：

```text
$DATA_ROOT/data_cached/qwen3-8b-maxprompt1024/dapo-math-17k.filtered.parquet   # train
$DATA_ROOT/data_cached/qwen3-8b-maxprompt1024/aime-2024.filtered.parquet       # val
```

> **数据只需过滤一次，六个例子共用。** Qwen3-8B-Base 与 Qwen3-30B-A3B-Base 的
> `tokenizer.json` / `vocab.json` / `merges.txt` 三个文件 md5 完全相同（vocab 151936），
> 所以按 8B tokenizer 过滤出的结果对 MoE 同样成立。
