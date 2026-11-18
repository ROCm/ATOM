"""Build and verify the exact lm-eval prompts used by the sparse-path check."""

import hashlib
import json
import random
from pathlib import Path


def build_prompts(fewshot):
    import numpy as np
    import torch
    from lm_eval.tasks import TaskManager

    random.seed(0)
    np.random.seed(1234)
    torch.manual_seed(1234)
    task = TaskManager().load(["gsm8k"])["tasks"]["gsm8k"]
    task.set_config(key="num_fewshot", value=fewshot)
    task.set_fewshot_seed(seed=1234)
    task.build_all_requests(
        limit=None,
        rank=0,
        world_size=1,
        cache_requests=False,
        rewrite_requests_cache=False,
        system_instruction=None,
        apply_chat_template=False,
        fewshot_as_multiturn=False,
        chat_template=None,
        tokenizer_name="",
    )
    prompts = {
        str(instance.doc_id): instance.arguments[0] for instance in task.instances
    }
    assert len(prompts) == 1319
    return prompts


def prepare(model, output, fewshot=20):
    from transformers import AutoTokenizer

    config = json.loads((Path(model) / "config.json").read_text())
    configs = [config]
    topks = set()
    while configs:
        current = configs.pop()
        if "index_topk" in current:
            topks.add(int(current["index_topk"]))
        configs.extend(value for value in current.values() if isinstance(value, dict))
    assert len(topks) == 1 and min(topks) > 0, topks
    topk = topks.pop()
    tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
    prompts = build_prompts(fewshot)
    records = {
        key: {
            "prompt_hash": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
            "tokens": len(tokenizer.encode(prompt, add_special_tokens=False)),
        }
        for key, prompt in prompts.items()
    }
    lengths = [record["tokens"] for record in records.values()]
    plan = {
        "fewshot": fewshot,
        "index_topk": topk,
        "min_tokens": min(lengths),
        "max_tokens": max(lengths),
        "samples": len(records),
        "records": records,
    }
    Path(output).write_text(json.dumps(plan, indent=2) + "\n")
    assert min(lengths) > topk, (min(lengths), topk)
    assert max(lengths) + 4096 <= 65536
    print(
        f"Prompt preflight: {len(records)} inputs, {min(lengths)}..{max(lengths)} tokens, index_topk={topk}",
        flush=True,
    )
    return plan


def verify_samples(folder, plan):
    files = list(Path(folder).rglob("samples_gsm8k*.jsonl"))
    assert len(files) == 1, files
    seen = set()
    with files[0].open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            key = (row["filter"], str(row["doc_id"]))
            assert key not in seen, key
            seen.add(key)
            assert row["prompt_hash"] == plan["records"][key[1]]["prompt_hash"], key
    expected = {
        (name, key)
        for name in ("strict-match", "flexible-extract")
        for key in plan["records"]
    }
    assert seen == expected, (len(seen), len(expected))
