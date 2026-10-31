#!/usr/bin/env python3
"""Run after the benchmark, on an otherwise idle budgeted Qwen server.

Evict a deterministic prompt's GPU checkpoint using unique short prompts on
the same DP session, then require both prefix reuse and identical output.
The filler count must exceed that worker's GPU checkpoint capacity while
remaining below its CPU checkpoint capacity (defaults: 1985 and 2496).
"""

import argparse
import json
import pathlib
import re
import time
import urllib.request
import uuid


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--model", default="/data/Qwen3.5-35B-A3B-MXFP4")
    parser.add_argument("--fillers", type=int, default=2200)
    parser.add_argument("--gpu-checkpoints", type=int, default=1985)
    parser.add_argument("--server-log", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    args = parser.parse_args()
    if not 0 < args.gpu_checkpoints < args.fillers:
        parser.error("fillers must exceed the positive GPU checkpoint capacity")
    session = "lmcache-restore-probe-" + uuid.uuid4().hex

    def completed_loads_high_water():
        # Worker lines share a log without rank labels. A new high-water mark
        # proves at least one worker completed a load during this idle probe.
        # An older high count on another worker can conservatively fail this
        # check; use a fresh benchmark server in that case.
        return max(
            (
                int(n)
                for n in re.findall(
                    r"state_offload_loads_completed=(\d+)",
                    args.server_log.read_text(errors="replace"),
                )
            ),
            default=0,
        )

    def stats():
        with urllib.request.urlopen(args.url + "/debug/cache_stats", timeout=30) as r:
            return json.load(r)

    def settled_stats(requests):
        deadline = time.monotonic() + 30
        while True:
            result = stats()
            if result["requests"] >= requests:
                return result
            if time.monotonic() >= deadline:
                raise RuntimeError("Cache statistics did not catch up with requests")
            time.sleep(0.2)

    def complete(prompt, max_tokens, *, ignore_eos=False):
        request = urllib.request.Request(
            args.url + "/v1/completions",
            data=json.dumps(
                {
                    "model": args.model,
                    "prompt": prompt,
                    "max_tokens": max_tokens,
                    "temperature": 0,
                    "ignore_eos": ignore_eos,
                }
            ).encode(),
            headers={"Content-Type": "application/json", "X-Session-ID": session},
        )
        with urllib.request.urlopen(request, timeout=180) as r:
            result = json.load(r)
        if "choices" not in result:
            raise RuntimeError(result)
        return result

    anchor = (
        session
        + " / anchor\n"
        + " ".join(
            f"Line {i}: the quick brown fox jumps over the lazy dog."
            for i in range(400)
        )
        + " Summarize the text above in one sentence."
    )
    initial = stats()
    cold = complete(anchor, 64)
    warm = complete(anchor, 64)
    before_fill = settled_stats(initial["requests"] + 2)
    loads_before = completed_loads_high_water()

    def fill(i):
        prompt = session + f" / filler {i}\n" + "This is unused context. " * 60
        return complete(prompt, 1)

    # One request at a time gives the state-store pipeline scheduler ticks to
    # drain. A burst can leave new checkpoints pinned in the store backlog,
    # so a count of unrelated evictions would not prove anchor displacement.
    for i in range(args.fillers):
        fill(i)
    before_restore = settled_stats(before_fill["requests"] + args.fillers)
    restored = complete(anchor, 64)
    after_restore = settled_stats(before_restore["requests"] + 1)
    # State load counters are logged every 100 decode scheduler ticks, not on
    # idle timers or the prefill early-return path. Even with MTP-3, 512 output
    # tokens require at least 128 decode steps and flush the load counter.
    complete(session + " / heartbeat", 512, ignore_eos=True)
    loads_after = completed_loads_high_water()

    texts = [r["choices"][0]["text"] for r in (cold, warm, restored)]
    evicted = before_restore["checkpoints_evicted"] - before_fill["checkpoints_evicted"]
    reused = after_restore["cached_tokens"] - before_restore["cached_tokens"]
    prompt_tokens = restored["usage"]["prompt_tokens"]
    result = {
        "session": session,
        "fillers": args.fillers,
        "gpu_checkpoint_capacity": args.gpu_checkpoints,
        "gpu_checkpoints_evicted_during_fill": evicted,
        "restored_prompt_tokens": prompt_tokens,
        "reused_prompt_tokens": reused,
        "identical_outputs": texts[0] == texts[1] == texts[2],
        "completed_loads_high_water_before": loads_before,
        "completed_loads_high_water_after": loads_after,
        "cold": cold,
        "warm": warm,
        "restored": restored,
        "before_fill": before_fill,
        "before_restore": before_restore,
        "after_restore": after_restore,
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    # The pool may start below capacity after benchmark orphan reclamation;
    # those initial fills need not evict anything. A full capacity's worth of
    # subsequent LRU evictions is enough to displace the anchor.
    assert evicted >= args.gpu_checkpoints, "Probe did not evict the GPU state window"
    assert reused >= prompt_tokens - 128, "Prompt was recomputed instead of restored"
    assert loads_after > loads_before, "No completed CPU state load was observed"
    assert result[
        "identical_outputs"
    ], "Cold, GPU-cached and CPU-restored outputs differ"
    print(
        json.dumps(
            {
                k: result[k]
                for k in (
                    "gpu_checkpoints_evicted_during_fill",
                    "restored_prompt_tokens",
                    "reused_prompt_tokens",
                    "identical_outputs",
                    "completed_loads_high_water_before",
                    "completed_loads_high_water_after",
                )
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
