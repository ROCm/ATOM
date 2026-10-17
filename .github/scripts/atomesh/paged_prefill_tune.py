#!/usr/bin/env python3
"""CI-only paged prefill shape sweep; validate every candidate before timing.

The old paged implementation is imported in a separate interpreter. Kernel
configurations are passed explicitly to low-level launchers, never injected
into model execution. Results include the unchanged gather + FMHA baseline.
"""

import argparse
import importlib
import json
import math
import os
import subprocess
import sys
from contextlib import ExitStack
from pathlib import Path


def cases():
    lengths = [
        ([1], [4097]),
        ([17], [32771]),
        ([128], [131077]),
        ([255], [524293]),
        ([256], [524293]),
        ([2048], [524293]),
        ([8192], [131077]),
        ([32768], [65539]),
        ([128], [1048573]),
        ([1, 17, 256, 2048], [8193, 32771, 131077, 524293]),
        ([1, 17, 64, 256] * 8, [8193, 16387, 32771, 65539] * 8),
        # Held-out lengths; no exact-shape production lookup is introduced.
        ([513], [77777]),
        ([1536], [333337]),
    ]
    return [
        {
            "id": f"{role}-{i}",
            "kv_heads": kh,
            "q_lens": q,
            "k_lens": k,
            "held_out": i >= 11,
        }
        for role, kh in [("target", 1), ("draft", 16)]
        for i, (q, k) in enumerate(lengths)
    ]


def fixture(case):
    import torch
    from aiter.ops.triton.utils.types import e4m3_dtype

    torch.manual_seed(20261016)
    page, d, h, kh = 128, 128, 16, case["kv_heads"]
    ql, kl = case["q_lens"], case["k_lens"]
    counts = [math.ceil(k / page) for k in kl]
    npages = sum(counts) + 1
    # Start in BF16 to avoid a large transient FP32 copy of the entire pool.
    k = torch.randn(npages, page, kh, d, dtype=torch.bfloat16, device="cuda").to(
        e4m3_dtype
    )
    v = torch.randn(npages, page, kh, d, dtype=torch.bfloat16, device="cuda").to(
        e4m3_dtype
    )
    ks = torch.rand(npages, kh, page, device="cuda") + 0.1
    vs = torch.rand(npages, kh, page, device="cuda") + 0.2
    table = torch.zeros(len(ql), max(counts), dtype=torch.int32, device="cuda")
    perm = torch.randperm(npages, device="cuda")
    begin = 0
    for i, count in enumerate(counts):
        table[i, :count] = perm[begin : begin + count]
        begin += count
    if len(ql) > 1:
        table[1:, 0] = table[0, 0]
    for i, length in enumerate(kl):
        if length % page:
            v[table[i, counts[i] - 1].long(), length % page :] = float("nan")
    kc = k.view(npages, page, kh, d // 16, 16).permute(0, 2, 3, 1, 4).contiguous()
    vc = v.view(npages, page // 16, 16, kh, d).permute(0, 3, 1, 4, 2).contiguous()
    q = torch.randn(sum(ql), h, d, dtype=torch.bfloat16, device="cuda")
    cu = torch.tensor(
        [0] + list(__import__("itertools").accumulate(ql)),
        dtype=torch.int32,
        device="cuda",
    )
    used = torch.tensor(kl, dtype=torch.int32, device="cuda")
    return {
        "q": q,
        "k": kc,
        "v": vc,
        "ks": ks,
        "vs": vs,
        "table": table,
        "cu": cu,
        "used": used,
        "out": torch.empty_like(q),
        "plain_k": k,
        "plain_v": v,
    }


def reference_rows(x, case):
    """Independent FP32 online softmax, bounded memory even at 1M context."""
    import torch

    selected, outputs = [], []
    begin = 0
    for i, (ql, kl) in enumerate(zip(case["q_lens"], case["k_lens"])):
        positions = sorted({0, ql // 2, ql - 1})
        selected.extend(begin + p for p in positions)
        query = x["q"][[begin + p for p in positions]].float().transpose(0, 1)
        m = torch.full((16, len(positions)), -float("inf"), device="cuda")
        denom = torch.zeros_like(m)
        acc = torch.zeros(16, len(positions), 128, device="cuda")
        for start in range(0, kl, 8192):
            pos = torch.arange(start, min(start + 8192, kl), device="cuda")
            pages, slots = x["table"][i, pos // 128].long(), pos % 128
            # Index before widening: do not convert the full pool per chunk.
            k = x["plain_k"].view(torch.uint8)[pages, slots].view(x["k"].dtype).float()
            v = x["plain_v"].view(torch.uint8)[pages, slots].view(x["v"].dtype).float()
            k = (k * x["ks"][pages, :, slots, None]).bfloat16().float()
            v = (v * x["vs"][pages, :, slots, None]).bfloat16().float()
            k = k.repeat_interleave(16 // case["kv_heads"], dim=1)
            v = v.repeat_interleave(16 // case["kv_heads"], dim=1)
            scores = query @ k.permute(1, 2, 0) / math.sqrt(128)
            mask = (
                pos[None, :]
                > (kl - ql + torch.tensor(positions, device="cuda"))[:, None]
            )
            scores.masked_fill_(mask[None], -float("inf"))
            next_m = torch.maximum(m, scores.amax(-1))
            alpha = torch.exp(m - next_m)
            probs = torch.exp(scores - next_m[..., None])
            acc = acc * alpha[..., None] + probs @ v.transpose(0, 1)
            denom = denom * alpha + probs.sum(-1)
            m = next_m
        outputs.append((acc / denom[..., None]).transpose(0, 1).bfloat16())
        begin += ql
    return selected, torch.cat(outputs)


def public_call(ua, x, case):
    return ua.unified_attention(
        x["q"],
        x["k"],
        x["v"],
        x["out"],
        x["cu"],
        max(case["q_lens"]),
        x["used"],
        max(case["k_lens"]),
        128**-0.5,
        True,
        (-1, -1),
        x["table"],
        0,
        None,
        None,
        None,
        backend="triton",
        shuffled_kv_cache=True,
        k_scale_cache=x["ks"],
        v_scale_cache=x["vs"],
    )


def parameters(ua, x, case):
    from aiter.ops.triton.utils.device_info import get_num_sms

    q, k = x["q"], x["k"]
    n, h, d = q.shape
    hk = k.shape[1]
    nq = h // hk
    blocks = (
        len(case["q_lens"])
        if max(case["q_lens"]) == 1
        else n // (16 // nq) + len(case["q_lens"])
    )
    return ua._UAParams(
        q=q,
        k=k,
        v=x["v"],
        out=x["out"],
        cu_seqlens_q=x["cu"],
        seqused_k=x["used"],
        block_table=x["table"],
        softmax_scale=d**-0.5,
        softcap=0,
        causal=True,
        sliding_window=0,
        max_seqlen_q=max(case["q_lens"]),
        max_seqlen_k=max(case["k_lens"]),
        num_tokens=n,
        num_query_heads=h,
        num_kv_heads=hk,
        num_queries_per_kv=nq,
        head_size=d,
        head_size_v=d,
        num_seqs=len(case["q_lens"]),
        total_num_q_blocks=blocks,
        num_2d_prgms=blocks * hk,
        num_blocks=k.shape[0],
        block_size=128,
        k_width=16,
        scale_k_width=4,
        block_scales_size=16,
        q_dtype=q.dtype,
        kv_cache_dtype=k.dtype,
        all_decode=max(case["q_lens"]) == 1,
        shuffled_kv_cache=True,
        use_alibi_slopes=False,
        use_qq_bias=False,
        num_sms=get_num_sms(),
        target_num_prgms=get_num_sms() * 4,
        k_descale=x["ks"],
        v_descale=x["vs"],
        per_token_kv_scales=True,
    )


def candidate(ua, p, block_m, warps, stages, splits):
    import torch

    cfg = {
        "BLOCK_M": block_m,
        "BLOCK_Q": block_m // p.num_queries_per_kv,
        "num_warps": warps,
        "num_stages": stages,
        "waves_per_eu": 2,
    }
    if splits == 1:
        cfg.update(TILE_SIZE=128, SPLIT_UNMASKED_LOOP=False)
        return lambda: ua._unified_attention_2d_triton(p, config=cfg)
    so = torch.empty(
        p.num_tokens,
        p.num_query_heads,
        splits,
        p.head_size,
        dtype=torch.float32,
        device="cuda",
    )
    sm = torch.empty(
        p.num_tokens, p.num_query_heads, splits, dtype=torch.float32, device="cuda"
    )
    se = torch.empty_like(sm)

    def run():
        ua._unified_attention_3d_triton(p, so, sm, se, splits, 128, config=cfg)
        ua._reduce_segments_triton(p, so, sm, se, splits, 128)

    return run


def gather_call(x, case):
    import aiter
    import torch

    from atom.model_ops.base_attention import cp_mha_gather_cache

    kl = case["k_lens"]
    cu_k = torch.tensor(
        [0] + list(__import__("itertools").accumulate(kl)),
        dtype=torch.int32,
        device="cuda",
    )
    batch_ids = torch.repeat_interleave(
        torch.arange(len(kl), device="cuda", dtype=torch.int32), x["used"].long()
    )
    starts = torch.zeros(len(kl), dtype=torch.int32, device="cuda")

    def run():
        k = torch.empty(
            sum(kl), case["kv_heads"], 128, dtype=torch.bfloat16, device="cuda"
        )
        v = torch.empty_like(k)
        cp_mha_gather_cache(
            x["k"],
            x["v"],
            k,
            v,
            x["table"],
            x["ks"],
            x["vs"],
            cu_k,
            batch_ids,
            starts,
            True,
            "SHUFFLE",
            sum(kl),
            True,
        )
        return aiter.flash_attn_varlen_func(
            x["q"],
            k,
            v,
            x["cu"],
            cu_k,
            max(case["q_lens"]),
            max(kl),
            min_seqlen_q=min(case["q_lens"]),
            softmax_scale=128**-0.5,
            causal=True,
        )

    return run


def measure(fn):
    import statistics

    import torch

    # Eager events include launch gaps, like the actual cached-prefill path.
    for _ in range(2):
        fn()
    torch.cuda.synchronize()
    results = []
    for _ in range(5):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(
            enable_timing=True
        )
        a.record()
        fn()
        b.record()
        b.synchronize()
        results.append(a.elapsed_time(b))
    return {"median_ms": statistics.median(results), "samples_ms": results}


def configurations():
    # Finalists from the broad sweep plus neighbouring split counts. The
    # production default is independently checked before these experiments.
    return [
        (128, 4, 1, 1),
        (256, 8, 1, 1),
        *[(64, 2, 1, split) for split in (8, 16, 32, 64)],
    ]


def configuration_name(config):
    bm, nw, ns, splits = config
    return f"m{bm}-w{nw}-s{ns}-kv{splits}"


def worker(args):
    import torch

    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    ua = importlib.import_module("aiter.ops.triton.attention.unified_attention")
    variant = configuration_name(args.config) if args.mode == "candidate" else args.mode
    result_path = Path(args.output) / f"{variant}-{args.shard}.jsonl"

    def record(row):
        with result_path.open("a") as f:
            f.write(json.dumps(row) + "\n")
        print(json.dumps(row), flush=True)

    record(
        {
            "device": torch.cuda.get_device_name(),
            "source": ua.__file__,
            "shard": args.shard,
        }
    )
    for idx, case in enumerate(cases()):
        if idx % args.shards != args.shard:
            continue
        if (
            args.mode == "candidate"
            and max(case["q_lens"]) == 1
            and args.config[0] > 16
        ):
            continue
        x = fixture(case)
        selected, reference = reference_rows(x, case)
        call = lambda x=x, case=case: public_call(ua, x, case)
        call()
        torch.testing.assert_close(x["out"][selected], reference, atol=0.003, rtol=0.03)
        expected = x["out"].clone()
        if args.mode != "candidate":
            record(dict(case=case, variant=variant, accuracy="passed", **measure(call)))
        if args.mode == "new":
            gather = gather_call(x, case)
            gathered = gather()
            torch.testing.assert_close(
                gathered[selected], reference, atol=0.003, rtol=0.03
            )
            torch.testing.assert_close(expected, gathered, atol=0.003, rtol=0.03)
            record(
                dict(
                    case=case,
                    variant="gather-fmha",
                    accuracy="passed",
                    **measure(gather),
                )
            )
            del gathered, gather
        elif args.mode == "candidate":
            p = parameters(ua, x, case)
            fn = candidate(ua, p, *args.config)
            record({"case": case, "variant": variant, "phase": "candidate-start"})
            try:
                fn()
            except Exception as exc:
                if type(exc).__name__ != "OutOfResources":
                    raise
                record(
                    {
                        "case": case,
                        "variant": variant,
                        "skipped": "OutOfResources",
                        "detail": str(exc),
                    }
                )
                return
            try:
                torch.testing.assert_close(x["out"], expected, atol=0.003, rtol=0.03)
                torch.testing.assert_close(
                    x["out"][selected], reference, atol=0.003, rtol=0.03
                )
            except AssertionError as exc:
                record(
                    {
                        "case": case,
                        "variant": variant,
                        "accuracy": "failed",
                        "detail": str(exc),
                    }
                )
                # End this isolated process; do not run another shape after a
                # potentially corrupting candidate or reuse its CUDA context.
                return
            record(dict(case=case, variant=variant, accuracy="passed", **measure(fn)))
            del fn, p
        del call, x, expected, reference
        torch.cuda.empty_cache()


def write_sweep_summary(output, rejected):
    """Keep all medians and correctness gates in a small API-readable snapshot."""
    measurements = {}
    for path in sorted(Path(output).glob("*.jsonl")):
        for line in path.read_text().splitlines():
            row = json.loads(line)
            if "case" not in row:
                continue
            key = (row["case"]["id"], row["variant"])
            measurements[key] = {
                "case_id": key[0],
                **{k: v for k, v in row.items() if k not in ("case", "samples_ms")},
            }
    summary = {
        "cases": cases(),
        "measurements": list(measurements.values()),
        "rejected_configs": list(rejected),
    }
    Path(output, "sweep-summary.json").write_text(json.dumps(summary) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--baseline-source")
    parser.add_argument("--mode", choices=["old", "new", "candidate"])
    parser.add_argument("--config", type=int, nargs=4)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--shards", type=int, default=8)
    args = parser.parse_args()
    Path(args.output).mkdir(parents=True, exist_ok=True)
    if args.mode:
        worker(args)
        return
    # All original and extended correctness cases run before any sweep timings.
    import aiter

    check = (
        Path(aiter.__file__).resolve().parents[1]
        / "op_tests/triton_tests/attention/test_unified_attention_scale_cache.py"
    )
    subprocess.run([sys.executable, str(check)], check=True)
    rejected = {}
    plan = [("old", None), ("new", None)] + [
        ("candidate", config) for config in configurations()
    ]
    for mode, config in plan:
        variant = configuration_name(config) if config else mode
        with ExitStack() as logs:
            procs = []
            for shard in range(args.shards):
                env = dict(os.environ, HIP_VISIBLE_DEVICES=str(shard))
                env.pop("ROCR_VISIBLE_DEVICES", None)
                if mode == "old":
                    env["PYTHONPATH"] = (
                        args.baseline_source + ":" + env.get("PYTHONPATH", "")
                    )
                log = logs.enter_context(
                    (Path(args.output) / f"{variant}-{shard}.log").open("w")
                )
                command = [
                    sys.executable,
                    str(Path(__file__).resolve()),
                    "--output",
                    args.output,
                    "--mode",
                    mode,
                    "--shard",
                    str(shard),
                    "--shards",
                    str(args.shards),
                ]
                if config:
                    command += ["--config", *map(str, config)]
                procs.append(
                    subprocess.Popen(
                        command, env=env, stdout=log, stderr=subprocess.STDOUT
                    )
                )
            status = [p.wait() for p in procs]
            write_sweep_summary(args.output, rejected)
            if mode != "candidate" and any(status):
                raise RuntimeError(f"{mode} workers failed: {status}; see per-GPU logs")
            if mode == "candidate":
                reasons = []
                if any(status):
                    reasons.append({"worker_exit_codes": status})
                for result in Path(args.output).glob(f"{variant}-*.jsonl"):
                    for line in result.read_text().splitlines():
                        row = json.loads(line)
                        if row.get("accuracy") == "failed" or row.get("skipped"):
                            reasons.append(row)
                if reasons:
                    rejected[variant] = reasons
                    print(f"[paged-prefill-tune] rejected {variant}", flush=True)
                (Path(args.output) / "rejected-configs.json").write_text(
                    json.dumps(rejected, indent=2) + "\n"
                )
                write_sweep_summary(args.output, rejected)
    print(
        "[paged-prefill-tune] default/reference gates passed; isolated candidate sweep complete",
        flush=True,
    )


if __name__ == "__main__":
    main()
