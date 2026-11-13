"""Measure native-VK versus copied-KV GDN snapshot allocator peaks on ROCm.

Run directly with Python on a ROCm GPU; this is an operator-level measurement,
not a model-service memory peak or a substitute for the RTP KV-pool budget.
"""

import argparse
import gc
import statistics

import torch

from atom.model_ops.fla_ops import gdn_flydsl


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", type=int, default=7680)
    parser.add_argument("--runs", type=int, default=5)
    args = parser.parse_args()
    if torch.version.hip is None or not torch.cuda.is_available():
        raise RuntimeError("This benchmark requires a ROCm GPU")

    device = torch.device("cuda")
    torch.manual_seed(7)
    q = torch.randn(1, args.tokens, 8, 128, device=device, dtype=torch.bfloat16) / 16
    k = torch.randn_like(q) / 16
    v = torch.randn(1, args.tokens, 24, 128, device=device, dtype=torch.bfloat16) / 16
    g = -torch.rand(1, args.tokens, 24, device=device, dtype=torch.bfloat16) / 8
    beta = torch.rand(1, args.tokens, 24, device=device, dtype=torch.bfloat16)
    initial = torch.zeros(1, 24, 128, 128, device=device, dtype=torch.float32)
    cu_seqlens = torch.tensor([0, args.tokens], device=device, dtype=torch.int32)
    metadata = gdn_flydsl.build_prefill_metadata((args.tokens,), cu_seqlens)
    inputs = (q, k, v, g, beta, initial, cu_seqlens, metadata, True)
    if not gdn_flydsl.prefill_supported(q, k, v, g, beta, metadata):
        raise RuntimeError("FlyDSL prefill is unavailable for this test shape")

    # Compile both specializations before measuring allocation peaks.
    for layout in ("kv", "vk"):
        result = gdn_flydsl.prefill(*inputs, snapshot_layout=layout)
        torch.cuda.synchronize()
        del result

    peaks = {"kv": [], "vk": []}
    for run in range(args.runs):
        for layout in (("kv", "vk") if run % 2 == 0 else ("vk", "kv")):
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            baseline = torch.cuda.memory_allocated()
            torch.cuda.reset_peak_memory_stats()
            result = gdn_flydsl.prefill(*inputs, snapshot_layout=layout)
            torch.cuda.synchronize()
            peaks[layout].append(torch.cuda.max_memory_allocated() - baseline)
            del result

    mib = 1024 * 1024
    print(f"tokens={args.tokens} runs={args.runs} dtype=bf16 heads=(8,24) K=V=128")
    for layout in ("kv", "vk"):
        print(
            f"{layout}: incremental PyTorch allocator peak MiB "
            f"{[round(value / mib, 2) for value in peaks[layout]]}; "
            f"median={statistics.median(peaks[layout]) / mib:.2f}"
        )
    delta = statistics.median(peaks["kv"]) - statistics.median(peaks["vk"])
    print(f"median KV-minus-VK peak MiB={delta / mib:.2f}")


if __name__ == "__main__":
    main()
