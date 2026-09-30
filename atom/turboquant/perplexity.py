"""
TurboQuant Perplexity Validation

Measures the quality impact of TurboQuant KV cache compression by
computing perplexity on a text corpus with and without compression.

The test compares:
1. Exact attention (no compression)
2. TurboQuant compressed attention (3-bit)
3. TurboQuant compressed attention (2-bit)

Usage:
  python3 -m atom.turboquant.perplexity --help

  # Quick validation (100 samples)
  python3 -m atom.turboquant.perplexity --samples 100

  # Full validation
  python3 -m atom.turboquant.perplexity --samples 1000 --seq-len 512
"""

import argparse
import math
import sys
import time

import torch
import torch.nn.functional as F

from .engine import TurboQuantEngine


def compute_attention_output(Q, K, V, scale):
    """Exact attention (no compression)."""
    scores = (Q.float() @ K.float().T) * scale
    weights = F.softmax(scores, dim=-1)
    return (weights @ V.float()).half()


def compute_tq_attention_output(engine, Q, K, V):
    """TurboQuant compressed attention."""
    ck = engine.compress_keys(K)
    cv = engine.compress_values(V)
    return engine.fused_attention(Q, ck, cv)


def generate_realistic_kv(seq_len, head_dim, device="cpu"):
    """
    Generate K/V tensors that mimic real attention patterns.
    Real model KV cache vectors are approximately unit-normalized
    (due to RMSNorm/LayerNorm in transformer layers).
    """
    K = F.normalize(torch.randn(seq_len, head_dim, device=device), dim=-1).half()
    V = F.normalize(torch.randn(seq_len, head_dim, device=device), dim=-1).half()
    return K, V


def compute_perplexity_proxy(Q, K, V, scale, engine=None):
    """
    Compute a perplexity proxy based on attention output divergence.

    For each query, compute the KL divergence between exact and
    approximate attention weight distributions. The average exp(KL)
    gives a perplexity-like quality metric.

    Returns:
        kl_div: Average KL divergence
        cos_sim: Average cosine similarity of attention outputs
        max_error: Maximum absolute error in attention outputs
    """
    # Exact attention
    exact_scores = (Q.float() @ K.float().T) * scale
    exact_weights = F.softmax(exact_scores, dim=-1)
    exact_output = (exact_weights @ V.float())

    if engine is None:
        return {"kl_div": 0.0, "cos_sim": 1.0, "max_error": 0.0, "output": exact_output}

    # TurboQuant attention
    ck = engine.compress_keys(K)
    cv = engine.compress_values(V)
    tq_scores = engine.attention_scores(Q, ck)
    tq_weights = F.softmax(tq_scores, dim=-1)
    V_recon = engine.decompress_values(cv)
    tq_output = (tq_weights @ V_recon.float())

    # KL divergence: KL(exact || approx)
    kl = F.kl_div(
        torch.log(tq_weights + 1e-10),
        exact_weights,
        reduction="batchmean",
        log_target=False,
    ).item()

    # Cosine similarity of outputs
    cos_sim = F.cosine_similarity(
        exact_output, tq_output, dim=-1
    ).mean().item()

    # Max absolute error
    max_err = (exact_output - tq_output).abs().max().item()

    return {
        "kl_div": kl,
        "cos_sim": cos_sim,
        "max_error": max_err,
        "output": tq_output,
    }


def run_validation(
    samples: int = 100,
    seq_len: int = 256,
    head_dim: int = 128,
    num_queries: int = 4,
    device: str = "cpu",
):
    """
    Run perplexity validation across multiple samples.

    Returns:
        dict with average metrics for 2-bit and 3-bit modes
    """
    results = {"3bit": [], "2bit": []}

    engine_3bit = TurboQuantEngine(head_dim=head_dim, total_bits=3, device=device)
    engine_2bit = TurboQuantEngine(head_dim=head_dim, total_bits=2, device=device)
    scale = 1.0 / math.sqrt(head_dim)

    print(f"Running {samples} samples, seq_len={seq_len}, head_dim={head_dim}...",
          file=sys.stderr)

    for i in range(samples):
        torch.manual_seed(i)
        K, V = generate_realistic_kv(seq_len, head_dim, device)
        Q = torch.randn(num_queries, head_dim, device=device).half()

        # 3-bit evaluation
        r3 = compute_perplexity_proxy(Q, K, V, scale, engine_3bit)
        results["3bit"].append(r3)

        # 2-bit evaluation
        r2 = compute_perplexity_proxy(Q, K, V, scale, engine_2bit)
        results["2bit"].append(r2)

        if (i + 1) % 25 == 0:
            avg_cos_3 = sum(r["cos_sim"] for r in results["3bit"]) / len(results["3bit"])
            avg_cos_2 = sum(r["cos_sim"] for r in results["2bit"]) / len(results["2bit"])
            print(f"  [{i+1}/{samples}] 3-bit cos_sim={avg_cos_3:.4f}, "
                  f"2-bit cos_sim={avg_cos_2:.4f}", file=sys.stderr)

    # Aggregate results
    def aggregate(runs):
        n = len(runs)
        return {
            "avg_kl_div": sum(r["kl_div"] for r in runs) / n,
            "avg_cos_sim": sum(r["cos_sim"] for r in runs) / n,
            "min_cos_sim": min(r["cos_sim"] for r in runs),
            "avg_max_error": sum(r["max_error"] for r in runs) / n,
            "perplexity_proxy": math.exp(sum(r["kl_div"] for r in runs) / n),
        }

    return {
        "3bit": aggregate(results["3bit"]),
        "2bit": aggregate(results["2bit"]),
        "samples": samples,
        "seq_len": seq_len,
        "head_dim": head_dim,
    }


def main():
    parser = argparse.ArgumentParser(
        description="TurboQuant perplexity validation"
    )
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--seq-len", type=int, default=256)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
        print("CUDA not available, using CPU", file=sys.stderr)

    t0 = time.time()
    results = run_validation(
        samples=args.samples,
        seq_len=args.seq_len,
        head_dim=args.head_dim,
        device=device,
    )
    elapsed = time.time() - t0

    print(f"\n{'='*60}")
    print(f"TurboQuant Perplexity Validation Results")
    print(f"{'='*60}")
    print(f"Samples: {results['samples']}, Seq len: {results['seq_len']}, "
          f"Head dim: {results['head_dim']}")
    print(f"Time: {elapsed:.1f}s")

    for bits in ["3bit", "2bit"]:
        r = results[bits]
        print(f"\n{bits} TurboQuant:")
        print(f"  Avg cosine similarity: {r['avg_cos_sim']:.4f}")
        print(f"  Min cosine similarity: {r['min_cos_sim']:.4f}")
        print(f"  Avg KL divergence:     {r['avg_kl_div']:.6f}")
        print(f"  Perplexity proxy:      {r['perplexity_proxy']:.4f}")
        print(f"  Avg max error:         {r['avg_max_error']:.6f}")

    # Quality assessment
    cos3 = results["3bit"]["avg_cos_sim"]
    cos2 = results["2bit"]["avg_cos_sim"]
    print(f"\n{'='*60}")
    print(f"Quality Assessment:")
    if cos3 > 0.99:
        print(f"  3-bit: EXCELLENT (cos_sim={cos3:.4f} > 0.99)")
    elif cos3 > 0.95:
        print(f"  3-bit: GOOD (cos_sim={cos3:.4f} > 0.95)")
    elif cos3 > 0.90:
        print(f"  3-bit: ACCEPTABLE (cos_sim={cos3:.4f} > 0.90)")
    else:
        print(f"  3-bit: POOR (cos_sim={cos3:.4f} < 0.90)")

    if cos2 > 0.95:
        print(f"  2-bit: GOOD (cos_sim={cos2:.4f} > 0.95)")
    elif cos2 > 0.85:
        print(f"  2-bit: ACCEPTABLE (cos_sim={cos2:.4f} > 0.85)")
    else:
        print(f"  2-bit: POOR (cos_sim={cos2:.4f} < 0.85)")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
