"""Microbench: do N separately-captured cudagraphs (sharing one pool) REUSE a
transient q_sa across each other, or reserve one per graph? And does handing them
ONE shared persistent q_sa buffer (out=) reduce reserved + stay correct?

Mirrors the AF-segmented attn pattern: per-layer q_sa = x@W1 (transient), then
out = q_sa@W2 (the layer's kept output). N layers = N separate graphs, shared pool.
"""
import torch

def run(share, capture_style, N=16, T=3072, D=1024, H=2048, dev="cuda"):
    torch.cuda.empty_cache(); torch.cuda.synchronize()
    pool = torch.cuda.graph_pool_handle()
    x = torch.randn(T, D, device=dev, dtype=torch.bfloat16)          # shared input
    W1 = [torch.randn(D, H, device=dev, dtype=torch.bfloat16) for _ in range(N)]
    W2 = [torch.randn(H, D, device=dev, dtype=torch.bfloat16) for _ in range(N)]
    refs = [(x @ W1[i]) @ W2[i] for i in range(N)]                   # eager reference
    qsa_shared = torch.empty(T, H, device=dev, dtype=torch.bfloat16) if share else None

    torch.cuda.synchronize(); r0 = torch.cuda.memory_reserved()
    graphs, outs = [], []
    for i in range(N):
        def body():
            qsa = qsa_shared if share else torch.empty(T, H, device=dev, dtype=torch.bfloat16)
            torch.matmul(x, W1[i], out=qsa)
            return qsa @ W2[i]
        if capture_style == "torch":
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g, pool=pool):
                o = body()
        else:  # raw capture_begin/end (what SegmentedCudaGraphCapture uses)
            g = torch.cuda.CUDAGraph()
            g.capture_begin(pool=pool, capture_error_mode="thread_local")
            o = body()
            g.capture_end()
        graphs.append(g); outs.append(o)
    torch.cuda.synchronize(); r1 = torch.cuda.memory_reserved()

    for g in graphs: g.replay()
    torch.cuda.synchronize()
    maxerr = max((outs[i].float() - refs[i].float()).abs().max().item() for i in range(N))
    print(f"  share={str(share):5} style={capture_style:5} | reserved grew "
          f"{(r1-r0)/1e6:7.1f}MB (N={N}) | maxerr={maxerr:.4f}")

if __name__ == "__main__":
    print("q_sa[3072,2048]bf16=12.6MB/graph, out[3072,1024]bf16=6.3MB/graph")
    for style in ("torch", "raw"):
        run(False, style)
        run(True, style)
