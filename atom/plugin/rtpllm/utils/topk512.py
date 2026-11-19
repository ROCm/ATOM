"""Small-row ROCm routing: preserve expert ties, shared scores and padding."""
import torch, triton
import triton.language as tl

@triton.jit
def _topk512(G, W, I, V, SG: tl.constexpr, SW: tl.constexpr, SI: tl.constexpr, SHARED: tl.constexpr, HAS_VALID: tl.constexpr):
    row = tl.program_id(0)
    ix = tl.arange(0, 512)
    val = tl.load(G + row * SG + ix).to(tl.float32)
    mx = tl.max(val, 0)
    ids = tl.full((16,), -1, tl.int32)
    scores = tl.full((16,), 0, tl.float32)
    o = tl.arange(0, 16)
    for j in tl.static_range(10):
        best = tl.max(val, 0)
        idx = tl.min(tl.where(val == best, ix, 2147483647), 0)
        ids = tl.where(o == j, idx, ids)
        scores = tl.where(o == j, tl.exp(best - mx), scores)
        val = tl.where(ix == idx, float('-inf'), val)
    scores = scores / tl.sum(scores, 0)
    active = True
    if HAS_VALID:
        active = row < tl.load(V)
    tl.store(W + row * SW + o, tl.where(active, scores, 0), o < 10)
    tl.store(I + row * SI + o, tl.where(active, ids, -1), o < 10)
    if SHARED:
        sh = tl.load(G + row * SG + 512).to(tl.float32)
        tl.store(W + row * SW + 10, tl.sigmoid(sh))

def run(g, w, i, valid=None):
    _topk512[g.shape[0],](g, w, i, valid, g.stride(0), w.stride(0), i.stride(0), g.shape[1] == 513, valid is not None, num_warps=1)
