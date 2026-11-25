"""Check all five prologue outputs, including graph replay on changed anchors.

Run on an idle, locked ROCm GPU:
  python tests/model_ops/check_fused_draft_prologue_gpu.py
Synthetic embeddings; no checkpoint or full-server accuracy claim.
"""

import torch
import torch.nn.functional as F

from atom.model_ops.deepseek_v41.dspark import (
    build_block_state,
    draft_step_indices,
    fused_draft_prologue,
)


def main():
    torch.manual_seed(0)
    vocab, hidden, width, hc, window = 129280, 5120, 5, 4, 128
    ring = window + width
    embed = torch.randn(vocab, hidden, device="cuda", dtype=torch.bfloat16)
    noise = embed[7].clone()
    cases = 0
    for batch in (0, 1, 11, 12, 13, 14, 15, 16):
        positions = torch.zeros(batch, device="cuda", dtype=torch.int64)
        ids = torch.zeros(batch, device="cuda", dtype=torch.int32)

        def native(positions=positions, ids=ids):
            return (
                *draft_step_indices(positions, ring, window, width),
                *build_block_state(F.embedding(ids, embed), noise, width, hc),
            )

        def fused(positions=positions, ids=ids):
            out = fused_draft_prologue(
                positions, ids, embed, noise, ring, window, width, hc
            )
            return (*out[:3], *out[3])

        def equal(a, b):
            assert len(a) == len(b) == 5
            for x, y in zip(a, b):
                assert x.shape == y.shape and x.dtype == y.dtype
                assert torch.equal(x, y)

        equal(native(), fused())
        if not batch:
            continue
        # Compile and warm both paths before capture, then mutate staged inputs.
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                native()
                fused()
        torch.cuda.current_stream().wait_stream(stream)
        graph_native, graph_fused = torch.cuda.CUDAGraph(), torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph_native):
            captured_native = native()
        with torch.cuda.graph(graph_fused):
            captured_fused = fused()
        for low, high in ((0, 5), (0, 300), (126, 140), (100000, 300000)):
            positions.copy_(torch.randint(low, high, (batch,), device="cuda"))
            ids.copy_(
                torch.randint(0, vocab, (batch,), device="cuda", dtype=torch.int32)
            )
            ids[0] = 0
            ids[-1] = vocab - 1
            equal(native(), fused())
            graph_native.replay()
            graph_fused.replay()
            torch.cuda.synchronize()
            equal(captured_native, captured_fused)
            equal(native(), captured_fused)
            cases += 1
    print(f"PASS: {cases} changed-input eager/graph cases plus empty batch")


if __name__ == "__main__":
    main()
