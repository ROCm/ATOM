# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Compare hi pooling to ATOM's actual fused-compress Triton entry, same inputs."""

import json
from types import SimpleNamespace

import pytest
import torch

from atom.models.deepseek_v41.mono.hi.blocks.base import Channel, Fmt

pytest.importorskip("flydsl", reason="the hi compressor emits FlyDSL programs")
from atom.model_ops.v4_kernels.fused_compress import fused_compress_attn
from atom.models.deepseek_v41.mono.hi.blocks.gfx950.compressor import Compressor
from atom.models.deepseek_v41.mono.hi.device.gfx950 import mailbox as mb
from atom.models.deepseek_v41.mono.hi.device.gfx950 import paged
from atom.models.deepseek_v41.mono.hi.device.gfx950.ops import ld_i32
from atom.models.deepseek_v41.mono.hi.plan.program import Program, Scenario, Stage
from atom.models.deepseek_v41.mono.hi.runtime.kernel import MonoKernel


def decode(x):
    q = x[..., :512].contiguous().view(torch.float8_e4m3fn).float()
    s = (x[..., 512:].int() << 23).view(torch.float32)
    return (q.reshape(-1, 16, 32) * s.reshape(-1, 16, 1)).reshape(-1, 512)


def metrics(reference, value):
    a, b = decode(reference), decode(value)
    return {
        "byte_equal": bool(torch.equal(reference, value)),
        "different_bytes": int((reference != value).sum()),
        "byte_match": float((reference == value).float().mean()),
        "scale_equal": bool(torch.equal(reference[..., 512:], value[..., 512:])),
        "finite": bool(torch.isfinite(a).all() and torch.isfinite(b).all()),
        "rel_l2": float((a - b).norm() / a.norm().clamp_min(1e-30)),
        "cos": float(
            torch.nn.functional.cosine_similarity(a.flatten(), b.flatten(), dim=0)
        ),
    }


def build(ratio, native_pool, sc):
    comp = Compressor(
        "comp",
        Channel("x", 5120, Fmt.BF16),
        ratio,
        Channel("entry", 512, Fmt.BF16),
        "pos",
        "rope",
        "cache",
        128,
        kv_state="kv_state" if ratio == 2 else "",
        score_state="score_state" if ratio == 2 else "",
        slots=7,
        kslices=1,
        qat=False,
        kv8=True,
        native_pool=native_pool,
    )
    plan = comp.plan(sc)
    parts = [(plan.kv_part, "partial")] + (
        [(plan.score_part, "scores")] if ratio == 2 else []
    )

    def feed(k, row):
        for name, arg in parts:
            word = ld_i32(k.args[arg], row * 512 + k.tid)
            mb.put(k.mb[name], row * 512 + k.tid, word, k.tag)

    writes = tuple(
        region
        for name, _ in parts
        for region in Channel(name, 512, Fmt.F32).regions(sc.rows, ("feed",))
    )
    stages = [
        Stage(
            "feed",
            sc.rows,
            feed,
            writes=writes,
            args=tuple(x[1] for x in parts),
            key=(tuple(parts),),
        )
    ]
    stages.append(comp.stages(sc)[-1])
    kernel = MonoKernel(
        Program(
            f"pool_numeric_r{ratio}_f32{int(native_pool)}",
            sc,
            stages,
            sgpr_spill_ok=True,
        )
    )
    return kernel, plan


def run_cases(out):
    args = SimpleNamespace(out=out)
    results = []
    nb, rq = 2, 6
    sc = Scenario(nb, rq)
    for ratio in (1, 2):
        built = {f: build(ratio, f, sc) for f in (False, True)}
        for seed in range(3):
            torch.manual_seed(781 + seed)
            partial = torch.randn(sc.rows, 512, device="cuda")
            partial *= (
                torch.exp(torch.randn(sc.rows, 16, 1, device="cuda") * 2)
                .expand(-1, -1, 32)
                .reshape(sc.rows, 512)
            )
            partial *= 2.0 ** (-20 if seed == 1 else 20 if seed == 2 else 0)
            if ratio == 1:
                partial = partial.bfloat16().float()
            scores = torch.randn_like(partial) * 5
            kv_state = torch.randn(nb, 7, 512, device="cuda")
            score_state = torch.randn_like(kv_state) * 5
            pos = torch.tensor([11, 22], dtype=torch.int32, device="cuda")
            gamma = (torch.randn(512, device="cuda") * 0.2 + 1).bfloat16()
            angle = torch.randn(256, 32, device="cuda")
            cosine, sine = angle.cos().bfloat16(), angle.sin().bfloat16()
            ptab = torch.arange(nb, dtype=torch.int32, device="cuda").view(nb, 1)
            slots = torch.arange(nb, dtype=torch.int32, device="cuda")
            req = torch.zeros(paged.REQ_WORDS, dtype=torch.int32, device="cuda")
            req[paged.GEO : paged.GEO + 3] = torch.tensor(
                [1, 128, 7 * 512], dtype=torch.int32, device="cuda"
            )
            req[paged.BID : paged.BID + nb] = slots
            req[paged.SLOT : paged.SLOT + nb] = slots
            req[paged.LIVE : paged.LIVE + nb] = 1
            req[paged.ROW_LIVE : paged.ROW_LIVE + sc.rows] = 1
            req[paged.ROW_LIVE + sc.rows - 1] = 0
            plan_rows = []
            indices = []
            for b, start in enumerate(pos.cpu().tolist()):
                for j in range(rq):
                    if (start + j + 1) % ratio == 0 and b * rq + j != sc.rows - 1:
                        plan_rows.append(
                            [b * rq + j, b, start + j, max(0, ratio - j - 1)]
                        )
                        indices.append(b * 128 + (start + j) // ratio)
            compress_plan = torch.tensor(plan_rows, dtype=torch.int32, device="cuda")
            reference = torch.full(
                (nb, 128, 528), 173, dtype=torch.uint8, device="cuda"
            )
            fused_compress_attn(
                kv_in=partial,
                score_in=scores if ratio == 2 else partial,
                kv_state=kv_state,
                score_state=score_state,
                plan=SimpleNamespace(
                    compress_plan_gpu=compress_plan, num_compress=len(plan_rows)
                ),
                state_slot_mapping=slots,
                ape=torch.zeros(ratio, 512, device="cuda"),
                rms_weight=gamma,
                rms_eps=1e-20,
                cos_cache=cosine,
                sin_cache=sine,
                kv_cache=reference,
                block_tables=ptab,
                k_per_block=128,
                overlap=False,
                ratio=ratio,
                head_dim=512,
                rope_head_dim=64,
                quant_mode="packed_fp8",
            )
            wanted = reference.view(-1, 528)[indices]
            entry = {"ratio": ratio, "seed": seed, "positions": pos.cpu().tolist()}
            for f, (kernel, plan) in built.items():
                output = torch.full_like(reference, 173)
                table = kernel.table(
                    dict(
                        partial=partial,
                        scores=scores,
                        pos=pos,
                        rope=cosine,
                        cache=output,
                        req=req,
                        ptab=ptab,
                        kv_state=kv_state.clone(),
                        score_state=score_state.clone(),
                        **{"rope.sin": sine, plan.aux: gamma},
                    )
                )
                buffers = kernel.buffers(uncached=True)
                kernel.launch(table, buffers)
                torch.cuda.synchronize()
                key = "candidate" if f else "baseline"
                entry[key] = metrics(wanted, output.view(-1, 528)[indices])
                untouched = torch.ones(nb * 128, dtype=torch.bool, device="cuda")
                untouched[indices] = False
                assert bool(
                    (output.view(-1, 528)[untouched] == 173).all()
                ), "unexpected pool write"
                assert entry[key]["finite"]
            results.append(entry)
            args.out.write_text(json.dumps(results, indent=2) + "\n")
            print(json.dumps(entry), flush=True)
    assert all(r["candidate"]["byte_equal"] for r in results)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="gfx950 GPU required")
def test_fp32_pool_matches_native_fp8(tmp_path):
    if not torch.cuda.get_device_properties(0).gcnArchName.startswith("gfx950"):
        pytest.skip("gfx950")
    run_cases(tmp_path / "pooling_results.json")
