# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Replay saved native layers through the public ATOM library (TP4)."""

import argparse
import copy
import json
import os
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist

from atom.models.minimax_m3.mono.library import (
    AtomM3Mono,
    CacheSpec,
    LayerSpec,
    StepMetadata,
    TPContext,
)


def error(a, b):
    a, b = a.float(), b.float()
    return {
        "relative_l2": float((a - b).norm() / b.norm().clamp_min(1e-30)),
        "cosine": float(
            torch.nn.functional.cosine_similarity(a.flatten(), b.flatten(), dim=0)
        ),
        "max_abs": float((a - b).abs().max()),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--directory", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--layer-id", type=int, default=3)
    p.add_argument("--k-scale-factor", type=float, default=1.0)
    p.add_argument("--v-scale-factor", type=float, default=1.0)
    args = p.parse_args()
    rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(rank)
    dist.init_process_group("gloo", timeout=timedelta(seconds=300))
    dev = torch.device("cuda", rank)
    directory = args.directory / f"rank-{rank}"
    weights = {
        k: v.to(dev)
        for k, v in torch.load(directory / "weights.pt", weights_only=True).items()
    }
    cfg = json.loads((directory / "info.json").read_text())["config"]
    ep = "block_sparse_moe.experts.routed_experts."

    def projection(name, shape):
        weight = weights[f"self_attn.{name}.weight"]
        if tuple(weight.shape) == shape[::-1]:
            weight = weight.t()
        if weight.dtype != torch.float8_e4m3fn:
            raise ValueError(
                "Capture native online PTPC weights; BF16 captures are unsupported"
            )
        return weight, weights[f"self_attn.{name}.weight_scale"].view(-1)

    spec = LayerSpec(
        args.layer_id,
        weights["input_layernorm.weight"],
        *projection("qkv_proj", (2560, 6144)),
        *[
            weights[f"self_attn.{k}_norm.weight"]
            for k in ("q", "k", "index_q", "index_k")
        ],
        weights["cos_sin"].bfloat16(),
        *projection("o_proj", (6144, 2048)),
        weights["post_attention_layernorm.weight"],
        weights["block_sparse_moe.gate.weight"],
        weights["block_sparse_moe.e_score_correction_bias"],
        *[
            weights[ep + k]
            for k in ("w13_weight", "w13_weight_scale", "w2_weight", "w2_weight_scale")
        ],
        cfg["rms_norm_eps"],
        cfg["routed_scaling_factor"],
        cfg["swiglu_limit"],
    )
    files = sorted(directory.glob("case-*.pt"))
    # Disjoint offsets exercise the separate main/index address spaces.
    main_cache = torch.zeros((512, 2, 128, 128), dtype=torch.float8_e4m3fn, device=dev)
    index_cache = torch.zeros((512, 128, 128), dtype=torch.float8_e4m3fn, device=dev)
    weights["k_scale"].mul_(args.k_scale_factor)
    weights["v_scale"].mul_(args.v_scale_factor)
    cache = CacheSpec(
        args.layer_id,
        main_cache,
        main_cache.view(-1, 1, 8, 16, 16),
        main_cache.view(-1, 1, 1, 128, 16)[8:],
        index_cache,
        weights["k_scale"],
        weights["v_scale"],
    )
    runtime = AtomM3Mono([spec], [cache], TPContext(dist.group.WORLD, rank, 4, dev))
    records = []
    cases = [(file.name, torch.load(file, weights_only=True)) for file in files]
    single = copy.copy(next(c for name, c in cases if c["hidden"].shape[0] == 4))
    for key in (
        "hidden",
        "residual",
        "positions",
        "output",
        "residual_out",
        "main_slots",
        "index_slots",
    ):
        single[key] = single[key][:1]
    single["seq_lens"] = single["seq_lens"] - 3
    single["query_len"] = 1
    cases.append(("derived-single-row", single))
    for case_name, c in cases:
        n = c["hidden"].shape[0]
        if bool((c["main_slots"][:n] < 0).any()) or bool(
            (c["index_slots"][:n] < 0).any()
        ):
            raise ValueError(
                "Supply an all-live capture; this harness injects padding itself"
            )
        qlen = c["query_len"]
        reqs = n // qlen
        for offset in (0, 1):
            # Request tables remain fixed-address across graph replays.
            mt = torch.zeros((reqs, 128), dtype=torch.int32, device=dev)
            it = torch.zeros_like(mt)
            lens = c["seq_lens"].to(dev)
            slots = torch.empty(n, dtype=torch.int64, device=dev)
            islots = torch.empty_like(slots)
            h = c["hidden"].to(dev)
            res = c["residual"].to(dev)
            pos = c["positions"].to(dev)
            md = StepMetadata(mt, it, lens, slots, islots, n, qlen)

            def fill(shift, c=c, mt=mt, it=it, n=n, slots=slots, islots=islots):
                mbase, ibase = 7 + shift, 101 + shift
                main_cache.view(torch.uint8).fill_(127)
                index_cache.view(torch.uint8).fill_(127)
                payload = c["main_cache"].to(dev).view(torch.float8_e4m3fn)
                payload[:, 0] = (payload[:, 0].float() / args.k_scale_factor).to(
                    payload.dtype
                )
                payload[:, 1] = (payload[:, 1].float() / args.v_scale_factor).to(
                    payload.dtype
                )
                main_cache[mbase : mbase + payload.shape[0]].copy_(payload)
                index_cache[ibase : ibase + c["index_cache"].shape[0]].copy_(
                    c["index_cache"].to(dev)
                )
                width = (int(c["seq_lens"].max()) + 127) // 128
                mt[:, :width] = (
                    (
                        (
                            torch.searchsorted(
                                c["main_ids"], c["main_table"][:, :width].long()
                            )
                            + mbase
                        )
                        * 2
                    )
                    .int()
                    .to(dev)
                )
                it[:, :width] = (
                    (
                        torch.searchsorted(
                            c["index_ids"], c["index_table"][:, :width].long()
                        )
                        + ibase
                    )
                    .int()
                    .to(dev)
                )
                ms, ins = c["main_slots"][:n], c["index_slots"][:n]
                slots.copy_(
                    (
                        (torch.searchsorted(c["main_ids"], ms // 128) + mbase) * 256
                        + ms % 128
                    ).to(dev)
                )
                islots.copy_(
                    (
                        (torch.searchsorted(c["index_ids"], ins // 128) + ibase) * 128
                        + ins % 128
                    ).to(dev)
                )

            def forward(md=md, h=h, res=res, pos=pos):
                runtime.prepare_step(md)
                return runtime.forward_layer(args.layer_id, h, res, pos)

            fill(offset)
            runtime.prepare_step(
                md
            )  # A prior step may have stopped before its first layer.
            out, rout = forward()
            ref, rr = out.clone(), rout.clone()
            torch.cuda.synchronize()
            assert bool(torch.isfinite(ref).all()), case_name
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                gout, gr = forward()
            for shift in (offset, offset + 3):
                fill(shift)
                graph.replay()
                torch.cuda.synchronize()
                torch.testing.assert_close(gout, ref, atol=0, rtol=0)
                torch.testing.assert_close(gr, rr, atol=0, rtol=0)
            # Padding must not contribute experts or retain previous step's tags.
            padding_checked = False
            if n >= 8:
                slots[4:] = -1
                islots[4:] = -1
                lens[1:] = 0
                h[4:] = 0
                res[4:] = 0
                graph.replay()
                torch.cuda.synchronize()
                torch.testing.assert_close(gout[:4], ref[:4], atol=0, rtol=0)
                live = gout[:4].clone()
                lr = gr[:4].clone()
                h[4:] = float("nan")
                res[4:] = float("nan")
                graph.replay()
                torch.cuda.synchronize()
                torch.testing.assert_close(gout[:4], live, atol=0, rtol=0)
                torch.testing.assert_close(gr[:4], lr, atol=0, rtol=0)
                padding_checked = True
            graph.reset()
            records.append(
                {
                    "case": case_name,
                    "offset": offset,
                    "output": error(ref, c["output"].to(dev)),
                    "residual": error(rr, c["residual_out"].to(dev)),
                    "k_scale_factor": args.k_scale_factor,
                    "v_scale_factor": args.v_scale_factor,
                    "graph_equal": True,
                    "poisoned_padding_equal": padding_checked,
                }
            )
            print(rank, records[-1], flush=True)
    runtime.close()
    runtime.close()
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / f"rank-{rank}.json").write_text(json.dumps(records, indent=2) + "\n")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
