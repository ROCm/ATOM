"""Current local-context Mono vs native, with explicit unsupported widths."""

import argparse
import gzip
import hashlib
import json
import os
import statistics
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import common
import numpy as np
import torch
import torch.distributed as dist
from aiter.dist.parallel_state import (
    destroy_distributed_environment,
    destroy_model_parallel,
    get_tp_group,
    init_distributed_environment,
    initialize_model_parallel,
)

from atom.config import (
    Config,
    QuantizationConfig,
    flatten_dsa_config,
    set_current_atom_config,
)
from atom.model_ops.fp4_mqa_ragged_metadata import Fp4MqaRaggedMetadata
from atom.model_ops.indexer.rows import fp4_rows, fp8_rows
from atom.model_ops.sparse_indexer_chunk import LogitsWorkspace
from atom.models.deepseek_v2 import Indexer, get_rope, is_neox_rope_style
from atom.mono.dsa_indexer import kernel as kix
from atom.mono.dsa_indexer.bind import IndexerHook, shape_of
from atom.mono.dsa_indexer.runner import IndexerRunner
from atom.mono.runtime.compile import launcher_resources
from atom.utils.forward_context import AttentionMetaData, Context, set_forward_context

ROOT = Path(__file__).resolve().parent
MODEL = None
PREFIX = "model.layers.0.self_attn.indexer"


def gather(value, world):
    result = [None] * world
    dist.all_gather_object(result, value, group=get_tp_group().cpu_group)
    return result


def overlap(first, second, world):
    matched = sum(len(set(a.tolist()) & set(b.tolist())) for a, b in zip(first, second))
    counts = gather((matched, sum(v.numel() for v in first)), world)
    return sum(v[0] for v in counts) / sum(v[1] for v in counts)


@torch.inference_mode()
def run_case(args, cfg, weights, device, rank, world, batch, context):
    qlen, page, dcp = args.qlen, args.page, args.dcp
    rows = batch * qlen
    text_config = getattr(cfg.hf_config, "text_config", cfg.hf_config)
    output = ROOT / "runs" / args.tag / f"bs{batch}_q{qlen}_c{context}"
    output.mkdir(parents=True, exist_ok=True)
    module = Indexer(
        cfg,
        text_config,
        text_config.hidden_size,
        text_config.q_lora_rank,
        cfg.quant_config,
        "fp8",
        prefix=PREFIX,
        external_k_weights=True,
    ).to(device)
    if module.rope_dim:
        params = text_config.rope_parameters
        assert params.get("factor", 1.0) in (None, 1.0)
        module.rotary_emb = get_rope(
            text_config.qk_rope_head_dim,
            rotary_dim=text_config.qk_rope_head_dim,
            max_position=text_config.max_position_embeddings,
            base=params.get("rope_theta") or 10000,
            rope_scaling=None,
            is_neox_style=is_neox_rope_style(
                text_config, "indexer_rope_interleave", default_is_neox=True
            ),
        ).to(device)
    assert (module.n_head, module.head_dim, module.topk_tokens) == (32, 128, 2048)
    for name in ("wq_b.weight", "k_norm.weight", "k_norm.bias"):
        module.get_parameter(name).copy_(weights[name])
    module.wq_b.process_weights_after_loading()
    assert (
        module.wq_b.weight.dtype == torch.bfloat16 and module.wq_b.quant_type.value == 0
    )
    assert module._indexer_fp4 == (args.cache == "fp4")
    torch.manual_seed(2701)
    hidden = torch.randn(rows, text_config.hidden_size, device=device)
    qr = torch.randn(rows, module.q_lora_rank, device=device)
    kw = torch.nn.functional.linear(
        hidden,
        torch.cat((weights["wk.weight"], weights["weights_proj.weight"])).bfloat16(),
    )
    k, hw = kw[:, :128], kw[:, 128:]
    columns = -(-context // (page * dcp))
    cache_rank = rank if dcp > 1 else 0
    cscale = None
    if args.cache == "fp4":
        if dcp > 1:
            torch.manual_seed(42 + rank)
        historical = torch.randn(batch * columns, 64, 128, device=device)
        hq, hs = common.quant(historical)
        cache = hq.reshape(-1, 64, 4, 16).permute(0, 2, 1, 3).contiguous()
        cscale = common.scales(hs, 64)
        module.k_cache.kv_cache_scale = cscale
        assert module._index_cache_layout == "mfma"
        del historical, hq, hs
    else:
        cache = common.make_cache(
            batch * columns, cache_rank, device, shape_of(module, page).row_bytes
        )
    generator = torch.Generator(device=device).manual_seed(42 + cache_rank)
    tables = (
        torch.randperm(batch * columns, generator=generator, device=device)
        .int()
        .view(batch, columns)
    )
    tables_cpu = tables.cpu().long()
    reverse = torch.full((batch * columns,), -1, dtype=torch.long)
    reverse[tables_cpu.flatten()] = torch.arange(batch * columns)
    contexts = torch.full((batch,), context, device=device, dtype=torch.int32)
    bids = torch.arange(batch, device=device, dtype=torch.int32).repeat_interleave(qlen)
    positions = (
        contexts.long()[:, None] - qlen + torch.arange(qlen, device=device)
    ).flatten()
    ends = (positions + 1).int()
    local_pos = positions // dcp
    slots = tables[bids.long(), local_pos // page].long() * page + local_pos % page
    if dcp > 1:
        slots.masked_fill_(positions % dcp != rank, -1)
    ptr = torch.arange(rows + 1, device=device, dtype=torch.int32) * 2048
    module.k_cache.kv_cache = [cache]
    module.sparse_kv_indices_buffer = torch.full(
        (rows * 2048,), -7, dtype=torch.int32, device=device
    )
    module.dcp_sparse_kv_indptr_buffer = torch.full(
        (rows + 1,), -7, dtype=torch.int32, device=device
    )
    module.dcp_owned_counts_buffer = torch.full(
        (rows,), -7, dtype=torch.int32, device=device
    )
    workspace = LogitsWorkspace.for_rows(256, cfg.max_model_len, device)
    starts = torch.arange(batch + 1, device=device, dtype=torch.int32) * qlen
    metadata = AttentionMetaData(
        slot_mapping=slots,
        context_lens=contexts,
        block_tables=tables,
        max_seqlen_q=qlen,
        max_seqlen_k=context,
        cu_seqlens_q=starts,
        kv_indptr=torch.arange(batch + 1, device=device, dtype=torch.int32)
        * columns
        * page,
        kv_indices=(
            tables.reshape(-1, 1) * page
            + torch.arange(page, device=device, dtype=torch.int32)
        ).flatten(),
        sparse_kv_indptr=ptr,
        batch_id_per_q_token=bids,
        sparse_cu_seqlens_q=torch.arange(rows + 1, device=device, dtype=torch.int32),
        dcp_local_context_lens=(
            (ends // dcp + ((ends % dcp) > rank).int()) if dcp > 1 else None
        ),
    )
    metadata.token_block_tables = tables.repeat_interleave(qlen, dim=0)
    metadata.index_rows = fp8_rows(contexts, context, workspace, tables, next_n=qlen)
    if args.cache == "fp4":
        metadata.index_rows = fp4_rows(
            ends,
            context,
            workspace,
            Fp4MqaRaggedMetadata(starts, qlen, tables),
            heads=32,
            page_size=page,
            layout="mfma",
        )
    step = Context(
        positions=positions,
        scheduled_bs=batch,
        scheduled_tokens=rows,
        running_bs=batch,
        running_tokens=rows,
    )
    set_forward_context(metadata, cfg, step, np.array([rows], dtype=np.int32))
    hook = IndexerHook({0: module})
    runner = IndexerRunner.alone(
        SimpleNamespace(
            layers=[SimpleNamespace(self_attn=SimpleNamespace(indexer=module))]
        ),
        hook,
        SimpleNamespace(index_logits_workspace=workspace),
    )
    graphs = {}
    try:
        key = runner.ix.key(rows, qlen)
        supported = runner.prepare(rows, qlen)
        refusal = kix.unfit(key)
        assert supported, (batch, rows, refusal)
        arms = ("native", "mono") if supported else ("native",)

        def call(arm):
            with runner.serving(rows, qlen) if arm == "mono" else nullcontext():
                module(hidden, qr, None, positions, projected_k_weights=(k, hw))

        def selected():
            offsets = (
                (module.dcp_sparse_kv_indptr_buffer if dcp > 1 else ptr).cpu().long()
            )
            indices = module.sparse_kv_indices_buffer.cpu().long()
            counts = offsets[1:] - offsets[:-1]
            assert (
                offsets[0] == 0
                and (counts >= 0).all()
                and offsets[-1] <= indices.numel()
            )
            if dcp > 1:
                assert torch.equal(counts, module.dcp_owned_counts_buffer.cpu().long())
                total = torch.stack(gather(counts, world)).sum(0)
                assert (total == 2048).all(), total
            else:
                assert (counts == 2048).all()
                assert (module.dcp_sparse_kv_indptr_buffer == -7).all()
                assert (module.dcp_owned_counts_buffer == -7).all()
            values = []
            for r in range(rows):
                value = indices[offsets[r] : offsets[r + 1]].sort().values.clone()
                assert value.unique().numel() == value.numel()
                assert (value >= 0).all() and (value < batch * columns * page).all()
                logical_page = reverse[value // page]
                assert (logical_page // columns == r // qlen).all(), (
                    "wrong request",
                    r,
                )
                logical_key = (
                    logical_page % columns * page + value % page
                ) * dcp + cache_rank
                assert (logical_key <= int(positions[r])).all(), ("future key", r)
                values.append(value)
            return values

        expected = {}
        for arm in arms:
            for _ in range(3):
                call(arm)
            common.sync()
            expected[arm] = selected()
            if arm == "native":
                native_cache = cache.clone()
                native_scale = cscale.clone() if cscale is not None else None
        assert torch.equal(native_cache, cache)
        if cscale is not None:
            assert torch.equal(native_scale, cscale)
        del native_cache, native_scale
        recall = same_q_recall = None
        q_diagnostic = None
        if supported:
            recall = overlap(expected["native"], expected["mono"], world)
            offset, size = kix.plain_layout(key)["iqb"]
            mono_q = (
                runner.peers.plain[offset : offset + size]
                .view(torch.bfloat16)
                .reshape(rows, 4096)
                .clone()
            )
            native_q = module.wq_b(qr, None)
            q_diagnostic = {
                "equal_fraction": (native_q == mono_q).float().mean().item(),
                "max_abs_difference": (native_q.float() - mono_q.float())
                .abs()
                .max()
                .item(),
            }
            original_forward = module.wq_b.forward
            try:
                module.wq_b.forward = lambda *_args, **_kwargs: mono_q
                call("native")
                common.sync()
                same_q = selected()
            finally:
                module.wq_b.forward = original_forward
            same_q_recall = overlap(same_q, expected["mono"], world)
            assert same_q_recall >= 0.9999, (recall, same_q_recall, q_diagnostic)

        for arm in arms:
            graphs[arm] = common.capture(lambda arm=arm: call(arm))
            actual = selected()
            assert all(torch.equal(a, b) for a, b in zip(expected[arm], actual)), (
                arm,
                "capture changed output",
            )
        independence = {}
        if args.boundary and dcp == 1 and context == 8192:
            saved_peer = runner.peers.bytes.clone()
            saved_state = runner.mailboxes._state.clone()
            slab_at, slab_size = kix.plain_layout(key)["slab"]
            slab = runner.peers.plain[slab_at : slab_at + slab_size]
            saved_slab = slab.clone()
            common.sync()
            if rank == 0:
                saved_addresses = runner.peers.addresses.clone()
                runner.peers.addresses[1:].zero_()
                module.sparse_kv_indices_buffer.fill_(-7)
                try:
                    for _ in range(3):
                        graphs["mono"].replay()
                    torch.cuda.synchronize()
                    actual = selected()
                    assert all(
                        torch.equal(a, b) for a, b in zip(expected["mono"], actual)
                    )
                finally:
                    runner.peers.addresses.copy_(saved_addresses)
                    torch.cuda.synchronize()
            common.sync()
            assert torch.equal(saved_peer, runner.peers.bytes)
            assert torch.equal(saved_state, runner.mailboxes._state)
            assert torch.equal(saved_slab, slab)
            independence = {
                "rank0_alone_remote_addresses_null": True,
                "mailbox_unchanged": True,
                "epoch_unchanged": True,
                "slab_unchanged": True,
            }
        timings = []
        for repeat in range(args.repeats):
            order = arms if repeat % 2 == 0 else arms[::-1]
            timings.append(
                {arm: common.time_graph(graphs[arm], args.replays) for arm in order}
            )
        median = {
            arm: statistics.median(t[arm]["max_rank_us"] for t in timings)
            for arm in arms
        }
        traces = {}
        for arm in arms:
            for _ in range(3):
                graphs[arm].replay()
            common.sync()
            actual = selected()
            assert all(torch.equal(a, b) for a, b in zip(expected[arm], actual)), (
                arm,
                "replay changed output",
            )
            path = output / f"{arm}_rank{rank}.json"
            common.profile_graph(graphs[arm], path, 1)
            with gzip.open(str(path) + ".gz", "rt") as handle:
                kernels = [
                    e
                    for e in json.load(handle)["traceEvents"]
                    if e.get("cat") == "kernel"
                ]
            names = [e["name"] for e in kernels]
            mono_names = [name for name in names if "mono_dsa_indexer_" in name]
            assert len(mono_names) == (common.GRAPH_STEPS if arm == "mono" else 0)
            assert all(
                f"_s{rows}_" in name
                and f"_q{qlen}_" in name
                and f"_rep{int(dcp == 1)}_" in name
                for name in mono_names
            )
            traces[arm] = {
                "path": str(path) + ".gz",
                "kernels": len(names),
                "mono_calls": len(mono_names),
                "kernel_names": sorted(set(names)),
            }
        transitions = []
        if args.boundary and context == 163840:
            patterns = [
                [8192] * batch,
                [8193] * batch,
                [8192] * (batch - 1) + [context],
                [context] * batch,
                [8192] * batch,
            ]
            for lens in patterns:
                contexts.copy_(torch.tensor(lens, device=device, dtype=torch.int32))
                positions.copy_(
                    (
                        contexts.long()[:, None]
                        - qlen
                        + torch.arange(qlen, device=device)
                    ).flatten()
                )
                ends.copy_((positions + 1).int())
                local_pos = positions // dcp
                slots.copy_(
                    tables[bids.long(), local_pos // page].long() * page
                    + local_pos % page
                )
                if dcp > 1:
                    slots.masked_fill_(positions % dcp != rank, -1)
                    metadata.dcp_local_context_lens.copy_(
                        ends // dcp + ((ends % dcp) > rank).int()
                    )
                for arm in arms:
                    call(arm)
                    common.sync()
                    eager = selected()
                    for _ in range(3):
                        graphs[arm].replay()
                        common.sync()
                        actual = selected()
                        assert all(torch.equal(a, b) for a, b in zip(eager, actual)), (
                            arm,
                            lens,
                            "transition",
                        )
                transitions.append(
                    {
                        "lengths": lens,
                        "same_graph": True,
                        "replays": 3,
                        "eager_equal": True,
                    }
                )
        result = {
            "batch": batch,
            "qlen": qlen,
            "rows": rows,
            "context": context,
            "max_context": cfg.max_model_len,
            "tp": world,
            "dcp": dcp,
            "cache": args.cache,
            "page": page,
            "route": "mono" if supported else "native_fallback",
            "refusal": refusal,
            "native_us": median["native"],
            "mono_us": median.get("mono"),
            "latency_reduction_percent": (
                100 * (1 - median["mono"] / median["native"]) if supported else None
            ),
            "beats_native_repeats": (
                sum(
                    t["mono"]["max_rank_us"] < t["native"]["max_rank_us"]
                    for t in timings
                )
                if supported
                else None
            ),
            "cache_bytes_equal": True,
            "causal_and_request_bounds_checked": True,
            "global_topk_counts_checked": True,
            "graph_output_equal": True,
            "topk_overlap": recall,
            "same_q_topk_overlap": same_q_recall,
            "independence": independence,
            "transitions": transitions,
            "q_diagnostic": gather(q_diagnostic, world),
            "timings": timings,
            "traces": gather(traces, world),
            "resources": (
                launcher_resources(runner.widths[rows, qlen]) if supported else []
            ),
            "mono_includes_reset": True,
            "native_dcp_includes_exchange": dcp > 1,
            "checkpoint": str(MODEL),
            "checkpoint_weight_prefix": PREFIX,
            "q_lora": module.q_lora_rank,
            "rope_dim": module.rope_dim,
            "short_gemm": key.short_gemm,
            "online_quant_config": None,
            "repeats": args.repeats,
            "replays": args.replays,
            "graph_steps": common.GRAPH_STEPS,
        }
        if rank == 0:
            (output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
            print(
                "CASE_DONE "
                + json.dumps(
                    {
                        k: v
                        for k, v in result.items()
                        if k not in ("q_diagnostic", "timings", "traces", "resources")
                    }
                ),
                flush=True,
            )
        return result
    finally:
        common.sync()
        for graph in graphs.values():
            graph.reset()
        runner.close()
        hook.remove()


def main():
    global MODEL, ROOT, PREFIX
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--weight-prefix", default=PREFIX)
    parser.add_argument("--output", type=Path, default=Path("dsa_indexer_results"))
    parser.add_argument("--cache", choices=("fp8", "fp4"), required=True)
    parser.add_argument("--dcp", type=int, choices=(1, 8), required=True)
    parser.add_argument(
        "--batches", type=int, nargs="+", default=[1, 8, 16, 24, 32, 40]
    )
    parser.add_argument("--contexts", type=int, nargs="+", default=[8192, 163840])
    parser.add_argument("--qlen", type=int, default=6)
    parser.add_argument("--repeats", type=int, default=9)
    parser.add_argument("--replays", type=int, default=64)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--boundary", action="store_true")
    args = parser.parse_args()
    MODEL, ROOT, PREFIX = args.model, args.output, args.weight_prefix
    common.MODEL, common.PREFIX = MODEL, PREFIX
    args.page = 64 if args.cache == "fp4" else 128
    assert args.qlen == 6 and 0 < min(args.batches) <= max(args.batches) <= 42
    rank, world = int(os.environ["LOCAL_RANK"]), int(os.environ["WORLD_SIZE"])
    assert world == 8
    torch.cuda.set_device(rank)
    torch.set_default_dtype(torch.bfloat16)
    device = torch.device("cuda", rank)
    common.PAGE = args.page
    cfg = Config(
        model=str(MODEL),
        trust_remote_code=True,
        tensor_parallel_size=world,
        decode_context_parallel_size=args.dcp,
        max_num_seqs=max(40, max(args.batches)),
        max_model_len=163840,
        max_num_batched_tokens=8192,
        kv_cache_block_size=args.page,
        kv_cache_dtype="fp8",
        index_cache_dtype=args.cache,
        online_quant_config=None,
        enable_prefix_caching=False,
    )
    checksum = hashlib.sha256((MODEL / "config.json").read_bytes()).hexdigest()
    text_config = getattr(cfg.hf_config, "text_config", cfg.hf_config)
    flatten_dsa_config(text_config)
    # Multimodal wrappers may omit this field and trigger Config's 8K default.
    # The standalone indexer must retain the same 160K capacity in every arm.
    cfg.max_model_len = min(163840, text_config.max_position_embeddings)
    assert cfg.max_model_len == 163840
    if (
        getattr(cfg.hf_config, "quantization_config", None) is None
        and getattr(text_config, "quantization_config", None) is not None
    ):
        cfg.quant_config = QuantizationConfig(text_config, cfg.online_quant_config)
    set_current_atom_config(cfg)
    init_distributed_environment(world_size=world, rank=rank, local_rank=rank)
    initialize_model_parallel(
        tensor_model_parallel_size=world, decode_context_model_parallel_size=args.dcp
    )
    try:
        weights = common.checkpoint_tensors(device)
        result = {
            "arguments": {
                k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()
            },
            "checkpoint_config_sha256": checksum,
            "cases": [],
        }
        for batch in args.batches:
            for context in args.contexts:
                if rank == 0:
                    print(
                        f"START bs={batch} q={args.qlen} context={context} dcp={args.dcp} cache={args.cache}",
                        flush=True,
                    )
                result["cases"].append(
                    run_case(args, cfg, weights, device, rank, world, batch, context)
                )
                if rank == 0:
                    (ROOT / "runs" / args.tag / "result.json").write_text(
                        json.dumps(result, indent=2) + "\n"
                    )
                torch.cuda.empty_cache()
        assert (
            hashlib.sha256((MODEL / "config.json").read_bytes()).hexdigest() == checksum
        )
        if rank == 0:
            print("SWEEP_DONE", flush=True)
    finally:
        destroy_model_parallel()
        destroy_distributed_environment()


if __name__ == "__main__":
    main()
