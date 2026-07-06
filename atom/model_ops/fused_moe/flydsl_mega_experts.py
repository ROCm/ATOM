# SPDX-License-Identifier: Apache-2.0
"""FlyDSL MegaMoE (ghu_moe_stage1) fused EP-MoE integration for ATOM.

ATOM_USE_FLYDSL_FUSED=1 replaces the whole EP experts step (dispatch + gemm1 +
quant + gemm2 + combine) with the new-PR ``MegaMoE`` single op:
  stage1 = FusedMoEMegaStage1 (single-launch dispatch + GEMM1, atom-logical a2)
  stage2 = FlyDSLMoeGemm2CombineOp (GEMM2 epilogue inlines EP combine)

Large-bs (prefill) works via FUSED_MEGA_COMPACT_ATOM=1 (compact dispatch +
atom-logical a2; stage2 unchanged). Validated bit-exact in FlyDSL notes
(2026-06-16) for v4_flash a8w4 bs 8..8192.

Memory: ONE MegaMoE is shared across all MoE layers (process-level cache keyed
by shape/quant/mtpr/tile); per-layer weights are swapped in before forward
(stage1.w1 / self.w2 are runtime pointer args, not baked into the kernel).
"""
from __future__ import annotations

import os
import sys

import torch

# compact+atom combo must be on so large bs (prefill) doesn't trip the
# atom_contract-vs-compact assert in FusedMoEMegaStage1.
os.environ.setdefault("FUSED_MEGA_COMPACT_ATOM", "1")
# ATOM EP holds only this rank's epr experts' weights, so index w1/bias by LOCAL
# expert id (kernel patch in fused_moe_gemm_2stage.py). We pass local-epr w1.
os.environ.setdefault("FUSED_MEGA_W1_LOCAL", "1")

_FLYDSL_KERNELS_PATH = os.environ.get("ATOM_FLYDSL_KERNELS_PATH", "/home/yashao/FlyDSL")
_FP4 = getattr(torch, "float4_e2m1fn_x2", None)
_MEGA_CACHE: dict = {}
_MEGA_DBG_N = 0
_MEGA_PRE_N = 0
_MEGA_BUILD_DBG = False
_MEGA_RAW_SAVED = False
_MEGA_BUILD_CNT = 0
_MEGA_DUMP_LAYERS = {int(v) for v in os.environ.get("ATOM_MEGA_DUMP_LAYERS", "0,30").split(",") if v.strip()}


def _os_env(k):
    return os.environ.get(k, "<unset>")


def _ensure_path():
    if _FLYDSL_KERNELS_PATH and _FLYDSL_KERNELS_PATH not in sys.path:
        sys.path.insert(0, _FLYDSL_KERNELS_PATH)


def _shuffle_fns():
    """The EXACT shuffle ops MegaMoE's reference uses (FlyDSL tests.utils /
    tests.kernels.utils), NOT aiter's shuffle_weight."""
    _ensure_path()
    from tests.utils import shuffle_weight
    from tests.kernels.utils import fp4_utils

    return shuffle_weight, fp4_utils


def build_mega_weights(layer) -> None:
    """From ATOM's RAW (pre-atom-shuffle) mxfp4 w13/w2 + e8m0 scales, build the
    MegaMoE-layout weights and stash on the layer. Must run BEFORE atom's own
    shuffle_weight in process_weights_after_loading (uses raw layout)."""
    shuffle_weight, fp4_utils = _shuffle_fns()

    w13 = layer.w13_weight.data  # [E_local, 2*inter, hidden//2] fp4-packed uint8
    E, two_inter, h_half = w13.shape
    w13f = w13.reshape(E * two_inter, h_half)
    if _FP4 is not None:
        w13f = w13f.view(_FP4)
    layer._mega_w1 = shuffle_weight(w13f).view(torch.uint8).contiguous()

    s1 = layer.w13_weight_scale.data  # [E, 2*inter, hidden//32]
    s1f = s1.reshape(E * s1.shape[1], s1.shape[2])
    layer._mega_w1_scale = fp4_utils.e8m0_shuffle(s1f).view(torch.uint8).contiguous()

    w2 = layer.w2_weight.data  # [E_local, hidden, inter//2] fp4-packed uint8
    E2, hh, i_half = w2.shape
    w2f = w2.reshape(E2 * hh, i_half)
    if _FP4 is not None:
        w2f = w2f.view(_FP4)
    layer._mega_w2 = shuffle_weight(w2f).view(torch.uint8).contiguous().view(-1)

    s2 = layer.w2_weight_scale.data
    s2f = s2.reshape(E2 * s2.shape[1], s2.shape[2])
    layer._mega_w2_scale = (
        fp4_utils.e8m0_shuffle(s2f).view(torch.uint8).contiguous().view(-1)
    )

    # stash RAW fp4 weights+scales (CPU) for a few layers for offline dequant compare
    global _MEGA_RAW_SAVED, _MEGA_BUILD_CNT
    _idx = _MEGA_BUILD_CNT
    _MEGA_BUILD_CNT += 1
    if os.environ.get("ATOM_MEGA_DUMP", "") and _idx in _MEGA_DUMP_LAYERS:
        layer._raw_w13 = w13.detach().cpu()
        layer._raw_w13_scale = s1.detach().cpu()
        layer._raw_w2 = w2.detach().cpu()
        layer._raw_w2_scale = s2.detach().cpu()
        layer._mega_dump_first = True
        layer._mega_layer_idx = _idx

    global _MEGA_BUILD_DBG
    if not _MEGA_BUILD_DBG:
        _MEGA_BUILD_DBG = True
        _w13b = getattr(layer, "w13_bias", None)
        _w2b = getattr(layer, "w2_bias", None)
        print(f"[MEGA-BUILD] w13={tuple(w13.shape)}{w13.dtype} w13_scale={tuple(s1.shape)}{s1.dtype} "
              f"w2={tuple(w2.shape)}{w2.dtype} w2_scale={tuple(s2.shape)}{s2.dtype} | "
              f"_mega_w1={tuple(layer._mega_w1.shape)} _mega_w1_scale={tuple(layer._mega_w1_scale.shape)} "
              f"_mega_w2={tuple(layer._mega_w2.shape)} _mega_w2_scale={tuple(layer._mega_w2_scale.shape)} | "
              f"w13_bias={None if _w13b is None else tuple(_w13b.shape)} "
              f"w2_bias={None if _w2b is None else tuple(_w2b.shape)} "
              f"GU_ITLV={_os_env('ATOM_MOE_GU_ITLV')}", flush=True)


def _next_pow2(n: int) -> int:
    p = 1
    while p < n:
        p <<= 1
    return p


_GEMM2_TILE_TABLE_CACHE: dict = {}


def _se_atom_sort_block() -> int:
    """Single source of truth for the fused gemm2 sort block = the granularity
    MegaMoE stage1 emits sorted_expert_ids (_se_atom) at. Read from the
    megakernel constant (SE_ATOM_SORT_BLOCK) so atom never hardcodes it; the
    fused gemm2 reads _se_atom at this block and its tile_m must divide it."""
    _ensure_path()
    try:
        from kernels.fused_moe_megakernel import SE_ATOM_SORT_BLOCK
        return int(SE_ATOM_SORT_BLOCK)
    except Exception:  # noqa: BLE001
        return 32


def build_gemm2_tile_table(model_dim, inter_dim, local_experts, topk, sblk=None):
    """Per-M GEMM2 tile table {pow2_token: (tile_m,tile_n,tile_k)} for the EP
    shape, parsed from aiter's dsv4 tuned CSV (the same kernelName2 that
    get_2stage_cfgs returns). cktile buckets (no FlyDSL tile) are aligned to the
    nearest FlyDSL bucket (tie -> smaller). Returns None if no CSV/rows so the
    op keeps its single default tile.

    The op (FlyDSLMoeGemm2CombineOp) then picks per forward by nextPow2(
    run_tokens), mirroring aiter's get_2stage_cfgs(get_padded_M(token_num)).

    ``sblk`` is the AUTHORITATIVE stage1 sorted-block (= MegaMoE
    ``stage1.sort_block_m`` = ``max(32, unit_size)``); gemm2 tile_m must divide
    it. Pass the real value from the built MegaMoE so the tuned prefill tile_m
    (e.g. 64) is kept when sblk==64 instead of being clamped to a stale 32.
    Falls back to ``_se_atom_sort_block()`` only when not given.
    """
    import csv as _csv
    import re as _re

    if sblk is None:
        sblk = _se_atom_sort_block()
    sblk = int(sblk)
    key = (model_dim, inter_dim, local_experts, topk, sblk)
    if key in _GEMM2_TILE_TABLE_CACHE:
        return _GEMM2_TILE_TABLE_CACHE[key]
    paths = [
        os.environ.get("AITER_CONFIG_FMOE", ""),
        "/app/aiter-test/aiter/configs/model_configs/dsv4_fp8fp4_tuned_fmoe.csv",
    ]
    csv_path = next((p for p in paths if p and os.path.exists(p)), None)
    if csv_path is None:
        _GEMM2_TILE_TABLE_CACHE[key] = None
        return None
    # gemm2 must sub-tile this block; tile_m | sblk (authoritative, from MegaMoE)
    raw: dict = {}  # token -> (tm,tn,tk) | None(cktile/miss)
    try:
        with open(csv_path, newline="") as f:
            for r in _csv.DictReader(f):
                try:
                    if int(r["model_dim"]) != model_dim or int(r["inter_dim"]) != inter_dim:
                        continue
                    if int(r["expert"]) != local_experts or int(r["topk"]) != topk:
                        continue
                    tok = int(r["token"])
                    kn2 = (r.get("kernelName2", "") or "").strip()
                except (KeyError, ValueError):
                    continue
                mm = _re.search(r"t(\d+)x(\d+)x(\d+)", kn2)
                if kn2.startswith("cktile") or mm is None:
                    raw[tok] = None
                else:
                    tmm, tnn, tkk = int(mm.group(1)), int(mm.group(2)), int(mm.group(3))
                    # The fused gemm2 reads _se_atom at stage1's emission block (sblk,
                    # from megakernel SE_ATOM_SORT_BLOCK), so gemm2 tile_m must divide
                    # sblk. aiter's tuned prefill tiles use tile_m=64; if 64 ∤ sblk
                    # (e.g. sblk=32) clamp tile_m down to sblk (keep tuned tile_n/k).
                    # When sblk grows (e.g. 64), larger tuned tile_m is allowed.
                    if sblk % tmm != 0:
                        tmm = sblk
                    raw[tok] = (tmm, tnn, tkk)
    except OSError:
        _GEMM2_TILE_TABLE_CACHE[key] = None
        return None
    flydsl_toks = sorted(t for t, v in raw.items() if v is not None)
    if not flydsl_toks:
        _GEMM2_TILE_TABLE_CACHE[key] = None
        return None
    table: dict = {}
    for t in sorted(raw):
        if raw[t] is not None:
            table[t] = raw[t]
        else:  # cktile -> nearest FlyDSL bucket (tie prefers smaller / align down)
            best = min(flydsl_toks, key=lambda ft: (abs(ft - t), ft > t))
            table[t] = raw[best]
    _GEMM2_TILE_TABLE_CACHE[key] = table
    print(f"[MEGA-GEMM2-TABLE] shape(md={model_dim},id={inter_dim},E={local_experts},k={topk}) "
          f"from {os.path.basename(csv_path)} -> {table}", flush=True)
    return table


def get_or_build_mega_moe(
    *,
    rank,
    world_size,
    model_dim,
    inter_dim,
    experts,
    topk,
    quant,
    mtpr,
    w1,
    w1_scale,
    w2,
    w2_scale,
    gemm2_tile=(-1, -1, -1),
    gemm2_tile_table=None,
):
    key = (rank, world_size, model_dim, inter_dim, experts, topk, quant, mtpr, gemm2_tile)
    m = _MEGA_CACHE.get(key)
    if m is None:
        _ensure_path()
        from kernels.mega_moe import MegaMoE

        tm, tn, tk = gemm2_tile
        m = MegaMoE(
            rank=rank,
            world_size=world_size,
            model_dim=model_dim,
            inter_dim=inter_dim,
            experts=experts,
            topk=topk,
            quant=quant,
            w1=w1,
            w1_scale=w1_scale,
            w2=w2,
            w2_scale=w2_scale,
            max_tok_per_rank=mtpr,
            gemm2_tile_m=tm,
            gemm2_tile_n=tn,
            gemm2_tile_k=tk,
            gemm2_persist_m=-1,
            # mega_moe_v1 (e9d3bfcf+) replaced stage2_mode="fused" with
            # enable_fused_stage1/enable_fused_stage2 (both default True = fused).
            enable_fused_stage1=True,
            enable_fused_stage2=True,
            gemm2_tile_table=gemm2_tile_table,
        )
        _MEGA_CACHE[key] = m
    return m


def run_mega_moe(
    layer,
    x: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    *,
    model_dim: int,
    inter_dim: int,
    experts: int,
    topk: int,
    quant: str = "a8w4",
    gemm2_tile=None,
) -> torch.Tensor:
    """Replace EP experts with MegaMoE. x: [tokens, model_dim] bf16 (this rank's
    local tokens, pre-dispatch). topk_ids: global expert ids. Returns
    [tokens, model_dim] bf16."""
    if gemm2_tile is None:
        # mega_moe_v1 auto-loads its per-M MegaGemm2 tune table when
        # gemm2_tile_m<=0 (native best-config path: prefill tile_m=64, decode
        # 16/32). Override via ATOM_MEGA_GEMM2_TILE="m,n,k" to force a fixed tile.
        _t = os.environ.get("ATOM_MEGA_GEMM2_TILE", "-1,-1,-1")
        gemm2_tile = tuple(int(v) for v in _t.split(","))
    from aiter.dist.parallel_state import get_ep_group

    am = get_ep_group().device_communicator.all2all_manager
    rank, world = int(am.rank), int(am.world_size)

    run_tokens = int(x.shape[0])
    mtpr = int(os.environ.get("ATOM_MEGA_MTPR", "8192"))
    if mtpr < run_tokens:
        mtpr = _next_pow2(run_tokens)

    # derive total experts robustly from this rank's local weight shape:
    # _mega_w1 = [E_local * 2*inter, hidden//2]
    local_E = int(layer._mega_w1.shape[0] // (2 * inter_dim))
    experts = local_E * world

    mega = get_or_build_mega_moe(
        rank=rank,
        world_size=world,
        model_dim=model_dim,
        inter_dim=inter_dim,
        experts=experts,
        topk=topk,
        quant=quant,
        mtpr=mtpr,
        w1=layer._mega_w1,
        w1_scale=layer._mega_w1_scale,
        w2=layer._mega_w2,
        w2_scale=layer._mega_w2_scale,
        gemm2_tile=tuple(gemm2_tile),
        gemm2_tile_table=None,
    )

    # Per-M GEMM2 tile selection now lives in FlyDSL: MegaMoE auto-loads its
    # MegaGemm2 tune JSON by default (symmetric with gemm1's MegaStage1 auto-tune)
    # and falls back to the op's single default tile on a miss. ATOM no longer
    # builds/forwards the table (we pass gemm2_tile_table=None to get_or_build).

    # per-layer weight swap (runtime pointer args; shapes identical across layers)
    mega.stage1.w1 = layer._mega_w1
    mega.stage1.w1_scale = layer._mega_w1_scale
    mega.w2 = layer._mega_w2
    mega.w2_scale = layer._mega_w2_scale

    wts = topk_weights.to(torch.float32).contiguous()
    ids = topk_ids.to(torch.int32).contiguous()
    # PRE-FORWARD routing diagnostic (prints BEFORE the crashing gemm2 kernel).
    global _MEGA_PRE_N
    if _MEGA_PRE_N < 4:
        _MEGA_PRE_N += 1
        try:
            _i64 = ids.to(torch.int64)
            _rs, _ = torch.sort(_i64, dim=1)
            _dups = int((_rs[:, 1:] == _rs[:, :-1]).sum().item())
            _cnt = torch.bincount(_i64.reshape(-1), minlength=experts)
            _emax = int(_cnt.max().item()); _eargmax = int(_cnt.argmax().item())
            _lo, _hi = rank * local_E, (rank + 1) * local_E
            _self_local = int(((_i64 >= _lo) & (_i64 < _hi)).sum().item())
            print(f"[MEGA-PRE] rank={rank} run_tokens={run_tokens} mtpr={mtpr} max_recv={world*mtpr} "
                  f"experts={experts} local_E={local_E} topk_arg={topk} ids.shape={tuple(ids.shape)} "
                  f"ids[min={int(ids.min())},max={int(ids.max())}] row_dups={_dups} "
                  f"per_expert_max={_emax}@e{_eargmax} self->local={_self_local} "
                  f"x.shape={tuple(x.shape)} w1={tuple(layer._mega_w1.shape)} "
                  f"stage1.w1_id={id(getattr(mega.stage1,'w1',None))} layer_w1_id={id(layer._mega_w1)}", flush=True)
            import sys as _sys; _sys.stdout.flush()
        except Exception as _e:  # noqa: BLE001
            print(f"[MEGA-PRE] print failed: {_e}", flush=True)
    # DUMP the exact warmup inputs (this rank) BEFORE the crashing forward, so we
    # can replay real routing in the standalone bench. Gated by ATOM_MEGA_DUMP_IDS=dir.
    _dd = os.environ.get("ATOM_MEGA_DUMP_IDS", "")
    if _dd and not getattr(layer, "_mega_ids_dumped", False):
        try:
            os.makedirs(_dd, exist_ok=True)
            torch.save({
                "rank": rank, "world": world, "run_tokens": run_tokens, "mtpr": mtpr,
                "experts": experts, "local_E": local_E, "topk": topk,
                "ids": ids.detach().cpu(), "wts": wts.detach().cpu(),
                "x": x.detach().to(torch.bfloat16).cpu(),
            }, f"{_dd}/atom_ids_rank{rank}.pt")
            layer._mega_ids_dumped = True
            print(f"[MEGA-DUMP-IDS] rank{rank} saved to {_dd}/atom_ids_rank{rank}.pt", flush=True)
        except Exception as _e:  # noqa: BLE001
            print(f"[MEGA-DUMP-IDS] failed: {_e}", flush=True)
    out = mega.forward_bf16(x.contiguous(), wts, ids)
    # one-shot dump for offline numeric compare (ATOM_MEGA_DUMP=/path, small bs only)
    _dump = os.environ.get("ATOM_MEGA_DUMP", "")
    if _dump and run_tokens <= 1024 and getattr(layer, "_mega_dump_first", False) \
            and getattr(layer, "_mega_dumped", False) is False \
            and not torch.cuda.is_current_stream_capturing() \
            and int(ids.unique().numel()) >= 32:
        try:
            from aiter.dist.parallel_state import get_ep_group as _gep
            _r = int(_gep().device_communicator.all2all_manager.rank)
            torch.save({
                "rank": _r, "run_tokens": run_tokens,
                "x": x.detach().to(torch.bfloat16).cpu(),
                "wts": wts.detach().cpu(), "ids": ids.detach().cpu(),
                "mega_w1": layer._mega_w1.detach().cpu(),
                "mega_w1_scale": layer._mega_w1_scale.detach().cpu(),
                "mega_w2": layer._mega_w2.detach().cpu(),
                "mega_w2_scale": layer._mega_w2_scale.detach().cpu(),
                "raw_w13": layer._raw_w13, "raw_w13_scale": layer._raw_w13_scale,
                "raw_w2": layer._raw_w2, "raw_w2_scale": layer._raw_w2_scale,
                "out": out.detach().to(torch.bfloat16).cpu(),
                "model_dim": model_dim, "inter_dim": inter_dim, "experts": experts, "topk": topk,
                "layer_idx": int(getattr(layer, "_mega_layer_idx", -1)),
            }, f"{_dump}/mega_dump_rank{_r}_L{int(getattr(layer, '_mega_layer_idx', -1))}.pt")
            layer._mega_dumped = True
            print(f"[MEGA-DUMP] rank{_r} L{int(getattr(layer,'_mega_layer_idx',-1))} saved run_tokens={run_tokens}", flush=True)
        except Exception as _e:  # noqa: BLE001
            print(f"[MEGA-DUMP] failed: {_e}", flush=True)
    global _MEGA_DBG_N
    if _MEGA_DBG_N < 2:
        _MEGA_DBG_N += 1
        try:
            _i64 = ids.to(torch.int64)
            # per-row distinct expert count (topk should be all-distinct)
            _rowsorted, _ = torch.sort(_i64, dim=1)
            _dups = int((_rowsorted[:, 1:] == _rowsorted[:, :-1]).sum().item())
            # per-GLOBAL-expert receive count (token-copies), and the max
            _cnt = torch.bincount(_i64.reshape(-1), minlength=experts)
            _emax = int(_cnt.max().item()); _eargmax = int(_cnt.argmax().item())
            # how many of THIS rank's local tokens hit each local-expert range (sanity)
            _lo, _hi = rank * local_E, (rank + 1) * local_E
            _self_local = int(((_i64 >= _lo) & (_i64 < _hi)).sum().item())
            print(f"[MEGA-DBG] rank={rank} run_tokens={run_tokens} mtpr={mtpr} max_recv={world*mtpr} "
                  f"experts={experts} local_E={local_E} topk={topk} "
                  f"ids[min={int(ids.min())},max={int(ids.max())}] row_dups={_dups} "
                  f"per_expert_max_count={_emax}@e{_eargmax} self->local={_self_local}", flush=True)
        except Exception as _e:  # noqa: BLE001
            print(f"[MEGA-DBG] print failed: {_e}", flush=True)
    return out
