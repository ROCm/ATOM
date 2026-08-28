# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""TP=1 weight twins, for asymmetric rapidserve (prefill TP=N, decode DP=N).

The decode process runs attention at TP=1, so a handful of its weights have no
counterpart among prefill's TP=N shards. Rather than have decode load them
itself — which meant 8 concurrent checkpoint reads and a partial-load path that
kept writing through IPC aliases into prefill's memory — **prefill produces both
copies from the one checkpoint pass it already does**, and decode imports
everything.

The enabling fact is that `weight_loader` receives the FULL checkpoint tensor and
narrows it itself (`linear.py:1041` / `:1899`, `embed_head.py:160`). Prefill
already has every byte decode needs and discards 7/8 of it. So for each module
whose TP=1 shape differs we build a twin at tp_size=1, feed it the same
`loaded_weight`, and post-process it with TP=1 semantics:

    weight_loader(param, loaded_weight)      <- full tensor
      |- real module (TP=N)  narrow -> shard <- prefill's own compute
      +- twin        (TP=1)  full            <- decode's copy

The twin must be built from RAW checkpoint data and processed independently —
prefill's processed shards cannot be concatenated into it. Per-shard quant scales
differ (`linear.py:639-652`), and preshuffled layouts are physically reordered:
`tools/check_row_parallel_strided_shard.py` measured
`gemm_a8w8_blockscale_bpreshuffle` producing 3.67M/3.67M wrong elements when fed
a non-native layout.

Deliberately NOT twinned:
  - `FusedMoE` — decode reaches the same 8-way split via `flatten_tp_across_dp`
    (`moe.py:133-139`), which turns its `tp_size=1, dp_size=8` into the
    `tp_size=8, tp_rank=<gpu>` that prefill gets from plain TP. With
    `--enable-expert-parallel` off (the default, `config.py:1328`) that split is
    a TENSOR shard of the intermediate dim (`moe.py:2558`); with it on it is a
    shard by whole expert (`moe.py:2727`). Either way both sides derive the same
    one, so those weights already alias.
    They are also where ~90% of the bytes live.
  - `ReplicatedLinear` / norms — already full size on every rank.
"""

import contextlib
import logging
from typing import Any

import torch
from torch import nn

logger = logging.getLogger("atom")


class _SoloGroup:
    """Stand-in for a TP group of one, used while constructing twins."""

    world_size = 1
    rank_in_group = 0
    rank = 0
    size = 1


@contextlib.contextmanager
def _tp_group_of_one():
    """Make `get_tp_group()` report a single-rank group inside this block.

    `LinearBase.__init__` (linear.py:424-425) and `VocabParallelEmbedding`
    (embed_head.py:148-149) read the group at construction to size their shards.
    Nothing else in those constructors consults it — quant config is resolved by
    `prefix` — so patching the symbol in the two modules that import it is
    sufficient to get full-size parameters.
    """
    from atom.model_ops import embed_head, linear

    patched = [m for m in (linear, embed_head) if hasattr(m, "get_tp_group")]
    saved = [m.get_tp_group for m in patched]
    for m in patched:
        m.get_tp_group = lambda: _SoloGroup()
    try:
        yield
    finally:
        for m, fn in zip(patched, saved):
            m.get_tp_group = fn


@contextlib.contextmanager
def record_ctor_args():
    """Stash each module's own constructor arguments on the instance.

    A twin is built by re-invoking the same class with the same arguments under
    `_tp_group_of_one`, so the arguments have to be captured when the real model
    is built. Recording happens AFTER the wrapped `__init__` returns, so for a
    subclass that calls `super().__init__()` the last writer is the outermost
    (most derived) call — which is the one whose signature we must replay.
    """
    from atom.model_ops.embed_head import VocabParallelEmbedding
    from atom.model_ops.linear import LinearBase

    targets = _concrete_subclasses(LinearBase) | _concrete_subclasses(
        VocabParallelEmbedding
    )
    saved = {}
    for cls in targets:
        if "__init__" not in cls.__dict__:
            continue
        saved[cls] = cls.__init__

        def _wrap(orig):
            def __init__(self, *args, **kwargs):
                orig(self, *args, **kwargs)
                self._twin_ctor_args = (args, kwargs)

            return __init__

        cls.__init__ = _wrap(cls.__dict__["__init__"])
    try:
        yield
    finally:
        for cls, orig in saved.items():
            cls.__init__ = orig


def _concrete_subclasses(base) -> set:
    out = {base}
    for sub in base.__subclasses__():
        out |= _concrete_subclasses(sub)
    return out


def needs_twin(mod: nn.Module) -> bool:
    """Whether this module's parameters differ in shape between TP=N and TP=1."""
    from atom.model_ops.embed_head import VocabParallelEmbedding
    from atom.model_ops.linear import LinearBase

    if isinstance(mod, VocabParallelEmbedding):
        return getattr(mod, "tp_size", 1) > 1
    if isinstance(mod, LinearBase):
        # tp_dim None => ReplicatedLinear and friends: already full size.
        return getattr(mod, "tp_dim", None) is not None and mod.tp_size > 1
    return False


class DecodeTwins:
    """TP=1 replicas of the modules decode cannot alias from prefill.

    Lifecycle, all inside the prefill process:
        build()      before load — construct twins, redirect the weight loaders
        finalize()   after load  — run TP=1 post-processing
        overrides()  at export   — {param name: tensor} to overlay on the handles
    """

    def __init__(self) -> None:
        self._device = None
        self._twins: dict[str, nn.Module] = {}
        # Bare (non-module) sharded Parameters, e.g. DeepSeek-V4's `attn_sink`
        # sized n_local_heads. They have no weight_loader of their own and are
        # narrowed by numel in `default_weight_loader`, so they are captured
        # directly rather than via a twin.
        self._bare_full: dict[str, torch.Tensor] = {}
        # Filled by _agree_recorder during the load: name -> True (agrees) /
        # False (disagrees) / None (geometry not mapped, so not judged).
        self._agree: dict[str, bool | None] = {}
        self._diag: list[str] = []

    # -- build -----------------------------------------------------------
    @classmethod
    def build(cls, model: nn.Module, device) -> "DecodeTwins":
        """Construct twins on `device` and redirect the real modules' loaders.

        `device` is pinned explicitly rather than inherited from the ambient
        default: these tensors are exported over CUDA IPC, and a twin that lands
        on CPU fails only much later, at `_share_cuda_`.
        """
        self = cls()
        self._device = device
        torch.set_default_device(device)
        try:
            for name, mod in model.named_modules():
                if needs_twin(mod):
                    twin = self._make_twin(name, mod)
                    if twin is not None:
                        self._twins[name] = twin
                        self._redirect_loaders(name, mod, twin)
        finally:
            torch.set_default_device(None)
        n_bare = self._capture_bare_params(model)
        self._assert_on_device()
        logger.info(
            "[TWINS] %d TP=1 twin module(s) built; %d bare parameter(s) armed "
            "for capture (they fill during the load)",
            len(self._twins),
            n_bare,
        )
        return self

    def _make_twin(self, name: str, mod: nn.Module) -> nn.Module | None:
        recorded = getattr(mod, "_twin_ctor_args", None)
        if recorded is None:
            logger.warning(
                "[TWINS] %s (%s) has no recorded constructor args — decode will "
                "be missing this weight. Was the model built inside "
                "record_ctor_args()?",
                name,
                type(mod).__name__,
            )
            return None
        args, kwargs = recorded
        with _tp_group_of_one():
            twin = type(mod)(*args, **kwargs)
        assert getattr(twin, "tp_size", 1) == 1, (
            f"{name}: twin came out at tp_size={twin.tp_size}; the group patch "
            "did not take effect"
        )
        return twin

    def _redirect_loaders(self, name: str, mod: nn.Module, twin: nn.Module) -> None:
        """Feed every `loaded_weight` to the twin as well as the real module.

        Both sides receive the FULL checkpoint tensor and narrow it themselves —
        the real module to its shard, the twin not at all — so one dispatch
        populates both. Rebinding on the Parameter is how layers already attach
        loaders (`linear.py:496`, `embed_head.py:158`).
        """
        twin._twin_targets = {}
        twin._twin_writes = {}
        for attr, param in list(mod.named_parameters(recurse=False)):
            tparam = twin._parameters.get(attr)
            if tparam is None:
                continue
            original = getattr(param, "weight_loader", None)
            if original is None:
                continue
            # Remember WHICH object the loader will write into. The closure below
            # pins that Parameter for the rest of the load, so if anything rebinds
            # `twin.<attr>` afterwards the write lands on an orphan while export
            # reads the replacement — a silent, weight-specific corruption.
            # verify_against_shards checks the two are still the same object.
            twin._twin_targets[attr] = tparam
            param.weight_loader = _dual_loader(
                original, twin, attr, tparam, self._agree_recorder(name, mod, twin)
            )

    def _agree_recorder(self, name: str, real: nn.Module, twin: nn.Module):
        """Record whether the twin agrees with the shard, AT LOAD TIME.

        This cannot be deferred. `load_model` runs the real model's
        `process_weights_after_loading` before it returns (loader.py:234), so by
        the time the load call finishes the real FP8 weights have been
        preshuffled into their kernel layout — physically reordered — while the
        twins are still raw. Comparing then reports every FP8 weight as broken
        and every BF16 weight as fine, which says nothing about either.

        The instant after both loaders have run is the only moment the two sides
        are supposed to hold identical bytes.

        Only the region THIS call wrote is compared. Weight loading is threaded
        (`loading_core.py:187-198`), and a merged layer's partitions arrive as
        separate checkpoint tensors on separate tasks — so gate and up land in
        one Parameter from two threads at once. Reading the whole Parameter here
        races with the sibling partition's writer and reports sporadic,
        rank-dependent disagreement on weights that are in fact fine. Each
        partition has exactly one writer, so per-partition comparison is stable.
        """
        rank = getattr(real, "tp_rank", 0)

        def on_write(rparam: nn.Parameter, attr: str, args: tuple) -> None:
            tparam = twin._parameters.get(attr)
            if tparam is None:
                return
            view = shard_view(real, twin, tparam, rparam, rank)
            if view is None or view.shape != rparam.data.shape:
                self._agree[f"{name}.{attr}"] = None  # geometry not mapped
                return
            part = _written_partition(real, args)
            key = f"{name}.{attr}"
            tview, rview = view, rparam.data
            if part is not None:
                # Must be the PARAMETER's partition sizes, not the weight's:
                # a per-1x128 scale has 1/128 the rows (see
                # partition_sizes_in_param_units).
                sizes = partition_sizes_in_param_units(
                    real, twin, rview.shape[0], tparam.data.shape[0]
                )
                if sizes is None:
                    self._agree[key] = None
                    return
                rops = sizes[0]
                off = sum(rops[:part])
                tview = tview.narrow(0, off, rops[part])
                rview = rview.narrow(0, off, rops[part])
                key = f"{key}#{part}"
            ok = _tensors_equal(tview, rview)
            self._agree[key] = ok
            if not ok and len(self._diag) < 4:
                self._diag.append(
                    f"{key}: " + diagnose_mismatch(real, twin, tparam, rparam, rank)
                )

        return on_write

    def _capture_bare_params(self, model: nn.Module) -> None:
        """Attach a capture loader to Parameters that have no loader of their own.

        Those go through `default_weight_loader`, which shards by numel
        (`loader.py:49-60`). We cannot tell in advance which are sharded, so we
        record the full tensor for all of them and discard the replicated ones in
        `overrides()` — replicated ones alias fine and only the genuinely sharded
        ones (attn_sink: ~31KB total) are kept.
        """
        store = self._bare_full
        armed = 0
        for mod_name, mod in model.named_modules():
            for attr, param in list(mod.named_parameters(recurse=False)):
                if hasattr(param, "weight_loader"):
                    continue
                full_name = f"{mod_name}.{attr}" if mod_name else attr
                param.weight_loader = _capture_loader(full_name, store)
                armed += 1
        return armed

    def _assert_on_device(self) -> None:
        """Every twin parameter must be on the GPU we will export from."""
        bad = [
            f"{name}.{attr}"
            for name, twin in self._twins.items()
            for attr, p in twin.named_parameters(recurse=False)
            if p.device.type != "cuda"
        ]
        if bad:
            raise RuntimeError(
                f"{len(bad)} twin parameter(s) were built off-GPU and cannot be "
                f"exported over CUDA IPC (torch's default device was not set "
                f"during construction): {bad[:8]}"
            )

    # -- after load -------------------------------------------------------
    def finalize(self, model: nn.Module) -> None:
        """Post-process the twins exactly as `load_model` post-processes the model.

        The traversal is `named_modules()` order — a pre-order walk, so PARENTS
        BEFORE CHILDREN — and matching it is load-bearing, not cosmetic. V4's
        attention dequants `wo_a` from FP8 to BF16 and deletes its scale
        (deepseek_v4.py:2453); `wo_a`'s own LinearBase hook must then see the
        BF16 result. Run the child first and it preshuffles the FP8 weight,
        after which the parent dequants a shuffled tensor — output that is
        structurally valid, numerically wrong, and never reaches EOS.

        For each module in that order:
          - a twinned module      -> run the TWIN's hook
          - a module owning twins -> run ITS hook with the twins swapped in

        The second case re-runs a hook that already ran during the load, so
        those hooks must be idempotent. V4's is, explicitly (it returns early
        once the weight is BF16). A hook that is not would double-apply to the
        parent's non-twinned children, so the count is reported.
        """
        own = swapped = 0
        for mod_name, mod in model.named_modules():
            twin = self._twins.get(mod_name)
            if twin is not None:
                hook = getattr(twin, "process_weights_after_loading", None)
                if hook is not None:
                    self._run_hook(hook, mod_name)
                    own += 1
                continue
            if getattr(mod, "process_weights_after_loading", None) is None:
                continue
            pairs = self._twinned_children(mod_name, mod)
            if not pairs:
                continue
            saved = [(attr, getattr(mod, attr)) for attr, _ in pairs]
            for attr, child_twin in pairs:
                setattr(mod, attr, child_twin)
            try:
                self._run_hook(mod.process_weights_after_loading, mod_name)
                swapped += 1
            finally:
                for attr, real in saved:
                    setattr(mod, attr, real)
        logger.info(
            "[TWINS] post-processed in loader order: %d twin hook(s), "
            "%d ancestor hook(s) re-run against their twins",
            own,
            swapped,
        )

    def _twinned_children(self, mod_name: str, mod: nn.Module) -> list:
        """(attr, twin) for this module's DIRECT children that have twins."""
        out = []
        for attr, _ in mod.named_children():
            child_name = f"{mod_name}.{attr}" if mod_name else attr
            twin = self._twins.get(child_name)
            if twin is not None:
                out.append((attr, twin))
        return out

    @staticmethod
    def _run_hook(hook, where: str) -> None:
        try:
            hook()
        except Exception:
            logger.exception("[TWINS] post-processing failed for %s", where)
            raise

    # -- export -----------------------------------------------------------
    def overrides(self) -> dict[str, torch.Tensor]:
        """{parameter name: TP=1 tensor} to overlay on the exported handles.

        Names match the producer's own parameters, so the consumer needs no
        special case — it simply receives a full matrix where the producer holds
        a shard.

        Covers both Parameters and plain tensor attributes, because a post-load
        hook may convert one into the other (`moe.py:1103-1108` does exactly
        that for the MoE scales) and the export keys purely on the name.
        """
        out: dict[str, torch.Tensor] = {}
        for mod_name, twin in self._twins.items():
            for attr, p in twin.named_parameters(recurse=False):
                out[f"{mod_name}.{attr}"] = p.data
            for attr, val in vars(twin).items():
                if isinstance(val, torch.Tensor) and not attr.startswith("_"):
                    out[f"{mod_name}.{attr}"] = val
        # Bare sharded Parameters (attn_sink); replicated ones were dropped by
        # drop_replicated_bare and are aliased from the producer as usual.
        out.update(self._bare_full)
        return out

    def module_attr_overrides(self) -> dict[str, dict]:
        """{module name: {attr: value}} produced by the twins' post-processing.

        Tensors are not the only output of `process_weights_after_loading`. It
        also stamps plain attributes that `forward` reads — `LinearBase` sets
        `is_output_padded` (linear.py:798) and `_output_size_before_padding`
        (:827), and `forward` slices the padded columns off using both (:985).

        Both are TP-dependent: whether the output needs padding is decided from
        `output_size`, which is exactly what differs between the TP=N module and
        its TP=1 twin. So the consumer must take the TWIN's values, not the
        producer's — and it cannot compute them itself, having never run the
        hook. Underscore-prefixed names are included deliberately;
        `_module_meta_attrs` skips them, which would drop
        `_output_size_before_padding` and leave `forward` raising AttributeError.
        """
        simple = (str, bool, int, float)
        out: dict[str, dict] = {}
        for mod_name, twin in self._twins.items():
            attrs = {
                k: v
                for k, v in vars(twin).items()
                if k != "training" and isinstance(v, simple)
            }
            if attrs:
                out[mod_name] = attrs
        return out

    def verify_against_shards(self, model: nn.Module, tp_rank: int) -> None:
        """Report the load-time twin/shard comparison and fail on disagreement.

        The comparison itself happened in `_agree_recorder`, during the load. It
        HAS to: `load_model` runs the real model's `process_weights_after_loading`
        before it returns (loader.py:234), so once the load call is over the real
        FP8 weights have been preshuffled into their kernel layout while the twins
        are still raw. An earlier version of this check compared here instead and
        reported all 244 FP8 weights as corrupt and every BF16 weight as fine —
        measuring the preshuffle, not the twins. Do not move the comparison back.

        Three faults are distinguished, because they need different fixes:
          - rebound: the loader wrote into a Parameter that was later replaced,
            so export ships an un-written tensor;
          - never written: the checkpoint pass never routed through
            `param.weight_loader` for that parameter;
          - disagrees: fed, alive, but not the bytes the shard got.
        """
        agreed = sum(1 for v in self._agree.values() if v is True)
        unjudged = sum(1 for v in self._agree.values() if v is None)
        bad = sorted(k for k, v in self._agree.items() if v is False)
        unwritten: list[str] = []
        rebound: list[str] = []
        for mod_name, twin in self._twins.items():
            targets = getattr(twin, "_twin_targets", {})
            writes = getattr(twin, "_twin_writes", {})
            for attr, target in targets.items():
                # Lifetime, not arithmetic: the loader wrote into targets[attr];
                # export reads twin._parameters[attr].
                if twin._parameters.get(attr) is not target:
                    rebound.append(f"{mod_name}.{attr}")
                if not writes.get(attr):
                    unwritten.append(f"{mod_name}.{attr}")
        logger.info(
            "[TWINS] shard cross-check rank %d: %d agree, %d DISAGREE, "
            "%d unjudged (geometry not mapped), %d never written, %d rebound",
            tp_rank,
            agreed,
            len(bad),
            unjudged,
            len(unwritten),
            len(rebound),
        )
        for line in self._diag:
            logger.error("[TWINS] %s", line)
        if rebound:
            raise RuntimeError(
                f"{len(rebound)} twin parameter(s) were replaced after the dual "
                f"loader captured them, so the load wrote into an orphaned tensor "
                f"and decode would export an unwritten one: {rebound[:12]}"
            )
        if unwritten:
            raise RuntimeError(
                f"{len(unwritten)} twin parameter(s) were never fed by the dual "
                f"loader — the checkpoint pass did not route through "
                f"param.weight_loader for them: {unwritten[:12]}"
            )
        if bad:
            raise RuntimeError(
                f"{len(bad)} twin parameter(s) disagree with the TP shard loaded "
                f"from the same checkpoint tensor, so decode would run on wrong "
                f"weights: {bad[:12]}"
            )

    def verify_post_processing(self, model: nn.Module, tp_rank: int, rows: int = 8):
        """After finalize(): does the twin COMPUTE what the shard computes?

        `verify_against_shards` proves the twin received the right bytes. It says
        nothing about `process_weights_after_loading`, which is where the FP8
        preshuffle and the dequant-from-an-ancestor-hook live and where a TP=1
        twin could plausibly diverge from a TP=8 shard. That is only checkable
        through the forward pass, because the post-processed tensors are
        legitimately different layouts.

        Column-parallel only, deliberately: a ColumnParallelLinear returns this
        rank's output slice with no collective, so `twin(x)` restricted to that
        slice must match `real(x)`. Row-parallel would need the TP all-reduce to
        compare, and running a collective from inside a verification pass risks
        desynchronising the ranks it is meant to be checking.

        Reports rather than raises. The tolerance below is a judgement call
        (FP8/FP4 weights, different kernel tile shapes on either side), and a
        tolerance that is merely too tight should not be able to abort a boot.
        """
        import torch as _torch

        worst, checked, bad = 0.0, 0, []
        for mod_name, twin in self._twins.items():
            real = model.get_submodule(mod_name)
            if getattr(real, "tp_dim", None) != 0 or getattr(real, "tp_size", 1) < 2:
                continue
            sizes = list(getattr(real, "output_partition_sizes", None) or [])
            tsizes = list(getattr(twin, "output_partition_sizes", None) or [])
            in_size = getattr(real, "input_size", None)
            if not sizes or len(sizes) != len(tsizes) or not in_size:
                continue
            try:
                p = next(twin.parameters())
                x = _torch.randn(rows, in_size, dtype=torch.bfloat16, device=p.device)
                with _torch.inference_mode():
                    yr, yt = real(x), twin(x)
                yr = yr[0] if isinstance(yr, tuple) else yr
                yt = yt[0] if isinstance(yt, tuple) else yt
                roff = toff = 0
                for rsz, tsz in zip(sizes, tsizes):
                    a = yr[..., roff : roff + rsz].float()
                    b = yt[..., toff + tp_rank * rsz : toff + (tp_rank + 1) * rsz]
                    denom = a.abs().max().clamp_min(1e-6)
                    err = float((a - b.float()).abs().max() / denom)
                    worst = max(worst, err)
                    if err > 0.05:
                        bad.append(f"{mod_name} (rel {err:.3f})")
                    roff += rsz
                    toff += tsz
                checked += 1
            except Exception as exc:  # diagnostic only — never break the boot
                logger.debug("[TWINS] post-process check skipped %s: %s", mod_name, exc)
        logger.info(
            "[TWINS] post-process forward check rank %d: %d column-parallel "
            "module(s) compared, worst relative error %.4f, %d over tolerance%s",
            tp_rank,
            checked,
            worst,
            len(bad),
            (": " + ", ".join(bad[:6])) if bad else "",
        )

    def drop_replicated_bare(self, model: nn.Module) -> None:
        """Forget captured bare tensors that were never sharded.

        A replicated Parameter's full tensor equals the rank-local one, so decode
        can alias it and holding a copy is pure waste. What survives is the
        genuinely head-sharded set — DeepSeek-V4's `attn_sink` and nothing else
        on current models, ~31KB in total.
        """
        params = dict(model.named_parameters())
        before = len(self._bare_full)
        for name in list(self._bare_full):
            p = params.get(name)
            if p is not None and p.numel() == self._bare_full[name].numel():
                del self._bare_full[name]
        logger.info(
            "[TWINS] bare parameters: %d captured, %d replicated (dropped), "
            "%d genuinely sharded and exported: %s",
            before,
            before - len(self._bare_full),
            len(self._bare_full),
            sorted(self._bare_full)[:6],
        )


def partition_sizes_in_param_units(real: nn.Module, twin: nn.Module, r_rows, t_rows):
    """`(real_sizes, twin_sizes)` for THIS parameter, or None if not derivable.

    `output_partition_sizes` counts WEIGHT rows. A companion parameter may be
    coarser: a per-1x128 block scale has one row per 128 weight rows
    (linear.py:1143-1145). The granularity is recovered from the shapes rather
    than read off the quant config, so any future block size works and a
    parameter whose rows are not a clean divisor is skipped rather than silently
    compared against the wrong slice.

    Every caller that indexes partitions of a parameter must go through this.
    Using the raw weight-row sizes to narrow a scale is how this first broke:
    `start (0) + length (384) exceeds dimension size (6)`.
    """
    rops = list(getattr(real, "output_partition_sizes", None) or [])
    tops = list(getattr(twin, "output_partition_sizes", None) or [])
    if not rops or len(rops) != len(tops) or not r_rows:
        return None
    if sum(rops) % r_rows:
        return None
    gran = sum(rops) // r_rows
    if gran < 1 or any(x % gran for x in rops) or any(x % gran for x in tops):
        return None
    rops = [x // gran for x in rops]
    tops = [x // gran for x in tops]
    if sum(rops) != r_rows or sum(tops) != t_rows:
        return None
    return rops, tops


def shard_view(real: nn.Module, twin: nn.Module, tparam, rparam, tp_rank: int):
    """The slice of a twin parameter that the real module's shard should equal.

    Returns None when the geometry is not one of the mapped cases, so an
    unrecognised layer is skipped rather than guessed at. Two cases are mapped:

      - `tp_dim == 0`: sharded on the output dim. Merged layers (qkv, gate_up)
        are laid out per-partition, so the real tensor is
        [gate_shard, up_shard] while the twin is [gate_full, up_full] — a single
        narrow would silently compare the wrong bytes. Walk the partitions
        (`linear.py:1141-1142`) and gather this rank's piece of each.
      - `tp_dim == 1`: sharded on the input dim, one contiguous narrow.

    Vocab-parallel modules have no `tp_dim`; they shard contiguously on dim 0 at
    `vocab_start_idx` (`embed_head.py:152-154`).
    """
    t, r = tparam.data, rparam.data
    if t.shape == r.shape:
        return t  # replicated (e.g. per-tensor scales) — must match exactly
    if t.dtype != r.dtype or t.dim() != r.dim():
        return None
    tp_size = getattr(real, "tp_size", 1)
    if tp_size < 2:
        return None

    start = getattr(real, "vocab_start_idx", None)
    if start is not None:
        if start + r.shape[0] > t.shape[0]:
            return None
        return t.narrow(0, start, r.shape[0])

    tp_dim = getattr(real, "tp_dim", None)
    if tp_dim == 1:
        if t.shape[1] != r.shape[1] * tp_size:
            return None
        return t.narrow(1, tp_rank * r.shape[1], r.shape[1])
    if tp_dim != 0:
        return None

    sizes = partition_sizes_in_param_units(real, twin, r.shape[0], t.shape[0])
    if sizes is None:
        return None
    rops, tops = sizes

    parts, off = [], 0
    for rsz, tsz in zip(rops, tops):
        if tsz != rsz * tp_size:
            return None
        parts.append(t.narrow(0, off + tp_rank * rsz, rsz))
        off += tsz
    return parts[0] if len(parts) == 1 else torch.cat(parts, 0)


def diagnose_mismatch(real, twin, tparam, rparam, tp_rank: int) -> str:
    """Explain HOW a twin disagrees with its shard — the useful half of a failure.

    A mismatch has two very different causes and they need opposite fixes:

      - the twin holds the right bytes but this checker looked in the wrong
        place (a rank or offset the module does not actually use), in which case
        the shard turns up at some OTHER offset and the twins are fine;
      - the twin genuinely never received those bytes, in which case the shard
        appears at no offset at all.

    So sweep every candidate offset and report which one matched. Also report the
    module's own `tp_rank`, since that disagreeing with the ModelRunner rank is
    the cheapest explanation for the first case.
    """
    t, r = tparam.data, rparam.data
    own = getattr(real, "tp_rank", None)
    tp_size = getattr(real, "tp_size", 1)
    tp_dim = getattr(real, "tp_dim", None)
    note = (
        f"tp_rank={own} (checked as {tp_rank}) tp_size={tp_size} tp_dim={tp_dim} "
        f"dtype={r.dtype} twin{tuple(t.shape)}@tp{getattr(twin, 'tp_size', '?')} "
        f"vs real{tuple(r.shape)}"
    )
    if tp_dim is None or t.dim() != r.dim() or t.shape[tp_dim] < r.shape[tp_dim]:
        return note + " shapes=" + f"{tuple(t.shape)} vs {tuple(r.shape)}"
    width = r.shape[tp_dim]
    hits = [
        k
        for k in range(tp_size)
        if k * width + width <= t.shape[tp_dim]
        and _tensors_equal(t.narrow(tp_dim, k * width, width), r)
    ]
    if hits:
        return f"{note} — shard found at offset index {hits} instead"
    return f"{note} — shard found at NO offset; the twin never received it"


def _written_partition(real: nn.Module, args: tuple) -> int | None:
    """Which output partition this loader call wrote, if it wrote just one.

    Merged layers take `loaded_shard_id` as the first positional argument
    (`linear.py:1141`). An int names one partition and that call is that
    partition's only writer. Anything else — no shard id (the fused-tensor path,
    where one call writes every partition), a string id, a list of ids — means
    the call is not scoped to a single partition, so return None and let the
    caller compare the whole parameter.
    """
    if not args or not isinstance(args[0], int) or isinstance(args[0], bool):
        return None
    sizes = getattr(real, "output_partition_sizes", None)
    if not sizes or len(sizes) < 2 or getattr(real, "tp_dim", None) != 0:
        return None
    return args[0] if 0 <= args[0] < len(sizes) else None


def _tensors_equal(a: torch.Tensor, b: torch.Tensor) -> bool:
    """Equality that also works for the fp8 dtypes, without a float32 blowup.

    `torch.equal` has no fp8 kernel, and widening to fp32 would quadruple a
    100MB weight during a load that is already memory-tight. Compare the raw
    bytes instead — exact, and at most one 1-byte-per-element temporary.
    """
    if a.shape != b.shape or a.dtype != b.dtype:
        return False
    if a.element_size() == 1:
        a = a.contiguous().view(torch.uint8)
        b = b.contiguous().view(torch.uint8)
    return bool(torch.equal(a, b))


def _dual_loader(
    original, twin: nn.Module, attr: str, tparam: nn.Parameter, on_write=None
):
    """Wrap a module's weight_loader so the twin is filled from the same tensor."""

    def load(param: nn.Parameter, loaded_weight: torch.Tensor, *args: Any, **kw: Any):
        original(param, loaded_weight, *args, **kw)
        twin_loader = getattr(twin, "weight_loader", None)
        if twin_loader is not None:
            twin_loader(tparam, loaded_weight, *args, **kw)
            # Counted so verify_against_shards can separate "the twin was fed
            # the wrong bytes" from "the twin was never fed at all". Bookkeeping
            # only — created on demand so it can never fail the actual load.
            writes = twin.__dict__.setdefault("_twin_writes", {})
            writes[attr] = writes.get(attr, 0) + 1
            if on_write is not None:
                # `args` carries the merged layer's loaded_shard_id, which is
                # what lets the recorder compare only the partition this call
                # wrote rather than racing the sibling partition's thread.
                on_write(param, attr, args)

    return load


def _capture_loader(name: str, store: dict[str, torch.Tensor]):
    """Record the full checkpoint tensor, then do the normal sharded copy."""

    def load(param: nn.Parameter, loaded_weight: torch.Tensor, *args: Any, **kw: Any):
        from atom.model_loader.loader import default_weight_loader

        store[name] = loaded_weight.detach().to(param.device).clone()
        default_weight_loader(param, loaded_weight)

    return load
