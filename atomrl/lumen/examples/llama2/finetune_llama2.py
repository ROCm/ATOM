# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

"""LLaMA2 Supervised Fine-Tuning — unified entry point.

Supports two backends, selected via ``--backend``:

- **megatron** — Megatron-LM-AMD ``pretrain`` driver with TP/PP/CP/VP/SP.
  Requires ``pip install megatron-lm``.
- **fsdp** — PyTorch FSDP + HuggingFace Transformers.
  Requires ``pip install transformers peft``.

Usage:
    # Megatron backend (default)
    torchrun --nproc_per_node=8 finetune_llama2.py --backend megatron <megatron args>

    # FSDP backend
    torchrun --nproc_per_node=8 finetune_llama2.py --backend fsdp <fsdp args>

See run_finetune.sh for a complete launch example.
"""

import os
import sys

from atomrl.lumen.models.utils import peek_backend


def _maybe_install_profiler():
    """Patch train_step to capture torch.profiler on steps [start, end].

    Activated by env LUMEN_PROF_START (default: off).
    LUMEN_PROF_END, LUMEN_PROF_OUTPUT, LUMEN_PROF_STOP_AFTER are optional.

    LUMEN_COPY_TRACE=1 enables Python-level copy_/contiguous tracing
    (monkey-patch approach, independent of ROCm profiler stack support).
    """
    prof_start = os.environ.get("LUMEN_PROF_START")
    if not prof_start:
        return

    import functools
    import traceback
    from collections import Counter, defaultdict

    import megatron.training.training as _mtt
    import torch
    import torch.distributed as dist

    start = int(prof_start)
    end = int(os.environ.get("LUMEN_PROF_END", str(start + 2)))
    output = os.environ.get(
        "LUMEN_PROF_OUTPUT",
        "/data1/lumen/results/tp1_fp8/profile_summary_new.txt",
    )
    stop_after = int(os.environ.get("LUMEN_PROF_STOP_AFTER", "0"))

    with_stack = os.environ.get("LUMEN_PROF_STACK", "0") == "1"
    trace_output = os.environ.get(
        "LUMEN_PROF_TRACE",
        "",
    )
    copy_trace = os.environ.get("LUMEN_COPY_TRACE", "0") == "1"

    _state = {"step": 0, "prof": None, "tracing": False}

    # Copy/contiguous tracing data
    _copy_counts = Counter()  # stack_key -> count
    _copy_nbytes = defaultdict(int)  # stack_key -> total bytes
    _contig_counts = Counter()
    _contig_nbytes = defaultdict(int)

    def _short_stack(skip=2):
        """Return a compact stack key: last 3 lumen/megatron frames."""
        frames = traceback.extract_stack()
        relevant = []
        for f in frames[:-skip]:
            fn = f.filename
            if any(k in fn for k in ("lumen/", "megatron/", "examples/")):
                short = fn.replace("/workspace/Lumen/", "").replace(
                    "/workspace/megatron_lm/", "meg:"
                )
                relevant.append(f"{short}:{f.lineno} {f.name}")
        if relevant:
            return " <- ".join(relevant[-3:])
        return (
            frames[-skip - 1].filename.split("/")[-1]
            + ":"
            + str(frames[-skip - 1].lineno)
        )

    if copy_trace:
        _orig_copy = torch.Tensor.copy_
        _orig_contig = torch.Tensor.contiguous

        def _traced_copy(self, src, non_blocking=False):
            if _state["tracing"]:
                key = _short_stack(skip=2)
                _copy_counts[key] += 1
                _copy_nbytes[key] += self.nelement() * self.element_size()
            return _orig_copy(self, src, non_blocking)

        def _traced_contiguous(self, memory_format=torch.contiguous_format):
            if _state["tracing"] and not self.is_contiguous(
                memory_format=memory_format
            ):
                key = _short_stack(skip=2)
                _contig_counts[key] += 1
                _contig_nbytes[key] += self.nelement() * self.element_size()
            return _orig_contig(self, memory_format=memory_format)

        torch.Tensor.copy_ = _traced_copy
        torch.Tensor.contiguous = _traced_contiguous
        print("[PROFILER] copy_/contiguous tracing installed", flush=True)

    _orig = _mtt.train_step

    @functools.wraps(_orig)
    def _profiled(*args, **kwargs):
        _state["step"] += 1
        step = _state["step"]
        rank = dist.get_rank() if dist.is_initialized() else 0

        if step == start and rank == 0:
            _state["prof"] = torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA,
                ],
                record_shapes=with_stack,
                profile_memory=False,
                with_stack=with_stack,
            )
            _state["prof"].__enter__()
            if copy_trace:
                _copy_counts.clear()
                _copy_nbytes.clear()
                _contig_counts.clear()
                _contig_nbytes.clear()
                _state["tracing"] = True
            print(
                f"[PROFILER] Started at step {step} (with_stack={with_stack}, copy_trace={copy_trace})",
                flush=True,
            )

        result = _orig(*args, **kwargs)

        if step == end and rank == 0 and _state["prof"] is not None:
            _state["tracing"] = False
            _state["prof"].__exit__(None, None, None)
            summary = (
                _state["prof"]
                .key_averages(group_by_stack_n=5 if with_stack else 0)
                .table(sort_by="self_cuda_time_total", row_limit=200)
            )
            with open(output, "w") as f:
                f.write(
                    f"Profile steps {start}-{end}, "
                    f"LUMEN_PREFER_HIPBLASLT="
                    f"{os.environ.get('LUMEN_PREFER_HIPBLASLT', '?')}\n\n"
                )
                f.write(summary)
            print(f"[PROFILER] Wrote {output}", flush=True)
            if trace_output:
                _state["prof"].export_chrome_trace(trace_output)
                print(f"[PROFILER] Wrote chrome trace {trace_output}", flush=True)

            # Dump copy_/contiguous trace (monkey-patch approach)
            if copy_trace:
                nsteps = end - start
                copy_out = output.replace(".txt", "_copy_trace.txt")
                with open(copy_out, "w") as cf:
                    cf.write(
                        f"copy_ trace — {nsteps} steps, {sum(_copy_counts.values())} total calls\n"
                    )
                    cf.write(f"{'='*120}\n\n")
                    cf.write("--- copy_ by call count (top 40) ---\n")
                    cf.write(
                        f"{'Calls':>8}  {'Calls/step':>10}  {'MB/step':>10}  Source\n"
                    )
                    cf.write(f"{'-'*120}\n")
                    for key, count in _copy_counts.most_common(40):
                        mb = _copy_nbytes[key] / 1e6 / nsteps
                        cf.write(
                            f"{count:>8}  {count/nsteps:>10.0f}  {mb:>10.1f}  {key}\n"
                        )

                    cf.write("\n--- copy_ by bytes (top 40) ---\n")
                    cf.write(
                        f"{'Calls':>8}  {'Calls/step':>10}  {'MB/step':>10}  Source\n"
                    )
                    cf.write(f"{'-'*120}\n")
                    by_bytes = sorted(_copy_nbytes.items(), key=lambda x: -x[1])
                    for key, nbytes in by_bytes[:40]:
                        count = _copy_counts[key]
                        mb = nbytes / 1e6 / nsteps
                        cf.write(
                            f"{count:>8}  {count/nsteps:>10.0f}  {mb:>10.1f}  {key}\n"
                        )

                    if _contig_counts:
                        cf.write(
                            "\n--- contiguous() (non-contiguous tensors only, top 40) ---\n"
                        )
                        cf.write(
                            f"{'Calls':>8}  {'Calls/step':>10}  {'MB/step':>10}  Source\n"
                        )
                        cf.write(f"{'-'*120}\n")
                        by_bytes_c = sorted(_contig_nbytes.items(), key=lambda x: -x[1])
                        for key, nbytes in by_bytes_c[:40]:
                            count = _contig_counts[key]
                            mb = nbytes / 1e6 / nsteps
                            cf.write(
                                f"{count:>8}  {count/nsteps:>10.0f}  {mb:>10.1f}  {key}\n"
                            )

                print(f"[PROFILER] Wrote copy trace {copy_out}", flush=True)

            _state["prof"] = None

        if stop_after and step >= stop_after:
            if rank == 0:
                print(f"[PROFILER] Stopping after step {step}", flush=True)
            sys.exit(0)

        return result

    _mtt.train_step = _profiled
    print(f"[PROFILER] Will profile steps {start}-{end}, output={output}", flush=True)


def _maybe_install_amax_trace():
    """Patch train_step to dump fused_amax_abs call-site traces per step.

    Activated by env LUMEN_TRACE_AMAX=1.
    """
    if os.environ.get("LUMEN_TRACE_AMAX", "0") != "1":
        return

    import functools

    import megatron.training.training as _mtt
    import torch.distributed as dist
    from atomrl.lumen.ops.quantize.quant_amax_fused import dump_amax_traces
    from atomrl.lumen.quantize.scaling_manager import dump_amax_tids

    _orig = _mtt.train_step
    _step = [0]

    @functools.wraps(_orig)
    def _traced(*args, **kwargs):
        result = _orig(*args, **kwargs)
        _step[0] += 1
        rank = dist.get_rank() if dist.is_initialized() else 0
        if rank == 0:
            dump_amax_traces(f"step {_step[0]}")
            dump_amax_tids(f"step {_step[0]}")
        return result

    _mtt.train_step = _traced
    print("[AMAX TRACE] Installed train_step amax trace hook", flush=True)


def _run_megatron():
    from atomrl.lumen.models.llama2.megatron import (
        add_finetune_args,
        apply_fp8_training,
        apply_lora,
        forward_step,
        lumen_gpt_builder,
        train_valid_test_datasets_provider,
    )
    from atomrl.lumen.models.megatron import make_lumen_model_provider
    from atomrl.lumen.patches import apply_training_patches
    from megatron.core.enums import ModelType
    from megatron.training import pretrain

    model_provider = make_lumen_model_provider(
        lumen_gpt_builder,
        lora_applier=apply_lora,
        fp8_applier=apply_fp8_training,
    )
    apply_training_patches(
        names={
            "fp8_param_gather_hook",
            "mxfp4_weight_cache_hook",
            "fp8_param_storage_hook",
            "hip_graphs_hook",
            "val_loss_early_stop_hook",
        }
    )
    _maybe_install_profiler()
    _maybe_install_amax_trace()

    train_valid_test_datasets_provider.is_distributed = True

    pretrain(
        train_valid_test_datasets_provider,
        model_provider,
        ModelType.encoder_or_decoder,
        forward_step,
        extra_args_provider=add_finetune_args,
        args_defaults={"tokenizer_type": "HuggingFaceTokenizer"},
    )


def _run_fsdp():
    from atomrl.lumen.models.llama2.fsdp import FSDPTrainer, get_args

    args = get_args()
    trainer = FSDPTrainer(args)
    trainer.train()


if __name__ == "__main__":
    backend = peek_backend()

    if backend == "megatron":
        _run_megatron()
    elif backend == "fsdp":
        _run_fsdp()
    else:
        print(f"ERROR: Unknown backend '{backend}'. Use 'megatron' or 'fsdp'.")
        sys.exit(1)
