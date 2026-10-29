"""Model evidence, bounded startup stacks and optional Prefill trace labels."""
import json
import math
import os
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path

from vllm.v1.worker.gpu_worker import Worker


def classname(value):
    return None if value is None else type(value).__module__ + "." + type(value).__name__


def inspect_model(model):
    result = []
    if model is None:
        return result
    for name, layer in model.named_modules():
        method = getattr(layer, "quant_method", None)
        if method is None:
            continue
        tensors = {}
        for key in ("weight", "weight_scale", "weight_scale_inv", "input_scale", "w13_weight", "w2_weight", "w13_weight_scale", "w2_weight_scale"):
            tensor = getattr(layer, key, None)
            if tensor is not None and hasattr(tensor, "shape"):
                tensors[key] = {"shape": list(tensor.shape), "dtype": str(tensor.dtype)}
        kernel = getattr(method, "kernel", None)
        result.append({"name": name, "layer": classname(layer), "quant_method": classname(method),
                       "fp8_linear": classname(getattr(method, "fp8_linear", None)),
                       "linear_kernel": classname(kernel), "use_asm_gemm": getattr(kernel, "use_asm_gemm", None),
                       "moe_kernel": classname(getattr(method, "moe_kernel", None)), "tensors": tensors})
    return result


def inspect_speculative(spec, expected_drafts, expected_al):
    rates = spec.synthetic_acceptance_rates
    resolved_al = 1 + sum(rates) if rates is not None else None
    if expected_al is None:
        if (spec.rejection_sample_method != "standard" or rates is not None
                or spec.synthetic_acceptance_length is not None
                or spec.num_speculative_tokens != expected_drafts):
            raise ValueError("C48 eval requires natural rejection sampling")
    elif (spec.rejection_sample_method != "synthetic" or rates is None
            or spec.num_speculative_tokens != expected_drafts
            or len(rates) != expected_drafts
            or not math.isclose(resolved_al, expected_al, rel_tol=0, abs_tol=1e-9)):
        raise ValueError(f"C48 synthetic acceptance mismatch: drafts={spec.num_speculative_tokens}, rates={rates}, expected_AL={expected_al}")
    return {"num_speculative_tokens": spec.num_speculative_tokens,
            "synthetic_acceptance_length": spec.synthetic_acceptance_length,
            "synthetic_acceptance_rates": rates,
            "resolved_acceptance_length": resolved_al,
            "rejection_sample_method": spec.rejection_sample_method}


def gemm_m_buckets(max_m):
    """ATOM model_runner.gemm_m_buckets: one M per aiter getPaddedM bucket up to max_m."""
    buckets = [m for m in (1, 2, 4, 8) if m <= max_m]
    for step, upper in ((16, 256), (32, 1024), (64, 4096), (128, max_m)):
        start = (buckets[-1] // step + 1) * step if buckets else step
        buckets.extend(range(start, min(upper, max_m) + 1, step))
    if max_m > 0 and buckets[-1] != max_m:
        buckets.append(max_m)
    return buckets


def warm_gemm_m_buckets(worker):
    """Run dummy forwards at every tuned-GEMM M bucket so first-use FlyDSL builds happen before serving.

    The optional second sweep (C48_P_GEMM_WARMUP_SWEEPS=2, the default) times the
    same sizes with every kernel already built; the difference between the sweeps
    is the first-use cost a serving prefill would pay.
    """
    import time
    import torch
    num_sweeps = int(os.environ.get("C48_P_GEMM_WARMUP_SWEEPS", "2"))
    if num_sweeps not in (1, 2):
        raise ValueError(f"C48_P_GEMM_WARMUP_SWEEPS must be 1 or 2, got {num_sweeps}")
    max_m = worker.scheduler_config.max_num_batched_tokens
    # Other tuned tables key exact M at multiples of 8 up to 512 (ATOM's linear pass).
    sizes = sorted(set(gemm_m_buckets(max_m)) | set(range(8, min(512, max_m) + 1, 8)) | {max_m - 1})
    sweeps = []
    for sweep in range(num_sweeps):
        times = []
        for m in sizes:
            if worker.rank == 0:
                print(f"[C48] P GEMM warmup sweep={sweep} M={m}", flush=True)
            start = time.perf_counter()
            worker.model_runner._dummy_run(m, skip_eplb=True)
            torch.cuda.synchronize()
            times.append((time.perf_counter() - start, m))
        sweeps.append(times)
    summary = (f"[C48] P GEMM M-bucket warmup rank={worker.rank} sizes={len(sizes)} "
               f"first={sum(t for t, _ in sweeps[0]):.1f}s ")
    if num_sweeps == 2:
        slowest = sorted(((a - b, m) for (a, m), (b, _) in zip(*sweeps)), reverse=True)[:8]
        summary += f"second={sum(t for t, _ in sweeps[1]):.1f}s first-minus-second "
    else:
        slowest = sorted(sweeps[0], reverse=True)[:8]
        summary += "slowest-first "
    print(summary + " ".join(f"M={m}:{d:.2f}s" for d, m in slowest), flush=True)


def mooncake_hca(local_rank, environ=os.environ):
    """ATOM's rail choice: rdma<N> for physical GPU N (ROCR masks before HIP)."""
    visible = (environ.get("ROCR_VISIBLE_DEVICES") or environ.get("HIP_VISIBLE_DEVICES")
               or environ.get("CUDA_VISIBLE_DEVICES") or "")
    devices = [d for d in visible.split(",") if d]
    return f"rdma{int(devices[local_rank]) if devices else local_rank}"


def nic_limits(device):
    """The firmware-set MR limits (max_mr, max_mr_size, page_size_cap) that bound Mooncake registrations."""
    import ctypes
    # struct ibv_device_attr through max_mr; the image has libibverbs but no ibv_devinfo (job 10092).
    fields = [("fw_ver", ctypes.c_char * 64), ("node_guid", ctypes.c_uint64), ("sys_image_guid", ctypes.c_uint64),
              ("max_mr_size", ctypes.c_uint64), ("page_size_cap", ctypes.c_uint64)]
    fields += [(name, ctypes.c_int) for name in ("vendor_id", "vendor_part_id", "hw_ver", "max_qp", "max_qp_wr",
                                                 "device_cap_flags", "max_sge", "max_sge_rd", "max_cq", "max_cqe", "max_mr")]
    attr = type("DeviceAttr", (ctypes.Structure,), {"_fields_": fields + [("rest", ctypes.c_byte * 512)]})()
    try:
        lib = ctypes.CDLL("libibverbs.so.1")
    except OSError as exc:
        return f"unavailable ({exc})"
    lib.ibv_get_device_list.restype = ctypes.POINTER(ctypes.c_void_p)
    lib.ibv_get_device_name.restype = ctypes.c_char_p
    lib.ibv_get_device_name.argtypes = lib.ibv_open_device.argtypes = lib.ibv_close_device.argtypes = [ctypes.c_void_p]
    lib.ibv_open_device.restype = ctypes.c_void_p
    lib.ibv_query_device.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    count = ctypes.c_int()
    devices = lib.ibv_get_device_list(ctypes.byref(count))
    try:
        names = [lib.ibv_get_device_name(devices[i]).decode() for i in range(count.value)]
        if device not in names:
            return f"unavailable ({device} not in {names})"
        context = lib.ibv_open_device(devices[names.index(device)])
        if not context:
            return f"unavailable (ibv_open_device {device} failed)"
        rc = lib.ibv_query_device(context, ctypes.byref(attr))
        lib.ibv_close_device(context)
    finally:
        if devices:
            lib.ibv_free_device_list(devices)
    if rc:
        return f"unavailable (ibv_query_device rc={rc})"
    return (f"fw_ver: {attr.fw_ver.decode()} max_mr_size: {attr.max_mr_size:#x} "
            f"page_size_cap: {attr.page_size_cap:#x} max_mr: {attr.max_mr}")


def prepare_store_memory(worker):
    """ATOM's Store memory recipe (pd_server_atom.sh prepare_mooncake_store_memory), once per node before the
    embedded segments pin: evict the models' page cache, then fault each NUMA node's share as THP from one bound
    thread. Mooncake pre-touches from many threads whose racing compactions leave 4 KiB pages, and ionic maps a
    whole MR at 4 KiB for one of them (job 10077)."""
    import fcntl
    import subprocess
    import sys
    import time
    here, out = Path(__file__).resolve().parent, Path(os.environ["C48_DIAG_DIR"])
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "mooncake-store-memory.lock", "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        done = out / "mooncake-store-memory.done"
        if done.exists():
            return
        from drop_page_cache import drop_page_cache
        spec = worker.vllm_config.speculative_config
        models = [worker.vllm_config.model_config.model] + ([spec.model] if spec and spec.model else [])
        root, start = os.path.commonpath(models), time.monotonic()
        drop_page_cache(root)
        print(f"[C48] Store memory: dropped the page cache of {root} in {time.monotonic() - start:.0f} s", flush=True)
        segments = (json.loads(Path(os.environ["MOONCAKE_CONFIG_PATH"]).read_text())["global_segment_size"]
                    * worker.parallel_config.world_size)
        nodes = [p for p in sorted(Path("/sys/devices/system/node").glob("node[0-9]*")) if (p / "cpulist").read_text().strip()]
        for node in nodes:
            info = {f[2].rstrip(":"): int(f[3]) << 10 for f in map(str.split, (node / "meminfo").read_text().splitlines())
                    if len(f) == 5}
            # A process bound to a node is killed out of memory past what the node can free; the workers hold theirs.
            gib = min(segments / len(nodes), info["MemFree"] + info["Active(file)"] + info["Inactive(file)"] - (64 << 30)) / 2**30
            print(f"[C48] Store memory: compacting {node.name} for {gib:.0f} of {segments / len(nodes) / 2**30:.0f} GiB"
                  f" (MemFree {info['MemFree'] >> 30} GiB, file {(info['Active(file)'] + info['Inactive(file)']) >> 30} GiB)",
                  flush=True)
            if gib < 1:
                continue
            # 1211 GiB on node1 reached ~830 GiB of THP in 600 s (job 10183) and enough in 1800 s (job 10122).
            try:
                subprocess.run([sys.executable, str(here / "numa_exec.py"), node.name.removeprefix("node"), sys.executable,
                                str(here / "compact_huge_pages.py"), f"{gib:.0f}"], check=False, timeout=1800)
            except subprocess.TimeoutExpired:
                print(f"[C48][WARN] Store memory: compacting {node.name} did not finish in 1800 s", flush=True)
        done.touch()


def enable_native_mla_96(config):
    """Opt-in native 96-head CPRR control for the pinned K3 decode recipe."""
    if os.environ.get("C48_D_MLA_NATIVE96") != "1":
        return None
    import hashlib
    import importlib

    pc = config.parallel_config
    topology = (pc.tensor_parallel_size, pc.decode_context_parallel_size,
                pc.pipeline_parallel_size, pc.prefill_context_parallel_size)
    if (os.environ.get("C48_DIAG_ROLE") != "decode" or topology != (8, 8, 1, 1)
            or pc.cp_kv_cache_interleave_size != 1
            or config.cache_config.cache_dtype != "fp8"
            or config.speculative_config.num_speculative_tokens != 4
            or config.model_config.hf_text_config.num_attention_heads != 96
            or os.environ.get("VLLM_ROCM_AITER_MLA_DCP_VERIFY") != "asm"
            or os.environ.get("AITER_MLA_USE_OPUS", "0") != "0"):
        raise ValueError("Native96 requires decode TP8/DCP8, PP1/PCP1, interleave=1, FP8, four drafts and ASM without OPUS")
    mla = importlib.import_module("aiter.mla")
    backend = importlib.import_module("vllm.v1.attention.backends.mla.rocm_aiter_mla")
    mla_hash = hashlib.sha256(Path(mla.__file__).read_bytes()).hexdigest()
    if mla.get_gfx() != "gfx950" or mla_hash != "6f6a902ab55fd78c8323e848144d181f72b683a4e0f7eaafb7550f529109050a":
        raise RuntimeError(f"Native96 requires the pinned gfx950 AITER dispatcher, got {mla_hash}")
    kernel = (Path(mla.__file__).resolve().parents[1] / "aiter_meta/hsa/gfx950/mla"
              / "mla_a8w8_qh32_qseqlen4_gqaratio32_lse_cprr_ps.co")
    if not kernel.is_file():
        raise RuntimeError(f"Native96 CPRR kernel is missing: {kernel}")
    original = (16, 32, 64, 128)
    enabled = original + (96,)
    if not getattr(backend, "_c48_native96_enabled", False):
        if backend._NATIVE_CPRR_HEADS != original or backend._asm_dcp_verify_heads(96) != 128:
            raise RuntimeError("Unexpected vLLM CPRR head selection before Native96")
        # Appending, rather than sorting, preserves padding for every other head count.
        backend._NATIVE_CPRR_HEADS = enabled
        backend._c48_native96_enabled = True
    if backend._NATIVE_CPRR_HEADS != enabled or backend._asm_dcp_verify_heads(96) != 96:
        raise RuntimeError("Native96 head selection did not activate")
    return {"selected_heads": 96, "original_padded_heads": 128,
            "native_head_choices": list(enabled), "aiter_mla_sha256": mla_hash,
            "backend_sha256": hashlib.sha256(Path(backend.__file__).read_bytes()).hexdigest(),
            "kernel_path": str(kernel), "kernel_sha256": hashlib.sha256(kernel.read_bytes()).hexdigest()}


def native_mla_96_builders(runner):
    """Verify actual target metadata builders, including every microbatch copy."""
    observed = []
    for groups in runner.attn_groups:
        for group in groups:
            for builder in group.metadata_builders:
                if getattr(builder, "_decode_num_heads", None) != 96:
                    continue
                if (not builder._asm_dcp_verify or builder._asm_dcp_verify_heads != 96
                        or builder._num_attention_heads != 96
                        or not 3 <= builder._mtp_decode_qlen <= 6):
                    raise RuntimeError("Native96 metadata builder is not using the supported CPRR shape")
                observed.append({"heads": builder._num_attention_heads, "max_query_length": builder._mtp_decode_qlen})
    if not observed:
        raise RuntimeError("Native96 has no matching target metadata builder")
    return observed


class ManifestWorker(Worker):
    def compile_or_warm_up_model(self):
        result = super().compile_or_warm_up_model()
        if os.environ.get("C48_P_GEMM_WARMUP") == "1":
            warm_gemm_m_buckets(self)
        return result

    def initialize_from_config(self, kv_cache_config):
        from pd_c48_timeline import startup_stage
        if os.environ.get("C48_MOONCAKE_HCA_PER_GPU") == "1":
            # Read by every Mooncake TE discovery in this process; the connectors are built below.
            os.environ["MC_TE_FILTERS"] = mooncake_hca(self.local_rank)
            print(f"[C48] Mooncake rank={self.rank} MC_TE_FILTERS={os.environ['MC_TE_FILTERS']}", flush=True)
            if self.local_rank == 0:
                print(f"[C48] Mooncake {os.environ['MC_TE_FILTERS']} limits: {nic_limits(os.environ['MC_TE_FILTERS'])}",
                      flush=True)
        store = os.environ.get("C48_MOONCAKE_STORE_COMPACT") == "1"
        if store:
            prepare_store_memory(self)
        with startup_stage(self, "initialize_from_config"):
            try:
                result = super().initialize_from_config(kv_cache_config)
                if getattr(self, "_c48_native96", None) is not None:
                    self._c48_native96["metadata_builders"] = native_mla_96_builders(self.model_runner)
                    path = Path(os.environ["C48_DIAG_DIR"]) / f'decode-rank-{self.rank}.json'
                    record = json.loads(path.read_text())
                    record["mla_native96"] = self._c48_native96
                    path.write_text(json.dumps(record, indent=2) + "\n")
                    print(f"[C48-NATIVE96] rank={self.rank} {json.dumps(self._c48_native96)}", flush=True)
                return result
            finally:
                if store:
                    with open("/proc/self/smaps_rollup") as f:
                        huge = next((" ".join(line.split()) for line in f if line.startswith("AnonHugePages:")), "?")
                    print(f"[C48] Store rank={self.rank} {huge}", flush=True)

    def annotate_profile(self, scheduler_output):
        original = super().annotate_profile(scheduler_output)
        if (os.environ.get("C48_PREFILL_TIMELINE") != "1"
                or os.environ["C48_DIAG_ROLE"] != "prefill"
                or not self.profiler or not self.profiler.should_annotate):
            return original
        import torch
        from pd_c48_timeline import batch_key

        @contextmanager
        def annotated():
            with original, torch.profiler.record_function(
                "C48-PREFILL-BATCH:" + batch_key(scheduler_output)
            ):
                yield
        return annotated()

    def load_model(self, *, load_dummy_weights=False):
        from pd_c48_memory import LoadingMemoryProbe
        self._c48_native96 = enable_native_mla_96(self.vllm_config)
        with LoadingMemoryProbe(self):
            super().load_model(load_dummy_weights=load_dummy_weights)
        runner = self.model_runner
        target = inspect_model(runner.get_model())
        draft = inspect_model(runner.get_draft_model())
        counts = {key: sum("Fp8PtpcOnlineLinearMethod" in item["quant_method"] for item in rows)
                  for key, rows in (("target", target), ("draft", draft))}
        spec = self.vllm_config.speculative_config
        data = {"role": os.environ["C48_DIAG_ROLE"], "rank": self.rank,
                "target": target, "draft": draft, "ptpc_counts": counts,
                "profiler_config": asdict(self.vllm_config.profiler_config),
                "speculative": inspect_speculative(spec,
                    int(os.environ["C48_DIAG_DRAFT_TOKENS"]),
                    float(os.environ["C48_DIAG_AL"]) if os.environ["C48_DIAG_AL"] else None)}
        if self._c48_native96 is not None:
            data["mla_native96"] = self._c48_native96
        if os.environ.get("C48_EXPECT_SHARED_PREFIX_CHECKPOINT") == "1":
            import hashlib
            from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.store import worker as store_worker
            digest = hashlib.sha256(Path(store_worker.__file__).read_bytes()).hexdigest()
            assert digest == "dae33150acc1332622c63641a864415d95ccaf719250555ffdbeea7cd8719b9f", "Unexpected checkpoint Store worker"
            data["mooncake_boundary_store"] = {"module_path": store_worker.__file__, "module_sha256": digest}
        if os.environ.get("AITER_MLA_USE_OPUS") == "1":
            import hashlib
            import importlib
            aiter = importlib.import_module("aiter")
            mla = importlib.import_module("aiter.mla")
            source = Path(mla.__file__).read_bytes()
            assert callable(getattr(aiter, "opus_mla_decode_fp8_fwd", None)), "AITER lacks FP8 MLA OPUS"
            assert b"AITER_MLA_USE_OPUS" in source, "AITER MLA dispatcher lacks the OPUS switch"
            data["aiter_mla_opus"] = {"operator_available": True, "module_path": mla.__file__,
                                      "module_sha256": hashlib.sha256(source).hexdigest()}
        if os.environ.get("C48_MOONCAKE_DCP_TRANSFER") == "1":
            import hashlib
            from vllm.distributed.kv_transfer.kv_connector.v1.mooncake import (
                dcp_staging, dcp_transfer, mooncake_connector,
            )
            assert hasattr(mooncake_connector.MooncakeConnectorWorker, "_send_staged_dcp")
            assert {"mla_block_sizes", "mamba_state_sizes", "req_num_tokens"}.issubset(
                mooncake_connector.MooncakeXferMetadata.__struct_fields__)
            data["mooncake_dcp"] = {
                module.__name__: {"path": module.__file__,
                                  "sha256": hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()}
                for module in (mooncake_connector, dcp_transfer, dcp_staging)
            }
        out = Path(os.environ["C48_DIAG_DIR"])
        out.mkdir(parents=True, exist_ok=True)
        (out / f'{data["role"]}-rank-{self.rank}.json').write_text(json.dumps(data, indent=2) + "\n")
        print(f"[C48] Loaded model quantization manifest rank={self.rank} PTPC={counts}", flush=True)
        assert bool(counts["target"]) == (os.environ["C48_DIAG_PTPC"] == "1"), counts
        expect = json.loads(os.environ.get("C48_EXPECT_QUANT", "{}"))
        got = {k: sum(item["quant_method"].endswith("." + k) for item in target) for k in expect}
        if expect and self.parallel_config.pipeline_parallel_size > 1:
            import torch
            from vllm.distributed.parallel_state import get_pp_group
            # The expected coverage names the whole target, split across PP stages.
            coverage = torch.tensor(list(got.values()), dtype=torch.int64, device="cpu")
            torch.distributed.all_reduce(coverage, group=get_pp_group().cpu_group)
            got = dict(zip(got, coverage.tolist()))
        assert got == expect, f"C48 quantization coverage mismatch: got={got} expected={expect}"
