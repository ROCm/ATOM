# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

"""vLLM + ATOM teacher engine for speculative distillation.

Launches a teacher model in a subprocess using vLLM with ATOM plugin
(auto-registered via pip entry points). Patched vLLM enables
``extract_hidden_states`` speculative mode: the engine captures hidden
states during prefill and stores them to **Mooncake** via AtomRL's
``MooncakeHiddenStatesConnector``.  The training process reads them
via ``EagleMooncakeStore.get()``.

Architecture::

    Training Process                 vLLM Subprocess (ATOM plugin)
    ─────────────────               ──────────────────────────────
    send input_ids via /dev/shm  →  llm.generate(prompts,
                                        sampling_params)
                                    → vLLM extract_hidden_states
                                    → MooncakeHiddenStatesConnector
                                      stores to Mooncake
    get hidden_states from       ←  returns mooncake keys + shapes
    EagleMooncakeStore

Requires:
    - vLLM with extract_hidden_states patches
    - ATOM pip-installed (auto-registers as vLLM platform/model plugin)
    - Mooncake Transfer Engine (RDMA or TCP)
"""

from __future__ import annotations

import json
import logging
import os
import signal
import socket
import subprocess
import sys
import tempfile
import textwrap
import time
from typing import Any

import torch

logger = logging.getLogger(__name__)


def _detect_local_ip() -> str:
    """Return the route-selected local IP, falling back to hostname lookup."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as route_socket:
            route_socket.connect(("8.8.8.8", 80))
            return route_socket.getsockname()[0]
    except OSError:
        return socket.gethostbyname(socket.gethostname())


_WORKER_SCRIPT = textwrap.dedent("""\
import json
import os
import sys
import time
import logging

# ---- Isolate from torchrun environment ----
for key in list(os.environ.keys()):
    if any(key.startswith(p) for p in [
        "MASTER_ADDR", "MASTER_PORT", "RANK", "LOCAL_RANK",
        "WORLD_SIZE", "LOCAL_WORLD_SIZE", "GROUP_RANK",
        "GROUP_WORLD_SIZE", "ROLE_RANK", "ROLE_WORLD_SIZE",
        "TORCHELASTIC_", "TORCH_NCCL_", "NCCL_ASYNC",
        "OMP_NUM_THREADS",
    ]):
        del os.environ[key]

logging.basicConfig(stream=sys.stderr, level=logging.INFO)
logger = logging.getLogger("vllm_atom_teacher")

import torch
from transformers import AutoConfig

cmd_fifo = sys.argv[1]
resp_fifo = sys.argv[2]
model_path = sys.argv[3]
tp_size = int(sys.argv[4])
gpu_ids_str = sys.argv[5]
quantization = sys.argv[6] if len(sys.argv) > 6 else ""
max_seq_len = int(sys.argv[7]) if len(sys.argv) > 7 else 8192
# argv[8]: JSON-encoded aux_layer_ids override (empty string => use auto).
aux_layer_ids_override_json = sys.argv[8] if len(sys.argv) > 8 else ""

gpu_ids = [int(x) for x in gpu_ids_str.split(",")]
os.environ["CUDA_VISIBLE_DEVICES"] = gpu_ids_str
if "HIP_VISIBLE_DEVICES" in os.environ:
    del os.environ["HIP_VISIBLE_DEVICES"]
if "ROCR_VISIBLE_DEVICES" in os.environ:
    del os.environ["ROCR_VISIBLE_DEVICES"]

# When no quantization is requested, fully disable ATOM to use native vLLM
# models (preserves vLLM's compressed-tensors INT4 MoE support).
if not quantization:
    os.environ["ATOM_DISABLE_VLLM_PLUGIN"] = "1"
    os.environ["VLLM_PLUGINS"] = ""
    logger.info("ATOM plugin disabled (no quantization requested)")
else:
    logger.info("ATOM plugin active for %s quantization", quantization)

# Get model hidden size from config
hf_config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
hf_config_text = getattr(hf_config, "text_config", hf_config)
hidden_size = getattr(hf_config_text, "hidden_size", None)
num_layers = getattr(hf_config_text, "num_hidden_layers", 32)

# Aux layer IDs in vLLM convention (capture-before-layer).
# Default vLLM auto-pick: [2, N//2, N-3] — yields the same hidden state
# as hooking layer output at [1, N//2-1, N-4].
# Config override wins when provided (e.g. NVIDIA gpt-oss-120b-Eagle3 uses
# [1, 17, 32]). The override is the literal list the draft model was built
# for; we pass it through unchanged so the captured features line up with
# the draft architecture.
if aux_layer_ids_override_json:
    aux_layer_ids = json.loads(aux_layer_ids_override_json)
    logger.info("Model: %s, hidden=%d, layers=%d, aux_layers=%s (config override)",
                model_path, hidden_size, num_layers, aux_layer_ids)
else:
    aux_layer_ids = [2, num_layers // 2, num_layers - 3]
    logger.info("Model: %s, hidden=%d, layers=%d, aux_layers=%s (auto)",
                model_path, hidden_size, num_layers, aux_layer_ids)

# Create vLLM Engine with extract_hidden_states + MooncakeHiddenStatesConnector
logger.info("Creating vLLM Engine (tp=%d, quant=%s, max_seq=%d)",
            tp_size, quantization or "none", max_seq_len)

from vllm import LLM, SamplingParams

engine_kwargs = dict(
    model=model_path,
    tensor_parallel_size=tp_size,
    trust_remote_code=True,
    distributed_executor_backend="mp",
    disable_custom_all_reduce=True,
    enable_prefix_caching=False,
    max_model_len=max_seq_len,
    max_num_batched_tokens=40000,
    max_num_seqs=128,
    gpu_memory_utilization=0.90,
    speculative_config={
        "method": "extract_hidden_states",
        "num_speculative_tokens": 1,
        "draft_model_config": {
            "hf_config": {
                "eagle_aux_hidden_state_layer_ids": list(aux_layer_ids)
            }
        },
    },
    kv_transfer_config={
        "kv_connector": "MooncakeHiddenStatesConnector",
        "kv_connector_module_path": (
            "atomrl.engine.inference.mooncake_hidden_states_connector"
        ),
        "kv_role": "kv_producer",
    },
)

engine_kwargs["kv_cache_dtype"] = "fp8_e4m3"

if quantization in ("mxfp4", "fp4"):
    engine_kwargs["quantization"] = "mxfp4"
    logger.info("MXFP4 online quantization enabled (ATOM/AITER ROCm)")
elif quantization:
    engine_kwargs["quantization"] = quantization
    logger.info("Quantization: %s", quantization)

llm = LLM(**engine_kwargs)

logger.info("vLLM Engine created successfully")

# Extract lm_head weight and save (from safetensors index — no GPU needed)
lm_head_path = f"/dev/shm/atomrl_vllm_lm_head_{os.getpid()}.pt"
try:
    from safetensors import safe_open
    idx_path = os.path.join(model_path, "model.safetensors.index.json")
    with open(idx_path) as _f:
        weight_map = json.load(_f)["weight_map"]
    lm_head_w = None
    for key in ["lm_head.weight", "language_model.lm_head.weight",
                "model.lm_head.weight"]:
        if key in weight_map:
            shard = weight_map[key]
            with safe_open(os.path.join(model_path, shard),
                           framework="pt", device="cpu") as sf:
                lm_head_w = sf.get_tensor(key)
            logger.info("lm_head loaded from safetensors: key=%s, shard=%s",
                        key, shard)
            break
    if lm_head_w is None:
        raise KeyError("lm_head.weight not found in safetensors index")
    torch.save(lm_head_w, lm_head_path)
    logger.info("lm_head.weight saved: %s", tuple(lm_head_w.shape))
except Exception as e:
    logger.warning("Could not extract lm_head: %s", e)
    lm_head_path = ""

num_aux = len(aux_layer_ids)
req_counter = 0

# Signal ready
resp_f = open(resp_fifo, "w")
resp_f.write(json.dumps({
    "status": "ready",
    "hidden_size": hidden_size,
    "num_aux_layers": num_aux,
    "aux_layer_ids": aux_layer_ids,
    "lm_head_path": lm_head_path,
}) + "\\n")
resp_f.flush()

# Command loop
cmd_f = open(cmd_fifo, "r")
sampling_params = SamplingParams(max_tokens=1, temperature=0)

for line in cmd_f:
    line = line.strip()
    if not line:
        continue
    msg = json.loads(line)
    cmd = msg["cmd"]

    if cmd == "extract_hidden":
        input_path = msg["input_path"]
        data = torch.load(input_path, map_location="cpu", weights_only=True)
        input_ids = data["input_ids"]
        attention_mask = data.get("attention_mask", None)

        B, T = input_ids.shape
        req_counter += 1

        # Strip padding using attention_mask so vLLM only processes real tokens.
        # Use bool indexing — works regardless of left or right padding.
        unpadded_prompts = []
        for i in range(B):
            if attention_mask is not None:
                ids = input_ids[i][attention_mask[i].bool()].tolist()
            else:
                ids = input_ids[i].tolist()
            unpadded_prompts.append({"prompt_token_ids": ids})

        mooncake_keys = []
        tensor_shapes_list = []
        tensor_dtypes_list = []
        seq_lens = []

        # Send all sequences at once — vLLM continuous batching handles
        # variable-length prefill efficiently without padding waste.
        results = llm.generate(unpadded_prompts, sampling_params, use_tqdm=False)

        for i, output in enumerate(results):
            kv_params = getattr(output, "kv_transfer_params", None)
            if kv_params is None:
                logger.error("No kv_transfer_params for request %d", i)
                continue

            mooncake_key = kv_params.get("mooncake_key", f"atomrl_{os.getpid()}_{req_counter}_{i}")
            shapes = kv_params.get("tensor_shapes", {})
            dtypes = kv_params.get("tensor_dtypes", {})
            prompt_tokens = len(output.prompt_token_ids)

            mooncake_keys.append(mooncake_key)
            tensor_shapes_list.append(shapes)
            tensor_dtypes_list.append(dtypes)
            seq_lens.append(prompt_tokens)

        resp_f.write(json.dumps({
            "status": "ok",
            "B": B,
            "T": T,
            "hidden_size": hidden_size,
            "num_aux_layers": num_aux,
            "mooncake_keys": mooncake_keys,
            "tensor_shapes": tensor_shapes_list,
            "tensor_dtypes": tensor_dtypes_list,
            "seq_lens": seq_lens,
        }) + "\\n")
        resp_f.flush()

        try:
            os.unlink(input_path)
        except OSError:
            pass

    elif cmd == "shutdown":
        if lm_head_path and os.path.exists(lm_head_path):
            os.unlink(lm_head_path)
        resp_f.write(json.dumps({"status": "ok"}) + "\\n")
        resp_f.flush()
        break

cmd_f.close()
resp_f.close()
""")


class VllmTeacherEngine:
    """Teacher engine: vLLM + ATOM plugin + Mooncake.

    Launches vLLM in a subprocess on dedicated inference GPUs.
    ATOM auto-registers as a vLLM plugin via pip entry points, providing
    MXFP4 online quantization through AITER kernels on ROCm.
    Patched vLLM enables extract_hidden_states mode: hidden states are
    stored to Mooncake by MooncakeHiddenStatesConnector, and the training
    process fetches them via ``EagleMooncakeStore.get()``.
    """

    def __init__(
        self,
        model_name: str,
        gpu_ids: list[int],
        mooncake_config: Any = None,
        *,
        tensor_parallel_size: int = 1,
        quantization: str = "",
        max_batch_size: int = 32,
        max_seq_len: int = 4096,
        local_device: torch.device | None = None,
        aux_layer_ids: list[int] | None = None,
    ):
        self._model_name = model_name
        self._gpu_ids = gpu_ids
        self._mooncake_config = mooncake_config
        self._tp_size = tensor_parallel_size or len(gpu_ids)
        self._quantization = quantization
        self._max_batch_size = max_batch_size
        self._max_seq_len = max_seq_len
        # When None, the worker auto-picks [2, N//2, N-3]. Otherwise this list
        # is the literal eagle_aux_hidden_state_layer_ids passed to vLLM ──
        # must match what the draft model was constructed for.
        self._aux_layer_ids_override = aux_layer_ids
        self._local_device = local_device or torch.device("cuda:0")

        self._proc: subprocess.Popen | None = None
        self._cmd_fifo: str | None = None
        self._resp_fifo: str | None = None
        self._cmd_f = None
        self._resp_f = None
        self._initialized = False

        self._hidden_size: int = 0
        self._num_aux_layers: int = 0
        self._aux_layer_ids: list[int] = []
        self._lm_head_weight: torch.Tensor | None = None
        self._mooncake_store: Any = None

    @property
    def is_alive(self) -> bool:
        return (
            self._initialized and self._proc is not None and self._proc.poll() is None
        )

    def start(self) -> None:
        """Start vLLM+ATOM subprocess and set up Mooncake store."""
        if self.is_alive:
            return

        fifo_dir = tempfile.mkdtemp(prefix="atomrl_vllm_teacher_")
        self._cmd_fifo = os.path.join(fifo_dir, "cmd")
        self._resp_fifo = os.path.join(fifo_dir, "resp")
        os.mkfifo(self._cmd_fifo)
        os.mkfifo(self._resp_fifo)

        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = ",".join(str(g) for g in self._gpu_ids)
        env.pop("HIP_VISIBLE_DEVICES", None)
        env.pop("ROCR_VISIBLE_DEVICES", None)
        env.pop("PYTORCH_CUDA_ALLOC_CONF", None)
        for key in list(env.keys()):
            if any(
                key.startswith(p)
                for p in [
                    "MASTER_ADDR",
                    "MASTER_PORT",
                    "RANK",
                    "LOCAL_RANK",
                    "WORLD_SIZE",
                    "LOCAL_WORLD_SIZE",
                    "GROUP_RANK",
                    "GROUP_WORLD_SIZE",
                    "ROLE_RANK",
                    "ROLE_WORLD_SIZE",
                    "TORCHELASTIC_",
                    "TORCH_NCCL_",
                    "NCCL_ASYNC",
                    "OMP_NUM_THREADS",
                ]
            ):
                del env[key]

        # Skip ATOM plugin when no quantization (use native vLLM models,
        # preserving compressed-tensors INT4 MoE support).
        if not self._quantization:
            env["ATOM_DISABLE_VLLM_PLUGIN"] = "1"
            env["VLLM_PLUGINS"] = ""

        # Set Mooncake env vars for MooncakeHiddenStatesConnector
        if self._mooncake_config is not None:
            mc = self._mooncake_config
            local_ip = _detect_local_ip()

            master_addr = getattr(mc, "master_server_address", "") or ""
            metadata_server = getattr(mc, "metadata_server", "") or ""
            protocol = getattr(mc, "protocol", "tcp") or "tcp"
            device_name = getattr(mc, "device_name", "") or ""

            env["MOONCAKE_LOCAL_HOSTNAME"] = local_ip
            env["MOONCAKE_MASTER_SERVER"] = master_addr
            env["MOONCAKE_METADATA_SERVER"] = metadata_server
            env["MOONCAKE_PROTOCOL"] = protocol
            env["MOONCAKE_DEVICE_NAME"] = device_name
            env["MOONCAKE_GLOBAL_SEGMENT_SIZE"] = str(
                mc.global_segment_size_bytes
                if hasattr(mc, "global_segment_size_bytes")
                else 17179869184
            )
            env["MOONCAKE_LOCAL_BUFFER_SIZE"] = str(
                mc.local_buffer_size_bytes
                if hasattr(mc, "local_buffer_size_bytes")
                else 4294967296
            )
            env["MOONCAKE_ENABLE_GPU_DIRECT"] = "0"
            env["MOONCAKE_ENABLE_HARD_PIN"] = "0"

            try:
                from transformers import AutoConfig as _AC

                _hf = _AC.from_pretrained(self._model_name, trust_remote_code=True)
                _hf_text = getattr(_hf, "text_config", _hf)
                _hdim = getattr(_hf_text, "hidden_size", 4096)
            except (ImportError, OSError, ValueError):
                _hdim = 4096
            from atomrl.transfer.eagle_mooncake_store import (
                calculate_eagle3_buffer_size,
            )

            vllm_host_buf = calculate_eagle3_buffer_size(
                max_seq_len=self._max_seq_len,
                batch_size=1,
                hidden_dim=_hdim,
                safety_margin=2.0,
            )
            env["MOONCAKE_HOST_BUFFER_SIZE"] = str(vllm_host_buf)
            logger.info(
                "MOONCAKE_HOST_BUFFER_SIZE for vLLM subprocess: %d (%.1fGB)",
                vllm_host_buf,
                vllm_host_buf / 1024**3,
            )

            logger.info(
                "Mooncake env vars for vLLM subprocess: local=%s, master=%s, "
                "metadata=%s, protocol=%s, device=%s",
                local_ip,
                master_addr,
                metadata_server,
                protocol,
                device_name,
            )

        logger.info(
            "VllmTeacherEngine: starting subprocess for %s "
            "(tp=%d, gpus=%s, quant=%s)",
            self._model_name,
            self._tp_size,
            self._gpu_ids,
            self._quantization or "none",
        )

        aux_override_arg = (
            json.dumps(list(self._aux_layer_ids_override))
            if self._aux_layer_ids_override
            else ""
        )
        self._proc = subprocess.Popen(
            [
                sys.executable,
                "-u",
                "-c",
                _WORKER_SCRIPT,
                self._cmd_fifo,
                self._resp_fifo,
                self._model_name,
                str(self._tp_size),
                ",".join(str(g) for g in self._gpu_ids),
                self._quantization or "",
                str(self._max_seq_len),
                aux_override_arg,
            ],
            stdin=subprocess.DEVNULL,
            stdout=None,
            stderr=None,
            env=env,
            start_new_session=True,
        )

        # FIFO stays open for the worker lifetime.
        self._resp_f = open(self._resp_fifo, "r")  # noqa: SIM115 -- lifetime handle
        resp_line = self._resp_f.readline()
        if not resp_line:
            raise RuntimeError(
                "vLLM teacher subprocess exited before ready. "
                "Check stderr for vLLM/ATOM loading errors."
            )
        resp = json.loads(resp_line)
        if resp.get("status") != "ready":
            raise RuntimeError(f"vLLM teacher failed to start: {resp}")

        # FIFO stays open for the worker lifetime.
        self._cmd_f = open(self._cmd_fifo, "w")  # noqa: SIM115 -- lifetime handle

        self._hidden_size = resp["hidden_size"]
        self._num_aux_layers = resp["num_aux_layers"]
        self._aux_layer_ids = resp.get("aux_layer_ids", [])

        lm_head_path = resp.get("lm_head_path", "")
        if lm_head_path and os.path.exists(lm_head_path):
            self._lm_head_weight = torch.load(
                lm_head_path, map_location="cpu", weights_only=True
            )
            logger.info("lm_head.weight loaded: %s", tuple(self._lm_head_weight.shape))

        # Set up training-side Mooncake store
        if self._mooncake_config is not None:
            try:
                from atomrl.transfer.eagle_mooncake_store import EagleMooncakeStore
                from atomrl.transfer.mooncake_config import MooncakeConfig

                mc = self._mooncake_config
                local_ip = _detect_local_ip()

                mc_cfg = MooncakeConfig(
                    master_server_address=getattr(mc, "master_server_address", ""),
                    metadata_server=getattr(mc, "metadata_server", ""),
                    local_hostname=local_ip,
                    protocol=getattr(mc, "protocol", "tcp"),
                    device_name=getattr(mc, "device_name", ""),
                    global_segment_size=getattr(mc, "global_segment_size", "16GB"),
                    local_buffer_size=getattr(mc, "local_buffer_size", "4GB"),
                    max_seq_len=self._max_seq_len,
                    hidden_dim=self._hidden_size,
                    get_retry_wait_seconds=getattr(mc, "get_retry_wait_seconds", 1.0),
                    get_retry_max_wait_seconds=getattr(
                        mc, "get_retry_max_wait_seconds", 90.0
                    ),
                )
                self._mooncake_store = EagleMooncakeStore(mc_cfg)
                self._mooncake_store.setup(self._local_device)
                logger.info("Training-side EagleMooncakeStore initialized")
            except Exception as e:
                logger.error("Failed to init training-side Mooncake: %s", e)
                raise

        self._initialized = True
        logger.info(
            "VllmTeacherEngine: ready (pid=%d, hidden_size=%d, "
            "aux_layers=%s, mooncake=%s)",
            self._proc.pid,
            self._hidden_size,
            self._aux_layer_ids,
            self._mooncake_store is not None,
        )

    def _send_cmd(self, cmd: dict) -> dict:
        if not self.is_alive:
            raise RuntimeError("vLLM teacher worker is not running")
        self._cmd_f.write(json.dumps(cmd) + "\n")
        self._cmd_f.flush()
        resp_line = self._resp_f.readline()
        if not resp_line:
            raise RuntimeError("vLLM teacher worker closed response FIFO")
        return json.loads(resp_line)

    def extract_hidden_states(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        recv_device: torch.device | None = None,
    ) -> dict[str, torch.Tensor]:
        """Run teacher forward via vLLM and fetch hidden states from Mooncake.

        Args:
            input_ids: ``[B, T]`` token ids.
            attention_mask: ``[B, T]`` mask (1 = valid, 0 = pad).
            recv_device: Device for received tensors. Defaults to local GPU.

        Returns:
            Dict with ``hidden_states`` ``[B, T, 3*D]`` (3 concatenated aux layers),
            ``token_embeds`` ``[B, T, D]`` (first aux layer as embed proxy),
            and ``input_ids``.
        """
        if not self.is_alive:
            self.start()

        input_path = f"/dev/shm/atomrl_vllm_input_{os.getpid()}.pt"
        torch.save(
            {"input_ids": input_ids.cpu(), "attention_mask": attention_mask.cpu()},
            input_path,
        )

        resp = self._send_cmd(
            {
                "cmd": "extract_hidden",
                "input_path": input_path,
            }
        )

        if resp.get("status") != "ok":
            raise RuntimeError(f"extract_hidden failed: {resp}")

        resp["B"]
        resp["T"]
        hidden_size = resp["hidden_size"]
        num_aux = resp["num_aux_layers"]
        mooncake_keys = resp["mooncake_keys"]
        resp["tensor_shapes"]
        resp["tensor_dtypes"]
        seq_lens = resp["seq_lens"]

        # MooncakeHiddenStatesConnector splits aux layers: first N-1 → _hs,
        # last → _lhs.  We retrieve both and concatenate to [T, N*H].
        training_hidden_size = max(num_aux - 1, 1) * hidden_size

        all_hidden = []
        all_embeds = []
        all_ids = []
        all_last_hs = []

        if recv_device is None:
            recv_device = self._local_device

        for i, key in enumerate(mooncake_keys):
            seq_len = seq_lens[i]
            shapes = {
                "hidden_states": (seq_len, training_hidden_size),
                "input_ids": (seq_len,),
                "last_hidden_states": (seq_len, hidden_size),
            }
            dtypes = {
                "hidden_states": torch.bfloat16,
                "input_ids": torch.int64,
                "last_hidden_states": torch.bfloat16,
            }

            if self._mooncake_store is not None:
                output = self._mooncake_store.get(
                    key,
                    shapes,
                    dtypes,
                    device=recv_device,
                )
                hs_training = output.hidden_states  # [T, (N-1)*H]
                hs_last = output.last_hidden_states  # [T, H]

                # NaN diagnostic: check received data per aux layer
                for li in range(hs_training.shape[-1] // hidden_size):
                    chunk = hs_training[:, li * hidden_size : (li + 1) * hidden_size]
                    nans = torch.isnan(chunk).any(dim=-1).sum().item()
                    if nans > 0:
                        logger.warning(
                            "NaN from Mooncake hs_training layer[%d]: %d/%d tokens, "
                            "seq=%d, key=%s",
                            li,
                            nans,
                            seq_len,
                            i,
                            key,
                        )
                nans_last = torch.isnan(hs_last).any(dim=-1).sum().item()
                if nans_last > 0:
                    logger.warning(
                        "NaN from Mooncake last_hidden_states: %d/%d tokens, "
                        "seq=%d, key=%s",
                        nans_last,
                        seq_len,
                        i,
                        key,
                    )

                # cat() copies both tensors, so hs_concat survives buffer free.
                hs_concat = torch.cat([hs_training, hs_last], dim=-1)  # [T, N*H]

                all_hidden.append(hs_concat.unsqueeze(0))  # [1, T, N*H]
                all_embeds.append(
                    hs_concat[:, :hidden_size].unsqueeze(0)  # first aux as embed proxy
                )
                all_ids.append(output.input_ids.clone().unsqueeze(0))
                # Use slice of hs_concat (already a copy) instead of hs_last
                # (which wraps a Mooncake buffer freed by remove below).
                all_last_hs.append(hs_concat[:, -hidden_size:].clone().unsqueeze(0))

                self._mooncake_store.remove_eagle3_tensors(
                    key,
                    has_last_hidden_states=True,
                    has_target=False,
                )
            else:
                raise RuntimeError(
                    "Mooncake store not available. vLLM extract_hidden_states "
                    "requires Mooncake for hidden state transfer."
                )

        # Pad variable-length results to max seq_len for batched training.
        max_T = max(seq_lens)
        D_hidden = all_hidden[0].shape[-1]
        D_embed = all_embeds[0].shape[-1]
        D_last = all_last_hs[0].shape[-1]
        B_out = len(all_hidden)

        hidden_states = torch.zeros(
            B_out, max_T, D_hidden, dtype=torch.bfloat16, device=recv_device
        )
        token_embeds = torch.zeros(
            B_out, max_T, D_embed, dtype=torch.bfloat16, device=recv_device
        )
        ret_ids = torch.zeros(B_out, max_T, dtype=torch.int64, device=recv_device)
        last_hidden_states = torch.zeros(
            B_out, max_T, D_last, dtype=torch.bfloat16, device=recv_device
        )

        for i in range(B_out):
            slen = seq_lens[i]
            hidden_states[i, :slen] = all_hidden[i].squeeze(0)
            token_embeds[i, :slen] = all_embeds[i].squeeze(0)
            ret_ids[i, :slen] = all_ids[i].squeeze(0)
            last_hidden_states[i, :slen] = all_last_hs[i].squeeze(0)

        return {
            "hidden_states": hidden_states,
            "token_embeds": token_embeds,
            "input_ids": ret_ids,
            "last_hidden_states": last_hidden_states,
        }

    def get_lm_head_weight(self) -> torch.Tensor:
        if self._lm_head_weight is None:
            logger.info(
                "lm_head not from engine; loading from safetensors: %s",
                self._model_name,
            )
            import json as _json

            from safetensors import safe_open

            model_dir = self._model_name
            idx_path = os.path.join(model_dir, "model.safetensors.index.json")
            if not os.path.exists(idx_path):
                from huggingface_hub import hf_hub_download

                idx_path = hf_hub_download(model_dir, "model.safetensors.index.json")
                model_dir = os.path.dirname(idx_path)

            with open(idx_path) as f:
                weight_map = _json.load(f)["weight_map"]

            for key in [
                "lm_head.weight",
                "language_model.lm_head.weight",
                "model.lm_head.weight",
            ]:
                if key in weight_map:
                    shard = weight_map[key]
                    shard_path = os.path.join(model_dir, shard)
                    if not os.path.exists(shard_path):
                        from huggingface_hub import hf_hub_download

                        shard_path = hf_hub_download(self._model_name, shard)
                    with safe_open(shard_path, framework="pt", device="cpu") as sf:
                        self._lm_head_weight = sf.get_tensor(key).to(torch.bfloat16)
                    break

            if self._lm_head_weight is None:
                raise KeyError(
                    f"lm_head.weight not found in {idx_path}. "
                    f"Available keys with 'lm_head': "
                    f"{[k for k in weight_map if 'lm_head' in k]}"
                )
            logger.info("lm_head.weight loaded: %s", tuple(self._lm_head_weight.shape))
        return self._lm_head_weight

    def get_embed_weight(self) -> torch.Tensor:
        """Return the teacher's embed_tokens.weight (lazy-loaded from safetensors)."""
        if not hasattr(self, "_embed_weight") or self._embed_weight is None:
            import json as _json

            from safetensors import safe_open

            model_dir = self._model_name
            idx_path = os.path.join(model_dir, "model.safetensors.index.json")
            if not os.path.exists(idx_path):
                from huggingface_hub import hf_hub_download

                idx_path = hf_hub_download(model_dir, "model.safetensors.index.json")
                model_dir = os.path.dirname(idx_path)

            with open(idx_path) as f:
                weight_map = _json.load(f)["weight_map"]

            self._embed_weight = None
            for key in [
                "model.embed_tokens.weight",
                "language_model.model.embed_tokens.weight",
                "embed_tokens.weight",
            ]:
                if key in weight_map:
                    shard = weight_map[key]
                    shard_path = os.path.join(model_dir, shard)
                    if not os.path.exists(shard_path):
                        from huggingface_hub import hf_hub_download

                        shard_path = hf_hub_download(self._model_name, shard)
                    with safe_open(shard_path, framework="pt", device="cpu") as sf:
                        self._embed_weight = sf.get_tensor(key).to(torch.bfloat16)
                    break

            if self._embed_weight is None:
                raise KeyError(
                    f"embed_tokens.weight not found in {idx_path}. "
                    f"Available keys with 'embed': "
                    f"{[k for k in weight_map if 'embed' in k.lower()]}"
                )
            logger.info(
                "embed_tokens.weight loaded: %s", tuple(self._embed_weight.shape)
            )
        return self._embed_weight

    def get_norm_weight(self) -> tuple[torch.Tensor, float]:
        """Return (norm_weight, rms_norm_eps) from the teacher model.

        vLLM extract_hidden_states returns pre-norm hidden states.
        These are needed to norm them before computing teacher logits.
        """
        if not hasattr(self, "_norm_weight") or self._norm_weight is None:
            import json as _json

            from safetensors import safe_open

            model_dir = self._model_name
            idx_path = os.path.join(model_dir, "model.safetensors.index.json")
            if not os.path.exists(idx_path):
                from huggingface_hub import hf_hub_download

                idx_path = hf_hub_download(model_dir, "model.safetensors.index.json")
                model_dir = os.path.dirname(idx_path)

            with open(idx_path) as f:
                weight_map = _json.load(f)["weight_map"]

            self._norm_weight = None
            for key in [
                "model.norm.weight",
                "language_model.model.norm.weight",
                "norm.weight",
            ]:
                if key in weight_map:
                    shard = weight_map[key]
                    shard_path = os.path.join(model_dir, shard)
                    if not os.path.exists(shard_path):
                        from huggingface_hub import hf_hub_download

                        shard_path = hf_hub_download(self._model_name, shard)
                    with safe_open(shard_path, framework="pt", device="cpu") as sf:
                        self._norm_weight = sf.get_tensor(key).to(torch.bfloat16)
                    break

            if self._norm_weight is None:
                raise KeyError(
                    f"norm.weight not found in {idx_path}. "
                    f"Available keys with 'norm': "
                    f"{[k for k in weight_map if 'norm' in k.lower()]}"
                )

            cfg_path = os.path.join(self._model_name, "config.json")
            self._rms_norm_eps = 1e-6
            if os.path.exists(cfg_path):
                with open(cfg_path) as f:
                    cfg = _json.load(f)
                tc = cfg.get("text_config", cfg)
                self._rms_norm_eps = tc.get("rms_norm_eps", 1e-6)

            logger.info(
                "norm.weight loaded: %s, eps=%e",
                tuple(self._norm_weight.shape),
                self._rms_norm_eps,
            )
        return self._norm_weight, self._rms_norm_eps

    def shutdown(self) -> None:
        """Terminate the vLLM subprocess tree and clean up.

        vLLM spawns EngineCore + per-rank Worker_TP* processes that share
        the wrapper subprocess's session (``start_new_session=True``).
        SIGTERM to the wrapper alone leaves the workers orphaned, and
        their Mooncake C++ Ping/FetchTasks threads then poll the dead
        master forever. Signal the whole process group instead.
        """
        proc = self._proc
        if proc is not None:
            # Capture pgid up-front: even if the wrapper exits cleanly, vLLM's
            # multiproc_executor grandchildren (EngineCore / Worker_TP*) stay
            # alive in the same process group with the wrapper's pid as pgid.
            # Linux keeps a pgid valid as long as any member exists.
            pgid = None
            try:
                pgid = os.getpgid(proc.pid)
            except (ProcessLookupError, PermissionError):
                pass

            if proc.poll() is None:
                # Best-effort graceful shutdown — fire-and-forget so we never
                # block on a wedged worker's response.
                cmd_f = self._cmd_f
                if cmd_f is not None:
                    try:
                        cmd_f.write(json.dumps({"cmd": "shutdown"}) + "\n")
                        cmd_f.flush()
                    except OSError:
                        pass
                try:
                    proc.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    pass

            # Always nuke the whole process group, even if `proc` itself
            # already exited.  The wrapper exits as soon as it forwards the
            # shutdown signal, but its multiproc_executor grandchildren can
            # hang in mooncake C++ destructors and survive as orphans.
            if pgid is not None:
                self._killpg_by_pgid(pgid, signal.SIGTERM)
                deadline = time.monotonic() + 5.0
                while time.monotonic() < deadline:
                    if not self._pgid_has_members(pgid):
                        break
                    time.sleep(0.2)

                if self._pgid_has_members(pgid):
                    self._killpg_by_pgid(pgid, signal.SIGKILL)
                    deadline = time.monotonic() + 5.0
                    while time.monotonic() < deadline:
                        if not self._pgid_has_members(pgid):
                            break
                        time.sleep(0.2)

                    if self._pgid_has_members(pgid):
                        logger.warning(
                            "vLLM teacher pgid=%d still has members after "
                            "SIGKILL; orphan workers may remain.",
                            pgid,
                        )

        if self._mooncake_store is not None:
            try:
                self._mooncake_store.close()
            # Mooncake close is best-effort across third-party store implementations.
            except Exception:  # noqa: BLE001,S110 -- cleanup boundary
                pass
            self._mooncake_store = None

        for f in [self._cmd_f, self._resp_f]:
            try:
                if f:
                    f.close()
            except OSError:
                pass
        for p in [self._cmd_fifo, self._resp_fifo]:
            try:
                if p:
                    os.unlink(p)
            except OSError:
                pass

        self._proc = None
        self._cmd_f = None
        self._resp_f = None
        self._initialized = False
        logger.info("VllmTeacherEngine: shutdown complete.")

    @staticmethod
    def _killpg(pid: int, sig: int) -> None:
        try:
            os.killpg(os.getpgid(pid), sig)
        except (ProcessLookupError, PermissionError):
            pass

    @staticmethod
    def _killpg_by_pgid(pgid: int, sig: int) -> None:
        try:
            os.killpg(pgid, sig)
        except (ProcessLookupError, PermissionError):
            pass

    @staticmethod
    def _pgid_has_members(pgid: int) -> bool:
        """Return True if any process is still in the given process group."""
        try:
            for entry in os.listdir("/proc"):
                if not entry.isdigit():
                    continue
                try:
                    if os.getpgid(int(entry)) == pgid:
                        return True
                except (ProcessLookupError, PermissionError):
                    continue
        except OSError:
            # /proc missing — fall back to signal-0 probe on the pgid itself.
            try:
                os.killpg(pgid, 0)
                return True
            except (ProcessLookupError, PermissionError):
                return False
        return False

    def __del__(self) -> None:
        proc = getattr(self, "_proc", None)
        if proc is None:
            return
        try:
            pgid = os.getpgid(proc.pid)
        except (ProcessLookupError, PermissionError):
            return
        self._killpg_by_pgid(pgid, signal.SIGKILL)


__all__ = ["VllmTeacherEngine"]
