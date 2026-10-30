"""Two real attention blocks and native mHC seams; bounded native loading."""

import contextlib
import hashlib
import json
from pathlib import Path

import torch
from aiter.dist.parallel_state import get_tp_group
from atom.model_loader.loader import default_weight_loader
from atom.model_loader.loading_core import load_weights_into_model
from atom.model_ops.base_config import QuantizeMethodBase
from atom.model_ops.layernorm import RMSNorm
from atom.model_ops.utils import atom_parameter
from atom.models.deepseek_v41.attention import Attention
from atom.models.deepseek_v41.config import AttentionMode, build_attention_topology
from atom.models.deepseek_v41.model import Block, DeepseekV41ForCausalLM
from atom.models.deepseek_v41.owner_service.owner_cost_attention_chain import (
    FullOwnerHeads,
)
from atom.models.deepseek_v41.weights import checkpoint_schema
from atom.utils import envs
from safetensors import safe_open

MODEL = Path("/models/DeepSeek-V4.1-Flash")


class AttentionWindow(torch.nn.Module):
    """Only parameter declarations differ from Block: no unused MoE or Engram.

    All three execution methods below are the original Block methods.
    """

    prepare_attention = Block.prepare_attention
    prepare_ffn = Block.prepare_ffn
    process_weights_after_loading = Block.process_weights_after_loading

    def __init__(self, config, spec, compress_stream, index_stream):
        super().__init__()
        assert spec.layer_id in (8, 9) and spec.layer_id not in config.engram_layer_ids
        self.attn = Attention(
            config, spec, compress_stream=compress_stream, index_stream=index_stream
        )
        self.engram = None
        self.attn_norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.ffn_norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.hc_options = {
            "rms_eps": config.rms_norm_eps,
            "hc_eps": config.hc_eps,
            "sinkhorn_iters": config.hc_sinkhorn_iters,
            "post_mult": 2.0,
        }
        hc = config.hc_mult
        for sublayer in ("attn", "ffn"):
            for suffix, shape in (
                ("fn", (hc * (hc + 2), hc * config.hidden_size)),
                ("base", (hc * (hc + 2),)),
                ("scale", (3,)),
            ):
                self.register_parameter(
                    f"hc_{sublayer}_{suffix}",
                    atom_parameter(torch.empty(shape, dtype=torch.float32)),
                )
            self.register_buffer(f"hc_{sublayer}_fn_packed", None, persistent=False)


class Windows(torch.nn.Module):
    weights_mapper = DeepseekV41ForCausalLM.weights_mapper
    weights_mapping = DeepseekV41ForCausalLM.weights_mapping
    packed_modules_mapping = DeepseekV41ForCausalLM.packed_modules_mapping

    def __init__(self, hf):
        super().__init__()
        self.hf = hf
        self.topology = build_attention_topology(hf)
        a, b = self.topology[8:10]
        assert a.mode == AttentionMode.FULL and b.mode == AttentionMode.REUSE
        assert a.topk_owner == b.topk_owner == a.kv_owner == b.kv_owner == 8
        level = envs.ATOM_DSV41_SIDE_STREAMS
        assert level in (0, 1, 2)
        self.compress_stream = torch.cuda.Stream() if level >= 1 else None
        self.index_stream = torch.cuda.Stream() if level == 2 else None
        with torch.device("cuda"):
            self.layers = torch.nn.ModuleDict(
                {
                    str(spec.layer_id): AttentionWindow(
                        hf, spec, self.compress_stream, self.index_stream
                    )
                    for spec in (a, b)
                }
            )
        self.heads = {}

    def plan_load(self):
        """Resolve all checkpoint targets and header shapes before any weight read."""
        index_bytes = (MODEL / "model.safetensors.index.json").read_bytes()
        index = json.loads(index_bytes)["weight_map"]
        schema = checkpoint_schema(self.hf)
        wanted = {
            name: spec
            for name, spec in schema.items()
            if name.startswith(("layers.8.", "layers.9.")) and ".ffn." not in name
        }
        covered = set()
        headers = {}
        with contextlib.ExitStack() as stack:
            handles = {}
            for name, spec in wanted.items():
                filename = index[name]
                if filename not in handles:
                    handles[filename] = stack.enter_context(
                        safe_open(MODEL / filename, framework="pt", device="cpu")
                    )
                view = handles[filename].get_slice(name)
                assert tuple(view.get_shape()) == spec.shape, (
                    name,
                    view.get_shape(),
                    spec.shape,
                )
                assert view.get_dtype() == spec.dtype, (
                    name,
                    view.get_dtype(),
                    spec.dtype,
                )
                target = (
                    name.removesuffix(".scale") + ".weight_scale_inv"
                    if name.endswith(".scale")
                    else name
                )
                shard = None
                for key, (replacement, part) in self.packed_modules_mapping.items():
                    if key in target:
                        target, shard = target.replace(key, replacement), part
                        break
                parameter = self.get_parameter(target)
                covered.add(id(parameter))
                if shard is not None:
                    assert hasattr(parameter, "weight_loader"), target
                headers[name] = {
                    "shape": spec.shape,
                    "dtype": spec.dtype,
                    "target": target,
                    "target_shape": tuple(parameter.shape),
                    "packed_shard": shard,
                    "tp_axis": spec.tp_axis,
                }
        assert covered == {
            id(p) for p in self.parameters()
        }, "Missing attention/mHC parameter bindings"
        return {
            "index_sha256": hashlib.sha256(index_bytes).hexdigest(),
            "tensors": headers,
        }, index

    def load(self, plan, index):
        group = get_tp_group()

        def iterator(path, disable_mmap, wants):
            assert Path(path) == MODEL
            # Use safe_open's bounded mmap reads, as the retained layer fixture
            # does; no whole-shard deserialization or whole-model allocation.
            with contextlib.ExitStack() as stack:
                handles = {}
                for name in plan["tensors"]:
                    assert wants(name)
                    filename = index[name]
                    if filename not in handles:
                        handles[filename] = stack.enter_context(
                            safe_open(MODEL / filename, framework="pt", device="cpu")
                        )
                    yield name, handles[filename].get_tensor(name)

        loaded = load_weights_into_model(
            self,
            str(MODEL),
            self.hf,
            default_weight_loader=default_weight_loader,
            fuse_shared_expert=lambda _shared, _routed: False,
            is_rank0=lambda: group.rank_in_group == 0,
            weights_iterator=iterator,
        )
        assert {v["target"] for v in plan["tensors"].values()} <= loaded
        # Same parent-first postload order as the production loader.
        for module in self.modules():
            if hasattr(module, "process_weights_after_loading"):
                module.process_weights_after_loading()
            method = getattr(module, "quant_method", None)
            if isinstance(method, QuantizeMethodBase):
                method.process_weights_after_loading(module)
        self.heads = {
            int(name): FullOwnerHeads.gather_native(layer.attn, group)
            for name, layer in self.layers.items()
        }
        return {
            "loaded": sorted(loaded),
            "real_attention_layers": [8, 9],
            "side_stream_level": envs.ATOM_DSV41_SIDE_STREAMS,
        }
