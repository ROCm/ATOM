# SPDX-License-Identifier: MIT
"""Compare every verify row at model, operator and collective boundaries."""

import torch


class LayerTrace:
    def __init__(
        self, model, dump_directory=None, *, max_layers=None, request_projections=True
    ):
        self.mode = None
        self.row = 0
        self.rows = None
        self.values = {"block": {}, "serial": {}}
        self.hooks = []
        self.layer = None
        self.calls = {}
        self.wrapped = []
        self.dump_directory = dump_directory
        self.dump_count = 0
        self.engram_weights = {
            i: layer.engram.gate_weight
            for i, layer in enumerate(model.layers)
            if layer.engram is not None
        }
        for i, layer in enumerate(model.layers[:max_layers]):

            def enter(module, inputs, i=i):
                self.layer = i
                self.calls = {}

            self.hooks.append(layer.register_forward_pre_hook(enter))
            modules = {
                "block": layer,
                "attn": layer.attn,
                "ffn": layer.ffn,
                "attn_norm": layer.attn_norm,
                "ffn_norm": layer.ffn_norm,
                "router": layer.ffn.gate,
            }
            modules.update(
                (f"attn.{name}", getattr(layer.attn, name))
                for name in ("wqkv_a", "q_norm", "wq_b", "kv_norm", "wo_b")
            )
            if layer.engram is not None:
                modules["engram"] = layer.engram
                modules["engram.wkv"] = layer.engram.wkv
            for name, module in modules.items():
                label = f"layers.{i}.{name}"

                def hook(module, inputs, output, label=label):
                    if self.mode is not None:
                        if inputs and isinstance(inputs[0], torch.Tensor):
                            self.save(label + ".input", inputs[0])
                        self.save(label + ".output", output)

                self.hooks.append(module.register_forward_hook(hook))

            def leave(module, inputs, output):
                self.layer = None

            self.hooks.append(layer.register_forward_hook(leave))

        from aiter.dist.parallel_state import get_tp_group
        from atom.model_ops.deepseek_v41 import mhc
        from atom.models.deepseek_v41 import attention

        self.wrap(mhc, "hc_projection")
        self.wrap(attention, "grouped_output_projection")
        self.wrap(attention, "sparse_attn_v4_paged_decode", flat_rows=True)
        self.wrap(attention, "packed_decode", flat_rows=True)
        self.wrap(get_tp_group(), "all_reduce")
        for layer in model.layers[:max_layers]:
            if request_projections and layer.attn.compressor is not None:
                self.wrap(layer.attn.compressor, "project")

    def wrap(self, owner, name, *, flat_rows=False):
        original = getattr(owner, name)

        def traced(*args, **kwargs):
            label = None
            if self.mode is not None and self.layer is not None:
                call = self.calls.get(name, 0)
                self.calls[name] = call + 1
                label = f"layers.{self.layer}.{name}.{call}"
                self.save(label + ".input", args[0], flat_rows=flat_rows)
                if name == "sparse_attn_v4_paged_decode":
                    self.save_attention_keys(label, *args[:4])
            output = original(*args, **kwargs)
            if label is not None:
                self.save(label + ".output", output, flat_rows=flat_rows)
            return output

        self.wrapped.append((owner, name, original))
        setattr(owner, name, traced)

    def save_attention_keys(self, label, query, pool, indices, indptr):
        # Fixed capacity is test-only, matching this diagnostic's 512-token limit.
        offsets = torch.arange(512, device=query.device)
        lengths = indptr[1:] - indptr[:-1]
        assert int(lengths.max()) <= offsets.numel()
        mask = offsets[None] < lengths[:, None]
        entries = indices[
            (indptr[:-1, None] + offsets[None]).clamp_max(indices.numel() - 1)
        ]
        mask = mask & (entries >= 0)
        safe_entries = entries.masked_fill(~mask, 0).long()
        assert int(safe_entries.max()) < pool.shape[0]
        keys = pool[safe_entries].masked_fill(~mask[..., None], 0)
        self.save(label + ".kv", keys, flat_rows=True)
        self.save(label + ".indices", entries.masked_fill(~mask, -1))

    def save(self, label, value, *, flat_rows=False):
        if hasattr(value, "residual"):
            self.save(label + ".residual", value.residual)
            self.save(label + ".pre_mix", value.pre_mix)
        elif isinstance(value, tuple):
            for i, item in enumerate(value):
                self.save(f"{label}.{i}", item)
        elif isinstance(value, torch.Tensor):
            rows = value.flatten(0, 1) if value.ndim >= 3 and not flat_rows else value
            if self.mode == "block":
                saved = rows.detach().clone()
                previous = self.values["block"].get(label)
                self.values["block"][label] = (
                    saved if previous is None else torch.cat((previous, saved))
                )
            else:
                indices = [self.row] if self.rows is None else self.rows
                if rows.shape[0] != len(indices):
                    raise ValueError(f"Trace row mapping does not match {label}")
                saved = self.values["serial"].setdefault(label, {})
                for local, index in enumerate(indices):
                    saved[index] = rows[local].detach().clone()

    def compare(self, selected_row):
        result = []
        if self.dump_directory is not None:
            selected = {}
            for label, block in self.values["block"].items():
                if not label.startswith(
                    ("layers.0.", "layers.1.", "layers.2.", "layers.3.")
                ):
                    continue
                if (
                    "engram" not in label
                    and ".project." not in label
                    and "sparse_attn_v4_paged_decode" not in label
                ):
                    continue
                selected[label] = {
                    "block": block.cpu(),
                    "serial": {
                        i: v.cpu() for i, v in self.values["serial"][label].items()
                    },
                }
            rank = torch.distributed.get_rank()
            torch.save(
                {
                    "values": selected,
                    "engram_weights": {
                        i: w.cpu() for i, w in self.engram_weights.items()
                    },
                },
                self.dump_directory / f"trace_{self.dump_count}_rank{rank}.pt",
            )
            self.dump_count += 1
        for label, block in self.values["block"].items():
            serial = torch.stack(
                [self.values["serial"][label][i] for i in range(block.shape[0])]
            )
            error = (block.float() - serial.float()).abs()
            row_error = error.flatten(1)
            row_unequal = (block != serial).flatten(1).sum(-1)
            row_relative = row_error.norm(dim=-1) / serial.float().flatten(1).norm(
                dim=-1
            ).clamp_min(1e-30)
            result.append(
                {
                    "stage": label,
                    "unequal": int(row_unequal[selected_row]),
                    "elements": block[0].numel(),
                    "max_error": float(row_error[selected_row].max()),
                    "relative_l2": float(row_relative[selected_row]),
                    "row_unequal": row_unequal.tolist(),
                    "row_max_error": row_error.amax(-1).tolist(),
                    "row_relative_l2": row_relative.tolist(),
                }
            )
        self.values = {"block": {}, "serial": {}}
        ranks = [None] * torch.distributed.get_world_size()
        torch.distributed.all_gather_object(ranks, result)
        return {"stages": result, "ranks": ranks}
