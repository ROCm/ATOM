# SPDX-License-Identifier: MIT
"""Compare the same prompt at operator boundaries under different batches."""

import torch

from atom.utils.forward_context import get_forward_context


class PrefillTrace:
    def __init__(self, runner, prompt, directory):
        self.prompt = prompt
        self.directory = directory
        self.case = None
        self.repeat = 0
        self.layer = None
        self.rows = []
        self.values = {}
        self.records = []
        self.hooks = []
        self._run = runner.run_model
        runner.run_model = self.run
        modules = {"embed": runner.model.embed, "norm": runner.model.norm}
        for i, layer in enumerate(runner.model.layers):

            def enter(module, inputs, i=i):
                self.layer = i

            self.hooks.append(layer.register_forward_pre_hook(enter))
            for name, module in layer.named_modules():
                if name in (
                    "",
                    "attn",
                    "attn_norm",
                    "ffn_norm",
                    "ffn",
                    "ffn.gate",
                    "engram",
                    "engram.wkv",
                    "attn.wqkv_a",
                    "attn.q_norm",
                    "attn.wq_b",
                    "attn.kv_norm",
                    "attn.wo_b",
                ):
                    modules[f"layers.{i}.{name or 'block'}"] = module
        for name, module in modules.items():

            def hook(module, inputs, output, name=name):
                if self.rows:
                    if inputs:
                        self.save(name + ".input", inputs[0])
                    self.save(name + ".output", output)

            self.hooks.append(module.register_forward_hook(hook))

        from atom.models.deepseek_v41 import attention

        for name in ("sparse_attn_v4_paged_decode", "grouped_output_projection"):
            original = getattr(attention, name)

            def traced(*args, name=name, original=original, **kwargs):
                label = f"layers.{self.layer}.{name}"
                flat = name == "sparse_attn_v4_paged_decode"
                if self.rows:
                    self.save(label + ".input", args[0], flat_rows=flat)
                output = original(*args, **kwargs)
                if self.rows:
                    self.save(label + ".output", output, flat_rows=flat)
                return output

            setattr(attention, name, traced)

    def save(self, name, value, *, flat_rows=False):
        if hasattr(value, "residual"):
            self.save(name + ".residual", value.residual)
            self.save(name + ".pre_mix", value.pre_mix)
        elif isinstance(value, tuple):
            for i, item in enumerate(value):
                self.save(f"{name}.{i}", item)
        elif isinstance(value, torch.Tensor):
            # Runtime model math uses [1, tokens, ...]; embedding/norm are flat.
            flat = value.flatten(0, 1) if value.ndim >= 3 and not flat_rows else value
            if flat.shape[0] > max(self.rows):
                self.values[name] = flat[self.rows].detach().clone()

    @torch.inference_mode()
    def run(self, input_ids, batch):
        metadata = get_forward_context().attn_metadata
        spans = [
            span
            for span in metadata.step.requests
            if span.position == 0
            and span.length == len(self.prompt)
            and input_ids[span.token_slice].tolist() == self.prompt
        ]
        if not spans:
            return self._run(input_ids, batch)
        self.rows = [
            row
            for span in spans
            for row in range(span.offset, span.offset + span.length)
        ]
        self.values = {}
        for layer, values in metadata.engram_embeddings.items():
            self.save(f"engram_embeddings.{layer}", values)
        try:
            logits, hidden = self._run(input_ids, batch)
            indices = [metadata.step.requests.index(span) for span in spans]
            self.values["logits"] = logits[indices].detach().clone()
            values = {name: tensor.cpu() for name, tensor in self.values.items()}
            rank = torch.distributed.get_rank()
            path = (
                self.directory
                / f"prefill_case{self.case}_repeat{self.repeat}_rank{rank}.pt"
            )
            torch.save(values, path)
            scores, ids = values["logits"].topk(8)
            self.records.append(
                {
                    "case": self.case,
                    "repeat": self.repeat,
                    "rank": rank,
                    "spans": [vars(span) for span in spans],
                    "top_ids": ids.tolist(),
                    "top_scores": scores.tolist(),
                    "tensor_file": path.name,
                }
            )
            return logits, hidden
        finally:
            self.rows = []
            self.values = {}
