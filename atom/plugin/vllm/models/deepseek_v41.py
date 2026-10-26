# SPDX-License-Identifier: MIT
"""DeepSeek-V4.1 as vLLM's DSpark speculator needs it.

ATOM's own drafter reads a target's auxiliary hidden states through forward
hooks the drafter installs, writing into buffers it owns
(`set_aux_hidden_state_rows`). vLLM's does not: it unpacks them from the
forward's return value -- `hidden_states, aux_hidden_states = model_output` --
and asks the target which layers to tap through `get_eagle3_aux_hidden_state_layers`.

That is a convention of one frontend, so it lives here rather than in
`atom.models.deepseek_v41`, which keeps the single-tensor return its own
runner expects. The registry in `atom.plugin.vllm.model_wrapper` points V4.1
at this class, the way it already does for V4 and Qwen3.5.

The cost is a copy of `forward_hidden`'s layer loop, which is deliberate: the
alternative is a collection hook in the shared model that is inert for every
caller but this one.
"""

import torch

from atom.model_ops.deepseek_v41.mhc import SinglePassHCState
from atom.models.deepseek_v41.runtime import DeepseekV41RuntimeModel


class DeepseekV41RuntimeModelVllm(DeepseekV41RuntimeModel):
    """The ATOM V4.1 runtime model, plus vLLM's aux-hidden-state protocol."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # vLLM's speculator looks for the target's embedding under
        # `embed_tokens`/`embedding` on `target.model`; ATOM names it `embed`.
        # An alias rather than a rename, so nothing native has to change.
        self.embed_tokens = self.embed
        self.aux_hidden_state_layers: tuple[int, ...] = ()
        self._aux_hidden_states: list = []

    def set_aux_hidden_state_layers(self, layers) -> None:
        """Layers whose collapsed output the draft reads, set by the speculator."""
        self.aux_hidden_state_layers = tuple(int(i) for i in layers)

    def get_eagle3_aux_hidden_state_layers(self) -> tuple[int, ...]:
        """The layers this checkpoint names, not the generic early/middle/late.

        Every other ATOM target returns `(2, n // 2, n - 3)`, which is vLLM's
        Eagle3 default and a guess made in the absence of anything better.
        V4.1 has something better: the checkpoint carries
        `dspark_target_layer_ids`, and its draft stages were trained against
        those layers. Returning the generic triple would hand the draft three
        layers it has never seen and nothing would say so -- the shapes agree,
        so it would simply propose badly.
        """
        ids = getattr(self.config, "dspark_target_layer_ids", None)
        if not ids:
            raise ValueError(
                "DeepSeek-V4.1 DSpark needs `dspark_target_layer_ids` from the "
                "checkpoint; this config carries none, and the generic Eagle3 "
                "triple would feed the draft layers it was not trained on"
            )
        return tuple(int(i) for i in ids)

    def get_aux_hidden_states(self) -> list:
        """What the last forward collected, in the order the layers were asked for."""
        return self._aux_hidden_states

    def forward(self, input_ids, positions, inputs_embeds=None):
        """The backbone's rows, and the draft's beside them when asked for.

        Delegates the whole forward and only reshapes the return, so the
        replay machinery below it is not duplicated here.
        """
        if self.aux_hidden_state_layers and self.replay:
            # A replay runs the late layers on a tail of the rows, and only
            # the hook path knows how to move those captures back to their
            # forward rows (`_late_on_tail`). Refuse rather than return a
            # tensor whose untapped rows hold whatever was there before.
            raise NotImplementedError(
                "DeepSeek-V4.1 aux hidden states and decoder SWA bounded "
                "replay cannot run together; pass "
                "--no-decoder-swa-bounded-replay to speculate on the plugin"
            )
        hidden = super().forward(input_ids, positions, inputs_embeds)
        if not self.aux_hidden_state_layers:
            return hidden
        return hidden, [aux.squeeze(0) for aux in self.get_aux_hidden_states()]

    def forward_hidden(
        self,
        token_ids,
        cache,
        step,
        engram_embeddings=None,
        *,
        inputs_embeds=None,
        image_mask=None,
    ):
        """`DeepseekV41ForCausalLM.forward_hidden`, collecting the aux rows.

        A copy of the shared loop with one branch added. `+ 1`, and the mean
        rather than `collapse()`, because both are what a DSpark draft was
        trained against: vLLM's own V4 target collects after layer `idx` under
        `if (idx + 1) in self.aux_hidden_state_layers`, so the id names a
        layer's *input*, and the value it appends is `aux_recon.mean(dim=1)`
        -- the reconstructed post-mhc stream averaged over hc_mult.

        `collapse()` is a different quantity: the streams weighted by the
        learned pre-mix and summed, which is the next sublayer's input
        projection. Feeding that instead cost nothing visible -- the answers
        stayed correct, because rejection sampling keeps the target's
        distribution -- and showed up only as a mean acceptance length of 1.04
        against 100 draft tokens/s.
        """
        hidden = (
            self.embed(token_ids.flatten()).view(
                *token_ids.shape, self.config.hidden_size
            )
            if inputs_embeds is None
            else inputs_embeds
        )
        engram_embeddings = {} if engram_embeddings is None else engram_embeddings
        self.begin_forward(hidden, engram_embeddings)
        state = SinglePassHCState.from_embeddings(hidden, self.config.hc_mult)
        aux_hidden_states: list[torch.Tensor] = []
        for spec, layer in zip(self.topology, self.layers):
            rope = self.global_rope if spec.ratio else self.window_rope
            state = layer(
                state,
                cache,
                step,
                rope,
                engram_embeddings.get(spec.layer_id),
                image_mask=image_mask,
            )
            if spec.layer_id + 1 in self.aux_hidden_state_layers:
                aux_hidden_states.append(state.settle().residual.mean(dim=-2))
        hidden = state.collapse()
        if aux_hidden_states:
            self._aux_hidden_states = aux_hidden_states
        self.end_forward(hidden, engram_embeddings)
        return hidden
