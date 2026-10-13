"""RiceBenchmark adapter for Botanic1 (https://huggingface.co/living-models).

The OneGenome-Rice harness loads models with ``AutoModel.from_pretrained`` and
reads ``model(input_ids=..., output_hidden_states=True).hidden_states[layer]``.
This subclass changes three things relative to the released ``Botanic1`` class,
without touching its weights:

1. ``hidden_states`` follows the Hugging Face convention: ``num_hidden_layers + 1``
   entries, index 0 is the token embedding, index ``i`` the output of block ``i``,
   and the last entry is ``last_hidden_state`` (after the final norm).
   The released class returns one entry per block, before the final norm.
2. Padding. Botanic1 does not mask pad tokens: they enter both SSM scans and
   change the embeddings of real tokens. The harness pads every batch to its
   longest sequence and passes no attention mask. Rows are therefore re-encoded
   at their own length (grouped by length), so each sequence's embedding is
   independent of what it was batched with. Pad positions get zero vectors; the
   harness excludes them from the masked mean.
3. No length cap. The released class refuses inputs above ``max_seqlen`` (8,194
   tokens, the training length plus specials). The harness feeds whole sequences
   (up to 128 kb) and has no chunking, so longer inputs run as a single scan.
   Botanic1 was trained on 8,192-bp windows; results above that length measure
   length extrapolation.
"""

from __future__ import annotations

import torch
from transformers.modeling_outputs import BaseModelOutput

from .botanic1_architecture import Botanic1


class Botanic1ForRiceBenchmark(Botanic1):
    def _hidden_states(self, input_ids: torch.Tensor) -> list[torch.Tensor]:
        x = self.tok_embeddings(input_ids)
        states = [x]
        for layer in self.layers:
            x = layer(x)
            states.append(x)
        states[-1] = self.post_encoder_norm(x)
        return states

    def forward(self, input_ids=None, output_hidden_states=False, **kwargs):
        del kwargs  # attention_mask etc.: pads are found from input_ids
        is_pad = input_ids == self.config.pad_idx
        lengths = (~is_pad).sum(dim=1)
        # Right padding only: every pad sits after the last real token.
        positions = torch.arange(input_ids.shape[1], device=input_ids.device)
        if not torch.equal(is_pad, positions[None, :] >= lengths[:, None]):
            raise ValueError("expected right-padded input_ids")

        if bool((lengths == input_ids.shape[1]).all()):
            states = self._hidden_states(input_ids)
        else:
            states = None
            for length in lengths.unique().tolist():
                rows = (lengths == length).nonzero(as_tuple=True)[0]
                group = self._hidden_states(input_ids[rows, :length])
                if states is None:
                    states = [
                        g.new_zeros(input_ids.shape + g.shape[-1:]) for g in group
                    ]
                for full, part in zip(states, group):
                    full[rows, :length] = part

        return BaseModelOutput(
            last_hidden_state=states[-1],
            hidden_states=tuple(states) if output_hidden_states else None,
        )
