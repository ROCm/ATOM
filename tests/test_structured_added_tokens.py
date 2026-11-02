import json
from types import SimpleNamespace as NS

import numpy as np
import pytest
import torch

pytest.importorskip("llguidance")
from tokenizers import AddedToken, Tokenizer, decoders, models, pre_tokenizers
from transformers import PreTrainedTokenizerFast

from atom.structured_output import StructuredOutput
from atom.structured_output.backend_guidance import grammar_tokenizer


@pytest.fixture
def tokenizer():
    vocab = {s: i for i, s in enumerate(sorted(pre_tokenizers.ByteLevel.alphabet()))}
    vocab["[EOS]"] = len(vocab)
    backend = Tokenizer(models.BPE(vocab, []))
    backend.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    backend.decoder = decoders.ByteLevel()
    backend.add_tokens([AddedToken("<think>"), AddedToken("</think>")])
    return PreTrainedTokenizerFast(tokenizer_object=backend, eos_token="[EOS]")


@pytest.mark.parametrize("speculative", [False, True])
def test_visible_added_tokens_agree_between_masks_and_commits(tokenizer, speculative):
    text = '{"v":"<think>literal</think>"}'
    ids = tokenizer.encode(text, add_special_tokens=False)
    assert tokenizer.convert_tokens_to_ids("<think>") in ids
    assert tokenizer.decode(ids, skip_special_tokens=True) == text
    ids.append(tokenizer.eos_token_id)
    controller = StructuredOutput(tokenizer, len(tokenizer), [tokenizer.eos_token_id])
    schema = json.dumps({"const": {"v": "<think>literal</think>"}})
    history = []
    width = 6 if speculative else 1
    for start in range(0, len(ids), width):
        selected = ids[start : start + width]
        drafts = selected[:-1]
        batch = NS(
            structured_requests={0: (schema, (), history[:])},
            req_ids=[1],
            is_final_chunk=None,
            total_seqs_num_prefill=0,
        )
        metadata = (
            NS(
                target_logits_indices=torch.arange(len(drafts)),
                bonus_logits_indices=torch.tensor([len(drafts)]),
                num_draft_tokens_np=np.array([len(drafts)]),
                input_ids=torch.tensor([0, *drafts]),
            )
            if speculative
            else None
        )
        logits = torch.zeros(len(selected), len(tokenizer))
        prefixes = controller.apply(logits, batch, metadata)
        for row, token in enumerate(selected):
            assert torch.isfinite(logits[row, token])
        controller.commit(torch.tensor([selected]), batch, prefixes)
        history.extend(selected)
    assert controller.states[1][3].is_terminated()


def test_added_token_adapter_preserves_eos_and_original_tokenizer(tokenizer):
    original = tokenizer.backend_tokenizer.to_str()
    info = grammar_tokenizer(tokenizer, len(tokenizer), [tokenizer.eos_token_id])
    for text in ("<think>", "</think>"):
        token = tokenizer.convert_tokens_to_ids(text)
        assert not info.is_special_token(token)
        assert info.decode_bytes([token]) == text.encode()
    assert info.is_special_token(tokenizer.eos_token_id)
    assert tokenizer.backend_tokenizer.to_str() == original
