from typing import List

import torch
import torch.nn.functional as F
from summarization_models.models.ncs_custom.onmt.CopyGenerator import (
    collapse_copy_scores,
)

from .vocabulary import Vocabulary


def print_sample_predictions(
    predictions: torch.Tensor, vocabulary: Vocabulary, target: torch.Tensor, index=0
):
    # get an example
    example = predictions[:, index, :]
    example = F.softmax(example, 1)
    example = torch.argmax(example, 1).tolist()

    example_tokens = vocabulary.decode(example)
    target_tokens = target[index].tolist()
    example_target = vocabulary.decode(target_tokens)

    print()
    print("-" * 80)
    print(f"prediction: {' '.join(example_tokens)}")
    print(f"actual label: {' '.join(example_target)}")
    print("-" * 80)
    print()


def decode_tokens(
    token_indices: List[int],
    target_vocabulary: Vocabulary,
    extended_vocabulary: Vocabulary,
    include_eos_token: bool = False,
) -> List[str]:

    resolved_tokens = []
    extended_vocab_offset = len(target_vocabulary)
    copy_count = 0
    total = 0
    unks = 0
    for token_index in token_indices:
        if token_index < extended_vocab_offset:
            token = target_vocabulary.decode_token(token_index)
        else:
            idx = token_index - extended_vocab_offset
            if idx not in extended_vocabulary.idx_to_token:
                unks += 1
                continue
            token = extended_vocabulary.decode_token(
                token_index - extended_vocab_offset
            )

            copy_count += 1
        if token == target_vocabulary.eos_token:
            if include_eos_token:
                resolved_tokens.append(token)
            break

        resolved_tokens.append(token)
        total += 1

    return resolved_tokens


def convert_scores_to_predictions(
    scores: torch.Tensor,
    target_vocabulary: Vocabulary,
    source_vocabularies: List[Vocabulary],
    batch_size: int,
):
    """[summary]

    Parameters
    ----------
    scores : torch.Tensor
        [batch_size * tgt_len, tgt_vocab_size + max(extended_vocab_size)]
    target_vocabulary : Vocabulary
        Targetvocabulary
    source_vocabularies : Vocabulary
        source vocabularies as list
    target : torch.Tensor
        [batch_size, tgt_len]
    index : int, optional
        which example to pick inside batch, by default 0
    """
    scores = scores.detach().cpu().view(batch_size, -1, scores.size(-1))
    scores = collapse_copy_scores(
        scores, target_vocabulary, src_vocabs=source_vocabularies
    )
    # logits = scores.softmax(2)
    logits = scores
    token_indices_per_example = torch.argmax(logits, 2).tolist()

    predictions = []

    for i, token_indices in enumerate(token_indices_per_example):
        prediction = decode_tokens(
            token_indices,
            target_vocabulary=target_vocabulary,
            extended_vocabulary=source_vocabularies[i],
        )
        predictions.append(prediction)

    return predictions

