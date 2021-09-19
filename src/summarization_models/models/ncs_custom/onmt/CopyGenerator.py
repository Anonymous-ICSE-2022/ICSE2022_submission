from typing import List

import torch
import torch.nn as nn
from summarization_models.models.ncs_custom.vocabulary import Vocabulary

from .misc import aeq


def collapse_copy_scores(
    scores: torch.Tensor, tgt_vocab: Vocabulary, src_vocabs: List[Vocabulary],
) -> torch.Tensor:
    """Collapses scores for words that exist both in the target vocab
    and the extended source vocab. 

    Parameters
    ----------
    scores : torch.Tensor
        output from copy generator, of shape [batch_size, seq_len, dim]
    tgt_vocab : Vocabulary
        target vocabulary
    src_vocabs : List[Vocabulary]
        source vocabularies (extended)

    Returns
    -------
    torch.Tensor
        scores of same shape
    """
    offset = len(tgt_vocab)
    for b in range(scores.size(0)):
        blank = []
        fill = []

        src_vocab = src_vocabs[b]

        for i in range(len(src_vocab)):
            if i == src_vocab.pad_idx and i == src_vocab.unk_idx:
                continue

            sw = src_vocab.decode_token(i)
            try:
                ti = tgt_vocab.encode_token(sw)
                if ti != 0:
                    blank.append(offset + i)
                    fill.append(ti)
            except:
                continue

        if blank:
            blank_mask = torch.tensor(blank).type_as(scores).long()
            fill_mask = torch.tensor(fill).type_as(scores).long()
            score = scores[b]
            score.index_add_(1, fill_mask, score.index_select(1, blank_mask))
            score.index_fill_(1, blank_mask, 1e-10)
    return scores


class CopyGenerator(nn.Module):
    """An implementation of pointer-generator networks
    :cite:`DBLP:journals/corr/SeeLM17`.
    These networks consider copying words
    directly from the source sequence.
    The copy generator is an extended version of the standard
    generator that computes three values.
    * :math:`p_{softmax}` the standard softmax over `tgt_dict`
    * :math:`p(z)` the probability of copying a word from
      the source
    * :math:`p_{copy}` the probility of copying a particular word.
      taken from the attention distribution directly.
    The model returns a distribution over the extend dictionary,
    computed as
    :math:`p(w) = p(z=1)  p_{copy}(w)  +  p(z=0)  p_{softmax}(w)`
    .. mermaid::
       graph BT
          A[input]
          S[src_map]
          B[softmax]
          BB[switch]
          C[attn]
          D[copy]
          O[output]
          A --> B
          A --> BB
          S --> D
          C --> D
          D --> O
          B --> O
          BB --> O
    Args:
       input_size (int): size of input representation
       output_size (int): size of output vocabulary
       pad_idx (int)
    """

    def __init__(self, input_size: int, generator: nn.Module, pad_idx: int):
        super(CopyGenerator, self).__init__()
        self.linear = generator
        self.linear_copy = nn.Linear(input_size, 1)
        self.pad_idx = pad_idx

    def forward(self, hidden, attn, src_map):
        """
        Compute a distribution over the target dictionary
        extended by the dynamic dictionary implied by copying
        source words.
        Args:
           hidden (FloatTensor): hidden outputs ``(batch x tlen, input_size)``
           attn (FloatTensor): attn for each ``(batch x tlen, slen)``
           src_map (FloatTensor):
               A sparse indicator matrix mapping each source word to
               its index in the "extended" vocab containing.
               ``(src_len, batch, extra_words)``
        """

        # CHECKS
        batch_by_tlen, _ = hidden.size()
        batch_by_tlen_, slen = attn.size()
        slen_, batch, cvocab = src_map.size()
        aeq(batch_by_tlen, batch_by_tlen_)
        aeq(slen, slen_)

        # Original probabilities.
        logits = self.linear(hidden)
        logits[:, self.pad_idx] = -float("inf")
        prob = torch.softmax(logits, 1)

        # Probability of copying p(z=1) batch.
        p_copy = torch.sigmoid(self.linear_copy(hidden))
        # Probability of not copying: p_{word}(w) * (1 - p(z))
        out_prob = torch.mul(prob, 1 - p_copy)
        mul_attn = torch.mul(attn, p_copy)
        copy_prob = torch.bmm(
            mul_attn.view(-1, batch, slen).transpose(0, 1), src_map.transpose(0, 1)
        ).transpose(0, 1)
        copy_prob = copy_prob.contiguous().view(-1, cvocab)
        return torch.cat([out_prob, copy_prob], 1)


class CopyGeneratorLoss(nn.Module):
    """Copy generator criterion."""

    def __init__(
        self, vocab_size, force_copy, unk_index=0, ignore_index=-100, eps=1e-20
    ):
        super(CopyGeneratorLoss, self).__init__()
        self.force_copy = force_copy
        self.eps = eps
        self.vocab_size = vocab_size
        self.ignore_index = ignore_index
        self.unk_index = unk_index

    def forward(self, scores, align, target):
        """
        Args:
            scores (FloatTensor): ``(batch_size*tgt_len)`` x dynamic vocab size
                whose sum along dim 1 is less than or equal to 1, i.e. cols
                softmaxed.
            align (LongTensor): ``(batch_size x tgt_len)``
            target (LongTensor): ``(batch_size x tgt_len)``
        """
        # probabilities assigned by the model to the gold targets
        vocab_probs = scores.gather(1, target.unsqueeze(1)).squeeze(1)

        # probability of tokens copied from source
        copy_ix = align.unsqueeze(1) + self.vocab_size
        copy_tok_probs = scores.gather(1, copy_ix).squeeze(1)
        # Set scores for unk to 0 and add eps
        copy_tok_probs[align == self.unk_index] = 0
        copy_tok_probs += self.eps  # to avoid -inf logs
        # find the indices in which you do not use the copy mechanism
        non_copy = align == self.unk_index
        if not self.force_copy:
            non_copy = non_copy | (target != self.unk_index)

        probs = torch.where(non_copy, copy_tok_probs + vocab_probs, copy_tok_probs)

        loss = -probs.log()  # just NLLLoss; can the module be incorporated?
        # Drop padding.
        loss[target == self.ignore_index] = 0.0
        return loss
