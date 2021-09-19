"""Base class for encoders and generic multi encoders."""

import torch
import torch.nn as nn

from .misc import aeq


class EncoderBase(nn.Module):
    """
    Base encoder class. Specifies the interface used by different encoder types
    and required by :class:`onmt.Models.NMTModel`.
    .. mermaid::
       graph BT
          A[Input]
          subgraph RNN
            C[Pos 1]
            D[Pos 2]
            E[Pos N]
          end
          F[Memory_Bank]
          G[Final]
          A-->C
          A-->D
          A-->E
          C-->F
          D-->F
          E-->F
          E-->G
    """

    @classmethod
    def from_opt(cls, opt, embeddings=None):
        raise NotImplementedError

    def _check_args(self, src, lengths=None, hidden=None):
        n_batch = src.size(1)
        if lengths is not None:
            (n_batch_,) = lengths.size()
            aeq(n_batch, n_batch_)

    def forward(self, src: torch.LongTensor, lengths=None):
        """
        Args:
            src (LongTensor):
               padded sequences of sparse indices ``(src_len, batch, nfeat)``
            lengths (LongTensor): length of each sequence ``(batch,)``
        Returns:
            (FloatTensor, FloatTensor, FloatTensor):
            * final encoder state, used to initialize decoder
            * memory bank for attention, ``(src_len, batch, hidden)``
            * lengths
        """

        raise NotImplementedError


import torch
import torch.nn as nn


class DecoderBase(nn.Module):
    """Abstract class for decoders.
    Args:
        attentional (bool): The decoder returns non-empty attention.
    """

    def __init__(self, attentional=True):
        super(DecoderBase, self).__init__()
        self.attentional = attentional

    @classmethod
    def from_opt(cls, opt, embeddings):
        """Alternate constructor.
        Subclasses should override this method.
        """

        raise NotImplementedError


# At the moment this class is only used by embeddings.Embeddings look-up tables
class Elementwise(nn.ModuleList):
    """
    A simple network container.
    Parameters are a list of modules.
    Inputs are a 3d Tensor whose last dimension is the same length
    as the list.
    Outputs are the result of applying modules to inputs elementwise.
    An optional merge parameter allows the outputs to be reduced to a
    single Tensor.
    """

    def __init__(self, merge=None, *args):
        assert merge in [None, "first", "concat", "sum", "mlp"]
        self.merge = merge
        super(Elementwise, self).__init__(*args)

    def forward(self, inputs):
        inputs_ = [feat.squeeze(2) for feat in inputs.split(1, dim=2)]
        assert len(self) == len(inputs_)
        outputs = [f(x) for f, x in zip(self, inputs_)]
        if self.merge == "first":
            return outputs[0]
        elif self.merge == "concat" or self.merge == "mlp":
            return torch.cat(outputs, 2)
        elif self.merge == "sum":
            return sum(outputs)
        else:
            return outputs


class Cast(nn.Module):
    """
    Basic layer that casts its input to a specific data type. The same tensor
    is returned if the data type is already correct.
    """

    def __init__(self, dtype):
        super(Cast, self).__init__()
        self._dtype = dtype

    def forward(self, x):
        return x.to(self._dtype)
