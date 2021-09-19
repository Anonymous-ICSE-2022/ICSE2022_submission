from typing import List, NamedTuple

import numpy as np
import torch
from torch.utils.data import Dataset

from .vocabulary import Vocabulary


class TransformerDataset(Dataset):
    def __init__(
        self,
        code_tokens: List[List[str]],
        comment_tokens: List[List[str]],
        code_vocabulary: Vocabulary,
        comment_vocabulary: Vocabulary,
    ) -> None:
        super().__init__()

        self.code_vocabulary = code_vocabulary
        self.comment_vocabulary = comment_vocabulary

        self.code_tokens = np.array(code_tokens, dtype=object)
        self.comment_tokens = np.array(comment_tokens, dtype=object)

    def __len__(self):
        return len(self.code_tokens)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        if isinstance(idx, int):
            idx = [idx]
        
        indices = idx
        code_tokens = self.code_tokens[idx]
        code_tokens_len = [len(tokens) for tokens in code_tokens]
        comment_tokens = self.comment_tokens[idx]
        comment_tokens_len = [len(tokens) for tokens in comment_tokens]

        return (indices, code_tokens, code_tokens_len, comment_tokens, comment_tokens_len)

