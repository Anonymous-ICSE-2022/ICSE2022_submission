import os

import pandas as pd
import torch
from torch.utils.data import Dataset


class CodeBertDataset(Dataset):
    """Parquet Dataset"""

    def __init__(self, filepath, transform=None):
        self.filepath = filepath
        self.transform = transform
        self.data = pd.read_parquet(filepath)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        if isinstance(idx, int):
            idx = [idx]
        entries = self.data.iloc[idx]
        if self.transform:
            entries = self.transform(entries)

        return entries
