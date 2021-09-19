import os

import pyarrow as pa
import pyarrow.parquet as pq

import torch
from torch.utils.data import Dataset


class ParquetDataset(Dataset):
    """Parquet Dataset"""

    def __init__(self, filepath, transform=None):
        self.filepath = filepath
        self.transform = transform
        mmap = pa.memory_map(filepath)
        self.table = pq.read_table(mmap)

    def __len__(self):
        return self.table.num_rows

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        if isinstance(idx, int):
            idx = [idx]

        entries = self.table.take(idx).to_pandas()

        if self.transform:
            entries = self.transform(entries)

        return entries
