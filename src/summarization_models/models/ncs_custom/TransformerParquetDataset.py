import logging
import os
import sys
from collections import defaultdict
from dataclasses import dataclass
from functools import partial
from typing import Dict, Iterable, Iterator, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm


class TransformerParquetDataset(Dataset):
    def __init__(
        self, filepath: str, transform=lambda entry: {k: v[0] for k, v in entry.items()}
    ):
        self.filepath = filepath
        self.transform = transform
        print(f"Loading jsonl for in-memory usage")
        # just load in-memory and hope for the best :)
        self.data = pd.read_json(filepath, lines=True, orient="records").to_dict(
            "records"
        )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        return self.data[idx]


def records_to_columnar(entries):
    if not entries:
        return {}

    columnar = {k: [] for k in entries[0]}

    for entry in entries:
        for k, v in entry.items():
            columnar[k].append(v)

    return columnar


def df_to_arrow(dfs: Iterable[pd.DataFrame], arrow_filepath: str) -> None:
    """Takes an iterable of pandas dataframes and writes it to an arrow file.

    Parameters
    ----------
    dfs : Iterator[pd.DataFrame]
        iterable of dataframes, 
        eg. pd.read_csv(..., chunksize=22)
    arrow_filepath : str
        path to arrow file to write to.
    """
    dfs = iter(dfs)
    with pa.OSFile(arrow_filepath, "wb") as sink:
        df = next(dfs)
        table = pa.Table.from_pandas(df)
        with pa.RecordBatchFileWriter(sink, table.schema) as writer:
            writer.write_table(table)
            for df in tqdm(dfs):
                table = pa.Table.from_pandas(df)
                writer.write_table(table)


logger = logging.getLogger(__name__)


class ArrowDataset(Dataset):
    """A memory mapped dataset backed by an arrow table
    """

    def __init__(self, filepath: str):
        self.filepath = filepath
        self.mmap = pa.memory_map(filepath)

        with pa.ipc.RecordBatchFileReader(self.mmap) as reader:
            self.table = reader.read_all()

        self.colnames = self.table.column_names

    def __len__(self):
        return self.table.num_rows

    def _get_record_fast(self, idx: int) -> Dict:
        return {colname: self.table[colname][idx].as_py() for colname in self.colnames}

    def __getitem__(self, idx: Union[int, str]):
        if isinstance(idx, str):
            assert (
                idx in self.table.column_names
            ), f"Column {idx} not in table. Columns in table: [' '.join(self.table.column_names)]"
            return self.table[idx]

        if isinstance(idx, (list, tuple, torch.Tensor)):
            logger.warn("Many indices requested")

        return self._get_record_fast(idx)


class InMemoryDataset(Dataset):
    """A memory mapped dataset backed by an arrow table
    """

    def __init__(self, filepath: str):
        self.filepath = filepath
        self.data = pd.read_json(filepath, lines=True, orient="records").to_dict(
            orient="records"
        )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: Union[int, str]):
        return self.data[idx]
