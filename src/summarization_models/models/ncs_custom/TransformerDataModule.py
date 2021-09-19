import os
from typing import Any, List, NamedTuple, Optional, Union

import pandas as pd
from summarization_models.utils.logger import get_logger

logger = get_logger(__name__)
import pytorch_lightning as pl
import torch
from summarization_models.models.ncs_custom.TransformerParquetDataset import (
    ArrowDataset, InMemoryDataset, TransformerParquetDataset, df_to_arrow)
from summarization_models.models.ncs_custom.vocabulary import Vocabulary
from summarization_models.models.util.samplers import BucketBatchSampler
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.sampler import (BatchSampler, RandomSampler,
                                      SequentialSampler)
from tqdm import tqdm

from .TransformerDataset import TransformerDataset


class NCSBatch(NamedTuple):
    indices: List[int]
    code_token_ids: torch.Tensor
    comment_token_ids: torch.Tensor
    code_lengths: torch.Tensor
    code_extended_vocabularies: List[Vocabulary]
    code_extended_token_ids: torch.Tensor
    comment_alignments: torch.Tensor
    code_tokens: List[List[str]]
    comment_tokens: List[List[str]]
    max_comment_len: int
    max_code_len: int
    src_map: torch.Tensor


class TransformerDataModule(pl.LightningDataModule):
    def __init__(
        self,
        *,
        cache_dir: str,
        train_path: str,
        valid_path: str,
        test_path: str,
        code_vocabulary: Vocabulary,
        comment_vocabulary: Vocabulary,
        batch_size=64,
        num_workers=4,
        shuffle_train=True,
        pin_memory=True,
        uncase=True,
        clear_cache: bool = False,
        use_arrow=False,
    ):
        super().__init__()

        self.train_path = train_path
        self.valid_path = valid_path
        self.test_path = test_path
        self.code_vocabulary = code_vocabulary
        self.comment_vocabulary = comment_vocabulary
        self.batch_size = batch_size
        self.shuffle_train = shuffle_train
        self.pin_memory = pin_memory
        self.num_workers = num_workers
        self.uncase = uncase
        self.clear_cache = clear_cache
        self.cache_dir = cache_dir
        self._train_arrow = os.path.join(cache_dir, "train.arrow")
        self._test_arrow = os.path.join(cache_dir, "test.arrow")
        self._valid_arrow = os.path.join(cache_dir, "valid.arrow")
        self.use_arrow = use_arrow

    def _make_dataset(self, arrow_path, inmemory_path):
        if self.use_arrow:
            return ArrowDataset(arrow_path)

        return InMemoryDataset(inmemory_path)

    def prepare_data(self):
        "just to make the type checker shut up"

        if self.use_arrow:
            if self.clear_cache:
                for filename in [
                    self._train_arrow,
                    self._test_arrow,
                    self._valid_arrow,
                ]:
                    os.remove(filename)
                origins = [self.train_path, self.test_path, self.valid_path]
                targets = [self._train_arrow, self._test_arrow, self._valid_arrow]

                logger.info(f"Cachine files")
                for origin, target in zip(origins, targets):
                    if not os.path.exists(target):
                        logger.info(f"Caching {origin} to arrow..")
                        df_to_arrow(
                            pd.read_json(
                                origin, orient="records", lines=True, chunksize=2 ** 28
                            ),
                            arrow_filepath=target,
                        )

    def setup(self, stage: Optional[str]) -> None:
        if stage == "fit":
            self.train_dataset = self._make_dataset(self._train_arrow, self.train_path)
            self.val_dataset = self._make_dataset(self._valid_arrow, self.valid_path)

        else:
            self.test_dataset = TransformerParquetDataset(filepath=self._test_arrow)

    def _collate_fn(self, items: List[Any]) -> NCSBatch:
        batch_indices = []
        batch_code_tokens = []
        batch_comment_tokens = []

        batch_code_token_ids = []
        batch_comment_token_ids = []

        batch_code_lens = []
        batch_comment_lens = []

        batch_code_extended_vocabularies = []
        batch_code_extended_token_ids = []

        batch_alignments = []
        for item in items:
            batch_indices.append(item["id"])
            batch_code_tokens.append(
                [
                    token.lower() if self.uncase else token
                    for token in item["code_tokens"]
                ]
            )
            batch_code_lens.append(len(item["code_tokens"]))
            batch_comment_tokens.append(
                [
                    token.lower() if self.uncase else token
                    for token in item["comment_tokens"]
                ]
            )
            batch_comment_lens.append(len(item["comment_tokens"]))
            batch_comment_token_ids.append(
                [
                    token.lower() if self.uncase else token
                    for token in item["comment_tokens"]
                ]
            )

        max_code_len = min(max(batch_code_lens), self.code_vocabulary.max_len)
        max_comment_len = min(max(batch_comment_lens), self.comment_vocabulary.max_len)

        # now encode tokens
        for i, code_tokens in enumerate(batch_code_tokens):
            # encode code tokens
            code_tokens = batch_code_tokens[i]
            comment_tokens = batch_comment_tokens[i]
            code_token_ids = self.code_vocabulary.encode(
                code_tokens, max_len=max_code_len
            )

            batch_code_token_ids.append(code_token_ids)

            # minimize the amount of padding by picking minimum
            # maximum code length
            batch_code_lens[i] = min(max_code_len, batch_code_lens[i])

            # create a vocabulary for each example
            extended_vocabulary = create_extended_vocabulary(
                code_tokens, max_len=max_code_len
            )
            extended_code_token_ids = extended_vocabulary.encode(code_tokens)

            batch_code_extended_vocabularies.append(extended_vocabulary)
            batch_code_extended_token_ids.append(extended_code_token_ids)

            # encode comment tokens
            comment_tokens = comment_tokens[
                : max_comment_len - 2
            ]  # make room for sos & eos tokens if needed
            batch_comment_token_ids[i] = self.comment_vocabulary.encode(
                comment_tokens, add_eos_sos=True, max_len=max_comment_len
            )

            # alignment
            alignment_mask = torch.tensor(
                extended_vocabulary.encode(
                    [extended_vocabulary.unk_token]
                    + comment_tokens
                    + [extended_vocabulary.unk_token],
                    max_len=max_comment_len,
                    pad=False,
                )
            ).long()

            batch_alignments.append(alignment_mask)

        # convert extended vocabulary mapping
        # to vectors for consumption by the model
        alignments = make_tgt(batch_alignments)
        src_maps = make_src_map(
            torch.tensor(batch_code_extended_token_ids),
            batch_code_extended_vocabularies,
            self.code_vocabulary,
        )

        return NCSBatch(
            code_token_ids=torch.LongTensor(batch_code_token_ids),
            comment_token_ids=torch.LongTensor(batch_comment_token_ids),
            code_lengths=torch.LongTensor(batch_code_lens),
            code_extended_token_ids=torch.Tensor(batch_code_extended_token_ids),
            code_extended_vocabularies=batch_code_extended_vocabularies,
            comment_alignments=alignments,
            code_tokens=batch_code_tokens,
            comment_tokens=batch_comment_tokens,
            max_comment_len=max_comment_len,
            max_code_len=max_code_len,
            src_map=src_maps,
            indices=batch_indices,
        )

    def train_dataloader(self) -> Any:
        base_sampler = RandomSampler(range(len(self.train_dataset)))
        sampler = BucketBatchSampler(
            base_sampler,
            self.batch_size,
            sort_key=lambda i: len(self.train_dataset["code_tokens"][i]),
        )

        distributed_sampler = sampler

        return DataLoader(
            self.train_dataset,
            batch_sampler=distributed_sampler,
            num_workers=self.num_workers,
            collate_fn=self._collate_fn,
            pin_memory=self.pin_memory,
        )

    def val_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        base_sampler = SequentialSampler(range(len(self.val_dataset)))

        sampler = BucketBatchSampler(
            base_sampler,
            self.batch_size,
            sort_key=lambda i: len(self.train_dataset["code_tokens"][i]),
        )
        return DataLoader(
            self.val_dataset,
            batch_sampler=sampler,
            num_workers=self.num_workers,
            collate_fn=self._collate_fn,
            pin_memory=self.pin_memory,
        )

    def test_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        base_sampler = SequentialSampler(range(len(self.test_dataset)))

        sampler = BatchSampler(base_sampler, self.batch_size, drop_last=False)

        return DataLoader(
            self.test_dataset,
            batch_sampler=sampler,
            num_workers=self.num_workers,
            collate_fn=self._collate_fn,
            pin_memory=self.pin_memory,
        )


def create_extended_vocabulary(
    source_tokens: List[str], max_len=Optional[int]
) -> Vocabulary:
    return Vocabulary.from_tokens(
        [source_tokens], boundary_tokens=None, max_len=max_len,
    )


def make_src_map(
    encoded_tokens: torch.Tensor,
    vocabularies: List[Vocabulary],
    origin_vocabulary: Vocabulary,
):
    """[summary]

    Parameters
    ----------
    data : torch.Tensor
        [batch_size, seq_len]

    Returns
    -------
    src_map
        (batch_size,  src_len,  dynamic vocab size)
    """
    vocab_size = max([len(vocab) for vocab in vocabularies])
    batch_size, sentence_length = encoded_tokens.size()
    alignment = torch.zeros(sentence_length, batch_size, vocab_size, dtype=torch.float,)
    for i, example in enumerate(encoded_tokens):
        for j, token_idx in enumerate(example):
            if token_idx != vocabularies[i].pad_idx:
                alignment[j, i, int(token_idx)] = 1
    return alignment


def make_tgt(data):
    # print(f"data item shape: {data[0].shape}")
    tgt_size = max([item.size(0) for item in data])
    # print(f"max tgt size: {tgt_size}")
    alignment = torch.zeros(len(data), tgt_size).long()
    for i, sent in enumerate(data):
        # print(f"sent shape: {sent.shape}")
        alignment[i, : sent.size(0)] = sent
    return alignment
