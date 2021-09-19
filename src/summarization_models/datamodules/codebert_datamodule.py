from functools import partial
from os import cpu_count
from pathlib import Path
from typing import Optional

from datasets import load_dataset
from pytorch_lightning.core.datamodule import LightningDataModule
from torch.utils.data import dataset
from torch.utils.data.dataloader import DataLoader
from transformers import RobertaTokenizer


def compute_features(
    source,
    target,
    *,
    tokenizer: RobertaTokenizer,
    max_target_length,
    max_source_length,
    stage,
):
    source_tokens = tokenizer.tokenize(source)[: max_source_length - 2]
    source_tokens = [tokenizer.cls_token] + source_tokens + [tokenizer.sep_token]
    source_ids = tokenizer.convert_tokens_to_ids(source_tokens)
    source_mask = [1] * (len(source_tokens))
    padding_length = max_source_length - len(source_ids)
    source_ids += [tokenizer.pad_token_id] * padding_length
    source_mask += [0] * padding_length

    # target
    if stage == "test":
        target_tokens = tokenizer.tokenize(target)[: max_target_length - 2]
    else:
        target_tokens = tokenizer.tokenize(target)[: max_target_length - 2]
    target_tokens = [tokenizer.cls_token] + target_tokens + [tokenizer.sep_token]
    target_ids = tokenizer.convert_tokens_to_ids(target_tokens)
    target_mask = [1] * len(target_ids)
    padding_length = max_target_length - len(target_ids)
    target_ids += [tokenizer.pad_token_id] * padding_length
    target_mask += [0] * padding_length

    return source_ids, target_ids, source_mask, target_mask


def batch_compute_features(
    batch, *, tokenizer: RobertaTokenizer, max_target_length, max_source_length, stage,
):
    n = len(batch["docstring_tokens"])

    all_source_ids = []
    all_target_ids = []
    all_source_mask = []
    all_target_mask = []

    for i in range(n):
        source_ids, target_ids, source_mask, target_mask = compute_features(
            " ".join(batch["code_tokens"][i]),
            " ".join(batch["docstring_tokens"][i]),
            tokenizer=tokenizer,
            max_source_length=max_source_length,
            max_target_length=max_target_length,
            stage=stage,
        )

        all_source_ids.append(source_ids)
        all_target_ids.append(target_ids)
        all_source_mask.append(source_mask)
        all_target_mask.append(target_mask)

    return {
        "source_ids": all_source_ids,
        "target_ids": all_target_ids,
        "source_mask": all_source_mask,
        "target_mask": all_target_mask,
    }


class CodeBertDataModule(LightningDataModule):
    def __init__(
        self,
        dataset_name_or_path,
        tokenizer: RobertaTokenizer,
        num_workers: int,
        batch_size: int,
        max_source_length: int,
        max_target_length: int,
        dataset_config_name: Optional[str] = None,
        cache_name: Optional[str] = None,
        clear_cache: bool = False,
        dataset_cache_dir: Optional[str] = None,
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.dataset_name_or_path = dataset_name_or_path
        self.dataset_config_name = dataset_config_name
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.dataset_cache_dir = dataset_cache_dir
        self.cache_name = cache_name
        self.clear_cache = clear_cache

        if dataset_cache_dir is not None:
            if not Path(dataset_cache_dir).exists():
                Path(dataset_cache_dir).mkdir()
        if cache_name is not None:
            cache_dir = Path(cache_name).parents[0]
            if not cache_dir.exists():
                cache_dir.mkdir()

    def setup(self, stage: Optional[str]):
        map_fn = partial(
            batch_compute_features,
            tokenizer=self.tokenizer,
            max_target_length=self.max_target_length,
            max_source_length=self.max_source_length,
            stage="train",
        )

        # Assign train/val datasets for use in dataloaders
        if stage == "fit" or stage is None:

            self.train_dataset = load_dataset(
                self.dataset_name_or_path,
                self.dataset_config_name,
                split="train",
                cache_dir=self.dataset_cache_dir,
            )
            self.train_dataset = self.train_dataset.map(
                map_fn,
                batched=True,
                num_proc=cpu_count(),
                cache_file_name=f"{self.cache_name}_train.arrow"
                if self.cache_name
                else None,
                # load_from_cache_file=False if self.clear_cache else True,
            )

            self.train_dataset.set_format(
                type="torch",
                columns=["source_ids", "target_ids", "source_mask", "target_mask"],
            )

            self.val_dataset = load_dataset(
                self.dataset_name_or_path,
                self.dataset_config_name,
                split="validation",
                cache_dir=self.dataset_cache_dir,
            ).map(
                map_fn,
                batched=True,
                num_proc=cpu_count(),
                cache_file_name=f"{self.cache_name}_validation.arrow"
                if self.cache_name is not None
                else None,
                # load_from_cache_file=False if self.clear_cache else True,
            )

            self.val_dataset.set_format(
                type="torch",
                columns=["source_ids", "target_ids", "source_mask", "target_mask"],
            )

        # Assign test dataset for use in dataloader(s)
        if stage == "test" or stage is None:
            self.test_dataset = load_dataset(
                self.dataset_name_or_path,
                self.dataset_config_name,
                split="test",
                cache_dir=self.dataset_cache_dir,
            ).map(
                map_fn,
                batched=True,
                num_proc=cpu_count(),
                cache_file_name=f"{self.cache_name}_test.arrow"
                if self.cache_name is not None
                else None,
                # load_from_cache_file=False if self.clear_cache else True,
            )
            self.test_dataset.set_format(
                type="torch",
                columns=[
                    "id",
                    "source_ids",
                    "target_ids",
                    "source_mask",
                    "target_mask",
                ],
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
        )
