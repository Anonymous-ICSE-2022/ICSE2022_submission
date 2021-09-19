import os
from argparse import ArgumentParser
from multiprocessing import cpu_count
from typing import Optional

from configargparse import Namespace
from pytorch_lightning.core.datamodule import LightningDataModule
from torch.utils.data.dataloader import DataLoader
from transformers import RobertaTokenizer

from .CodeBertDataset import CodeBertDataset
from .CodeBertTransforms import CodeBertTransforms


class CodeBertDataModule(LightningDataModule):
    def __init__(self, args, tokenizer: RobertaTokenizer):
        super().__init__()
        self.tokenizer = tokenizer
        self.train_file_path = os.path.join(args.base_dir, args.train_file_path)
        self.val_file_path = os.path.join(args.base_dir, args.val_file_path)
        self.test_file_path = os.path.join(args.base_dir, args.test_file_path)
        self.max_source_length = args.max_source_length
        self.max_target_length = args.max_target_length
        self.batch_size = args.batch_size
        self.num_workers = min(cpu_count(), args.num_workers)

    @staticmethod
    def add_args(parent_parser: ArgumentParser):
        parser = parent_parser.add_argument_group("CodeBERT Data Module")
        parser.add_argument("--train_file_path", help="filepath")
        parser.add_argument("--test_file_path", help="filepathas")
        parser.add_argument("--val_file_path", help="filepathasd")
        parser.add_argument("--batch_size", help="filepathasdasdaa", type=int)
        parser.add_argument(
            "--num_workers", help="filepathasdasdaa", type=int, default=48
        )
        parser.add_argument(
            "--base_dir", help="filepathasdasdaa", type=str, default="./"
        )

        return parent_parser

    def setup(self, stage: Optional[str]):

        # Assign train/val datasets for use in dataloaders
        if stage == "fit" or stage is None:
            self.train_dataset = CodeBertDataset(
                self.train_file_path,
                transform=CodeBertTransforms.from_args(
                    Namespace(
                        max_source_length=self.max_source_length,
                        max_target_length=self.max_target_length,
                        stage="train",
                    ),
                    tokenizer=self.tokenizer,
                ),
            )

            self.val_dataset = CodeBertDataset(
                self.val_file_path,
                transform=CodeBertTransforms.from_args(
                    Namespace(
                        max_source_length=self.max_source_length,
                        max_target_length=self.max_target_length,
                        stage="train",
                    ),
                    tokenizer=self.tokenizer,
                ),
            )
        # Assign test dataset for use in dataloader(s)
        if stage == "test" or stage is None:
            self.test_dataset = CodeBertDataset(
                self.test_file_path,
                transform=CodeBertTransforms.from_args(
                    Namespace(
                        max_source_length=self.max_source_length,
                        max_target_length=self.max_target_length,
                        stage="test",
                    ),
                    tokenizer=self.tokenizer,
                ),
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
