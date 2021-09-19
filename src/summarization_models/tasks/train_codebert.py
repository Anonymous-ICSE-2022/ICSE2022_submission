from dataclasses import dataclass
from typing import Literal

import hydra
from omegaconf import MISSING
from summarization_models.models.codebert.codebert import CodeBERT, CodeBERTConfig

from .task import Task


@dataclass
class TokenizerConfig:
    name: Literal["roberta-base"] = "roberta-base"
    uncased: bool = True
    max_source_length: int = 256
    max_target_length: int = 128


@dataclass
class CodeBERTTaskConfig:
    model: CodeBERTConfig = CodeBERTConfig()
    tokenizer: TokenizerConfig = TokenizerConfig()

