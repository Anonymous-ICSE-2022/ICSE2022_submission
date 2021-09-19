from dataclasses import dataclass
from typing import Dict

import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig
from omegaconf.omegaconf import II, OmegaConf
from summarization_models.config.config import register_module

from .TransformerConfig import TransformerAhmadConfig

cs = ConfigStore.instance()


@dataclass
class DataConfig:
    base_dir: str
    batch_size: str


@dataclass
class Module:
    _target_: str


@register_module("c2nl")
def launch_module(cfg: Module) -> None:
    print("Inside")
