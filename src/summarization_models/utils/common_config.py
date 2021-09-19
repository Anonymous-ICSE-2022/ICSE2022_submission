from dataclasses import dataclass
from typing import List, Optional, Type, TypeVar

from omegaconf import MISSING, OmegaConf
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks.base import Callback


@dataclass
class TrainerConfig:
    # hardware
    gpus: str = "-1"
    accelerator: Optional[str] = "ddp"
    num_nodes: int = 1
    precision: int = 16
    num_tpus: Optional[int] = None

    # training
    gradient_clip_norm: float = 5.0
    max_epochs: int = 200
    check_val_every_n_epoch: int = 1
    batch_size: int = 64

    # pl lightning
    plugins: Optional[str] = ""

    # misc
    weights_summary: Optional[str] = None

    # debug/dev
    limit_train_batches: float = 1.0
    limit_val_batches: float = 1.0

    # checkpoint_path
    checkpoint_dir: str = MISSING
    resume_from_checkpoint: Optional[str] = None


@dataclass
class WandbConfig:
    enabled: bool = False
    experiment_name: Optional[str] = None
    project: Optional[str] = None


T = TypeVar("T")


def get_config(cls: Type[T]) -> T:
    config: T = OmegaConf.merge(
        OmegaConf.structured(cls), OmegaConf.from_cli()
    )  # type:ignore
    return config


def trainer_from_config(
    config: TrainerConfig, callbacks: List[Callback] = [], logger=None,
) -> Trainer:
    return Trainer(
        gpus=config.gpus,
        accelerator=config.accelerator,
        tpu_cores=config.num_tpus,
        num_nodes=config.num_nodes,
        logger=logger,
        gradient_clip_val=config.gradient_clip_norm,
        max_epochs=config.max_epochs,
        precision=config.precision,
        resume_from_checkpoint=config.resume_from_checkpoint,
        callbacks=callbacks,
        weights_summary=config.weights_summary,
        limit_val_batches=config.limit_val_batches,
        limit_train_batches=config.limit_train_batches,
        check_val_every_n_epoch=config.check_val_every_n_epoch,
    )
