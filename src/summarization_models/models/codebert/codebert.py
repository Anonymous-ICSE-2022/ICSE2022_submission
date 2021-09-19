from dataclasses import dataclass
from logging import getLogger
from typing import Any, List, Literal, Optional

import numpy as np
import pytorch_lightning as pl
from summarization_models.models.codebert.model import Seq2Seq
from torch import nn
from transformers import (
    AdamW,
    RobertaConfig,
    RobertaModel,
    RobertaTokenizer,
    get_linear_schedule_with_warmup,
)

logger = getLogger(__name__)

MODEL_CLASSES = {"roberta": (RobertaConfig, RobertaModel, RobertaTokenizer)}


@dataclass
class CodeBERTConfig:
    max_target_length: int = 128
    max_source_length: int = 256
    pretrained_model: Literal[
        "microsoft/codebert-base", "roberta-base"
    ] = "microsoft/codebert-base"
    num_decoder_layers: int = 6
    beam_size: int = 10
    lr: float = 1e-5
    weight_decay: float = 0.0
    warmup_steps: float = 0.1
    uncased: bool = True
    cls_token_id: int = 0
    sep_token_id: int = 2
    adam_epsilon: float = 1e-8
    pretrained_weights_cache_dir: Optional[str] = None


from pathlib import Path

from omegaconf import OmegaConf


class CodeBERT(pl.LightningModule):
    def __init__(self, cfg: CodeBERTConfig):
        super().__init__()

        self.save_hyperparameters(cfg.__dict__)
        self.cfg = cfg
        config_class = RobertaConfig
        model_class = RobertaModel
        weights_cache_dir = (
            Path(cfg.pretrained_weights_cache_dir)
            if cfg.pretrained_weights_cache_dir is not None
            else None
        )

        if weights_cache_dir is not None:
            if not weights_cache_dir.exists():
                weights_cache_dir.mkdir()

        pretrain_config = config_class.from_pretrained(
            cfg.pretrained_model, cache_dir=cfg.pretrained_weights_cache_dir
        )
        encoder = model_class.from_pretrained(
            cfg.pretrained_model, config=pretrain_config
        )

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=pretrain_config.hidden_size,
            nhead=pretrain_config.num_attention_heads,
        )
        decoder = nn.TransformerDecoder(
            decoder_layer, num_layers=cfg.num_decoder_layers
        )
        self.model = Seq2Seq(
            encoder=encoder,
            decoder=decoder,
            config=pretrain_config,
            beam_size=cfg.beam_size,
            max_length=cfg.max_target_length,
            sos_id=cfg.cls_token_id,
            eos_id=cfg.sep_token_id,
        )

    def forward(self, batch):
        # in lightning, forward defines the prediction/inference actions
        index, source_ids, target_ids, source_mask, target_mask = batch
        _, loss, num = self.model(
            source_ids=source_ids,
            source_mask=source_mask,
            target_ids=target_ids,
            target_mask=target_mask,
        )

        return loss

    def on_post_move_to_device(self) -> None:
        # To maintain compatibility with TPUs
        self.model.tie_weights()

    def training_step(self, batch, batch_idx):
        # training_step defined the train loop.
        # It is independent of forward
        loss, _, _ = self.model(
            source_ids=batch["source_ids"],
            source_mask=batch["source_mask"],
            target_ids=batch["target_ids"],
            target_mask=batch["target_mask"],
        )

        self.log("train/loss", loss)
        return loss

    def validation_step(self, batch, *args):
        _, loss, num = self.model(
            source_ids=batch["source_ids"],
            source_mask=batch["source_mask"],
            target_ids=batch["target_ids"],
            target_mask=batch["target_mask"],
        )

        return {"val_loss": loss.sum().item(), "tokens_num": num.sum().item()}

    def validation_epoch_end(self, outputs: List[Any]) -> None:
        total_tokens = 0
        total_loss = 0
        for output in outputs:
            total_loss += output["val_loss"]
            total_tokens += output["tokens_num"]

        val_loss = total_loss / total_tokens
        val_ppl = round(np.exp(val_loss), 5)

        self.log("validation/ppl", val_ppl)
        self.log("validation/loss", val_loss)

    def test_step(self, batch, *args, **kwargs):
        index, source_ids, target_ids, source_mask, target_mask = batch
        preds = self.model(source_ids=source_ids, source_mask=source_mask)

        preds = preds[:, 0, :]
        return {
            "preds": preds,
            "refs": target_ids,
            "index": index,
        }

    def test_step_end(self, batch_parts):
        # Need this so that pytorch does not
        return batch_parts

    def predict_step(self, batch: Any, *args, **kwargs,) -> Any:
        preds = self.model(
            source_ids=batch["source_ids"], source_mask=batch["source_mask"]
        )
        preds = preds[:, 0, :].detach()

        return preds

    # def test_epoch_end(self, outputs: List[Any]) -> None:
    #     try:
    #         if self.hparams.save_test_output:
    #             # gather results from different batches
    #             preds = list(map(itemgetter("preds"), outputs))
    #             refs = np.array(list(map(itemgetter("refs"), outputs)))
    #             indices = list(map(itemgetter("index"), outputs))
    #             all_preds = [pred.cpu().numpy() for pred in preds]
    #             all_refs = [ref.cpu().numpy() for ref in refs]
    #             all_indices = [index.cpu().numpy() for index in indices]
    #             all_preds = np.row_stack(all_preds)
    #             all_refs = np.row_stack(all_refs)
    #             all_indices = np.row_stack(all_indices)
    #             with open(self.hparams.save_test_output, "wb") as outfile:
    #                 pickle.dump(
    #                     {"preds": all_preds, "refs": all_refs, "index": all_indices,},
    #                     outfile,
    #                 )
    #     except Exception as e:
    #         print()
    #         print(f"=>>>> Encountered exception")
    #         print(e)

    @property
    def num_training_steps(self) -> int:
        """Total training steps inferred from datamodule and devices.

            Adapted from: https://github.com/PyTorchLightning/pytorch-lightning/issues/5449
        """
        if self.trainer.max_steps:
            return self.trainer.max_steps

        limit_batches = self.trainer.limit_train_batches
        batches = len(self.train_dataloader())
        batches = (
            min(batches, limit_batches)
            if isinstance(limit_batches, int)
            else int(limit_batches * batches)
        )

        num_devices = max(1, self.trainer.num_gpus, self.trainer.num_processes)
        if self.trainer.tpu_cores:
            num_devices = max(num_devices, self.trainer.tpu_cores)

        effective_accum = self.trainer.accumulate_grad_batches * num_devices
        return (batches // effective_accum) * self.trainer.max_epochs

    def configure_optimizers(self):
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p
                    for n, p in self.model.named_parameters()
                    if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": self.cfg.weight_decay,
            },
            {
                "params": [
                    p
                    for n, p in self.model.named_parameters()
                    if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]
        logger.warning(
            f"total training steps: {self.num_training_steps}, warmup steps: {self.num_training_steps * self.cfg.warmup_steps}, lr: {self.cfg.lr}"
        )
        optimizer = AdamW(
            optimizer_grouped_parameters, lr=self.cfg.lr, eps=self.cfg.adam_epsilon,
        )
        lr_scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(self.num_training_steps * self.cfg.warmup_steps),
            num_training_steps=self.num_training_steps,
        )

        return [optimizer], [{"scheduler": lr_scheduler, "interval": "step"}]
