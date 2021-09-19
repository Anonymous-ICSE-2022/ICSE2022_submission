import pytorch_lightning as pl
import sacrebleu
import torch
import torch.nn.functional as F
from nltk.translate.bleu_score import corpus_bleu
from summarization_models.models.ncs_custom.onmt.CopyGenerator import CopyGeneratorLoss
from summarization_models.models.ncs_custom.TransformerDataModule import NCSBatch
from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR

from .prediction_utils import convert_scores_to_predictions
from .TransformerAhmad import TransformerAhmad, TransformerAhmadConfig
from .vocabulary import Vocabulary


class TransformerLightningModule(pl.LightningModule):
    def __init__(
        self,
        config: TransformerAhmadConfig,
        code_vocabulary: Vocabulary,
        comment_vocabulary: Vocabulary,
    ):
        super(TransformerLightningModule, self).__init__()

        self.save_hyperparameters()

        # model creation
        self.model = TransformerAhmad(config)
        if config.copy_attention:
            self.criterion = CopyGeneratorLoss(
                len(comment_vocabulary),
                force_copy=config.force_copy,
                unk_index=comment_vocabulary.unk_idx,
                ignore_index=comment_vocabulary.pad_idx,
            )
        else:
            self.criterion = nn.CrossEntropyLoss(
                ignore_index=comment_vocabulary.pad_idx
            )

    def on_post_move_to_device(self) -> None:
        # To maintain compatibility with TPUs
        if self.hparams.config.share_decoder_weights:
            self.model.tie_weights()

    def forward(self, batch: NCSBatch) -> torch.Tensor:
        """Forward run

        Parameters
        ----------
        batch : ExampleBatch
            Batch of examples for prediction

        Returns
        -------
        torch.Tensor
            Tensor of size [batch_size * target_length, target_vocab_Size]
        """
        (code, code_tokens_len, comment) = (
            batch.code_token_ids,
            batch.code_lengths,
            batch.comment_token_ids,
        )
        code = code.T.unsqueeze(2)
        comment = comment.T.unsqueeze(2)

        src_maps = None
        if self.hparams.config.copy_attention:

            scores, _, _ = self.model(
                source=code,
                source_lengths=code_tokens_len,
                target=comment,
                src_map=batch.src_map,
            )  # [batch_size * tgt_len, target_vocab_size]
        else:
            scores, _, _ = self.model(
                source=code,
                source_lengths=code_tokens_len,
                target=comment,
                src_map=src_maps,
            )
            scores = scores.transpose(0, 1).contiguous()
            scores = scores.view(-1, scores.size(-1))

        return scores

    def training_step(self, batch: NCSBatch, batch_idx):
        original_comment = batch.comment_token_ids
        scores = self(batch)
        bsz = batch.comment_token_ids.size(0)

        # scores: [bsz * tgt_len, extended_vocab_size]
        if self.hparams.config.copy_attention:
            alignments = batch.comment_alignments
            loss = self.criterion(
                scores,
                alignments[:, 1:].contiguous().view(-1),
                original_comment[:, 1:].contiguous().view(-1),
            )

            loss = loss.view(bsz, -1)
            loss = loss.sum(1)

            loss = loss.mean()

            if batch_idx % 50 == 0:
                predictions = convert_scores_to_predictions(
                    scores,
                    self.hparams.comment_vocabulary,
                    batch.code_extended_vocabularies,
                    batch_size=original_comment.size(0),
                )
                sample_pred = " ".join(predictions[0])
                sample_comment = " ".join(batch.comment_tokens[0])
                print()
                print(f"=> Sample prediction: {sample_pred}")
                print(f"=>             Label: {sample_comment}")
                print()
        else:
            target = original_comment[:, 1:].contiguous().view(-1)
            loss = self.criterion(scores, target)

        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch: NCSBatch, batch_idx):
        original_comment = batch.comment_token_ids
        scores = self(batch)
        bsz = batch.comment_token_ids.size(0)

        if self.hparams.config.copy_attention:
            alignments = batch.comment_alignments

            loss = self.criterion(
                scores,
                alignments[:, 1:].contiguous().view(-1),
                original_comment[:, 1:].contiguous().view(-1),
            )
            loss = loss.view(original_comment.size(0), -1)

            loss = loss.sum(1)
            loss_per_token = loss.detach().div(original_comment.size(1) - 1).mean()

            loss = loss.mean()

            predictions = convert_scores_to_predictions(
                scores,
                self.hparams.comment_vocabulary,
                batch.code_extended_vocabularies,
                batch_size=original_comment.size(0),
            )
        else:
            target = original_comment[:, 1:].contiguous().view(-1)
            loss = self.criterion(scores, target)

            # TODO
            predictions = ["asdasd", "Asdasd"]
            # references = ["Asdasdas", "Asdasd"]

        return {
            "loss": loss,
            "ppl": loss_per_token,
            "predictions": predictions,
            "references": batch.comment_tokens[: len(predictions)],
        }

    def validation_step_end(self, validation_outputs):
        self.log("validation_loss", validation_outputs["loss"].mean())

        return validation_outputs

    # compute bleu scores
    def validation_epoch_end(self, validation_outputs):
        all_losses = []
        all_predictions = []
        all_references = []

        for validation_output in validation_outputs:
            all_losses.append(validation_output["loss"].item())
            all_predictions.extend(validation_output["predictions"])
            all_references.extend(validation_output["references"])

        bleu_score = sacrebleu.corpus_bleu(
            [" ".join(p) for p in all_predictions],
            [[" ".join(r) for r in all_references]],
        )
        nltk_bleu_score = corpus_bleu(all_references, all_predictions)
        self.log("validation_bleu", bleu_score.score, prog_bar=True)
        self.log("nltk_bleu", nltk_bleu_score, prog_bar=True)

    def configure_optimizers(self):
        optimizer = Adam(
            [
                parameter
                for parameter in self.model.parameters()
                if parameter.requires_grad
            ],
            lr=self.hparams.config.learning_rate,
        )
        scheduler = StepLR(optimizer, step_size=1, gamma=0.99)
        # scheduler = get_linear_schedule_with_warmup(
        #     optimizer,
        #     num_warmup_steps=self.hparams.config.num_warmup_steps,
        #     num_training_steps=self.num_training_steps,
        # )
        return (
            [optimizer],
            [
                {"scheduler": scheduler, "interval": "epoch", "frequency": 1,}
                # scheduler
            ],
        )

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

