from typing import List, NamedTuple

import torch
from summarization_models.models.ncs_custom.onmt.GlobalAttention import GlobalAttention

from .onmt.CopyGenerator import CopyGenerator
from .onmt.embeddings import Embeddings
from .onmt.model import NMTModel
from .onmt.transformer_decoder import TransformerDecoder
from .onmt.transformer_encoder import TransformerEncoder
from .TransformerConfig import TransformerAhmadConfig


class TransformerAhmad(torch.nn.Module):
    def __init__(self, config: TransformerAhmadConfig) -> None:
        super(TransformerAhmad, self).__init__()

        self.config = config

        source_embedding = Embeddings(
            config.d_model,
            config.source_vocab_size,
            word_padding_idx=config.code_pad_idx,
            dropout=config.embedding_dropout,
        )
        target_embedding = Embeddings(
            config.d_model,
            config.target_vocab_size,
            word_padding_idx=config.comment_pad_idx,
            position_encoding=True,
            dropout=config.embedding_dropout,
            max_len_pos=config.max_target_length,
        )

        self.encoder = TransformerEncoder(
            num_layers=config.num_encoder_layers,
            d_model=config.d_model,
            heads=config.n_heads,
            d_ff=config.d_ff,
            dropout=config.general_dropout,
            attention_dropout=config.attention_dropout,
            embeddings=source_embedding,
            max_relative_positions=config.max_relative_positions,
        )

        self.decoder = TransformerDecoder(
            num_layers=config.num_decoder_layers,
            d_model=config.d_model,
            heads=config.n_heads,
            d_ff=config.d_ff,
            dropout=config.general_dropout,
            attention_dropout=config.attention_dropout,
            embeddings=target_embedding,
            self_attn_type="scaled-dot",
            alignment_heads=0,
            alignment_layer=0,
            copy_attn=config.copy_attention,
        )

        if config.copy_attention:
            self.generator_projection = torch.nn.Linear(
                config.d_model, config.target_vocab_size
            )
            self.copy_attention = GlobalAttention(
                dim=config.d_model, attn_type="general"
            )

            self.generator = CopyGenerator(
                config.d_model, self.generator_projection, config.comment_pad_idx
            )

            self.tie_weights()
            # print(target_embedding.emb_luts)
            # assert (
            #     self.generator.linear.weight.shape == target_embedding.word_lut.weight
            # ), f"Expected generator weight ({self.generator.linear.weight.shape}) to equal word lut shape ({target_embedding.word_lut.weight.shape})"

            # self.generator.linear.weight = target_embedding.word_lut.weight
        else:
            self.generator = torch.nn.Linear(
                config.d_model, config.target_vocab_size, bias=False
            )

    def tie_weights(self):
        self.generator_projection.weight = (
            self.decoder.embeddings.word_lut.weight
        )  # target_embedding.word_lut.weight

    def forward(
        self,
        source: torch.Tensor,
        source_lengths: torch.Tensor,
        target: torch.Tensor,
        src_map=torch.LongTensor,
    ):
        """Forward pass

        Parameters
        ----------
        source : torch.Tensor
            [src_len, batch_size, 1]
        source_lengths : torch.Tensor
            [batch_size]
        target : torch.Tensor
            [tgt_len, batch_size, 1]
        src_map : [type], optional
            [description], by default torch.LongTensor

        Returns
        -------
        [type]
            [batch_size * tgt_len, extended_vocab_size]
        """
        src_len = source_lengths.size(0)
        tgt_len = target.size(0)
        bsz = source.size(1)

        dec_in = target[:-1]  # exclude last target from inputs

        enc_state, memory_bank, lengths = self.encoder(source, source_lengths)
        self.decoder.init_state(source, memory_bank, enc_state)

        decoder_outs, decoder_attn = self.decoder(
            dec_in, memory_bank, memory_lengths=lengths, with_align=False
        )
        # decoder_outs: [tgt_len, bsz, dim]
        # memory_bank: [seq_len, batch_size, dim]

        # NOTE Replace any copied words to <UNK> when decoding.
        if self.config.copy_attention:
            decoder_outs = decoder_outs.transpose(
                0, 1
            ).contiguous()  # decoder outs: [batch_size, tgt_len, dim]
            memory_bank = memory_bank.transpose(
                0, 1
            ).contiguous()  # mem_bank: [batch_size, seq_len, dim]

            _, copy_attention = self.copy_attention(
                decoder_outs, memory_bank, memory_lengths=source_lengths
            )  # copy attention: (tgt_len, batch, src_len)

            # TODO: Add a flag to switch between separate attention
            # and using decoder attention
            # copy_attention = decoder_attn["copy"]

            # what we need for generator is: (batch * tgt_len, src_len)
            copy_attention = copy_attention.transpose(
                0, 1
            ).contiguous()  # (batch, tgt_len, src_len)

            # For debugging
            # assert copy_attention.shape[0] == bsz
            # assert decoder_outs.shape[0] == bsz
            # assert copy_attention.shape[0] == decoder_outs.shape[0]
            # assert copy_attention.shape[1] == decoder_outs.shape[1]

            copy_attention = copy_attention.view(
                -1, copy_attention.size(2)
            )  #  [batch_size * tgt_len, dim]
            decoder_outs = decoder_outs.contiguous().view(
                -1, decoder_outs.size(2)
            )  # batch_size * tgt_len, dim

            with torch.cuda.amp.autocast_mode.autocast(enabled=False):
                # IMPORTANT:
                # We need to disable fp16 for CopyGenerator
                # Not really sure what the problem is but it simply
                # doesn't work with fp16
                generated_targets = self.generator(
                    decoder_outs, copy_attention, src_map,
                )

        else:
            print(f"=> Decoder output shape: {decoder_outs.shape}")
            generated_targets = self.generator(decoder_outs)

        # generated_targets; (bsz * tgt_len, extended_vocab_size)
        return generated_targets, decoder_outs, decoder_attn

