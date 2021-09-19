from dataclasses import dataclass


@dataclass
class TransformerAhmadConfig:

    # data config
    max_source_length: int = 150
    max_target_length: int = 50

    source_vocab_size: int = 50000
    target_vocab_size: int = 30000

    code_pad_idx: int = 0
    comment_pad_idx: int = 0

    # encoder + decoder
    d_ff: int = 2048
    d_model: int = 512
    num_encoder_layers: int = 6
    num_decoder_layers: int = 6

    # Reuse target embeddings for decoder
    share_decoder_weights: bool = True

    embedding_dropout: float = 0.2
    general_dropout: float = 0.2

    # attention
    n_heads: int = 8
    max_relative_positions: int = 32
    copy_attention: bool = True
    attention_dropout: float = 0.2

    # copy mechanism:
    force_copy: bool = False
    # optimizer
    learning_rate: float = 0.0001

    num_warmup_steps: int = 5000
    weight_decay: float = 0.99

