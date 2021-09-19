from typing import NamedTuple


class SpecialTokensBase(NamedTuple):
    PAD: str = "<P>"
    SOS: str = "<S>"
    EOS: str = "</S>"
    UNK: str = "<UNK>"


SpecialTokens = SpecialTokensBase()

ALL_SPECIAL_TOKENS = (
    SpecialTokens.PAD,
    SpecialTokens.SOS,
    SpecialTokens.EOS,
    SpecialTokens.UNK,
)

