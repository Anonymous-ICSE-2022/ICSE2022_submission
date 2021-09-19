from __future__ import annotations

import itertools
from collections import Counter
from copy import deepcopy
from typing import Any, Dict, Iterable, List, NamedTuple, Optional, Tuple, TypeVar

import yaml

try:
    from yaml import CDumper as Dumper
    from yaml import CLoader as Loader
except ImportError:
    from yaml import Loader, Dumper

from summarization_models.models.ncs_custom.vocabulary.SpecialTokens import (
    SpecialTokens,
)

T = TypeVar("T")


def first_not_none(items: Iterable[Optional[T]]) -> T:
    for item in items:
        if item is not None:
            return item
    raise ValueError("All items are None")


def load_yaml(filepath: str) -> dict:
    with open(filepath) as f:
        return yaml.load(f, Loader=Loader)


def dump_yaml(object: Any, filepath: str) -> None:
    with open(filepath, "w") as f:
        yaml.dump(object, f, Dumper=Dumper)


def first_not_none(items: Iterable[Optional[T]]) -> T:
    for item in items:
        if item is not None:
            return item
    raise ValueError("All items are None")


class Vocabulary:
    def __init__(
        self,
        token_to_idx: Dict[str, int] = {},
        idx_to_token: Dict[int, str] = {},
        max_length: Optional[int] = None,
        boundary_tokens: Optional[Tuple[str, str]] = (
            SpecialTokens.SOS,
            SpecialTokens.EOS,
        ),
        pad_token: Optional[str] = SpecialTokens.PAD,
        unk_token: Optional[str] = SpecialTokens.UNK,
    ):
        self.token_to_idx: Dict[str, int] = token_to_idx
        self.idx_to_token: Dict[int, str] = idx_to_token
        self.pad_token = pad_token
        self.unk_token = unk_token
        self.sos_token: Optional[str] = None
        self.eos_token: Optional[str] = None
        if boundary_tokens:
            assert (
                len(boundary_tokens) == 2
            ), f"Expecting len(boundary_tokens) to be 2, but found {len(boundary_tokens)}"
            self.sos_token, self.eos_token = boundary_tokens

        self.max_len: Optional[int] = max_length

    def save(self, filepath: str) -> None:
        """Serializes and saves this Vocabulary to a yaml file

        Parameters
        ----------
        filepath : str
            path to yaml file to save to
        """
        state_dict = deepcopy(self.__dict__)
        state_dict.pop("idx_to_token")

        assert "idx_to_token" not in state_dict

        dump_yaml(state_dict, filepath)

    @staticmethod
    def load(filepath: str) -> Vocabulary:
        vocabulary = Vocabulary()
        state_dict = load_yaml(filepath)

        vocabulary.__dict__ = state_dict
        vocabulary.idx_to_token = {v: k for k, v in vocabulary.token_to_idx.items()}
        return vocabulary

    @staticmethod
    def from_tokens(
        tokens_per_example: Iterable[Iterable[str]],
        size: int = None,
        max_len: Optional[int] = None,
        boundary_tokens: Optional[Tuple[str, str]] = (
            SpecialTokens.SOS,
            SpecialTokens.EOS,
        ),
        pad_token: str = SpecialTokens.PAD,
        unk_token: str = SpecialTokens.UNK,
        start_index: int = 0,
        uncase: bool = True,
    ) -> Vocabulary:
        vocabulary = Vocabulary(
            max_length=max_len,
            boundary_tokens=boundary_tokens,
            pad_token=pad_token,
            unk_token=unk_token,
        )
        token_to_idx, idx_to_token = mappings_from_iterable(
            tokens_per_example,
            special_tokens=vocabulary.special_tokens,
            size=size,
            start_index=start_index,
            uncase=uncase,
        )

        vocabulary.token_to_idx = token_to_idx
        vocabulary.idx_to_token = idx_to_token

        return vocabulary

    @property
    def special_tokens(self):
        all_special_tokens = [
            self.pad_token,
            self.unk_token,
            self.sos_token,
            self.eos_token,
        ]

        return [token for token in all_special_tokens if token is not None]

    @property
    def pad_idx(self):
        assert self.pad_token in self.token_to_idx, "PAD Token not found in vocabulary"
        return self.token_to_idx[self.pad_token]

    @property
    def unk_idx(self):
        assert self.unk_token in self.token_to_idx, "UNK Token not found in vocabulary"
        return self.token_to_idx[self.unk_token]

    @property
    def eos_idx(self):
        assert (
            self.eos_token is not None and self.eos_token in self.token_to_idx
        ), "EOS token is none or not found in vocabulary"
        return self.token_to_idx[self.eos_token]

    @property
    def sos_idx(self):
        assert (
            self.sos_token is not None and self.sos_token in self.token_to_idx
        ), "SOS token is none or not found in vocabulary"
        return self.token_to_idx[self.sos_token]

    def __len__(self):
        return len(self.token_to_idx)

    def encode(
        self, tokens: List[str], pad=True, add_eos_sos=False, max_len=None, add_unk=True
    ) -> List[int]:
        truncate_for_boundary = self.max_len is not None or max_len is not None
        max_len = first_not_none((max_len, self.max_len, len(tokens)))
        tokens = tokens[:max_len]
        if add_unk:
            encoded = [
                self.token_to_idx[token] if token in self.token_to_idx else self.unk_idx
                for token in tokens
            ]
        else:
            encoded = [self.token_to_idx[token] for token in tokens]

        if add_eos_sos:
            if truncate_for_boundary:
                encoded = encoded[: max_len - 2]

            encoded.insert(0, self.sos_idx)
            encoded.append(self.eos_idx)

        if pad and len(encoded) < max_len:
            encoded = encoded + (max_len - len(encoded)) * [self.pad_idx]
            assert (
                len(encoded) == max_len
            ), f"Expected len(encoded) == max_len({max_len}), but it was {len(encoded)}"
        return encoded

    def decode(self, tokens: List[int], remove_padding=True) -> List[str]:
        decoded = []
        max_index = len(tokens)
        if remove_padding:
            try:
                max_index = tokens.index(self.eos_idx) + 1
            except Exception as e:
                pass

        for token_idx in tokens[:max_index]:
            decoded.append(self.idx_to_token.get(token_idx, self.unk_token))
        return decoded

    def decode_token(self, index: int) -> str:
        return self.idx_to_token.get(index, self.unk_token)

    def encode_token(self, token: str) -> int:
        assert (
            token in self.token_to_idx
        ), "Tried to encode a token that is not in the vocabulary"
        return self.token_to_idx[token]


def mappings_from_iterable(
    tokens_per_example: Iterable[Iterable[str]],
    special_tokens: List[str] = [],
    size: Optional[int] = None,
    start_index: int = 0,
    uncase: bool = True,
) -> Tuple[Dict[str, int], Dict[int, str]]:
    if size:
        assert size - len(special_tokens) > 0

    counter: Counter[str] = Counter({})

    counter.update(
        [
            token.lower() if uncase else token
            for token in itertools.chain(*tokens_per_example)
        ]
    )

    counter_items = (
        counter.most_common(size - len(special_tokens))
        if size is not None
        else list(counter.items())
    )
    vocab_tokens: List[str] = [token for token, _ in counter_items]
    token_to_idx: Dict[str, int] = {}
    idx_to_token: Dict[int, str] = {}

    for token in itertools.chain(special_tokens, vocab_tokens):
        idx = len(token_to_idx) + start_index
        token_to_idx[token] = idx
        idx_to_token[idx] = token

    assert None not in token_to_idx, "None found in vocabulary, something is wrong"

    return token_to_idx, idx_to_token
