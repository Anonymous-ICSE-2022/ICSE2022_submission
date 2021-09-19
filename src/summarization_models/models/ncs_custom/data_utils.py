import itertools
import json
import os
from collections import Counter
from functools import partial
from math import ceil
from operator import itemgetter
from typing import Dict, Iterable, List, Literal, Optional, Tuple

import pandas as pd

from .TransformerDataModule import TransformerDataModule
from .vocabulary.Vocabulary import Vocabulary


def process_tokens(tokens, uncase: bool = True):
    return [token.lower() if uncase else token for token in tokens]


def load_tokens_parquet(filepath: str, uncase: bool = True):
    entries = pd.read_parquet(filepath)

    non_empty_comments = entries.comment_tokens.str.len() > 0
    non_empty_code = entries.code_tokens.str.len() > 0

    entries = entries[non_empty_code & non_empty_comments]

    return (
        [
            process_tokens(list(tokens), uncase=uncase)
            for tokens in entries["code_tokens"].tolist()
        ],
        [
            process_tokens(list(tokens), uncase=uncase)
            for tokens in entries["comment_tokens"].tolist()
        ],
    )


def jsonl(filename, map_fn=lambda x: x):
    with open(filename) as f:
        for line in f:
            yield map_fn(json.loads(line))


def prepare_vocabularies(
    base_dir: str,
    train_file: str = "train.json",
    test_file: str = "test.json",
    valid_file: str = "valid.json",
    comment_vocab_size=30000,
    code_vocab_size=50000,
    code_max_len=150,
    comment_max_len=50,
    type="parquet",
    uncase=True,
    num_workers=4,
):
    train_file = os.path.join(base_dir, train_file)
    test_file = os.path.join(base_dir, test_file)
    valid_file = os.path.join(base_dir, valid_file)

    print(f"=> Generating code vocabulary")
    from tqdm import tqdm

    code_token_iterator = partial(jsonl, map_fn=itemgetter("code_tokens"))
    bar = tqdm(
        itertools.chain(
            code_token_iterator(train_file), code_token_iterator(valid_file),
        ),
    )
    code_vocabulary = Vocabulary.from_tokens(
        bar,
        size=code_vocab_size,
        max_len=code_max_len,
        boundary_tokens=None,
        uncase=uncase,
    )
    comment_token_iterator = partial(jsonl, map_fn=itemgetter("comment_tokens"))
    print(f"=> Generating comment vocabulary")
    bar = tqdm(
        itertools.chain(
            comment_token_iterator(train_file), comment_token_iterator(valid_file)
        )
    )
    comment_vocabulary = Vocabulary.from_tokens(
        bar, size=comment_vocab_size, max_len=comment_max_len, uncase=uncase,
    )

    return code_vocabulary, comment_vocabulary


def prepare_data(
    base_dir: str,
    *,
    cache_dir: str,
    train_file: str = "train.jsonl.gz",
    test_file: str = "test.jsonl.gz",
    valid_file: str = "valid.jsonl.gz",
    comment_vocab_file: str = "comment.vocab.yaml",
    code_vocab_file: str = "code.vocab.yaml",
    comment_vocab_size=5000,
    code_vocab_size=5000,
    code_max_len=150,
    comment_max_len=50,
    batch_size=8,
    uncase=True,
    num_workers=4,
    use_cache=False,
):

    _train_file = os.path.join(base_dir, train_file)
    _test_file = os.path.join(base_dir, test_file)
    _valid_file = os.path.join(base_dir, valid_file)
    _code_vocab_file = os.path.join(base_dir, code_vocab_file)
    _comment_vocab_file = os.path.join(base_dir, comment_vocab_file)

    if os.path.exists(_comment_vocab_file) and os.path.exists(_code_vocab_file):
        comment_vocabulary = Vocabulary.load(_comment_vocab_file)
        code_vocabulary = Vocabulary.load(_code_vocab_file)
    else:
        print("=> Vocabulary files not found, preparing....")
        code_vocabulary, comment_vocabulary = prepare_vocabularies(
            base_dir=base_dir,
            train_file=train_file,
            valid_file=valid_file,
            code_vocab_size=code_vocab_size,
            comment_vocab_size=comment_vocab_size,
            code_max_len=code_max_len,
            comment_max_len=comment_max_len,
            uncase=uncase,
        )
        comment_vocabulary.save(os.path.join(base_dir, "comment.vocab.yaml"))
        code_vocabulary.save(os.path.join(base_dir, "code.vocab.yaml"))

    assert (
        len(comment_vocabulary) <= comment_vocab_size
    ), f"Expected {len(comment_vocabulary)} < {comment_vocab_size}"
    assert (
        len(code_vocabulary) <= code_vocab_size
    ), f"Expected {len(code_vocabulary)} < {code_vocab_size}"

    data_module = TransformerDataModule(
        train_path=_train_file,
        valid_path=_valid_file,
        test_path=_test_file,
        code_vocabulary=code_vocabulary,
        comment_vocabulary=comment_vocabulary,
        batch_size=batch_size,
        num_workers=num_workers,
        cache_dir=cache_dir,
        use_arrow=use_cache,
    )

    return data_module, code_vocabulary, comment_vocabulary


import pyarrow as pa
import pyarrow.parquet as pq


def parquet_generator(filepath: str, column_name: str = ""):
    mmap = pa.memory_map(filepath)
    table = pq.read_table(mmap).select([column_name])
    batches = table.to_batches()
    for batch in batches:
        entries = batch.to_pydict()[column_name]

        for entry in entries:
            yield entry


def get_file_path(
    base_dir: str, split="train", file_type: Literal["code", "comment"] = "code"
):
    assert file_type in ["code", "comment"]
    filename = "code.original_subtoken" if file_type == "code" else "javadoc.original"

    return os.path.join(base_dir, split, filename)


def load_tokens_from_files(
    base_dir: str, max_n=-1,
):
    raw_train_code_tokens, raw_train_comment_tokens = map(
        lambda file_type: load_file(
            get_file_path(base_dir=base_dir, split="train", file_type=file_type)
        ),
        ("code", "comment"),
    )
    train_code_tokens: List[List[str]] = []
    train_comment_tokens: List[List[str]] = []

    for i in range(len(raw_train_code_tokens)):
        if len(raw_train_code_tokens[i]) > 0 and len(raw_train_comment_tokens[i]) > 0:
            train_code_tokens.append(raw_train_code_tokens[i])
            train_comment_tokens.append(raw_train_comment_tokens[i])

    if max_n >= 0:
        train_code_tokens = train_code_tokens[:max_n]
        train_comment_tokens = train_comment_tokens[:max_n]

    raw_valid_code_tokens, raw_valid_comment_tokens = map(
        lambda file_type: load_file(
            get_file_path(base_dir=base_dir, split="dev", file_type=file_type)
        ),
        ("code", "comment"),
    )

    valid_code_tokens: List[List[str]] = []
    valid_comment_tokens: List[List[str]] = []

    for i in range(len(raw_valid_code_tokens)):
        if len(raw_valid_code_tokens[i]) > 0 and len(raw_valid_comment_tokens[i]) > 0:
            valid_code_tokens.append(raw_valid_code_tokens[i])
            valid_comment_tokens.append(raw_valid_comment_tokens[i])

    raw_test_code_tokens, raw_test_comment_tokens = map(
        lambda file_type: load_file(
            get_file_path(base_dir=base_dir, split="test", file_type=file_type)
        ),
        ("code", "comment"),
    )

    test_code_tokens: List[List[str]] = []
    test_comment_tokens: List[List[str]] = []

    for i in range(len(raw_test_code_tokens)):
        if len(raw_test_code_tokens[i]) > 0 and len(raw_test_comment_tokens[i]) > 0:
            test_code_tokens.append(raw_test_code_tokens[i])
            test_comment_tokens.append(raw_test_comment_tokens[i])

    return (
        train_code_tokens,
        train_comment_tokens,
        valid_code_tokens,
        valid_comment_tokens,
        test_code_tokens,
        test_comment_tokens,
    )


def load_file(filepath: str, uncase=True):
    data = []
    with open(filepath) as f:
        for line in f:
            tokens = line.split()
            tokens = [token.lower() if uncase else token for token in tokens]
            data.append(tokens)

    return data

