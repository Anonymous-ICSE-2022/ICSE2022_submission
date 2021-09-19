# from tokenizers import Tokenizer
import torch
from configargparse import Namespace
from transformers import RobertaTokenizer


def compute_features(
    tokenizer: RobertaTokenizer,
    source,
    target,
    max_target_length,
    max_source_length,
    stage,
):
    source_tokens = tokenizer.tokenize(source)[: max_source_length - 2]
    source_tokens = [tokenizer.cls_token] + source_tokens + [tokenizer.sep_token]
    source_ids = tokenizer.convert_tokens_to_ids(source_tokens)
    source_mask = [1] * (len(source_tokens))
    padding_length = max_source_length - len(source_ids)
    source_ids += [tokenizer.pad_token_id] * padding_length
    source_mask += [0] * padding_length

    # target
    if stage == "test":
        target_tokens = tokenizer.tokenize(target)[: max_target_length - 2]
    else:
        target_tokens = tokenizer.tokenize(target)[: max_target_length - 2]
    target_tokens = [tokenizer.cls_token] + target_tokens + [tokenizer.sep_token]
    target_ids = tokenizer.convert_tokens_to_ids(target_tokens)
    target_mask = [1] * len(target_ids)
    padding_length = max_target_length - len(target_ids)
    target_ids += [tokenizer.pad_token_id] * padding_length
    target_mask += [0] * padding_length

    return source_ids, target_ids, source_mask, target_mask


class CodeBertTransforms:
    def __init__(
        self,
        tokenizer: RobertaTokenizer,
        max_source_length: int,
        max_target_length: int,
        stage: str,
    ):
        self.max_source_length = max_source_length
        self.tokenizer = tokenizer
        self.max_target_length = max_target_length
        self.stage = stage

    @staticmethod
    def from_args(args: Namespace, tokenizer: RobertaTokenizer):
        return CodeBertTransforms(
            tokenizer=tokenizer,
            max_source_length=args.max_source_length,
            max_target_length=args.max_target_length,
            stage=args.stage,
        )

    def __call__(self, df):
        # code_tokens = [[self.tokenizer.cls_token] + code_tokens + [self.tokenizer.sep_token] for code_tokens in df["code_tokens"].values]
        # comment_tokens = [[self.tokenizer.cls_token] + code_tokens + [self.tokenizer.sep_token] for code_tokens in df["comment_tokens"].values]

        # source_tokens = self.tokenizer.batch_encode_plus(code_tokens, max_length=self.max_source_length - 2)
        # target_tokens = self.tokenizer.batch_encode_plus(comment_tokens, max_length=self.max_target_length)

        all_source_ids = []
        all_target_ids = []
        all_source_masks = []
        all_target_masks = []
        for entry in df.itertuples():
            source_ids, target_ids, source_mask, target_mask = compute_features(
                self.tokenizer,
                " ".join(entry.code_tokens),
                " ".join(entry.comment_tokens),
                max_target_length=self.max_target_length,
                max_source_length=self.max_source_length,
                stage=self.stage,
            )

            all_source_ids.append(source_ids)
            all_target_ids.append(target_ids)
            all_source_masks.append(source_mask)
            all_target_masks.append(target_mask)

        return tuple(
            torch.tensor(item, dtype=torch.long)
            for item in (
                df["id"].values,
                all_source_ids[0],
                all_target_ids[0],
                all_source_masks[0],
                all_target_masks[0],
            )
        )

