import json
from dataclasses import dataclass
from pathlib import Path
from typing import List

import datasets

_DESCRIPTION = "TL-CodeSum Dataset"
_CITATION = 'Hu, Xing, et al. "Deep code comment generation with hybrid lexical and syntactical information." Empirical Software Engineering 25.3 (2020): 2179-2217.'

_URL = "https://storage.googleapis.com/summarization-datsets/codexglue-constants-masked.tar.gz"


@dataclass
class CodeXGLUEMasked(datasets.BuilderConfig):
    """BuilderConfig for CSV."""

    pass


DATASET_KEYS = ["id", "code", "code_tokens", "docstring", "docstring_tokens"]


class CodeXGLUEMasked(datasets.GeneratorBasedBuilder):
    BUILDER_CONFIG_CLASS = CodeXGLUEMasked
    BUILDER_CONFIGS = [CodeXGLUEMasked(name="default", description="default config")]
    DEFAULT_CONFIG_NAME = "default"

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(
                {
                    "id": datasets.Value("int32"),
                    "code": datasets.Value("string"),
                    "code_tokens": datasets.Sequence(datasets.Value("string")),
                    "docstring": datasets.Value("string"),
                    "docstring_tokens": datasets.Sequence(datasets.Value("string")),
                }
            ),
            # No default supervised_keys (as we have to pass both question
            # and context as input).
            supervised_keys=None,
            homepage="https://github.com/xing-hu/EMSE-DeepCom",
            citation=_CITATION,
        )

    def _split_generators(
        self, dl_manager: datasets.DownloadManager
    ) -> List[datasets.SplitGenerator]:
        downloaded_files = dl_manager.download_and_extract(_URL)
        downloaded_files = Path(downloaded_files)
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={"filepath": downloaded_files / "train.jsonl"},
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={"filepath": downloaded_files / "valid.jsonl"},
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={"filepath": downloaded_files / "test.jsonl"},
            ),
        ]

    def _generate_examples(self, filepath):
        with open(filepath) as f:
            for line in f:
                entry = json.loads(line)
                yield entry["id"], {k: v for k, v in entry.items() if k in DATASET_KEYS}
