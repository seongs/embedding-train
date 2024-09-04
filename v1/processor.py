import os
import json
import dataclasses
from dataclasses import dataclass
from typing import List, Optional

from transformers.utils import logging
from transformers.data.processors.utils import DataProcessor

from datasets import Dataset

logger = logging.get_logger(__name__)

DEPRECATION_WARNING = (
    "This {0} will be removed from the library soon, preprocessing should be handled with the 🤗 Datasets "
    "library. You can have a look at this example script for pointers: "
    "https://github.com/huggingface/transformers/blob/main/examples/pytorch/text-classification/run_glue.py"
)


@dataclass
class InputExample:
    query: str
    positive_passage: str
    negative_passage: str

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(dataclasses.asdict(self), indent=2) + "\n"


@dataclass
class E5InputExample(InputExample):
    def __init__(self, query: str, positive_passage: str, negative_passage: str):
        super().__init__(
            f"query: {query}",
            f"passage: {positive_passage}",
            f"passage: {negative_passage}",
        )


@dataclass(frozen=True)
class InputFeatures:
    # question
    input_ids: List[int]
    token_type_ids: List[int]
    attention_mask: Optional[List[int]] = None

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(dataclasses.asdict(self)) + "\n"


class KoE5MRCProcessor(DataProcessor):
    """Processor for the KoE5 data set."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def get_train_examples(self, data_dir):
        """See base class."""
        if "train.json" in os.listdir(data_dir):
            return self._create_examples(
                self._read_json(os.path.join(data_dir, "train.json")), "train"
            )
        else:
            return self._create_examples(
                self._read_jsonl(os.path.join(data_dir, "train.jsonl")), "train"
            )

    def get_dev_examples(self, data_dir):
        """See base class."""
        if "dev.json" in os.listdir(data_dir):
            return self._create_examples(
                self._read_json(os.path.join(data_dir, "dev.json")), "dev"
            )
        else:
            return self._create_examples(
                self._read_jsonl(os.path.join(data_dir, "dev.jsonl")), "dev"
            )

    def get_test_examples(self, data_dir):
        """See base class."""
        if "test.json" in os.listdir(data_dir):
            return self._create_examples(
                self._read_json(os.path.join(data_dir, "test.json")), "test"
            )
        else:
            return self._create_examples(
                self._read_jsonl(os.path.join(data_dir, "test.jsonl")), "test"
            )

    def get_labels(self):
        """All Classes"""
        return ["0", "1"]

    def _create_examples(self, datas, set_type):
        """Creates examples for the training, dev and test sets."""
        queries = []
        positives = []
        hard_negatives = []

        for i, data in enumerate(datas):
            if isinstance(data["query"], list):
                query = data["query"][0]
            else:
                query = data["query"]
            if isinstance(data["document"], list):
                document = data["document"][0]
            else:
                document = data["document"]
            if "hard_negative" in data:
                if isinstance(data["hard_negative"], list):
                    hard_negative = data["hard_negative"][0]
                else:
                    hard_negative = data["hard_negative"]
            else:
                hard_negative = None

            queries.append(f"query: {query}")
            positives.append(f"passage: {document}")
            hard_negatives.append(f"passage: {hard_negative}")

        data = Dataset.from_dict({
            "anchor": queries,
            "positive": positives,
            # "negative": hard_negatives
        })
        return data

    @classmethod
    def _read_json(cls, input_file):
        """Reads a tab separated value file."""
        with open(input_file, "r", encoding="utf-8-sig") as f:
            return json.load(f)

    def _read_jsonl(file_path, input_file):
        """Read a JSONL file and return a list of dictionaries."""
        data = []
        with open(input_file, "r", encoding="utf-8-sig") as file:
            for line in file:
                data.append(json.loads(line))
        return data
