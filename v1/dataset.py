import os
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Union

import torch
from filelock import FileLock
from torch.utils.data import Dataset

from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers.utils import logging
from transformers.data.processors.utils import InputFeatures
from processor import KoE5MRCProcessor, convert_examples_to_features


logger = logging.get_logger(__name__)


@dataclass
class KoE5DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.

    Using `HfArgumentParser` we can turn this class into argparse arguments to be able to specify them on the command
    line.
    """

    data_dir: str = field(
        metadata={
            "help": "The input data dir. Should contain the .tsv files (or other data files) for the task."
        }
    )
    max_seq_length: int = field(
        default=128,
        metadata={
            "help": (
                "The maximum total input sequence length after tokenization. Sequences longer "
                "than this will be truncated, sequences shorter will be padded."
            )
        },
    )
    overwrite_cache: bool = field(
        default=False,
        metadata={"help": "Overwrite the cached training and evaluation sets"},
    )

    def __post_init__(self):
        self.task_name = self.task_name.lower()


class Split(Enum):
    train = "train"
    dev = "dev"
    test = "test"


class KoE5Dataset(Dataset):
    args: KoE5DataTrainingArguments
    features: List[InputFeatures]

    def __init__(
        self,
        args: KoE5DataTrainingArguments,
        tokenizer: PreTrainedTokenizerBase,
        limit_length: Optional[int] = None,
        mode: Union[str, Split] = Split.train,
        cache_dir: Optional[str] = None,
        test: Optional[bool] = False,
    ):
        self.args = args
        self.processor = KoE5MRCProcessor()
        if isinstance(mode, str):
            try:
                mode = Split[mode]
            except KeyError:
                raise KeyError("mode is not a valid split name")

        # Load data features from cache or dataset file
        cached_features_file = os.path.join(
            cache_dir if cache_dir is not None else args.data_dir,
            f"cached_{mode.value}_{tokenizer.__class__.__name__}_{args.max_seq_length}",
        )

        # Make sure only the first process in distributed training processes the dataset,
        # and the others will use the cache.
        lock_path = cached_features_file + ".lock"

        logger.debug(f"cache file exits: {os.path.exists(cached_features_file)}")

        with FileLock(lock_path):
            if os.path.exists(cached_features_file) and not args.overwrite_cache:
                logger.info("Start loading features...")
                start = time.time()
                self.features = torch.load(cached_features_file)
                logger.info(
                    f"Loading features from cached file {cached_features_file} [took %.3f s]",
                    time.time() - start,
                )

            else:
                lock_path = cached_features_file + ".lock"
                with FileLock(lock_path):
                    logger.info(
                        f"No cache files. Creating features from dataset file at {args.data_dir}"
                    )
                    if mode == Split.dev:
                        examples = self.processor.get_dev_examples(args.data_dir)
                    elif mode == Split.test:
                        examples = self.processor.get_test_examples(args.data_dir)
                    else:
                        examples = self.processor.get_train_examples(args.data_dir)

                    if limit_length is not None:
                        logger.info(f"Using limit length: {limit_length}")
                        examples = examples[:limit_length]

                    if test:
                        examples = examples[:100]
                        logger.info("Test mode activated: Got 100 examples!")

                    self.features = convert_examples_to_features(
                        examples, tokenizer, args.max_seq_length
                    )
                    logger.info("Converted examples to features!")

                    start = time.time()
                    torch.save(self.features, cached_features_file)
                    logger.info(
                        f"Saving features into cached file {cached_features_file} [took {time.time() - start:.3f} s]"
                    )

    def __len__(self):
        return len(self.features)

    def __getitem__(self, i) -> InputFeatures:
        return self.features[i]

    def get_labels(self):
        return [1, 0]
