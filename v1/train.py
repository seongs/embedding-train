import os
import sys
import fire
import torch

from dataclasses import dataclass, field
from transformers import (
    AutoTokenizer,
    AutoModel,
    HfArgumentParser,
    TrainingArguments,
    set_seed,
)

from data_collator import DataCollatorForKoE5
from dataset import KoE5Dataset
from trainer import CustomTrainer

from transformers.trainer_utils import get_last_checkpoint
from utils import _setup_logger
from typing import Tuple, Dict, Optional
from setproctitle import setproctitle

torch.autograd.set_detect_anomaly(True)


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={
            "help": "Path to pretrained model or model identifier from huggingface.co/models"
        }
    )
    config_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "Pretrained config name or path if not the same as model_name"
        },
    )
    tokenizer_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "Pretrained tokenizer name or path if not the same as model_name"
        },
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={
            "help": "Where to store the pretrained models downloaded from huggingface.co"
        },
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={
            "help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."
        },
    )
    model_revision: str = field(
        default="main",
        metadata={
            "help": "The specific model version to use (can be a branch name, tag name or commit id)."
        },
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": (
                "Will use the token generated when running `transformers-cli login` (necessary to use this script "
                "with private models)."
            )
        },
    )
    resize_position_embeddings: Optional[bool] = field(
        default=None,
        metadata={
            "help": (
                "Whether to automatically resize the position embeddings if `max_source_length` exceeds "
                "the model's position embeddings."
            )
        },
    )
    init_checkpoint: Optional[str] = field(
        default=None,
        metadata={"help": "A model bin file."},
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    lang: str = field(default=None, metadata={"help": "Language id for summarization."})
    dataset_name: Optional[str] = field(
        default=None,
        metadata={"help": "The name of the dataset to use (via the datasets library)."},
    )
    data_dir: Optional[str] = field(
        default=None, metadata={"help": "directory to the processed data"}
    )
    dataset_config_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "The configuration name of the dataset to use (via the datasets library)."
        },
    )
    sample_selection_train_file_path: Optional[str] = field(
        default=None, metadata={"help": "sample_selection_train_file_path"}
    )
    text_column: Optional[str] = field(
        default=None,
        metadata={
            "help": "The name of the column in the datasets containing the full texts (for summarization)."
        },
    )
    summary_column: Optional[str] = field(
        default=None,
        metadata={
            "help": "The name of the column in the datasets containing the summaries (for summarization)."
        },
    )
    train_file: Optional[str] = field(
        default=None,
        metadata={"help": "The input training data file (a jsonlines or csv file)."},
    )
    validation_file: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "An optional input evaluation data file to evaluate the metrics (rouge) on (a jsonlines or csv file)."
            )
        },
    )
    test_file: Optional[str] = field(
        default=None,
        metadata={
            "help": "An optional input test data file to evaluate the metrics (rouge) on (a jsonlines or csv file)."
        },
    )
    overwrite_cache: bool = field(
        default=False,
        metadata={"help": "Overwrite the cached training and evaluation sets"},
    )
    def_only: bool = field(default=False, metadata={"help": "def_only"})
    add_prompt_to_document: bool = field(
        default=True, metadata={"help": "add_prompt_to_document"}
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    debug_mode: Optional[int] = field(
        default=None,
        metadata={"help": "debug mode"},
    )
    max_examples: Optional[int] = field(
        default=None,
        metadata={"help": "debug mode"},
    )
    cl_temperature: Optional[float] = field(
        default=None,
        metadata={"help": "temperature"},
    )
    max_seq_length: Optional[int] = field(
        default=512,
        metadata={
            "help": (
                "The maximum total input sequence length after tokenization. Sequences longer "
                "than this will be truncated, sequences shorter will be padded."
            )
        },
    )
    sub_sample_ratio: Optional[float] = field(
        default=2.0,
        metadata={"help": ("sub_sample_ratio")},
    )
    val_max_target_length: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "The maximum total sequence length for validation target text after tokenization. Sequences longer "
                "than this will be truncated, sequences shorter will be padded. Will default to `max_target_length`."
                "This argument is also used to override the ``max_length`` param of ``model.generate``, which is used "
                "during ``evaluate`` and ``predict``."
            )
        },
    )
    pad_to_max_length: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether to pad all samples to model maximum sentence length. "
                "If False, will pad the samples dynamically when batching to the maximum length in the batch. More "
                "efficient on GPU but very bad for TPU."
            )
        },
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of training examples to this "
                "value if set."
            )
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                "value if set."
            )
        },
    )
    max_predict_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of prediction examples to this "
                "value if set."
            )
        },
    )
    num_beams: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "Number of beams to use for evaluation. This argument will be passed to ``model.generate``, "
                "which is used during ``evaluate`` and ``predict``."
            )
        },
    )
    ignore_pad_token_for_loss: bool = field(
        default=True,
        metadata={
            "help": "Whether to ignore the tokens corresponding to padded labels in the loss computation or not."
        },
    )
    source_prefix: Optional[str] = field(
        default="",
        metadata={
            "help": "A prefix to add before every source text (useful for T5 models)."
        },
    )

    forced_bos_token: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "The token to force as the first generated token after the decoder_start_token_id."
                "Useful for multilingual models like mBART where the first generated token"
                "needs to be the target language token (Usually it is the target language token)"
            )
        },
    )


def evaluate(args, **kwargs) -> Tuple[Dict, Dict]:
    NotImplemented


def train(
    test: bool = False,
    fp16: bool = False,
    gradient_accumulation_steps: int = 4,
    per_device_train_batch_size: int = 128,
    per_device_eval_batch_size: int = 128,
    evaluation_strategy: str = "steps",
    eval_steps: int = 300,
    prediction_loss_only: bool = False,
):
    setproctitle("dew1701 KoE5")

    parser = HfArgumentParser(
        (ModelArguments, DataTrainingArguments, TrainingArguments)
    )
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args = parser.parse_json_file(
            json_file=os.path.abspath(sys.argv[1])
        )
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    real_name_or_path = model_args.model_name_or_path
    data_args.output_dir = training_args.output_dir
    cache_dir = model_args.cache_dir
    data_args.model_name_or_path = model_args.model_name_or_path
    data_args.tokenizer_name_or_path = model_args.model_name_or_path
    training_args.cl_temperature = data_args.cl_temperature
    training_args.remove_unused_columns = False
    training_args.report_to = ["wandb"]
    training_args.per_device_train_batch_size = per_device_train_batch_size
    training_args.per_device_eval_batch_size = per_device_eval_batch_size
    training_args.fp16 = fp16
    training_args.gradient_accumulation_steps = gradient_accumulation_steps
    # training_args.evaluation_strategy = evaluation_strategy
    # training_args.eval_steps = eval_steps
    training_args.prediction_loss_only = prediction_loss_only

    if not os.path.isdir(data_args.output_dir):
        os.makedirs(data_args.output_dir, exist_ok=True)

    # Setup Logging modules (Logging, Wandb)
    logger = _setup_logger()
    logger.info(f"Running in {'test' if test else 'normal'} mode")
    logger.info(f"Training arguments: {training_args}")
    logger.info(f"Data arguments: {data_args}")

    last_checkpoint = None
    if (
        os.path.isdir(training_args.output_dir)
        and training_args.do_train
        and not training_args.overwrite_output_dir
    ):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif (
            last_checkpoint is not None and training_args.resume_from_checkpoint is None
        ):
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    if test:
        cache_dir = os.path.join(cache_dir, "test")

    tokenizer = AutoTokenizer.from_pretrained(
        (
            model_args.tokenizer_name
            if model_args.tokenizer_name
            else model_args.model_name_or_path
        ),
        cache_dir=cache_dir,
        use_fast=model_args.use_fast_tokenizer,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )

    set_seed(training_args.seed)
    model = AutoModel.from_pretrained(real_name_or_path)

    logger.info("Loading train_dataset...")
    train_dataset = KoE5Dataset(
        args=data_args,
        tokenizer=tokenizer,
        limit_length=None,
        mode="train",
        cache_dir=cache_dir,
        test=test,
    )
    logger.info("Finished loading train dataset!")

    logger.info("Loading eval_dataset...")
    eval_dataset = KoE5Dataset(
        args=data_args, tokenizer=tokenizer, mode="dev", cache_dir=cache_dir, test=test
    )
    logger.info("Finished loading eval dataset!")

    data_collator = DataCollatorForKoE5(
        tokenizer=tokenizer,
        padding=True,
        max_length=None,
        pad_to_multiple_of=None,
        label_pad_token_id=-100,
        return_tensors="pt",
    )

    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    checkpoint = None
    if training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint
    elif last_checkpoint is not None:
        checkpoint = last_checkpoint

    if model_args.init_checkpoint is not None:
        print(f"Loading from {model_args.init_checkpoint} ...")
        state_dict = torch.load(
            os.path.join(model_args.init_checkpoint, "pytorch_model.bin")
        )
        model.load_state_dict(state_dict)

    train_result = trainer.train(resume_from_checkpoint=checkpoint)
    trainer.save_model(output_dir=training_args.output_dir)


if __name__ == "__main__":
    fire.Fire(train)
