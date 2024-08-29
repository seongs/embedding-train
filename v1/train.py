import os
import fire

from sentence_transformers import SentenceTransformer, SentenceTransformerTrainer, losses
from sentence_transformers.training_args import SentenceTransformerTrainingArguments
from losses import CachedMultipleNegativesSymmetricRankingLoss

from datasets import load_dataset

from utils import _setup_logger, change2e5format, change2sentencetransformersformat
from setproctitle import setproctitle
from processor import KoE5MRCProcessor


def train(
    model_name_or_path: str = "intfloat/multilingual-e5-large",
    output_dir: str = "/data/ONTHEIT/MODELS",
    data_dir: str = "/data/ONTHEIT/DATA",
    num_epochs: int = 1,
    learning_rate: float = 2e-5,
    per_device_train_batch_size: int = 512,
    mini_batch_size: int = 32,
    per_device_eval_batch_size: int = 512,
    warmup_steps: int = 100,
    logging_steps: int = 2,
    use_hf_dataset: bool = False,
    max_seq_length: int = 512,
    save_steps: int = 100,
    cl_temperature: float = 0.02,
    test: bool = False,
    fp16: bool = True,
    gradient_accumulation_steps: int = 1,
    evaluation_strategy: str = "no",
    eval_steps: int = 100,
    prediction_loss_only: bool = False,
    use_wandb: bool = True,
    resume_from_checkpoint: bool = False,
):
    setproctitle("dew1701 KUKE")
    if int(os.environ.get("LOCAL_RANK", 0)) == 0:
        print(
            f"Training Embedding model with params:\n"
            f"base_model: {model_name_or_path}\n"
            f"{fp16=}\n"
            f"output_dir: {output_dir}\n"
            f"data_dir: {data_dir}\n"
            f"per_device_train_batch_size: {per_device_train_batch_size}\n"
            f"gradient_accumulation_steps: {gradient_accumulation_steps}\n"
            f"num_epochs: {num_epochs}\n"
            f"learning_rate: {learning_rate}\n"
            f"{use_wandb=}\n"
            f"resume_from_checkpoint: {resume_from_checkpoint or False}\n"
        )

    if not os.path.isdir(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    logger = _setup_logger()
    logger.info(f"Running in {'test' if test else 'normal'} mode")
    last_checkpoint = None

    model = SentenceTransformer(model_name_or_path, trust_remote_code=True)

    logger.info("Loading train_dataset...")
    if use_hf_dataset:
        dataset = load_dataset(data_dir)
        train_dataset = change2e5format(dataset["train"])
    else:
        processor = KoE5MRCProcessor()
        train_dataset = processor.get_train_examples(data_dir)
        print(train_dataset)
    logger.info("Finished loading train dataset!")

    # logger.info("Loading eval_dataset...")
    # if use_hf_dataset:
    #     dataset = load_dataset(data_dir)
    #     eval_dataset = change2e5format(dataset["valid"])
    # else:
    #     processor = KoE5MRCProcessor()
    #     eval_dataset = processor.get_dev_examples(data_dir)
    # logger.info("Finished loading eval dataset!")

    loss = losses.CachedMultipleNegativesRankingLoss(model=model, mini_batch_size=mini_batch_size)
    trainer = SentenceTransformerTrainer(
        model=model,
        args=SentenceTransformerTrainingArguments(
            per_device_train_batch_size=per_device_train_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            warmup_steps=warmup_steps,
            # warmup_ratio=warmup_ratio,
            num_train_epochs=num_epochs,
            learning_rate=learning_rate,
            logging_steps=logging_steps,
            # optim="adamw_torch",  # since we use DS optim?
            eval_strategy=evaluation_strategy,
            save_strategy="epoch",
            eval_steps=eval_steps,
            # save_steps=save_steps,
            output_dir=output_dir,
            report_to="wandb" if use_wandb else [],
            fp16=fp16,
            gradient_checkpointing=False
        ),
        train_dataset=train_dataset,
        eval_dataset=None,
        loss=loss
    )

    checkpoint = None
    if resume_from_checkpoint is not None:
        checkpoint = resume_from_checkpoint
    elif last_checkpoint is not None:
        checkpoint = last_checkpoint

    # if model_args.init_checkpoint is not None:
    #     print(f"Loading from {model_args.init_checkpoint} ...")
    #     state_dict = torch.load(
    #         os.path.join(model_args.init_checkpoint, "pytorch_model.bin")
    #     )
    #     model.load_state_dict(state_dict)

    trainer.train(resume_from_checkpoint=checkpoint)
    trainer.save_model(output_dir=output_dir)


if __name__ == "__main__":
    fire.Fire(train)
