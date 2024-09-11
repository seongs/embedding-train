import torch
from torch import nn
import torch.nn.functional as F

from torch.utils.data import SequentialSampler
from torch.utils.data.distributed import DistributedSampler

from transformers import Trainer
from transformers.modeling_outputs import BaseModelOutput

from utils import average_pool, has_length


torch.autograd.set_detect_anomaly(True)


class CustomTrainer(Trainer):
    def __init__(
        self, model, args, train_dataset, eval_dataset, tokenizer, data_collator
    ):
        super().__init__(
            model=model,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            data_collator=data_collator,
        )

    def _get_train_sampler(self):
        if self.train_dataset is None or not has_length(self.train_dataset):
            return None

        generator = None
        if self.args.world_size <= 1:
            generator = torch.Generator()
            if self.args.data_seed is None:
                seed = int(torch.empty((), dtype=torch.int64).random_().item())
            else:
                seed = self.args.data_seed
            generator.manual_seed(seed)

        seed = (
            self.args.data_seed if self.args.data_seed is not None else self.args.seed
        )

        if self.args.world_size <= 1:
            return SequentialSampler(self.train_dataset)
        else:
            return DistributedSampler(
                self.train_dataset,
                num_replicas=self.args.world_size,
                rank=self.args.process_index,
                seed=seed,
            )

    def compute_loss(self, model, inputs, return_outputs=False):
        embeddings = {}

        for k in ["query", "document", "hard_negative"]:
            input_ids = inputs[f"{k}_input_ids"]
            attention_mask = inputs[f"{k}_attention_mask"]

            input_dict = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "token_type_ids": torch.zeros_like(input_ids),
            }

            output: BaseModelOutput = model(**input_dict)
            pooled_output = average_pool(
                output.last_hidden_state, input_dict["attention_mask"]
            )
            embeddings[k] = F.normalize(pooled_output, p=2, dim=1)

        query_embeddings = embeddings["query"]
        positive_embeddings = embeddings["document"]
        negative_embeddings = embeddings["hard_negative"]

        similarity_fct = nn.CosineSimilarity(dim=-1)
        tau = self.args.cl_temperature

        positive_scores = similarity_fct(query_embeddings, positive_embeddings) / tau
        positive_negative_scores = (
            similarity_fct(positive_embeddings.unsqueeze(1), positive_embeddings) / tau
        )
        query_negative_scores = (
            similarity_fct(query_embeddings.unsqueeze(1), negative_embeddings) / tau
        )

        max_positive_scores = torch.max(positive_scores, dim=0, keepdim=True)[0]
        max_positive_negative_scores = torch.max(
            positive_negative_scores, dim=1, keepdim=True
        )[0]
        max_query_negative_scores = torch.max(
            query_negative_scores, dim=1, keepdim=True
        )[0]

        max_intermediate_scores = torch.max(
            max_positive_scores, max_positive_negative_scores
        )
        max_scores = torch.max(max_intermediate_scores, max_query_negative_scores)

        stable_positive_scores = positive_scores - max_scores
        stable_positive_negative_scores = (
            positive_negative_scores - max_scores.unsqueeze(1)
        )
        stable_query_negative_scores = query_negative_scores - max_scores.unsqueeze(1)

        exp_positive_scores = torch.exp(stable_positive_scores)  # 분자
        exp_negative_scores = torch.exp(
            (stable_positive_negative_scores + stable_query_negative_scores)
        )

        total_scores_sum = exp_positive_scores + exp_negative_scores.sum(dim=1)  # 분모
        log_prob = torch.log(exp_positive_scores / total_scores_sum)

        loss = -log_prob.mean()

        return loss