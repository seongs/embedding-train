from __future__ import annotations

from contextlib import nullcontext
from functools import partial
from typing import Any, Iterable, Iterator

import torch
import tqdm
from torch import Tensor, nn
from torch.utils.checkpoint import get_device_states, set_device_states

from sentence_transformers import SentenceTransformer, util


class RandContext:
    """
    Random-state context manager class. Reference: https://github.com/luyug/GradCache.

    This class will back up the pytorch's random state during initialization. Then when the context is activated,
    the class will set up the random state with the backed-up one.
    """

    def __init__(self, *tensors) -> None:
        self.fwd_cpu_state = torch.get_rng_state()
        self.fwd_gpu_devices, self.fwd_gpu_states = get_device_states(*tensors)

    def __enter__(self) -> None:
        self._fork = torch.random.fork_rng(devices=self.fwd_gpu_devices, enabled=True)
        self._fork.__enter__()
        torch.set_rng_state(self.fwd_cpu_state)
        set_device_states(self.fwd_gpu_devices, self.fwd_gpu_states)

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self._fork.__exit__(exc_type, exc_val, exc_tb)
        self._fork = None


def _backward_hook(
    grad_output: Tensor,
    sentence_features: Iterable[dict[str, Tensor]],
    loss_obj: CachedMultipleNegativesSymmetricRankingLoss,
) -> None:
    """A backward hook to backpropagate the cached gradients mini-batch by mini-batch."""
    assert loss_obj.cache is not None
    assert loss_obj.random_states is not None
    with torch.enable_grad():
        for sentence_feature, grad, random_states in zip(
            sentence_features, loss_obj.cache, loss_obj.random_states
        ):
            for (reps_mb, _), grad_mb in zip(
                loss_obj.embed_minibatch_iter(
                    sentence_feature=sentence_feature,
                    with_grad=True,
                    copy_random_state=False,
                    random_states=random_states,
                ),
                grad,
            ):
                surrogate = (
                    torch.dot(reps_mb.flatten(), grad_mb.flatten()) * grad_output
                )
                surrogate.backward()


class CachedMultipleNegativesSymmetricRankingLoss(nn.Module):
    def __init__(
        self,
        model: SentenceTransformer,
        scale: float = 20.0,
        similarity_fct: callable[[Tensor, Tensor], Tensor] = util.cos_sim,
        mini_batch_size: int = 32,
        show_progress_bar: bool = False,
    ) -> None:
        """
        Cached version of MultipleNegativesSymmetricRankingLoss with GradCache optimization.

        This loss is an adaptation of MultipleNegativesRankingLoss and CachedMultipleNegativesRankingLoss.
        It computes the following loss:

        For a given anchor and a list of candidates, find the positive candidate (symmetric version).

        In CachedMultipleNegativesSymmetricRankingLoss, we add another loss term: Given the positive and a list of all anchors,
        find the correct (matching) anchor. This creates a symmetric loss function.

        The caching mechanism allows for processing of larger batch sizes without increasing memory usage,
        inspired by the GradCache technique (https://arxiv.org/pdf/2101.06983.pdf).

        For the example of question-answering: You have (question, answer)-pairs. This loss computes
        the loss to find the answer for a given question AND the loss to find the question for a given answer.

        Note: If you pass triplets, the negative entry will be ignored. An anchor is just searched for the positive.

        Args:
            model: SentenceTransformer model
            scale: Output of similarity function is multiplied by scale value
            similarity_fct: similarity function between sentence embeddings. By default, cos_sim.
                            Can also be set to dot product (and then set scale to 1)
            mini_batch_size: Mini-batch size for the forward pass, this denotes how much memory is actually used during
                training and evaluation. The larger the mini-batch size, the more memory efficient the training is, but
                the slower the training will be.
            show_progress_bar: If True, shows progress bar during processing

        Requirements:
            1. (anchor, positive) pairs
            2. Should be used with large batch sizes for superior performance, but has slower training time than non-cached versions

        Relations:
            - Like :class:`MultipleNegativesRankingLoss`, but with an additional symmetric loss term and caching mechanism.
            - Inspired by :class:`CachedMultipleNegativesRankingLoss`, adapted for symmetric loss calculation.

        Inputs:
            +---------------------------------------+--------+
            | Texts                                 | Labels |
            +=======================================+========+
            | (anchor, positive) pairs              | none   |
            +---------------------------------------+--------+

        Example:
            ::

                from sentence_transformers import SentenceTransformer, SentenceTransformerTrainer, losses
                from datasets import Dataset

                model = SentenceTransformer("microsoft/mpnet-base")
                train_dataset = Dataset.from_dict({
                    "anchor": ["It's nice weather outside today.", "He drove to work."],
                    "positive": ["It's so sunny.", "He took the car to the office."],
                })
                loss = losses.CachedMultipleNegativesSymmetricRankingLoss(model, mini_batch_size=32)

                trainer = SentenceTransformerTrainer(
                    model=model,
                    train_dataset=train_dataset,
                    loss=loss,
                )
                trainer.train()

        References:
            - Efficient Natural Language Response Suggestion for Smart Reply, Section 4.4: https://arxiv.org/pdf/1705.00652.pdf
            - Scaling Deep Contrastive Learning Batch Size under Memory Limited Setup: https://arxiv.org/pdf/2101.06983.pdf
        """
        super().__init__()
        self.model = model
        self.scale = scale
        self.similarity_fct = similarity_fct
        self.cross_entropy_loss = nn.CrossEntropyLoss()
        self.mini_batch_size = mini_batch_size
        self.cache: list[list[Tensor]] | None = None
        self.random_states: list[list[RandContext]] | None = None
        self.show_progress_bar = show_progress_bar

    def embed_minibatch(
        self,
        sentence_feature: dict[str, Tensor],
        begin: int,
        end: int,
        with_grad: bool,
        copy_random_state: bool,
        random_state: RandContext | None = None,
    ) -> tuple[Tensor, RandContext | None]:
        """Embed a mini-batch of sentences."""
        grad_context = nullcontext if with_grad else torch.no_grad
        random_state_context = nullcontext() if random_state is None else random_state
        sentence_feature_minibatch = {
            k: v[begin:end] for k, v in sentence_feature.items()
        }
        with random_state_context:
            with grad_context():
                random_state = (
                    RandContext(*sentence_feature_minibatch.values())
                    if copy_random_state
                    else None
                )
                reps = self.model(sentence_feature_minibatch)["sentence_embedding"]
        return reps, random_state

    def embed_minibatch_iter(
        self,
        sentence_feature: dict[str, Tensor],
        with_grad: bool,
        copy_random_state: bool,
        random_states: list[RandContext] | None = None,
    ) -> Iterator[tuple[Tensor, RandContext | None]]:
        """Iterate over mini-batches of sentences for embedding."""
        input_ids: Tensor = sentence_feature["input_ids"]
        bsz, _ = input_ids.shape
        for i, b in enumerate(
            tqdm.trange(
                0,
                bsz,
                self.mini_batch_size,
                desc="Embed mini-batches",
                disable=not self.show_progress_bar,
            )
        ):
            e = b + self.mini_batch_size
            reps, random_state = self.embed_minibatch(
                sentence_feature=sentence_feature,
                begin=b,
                end=e,
                with_grad=with_grad,
                copy_random_state=copy_random_state,
                random_state=None if random_states is None else random_states[i],
            )
            yield reps, random_state

    def calculate_loss_and_cache_gradients(self, reps: list[list[Tensor]]) -> Tensor:
        """Calculate the symmetric loss and cache gradients."""
        embeddings_a = torch.cat(reps[0])  # (bsz, hdim)
        embeddings_b = torch.cat(
            [torch.cat(r) for r in reps[1:]]
        )  # ((1 + nneg) * bsz, hdim)

        batch_size = len(embeddings_a)
        labels = torch.arange(batch_size, device=embeddings_a.device)

        losses: list[torch.Tensor] = []
        for b in tqdm.trange(
            0,
            batch_size,
            self.mini_batch_size,
            desc="Preparing caches",
            disable=not self.show_progress_bar,
        ):
            e = b + self.mini_batch_size
            scores: Tensor = (
                self.similarity_fct(embeddings_a[b:e], embeddings_b) * self.scale
            )
            forward_loss: torch.Tensor = self.cross_entropy_loss(scores, labels[b:e])

            positive_scores = scores[:, b:e]
            backward_loss: torch.Tensor = self.cross_entropy_loss(
                positive_scores.t(), labels[: len(positive_scores)]
            )

            loss_mbatch = (forward_loss + backward_loss) / 2
            loss_mbatch.backward()
            losses.append(loss_mbatch.detach())

        loss = sum(losses) / len(losses)
        loss = loss.requires_grad_()

        self.cache = [[r.grad for r in rs] for rs in reps]

        return loss

    def calculate_loss(self, reps: list[list[Tensor]]) -> Tensor:
        """Calculate the symmetric loss without caching gradients (for evaluation)."""
        embeddings_a = torch.cat(reps[0])  # (bsz, hdim)
        embeddings_b = torch.cat(
            [torch.cat(r) for r in reps[1:]]
        )  # ((1 + nneg) * bsz, hdim)

        batch_size = len(embeddings_a)
        labels = torch.arange(batch_size, device=embeddings_a.device)

        losses: list[torch.Tensor] = []
        for b in tqdm.trange(
            0,
            batch_size,
            self.mini_batch_size,
            desc="Calculating loss",
            disable=not self.show_progress_bar,
        ):
            e = b + self.mini_batch_size
            scores: Tensor = (
                self.similarity_fct(embeddings_a[b:e], embeddings_b) * self.scale
            )
            forward_loss: torch.Tensor = self.cross_entropy_loss(scores, labels[b:e])

            positive_scores = scores[:, b:e]
            backward_loss: torch.Tensor = self.cross_entropy_loss(
                positive_scores.t(), labels[: len(positive_scores)]
            )

            loss_mbatch = (forward_loss + backward_loss) / 2
            losses.append(loss_mbatch)

        loss = sum(losses) / len(losses)
        return loss

    def forward(
        self, sentence_features: Iterable[dict[str, Tensor]], labels: Tensor
    ) -> Tensor:
        """Forward pass of the loss function."""
        reps = []
        self.random_states = []
        for sentence_feature in sentence_features:
            reps_mbs = []
            random_state_mbs = []
            for reps_mb, random_state in self.embed_minibatch_iter(
                sentence_feature=sentence_feature,
                with_grad=False,
                copy_random_state=True,
            ):
                reps_mbs.append(reps_mb.detach().requires_grad_())
                random_state_mbs.append(random_state)
            reps.append(reps_mbs)
            self.random_states.append(random_state_mbs)

        if torch.is_grad_enabled():
            loss = self.calculate_loss_and_cache_gradients(reps)
            loss.register_hook(
                partial(
                    _backward_hook, sentence_features=sentence_features, loss_obj=self
                )
            )
        else:
            loss = self.calculate_loss(reps)

        return loss

    def get_config_dict(self) -> dict[str, Any]:
        """Get the configuration of the loss function."""
        return {
            "scale": self.scale,
            "similarity_fct": self.similarity_fct.__name__,
            "mini_batch_size": self.mini_batch_size,
        }
