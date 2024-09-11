from dataclasses import dataclass
from typing import Any, Callable, Dict, List, NewType, Optional, Union

import numpy as np
from transformers.data.data_collator import DataCollatorMixin
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers.utils import PaddingStrategy

InputDataClass = NewType("InputDataClass", Any)
DataCollator = NewType("DataCollator", Callable[[List[InputDataClass]], Dict[str, Any]])


def pad_without_fast_tokenizer_warning(tokenizer, *pad_args, **pad_kwargs):
    """
    Pads without triggering the warning about how using the pad function is sub-optimal when using a fast tokenizer.
    """

    # To avoid errors when using Feature extractors
    if not hasattr(tokenizer, "deprecation_warnings"):
        return tokenizer.pad(*pad_args, **pad_kwargs)

    # Save the state of the warning, then disable it
    warning_state = tokenizer.deprecation_warnings.get(
        "Asking-to-pad-a-fast-tokenizer", False
    )
    tokenizer.deprecation_warnings["Asking-to-pad-a-fast-tokenizer"] = True

    try:
        padded = tokenizer.pad(*pad_args, **pad_kwargs)
    finally:
        # Restore the state of the warning.
        tokenizer.deprecation_warnings["Asking-to-pad-a-fast-tokenizer"] = warning_state

    return padded


@dataclass
class DataCollatorForKoE5(DataCollatorMixin):
    """
    Data collator that will dynamically pad the inputs received, as well as the labels.

    Args:
        tokenizer ([`PreTrainedTokenizer`] or [`PreTrainedTokenizerFast`]):
            The tokenizer used for encoding the data.
        padding (`bool`, `str` or [`~utils.PaddingStrategy`], *optional*, defaults to `True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
            among:

            - `True` or `'longest'` (default): Pad to the longest sequence in the batch (or no padding if only a single
              sequence is provided).
            - `'max_length'`: Pad to a maximum length specified with the argument `max_length` or to the maximum
              acceptable input length for the model if that argument is not provided.
            - `False` or `'do_not_pad'`: No padding (i.e., can output a batch with sequences of different lengths).
        max_length (`int`, *optional*):
            Maximum length of the returned list and optionally padding length (see above).
        pad_to_multiple_of (`int`, *optional*):
            If set will pad the sequence to a multiple of the provided value.

            This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >=
            7.5 (Volta).
        label_pad_token_id (`int`, *optional*, defaults to -100):
            The id to use when padding the labels (-100 will be automatically ignore by PyTorch loss functions).
        return_tensors (`str`, *optional*, defaults to `"pt"`):
            The type of Tensor to return. Allowable values are "np", "pt" and "tf".
    """

    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    label_pad_token_id: int = -100
    return_tensors: str = "pt"

    def torch_call(self, features):
        import torch

        label_name = "label" if "label" in features[0].keys() else "labels"
        labels = (
            [feature[label_name] for feature in features]
            if label_name in features[0].keys()
            else None
        )

        # 정규 형식 (input_ids, attention_mask 로 처리한 뒤 다시 prefix (query, document, hard_negative) 붙여줌
        cat_batch = {}
        for k in ["query", "document", "hard_negative"]:
            batch_features = [
                {
                    "input_ids": feature[f"{k}_input_ids"],
                    "attention_mask": feature[f"{k}_attention_mask"],
                }
                for feature in features
                if f"{k}_input_ids" in feature and f"{k}_attention_mask" in feature
            ]

            if batch_features:
                batch = pad_without_fast_tokenizer_warning(
                    self.tokenizer,
                    batch_features,
                    padding=self.padding,
                    max_length=self.max_length,
                    pad_to_multiple_of=self.pad_to_multiple_of,
                    return_tensors="pt",
                )

                cat_batch[f"{k}_input_ids"] = batch["input_ids"]
                cat_batch[f"{k}_attention_mask"] = batch["attention_mask"]

        if labels is None:
            return cat_batch
