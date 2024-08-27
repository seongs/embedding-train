import logging
from torch import Tensor, FloatTensor
from datasets import Dataset

def _setup_logger():
    log_format = logging.Formatter("[%(asctime)s %(levelname)s] %(message)s")
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(log_format)
    logger.handlers = [console_handler]

    return logger

def change2e5format(data: Dataset):
    queries = []
    positives = []
    for entry in data:
        queries.append(f"query: {entry['query']}")
        positives.append(f"passage: {entry['answer']}")

    formatted_data = Dataset.from_dict({
        "anchor": queries,
        "positive": positives,
    })
    return formatted_data


def average_pool(last_hidden_states: Tensor, attention_mask: Tensor) -> FloatTensor:
    last_hidden_states = last_hidden_states.masked_fill(
        ~attention_mask[..., None].bool(), 0.0
    )
    return last_hidden_states.sum(dim=1) / attention_mask.sum(dim=1)[..., None]


def has_length(dataset):
    """
    Checks if the dataset implements __len__() and it doesn't raise an error
    """
    try:
        return len(dataset) is not None
    except TypeError:
        # TypeError: len() of unsized object
        return False
