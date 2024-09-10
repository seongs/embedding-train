# pip install git+https://github.com/taeminlee/mteb.git@ontheit 후 사용
# streamlit run leaderboard.py 로 결과 확인

"""Example script for benchmarking all datasets constituting the MTEB Korean leaderboard & average scores"""
from __future__ import annotations

import os
import logging

from sentence_transformers import SentenceTransformer
from transformers import AutoModel

from mteb import MTEB, get_model, get_tasks
from mteb.models.e5_models import E5Wrapper

logging.basicConfig(level=logging.INFO)

logger = logging.getLogger("main")

TASK_LIST_CLASSIFICATION = []

TASK_LIST_CLUSTERING = []

TASK_LIST_PAIR_CLASSIFICATION = []

TASK_LIST_RERANKING = []

# TASK_LIST_RETRIEVAL = ["Ko-StrategyQA", "Ko-mrtydi", "Ko-miracl"]
# TASK_LIST_RETRIEVAL = ["Ko-StrategyQA"]
TASK_LIST_RETRIEVAL = [
    "Ko-StrategyQA",
    # "OntheITBM1-filtered-split",
    # "OntheITBM2-filtered-split",
    "Markers_bm",
    # "MIRACLRetrieval",
    # "PublicHealthQA"
]

TASK_LIST_STS = []

TASK_LIST = (
    TASK_LIST_CLASSIFICATION
    + TASK_LIST_CLUSTERING
    + TASK_LIST_PAIR_CLASSIFICATION
    + TASK_LIST_RERANKING
    + TASK_LIST_RETRIEVAL
    + TASK_LIST_STS
)


def get_subdirectories(directory):
    return [
        os.path.join(directory, d)
        for d in os.listdir(directory)
        if os.path.isdir(os.path.join(directory, d))
    ]


directories = ["/data/ONTHEIT/MODELS/", "/data/yjoonjang/KUKE"]

model_names = sum([get_subdirectories(directory) for directory in directories], [])
# model_names = ['intfloat/multilingual-e5-base', 'intfloat/multilingual-e5-large', 'Alibaba-NLP/gte-multilingual-base', 'Alibaba-NLP/gte-multilingual-mlm-base'] + model_names
# model_names = ["/data/yjoonjang/KUKE/KUKE-ft-after-pt-bs=32768-ep=1-lr=1e-5-240902"]

model_names = [
    # "/data/yjoonjang/KUKE/KUKE-ft_with_openqp_pair-bs=32768-ep=1-lr=1e-5-240903"
    # '/data/yjoonjang/KUKE/KUKE-loss=CachedMultipleNegativesSymmetricRankingLoss-bs=64-ep=1-lr=2e-5-240827',
    # '/data/yjoonjang/KUKE/KUKE-ft-bs=512-ep=1-lr=2e-5-240826'
    "/data/yjoonjang/KUKE/KUKE-ft_with_openqp_pair_without_hn-loss=symmetric-bs=512-ep=2-lr=2e-5-240905",
    "/data/yjoonjang/KUKE/KUKE-ft_with_openqp_pair_without_hn-loss=symmetric-bs=2048-ep=2-lr=2e-5-240905",
    "/data/yjoonjang/KUKE/KUKE-ft_with_openqp_pair_without_hn-loss=symmetric-bs=4096-ep=2-lr=2e-5-240905",
]
print(model_names)

for model_name in model_names:
    try:
        model = None
        if not os.path.exists(model_name):
            model = get_model(model_name)
        else:
            file_name = os.path.join(model_name, "model.safetensors")
            if os.path.exists(file_name):
                model = E5Wrapper(model_name)

        if model:
            # logger.info(f"Running task: {task} / {model_name}")
            print(f"Running task: {TASK_LIST} / {model_name}")
            evaluation = MTEB(
                tasks=get_tasks(tasks=TASK_LIST, languages=["kor-Kore", "kor-Hang"])
            )
            batch_size = 512
            if "bge-m3" in model_name:
                batch_size = 4
            evaluation.run(
                model,
                output_folder=f"/data/ONTHEIT/results/{model_name}",
                encode_kwargs={"batch_size": batch_size},
            )
    except Exception as ex:
        print(ex)
