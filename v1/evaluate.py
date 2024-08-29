# pip install git+https://github.com/taeminlee/mteb.git@ontheit 후 사용
# streamlit run leaderboard.py 로 결과 확인

"""Example script for benchmarking all datasets constituting the MTEB Korean leaderboard & average scores"""
from __future__ import annotations

import os
import logging

from sentence_transformers import SentenceTransformer
from mteb import MTEB

logging.basicConfig(level=logging.INFO)

logger = logging.getLogger("main")

TASK_LIST_CLASSIFICATION = []

TASK_LIST_CLUSTERING = []

TASK_LIST_PAIR_CLASSIFICATION = []

TASK_LIST_RERANKING = []

# TASK_LIST_RETRIEVAL = ["Ko-StrategyQA", "Ko-mrtydi", "Ko-miracl"]
# TASK_LIST_RETRIEVAL = ["Ko-StrategyQA"]
TASK_LIST_RETRIEVAL = ["Ko-StrategyQA", "OntheITBM1-filtered-split", "OntheITBM2-filtered-split"]

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
    return [os.path.join(directory, d) for d in os.listdir(directory) if os.path.isdir(os.path.join(directory, d))]

directory = '/data/yjoonjang/KUKE'

# model_names = get_subdirectories(directory)
model_names = ["/data/yjoonjang/kuke-pt-result"]

model_names =  model_names
print(model_names)

for model_name in model_names:
    try:
        if not os.path.exists(model_name):
            model = SentenceTransformer(model_name, trust_remote_code=True)
        else:
            file_name = os.path.join(model_name, 'model.safetensors')
            if os.path.exists(file_name):
                model = SentenceTransformer(model_name)

        for task in TASK_LIST:
            logger.info(f"Running task: {task}")
            evaluation = MTEB(
                tasks=[task],
                task_langs=["ko"],
            )
            evaluation.run(model, output_folder=f"/data/ONTHEIT/results/{model_name}", encode_kwargs={"batch_size": 64})
    except Exception as ex:
        print(ex)
