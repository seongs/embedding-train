from datasets import Dataset, DatasetDict
from huggingface_hub import HfApi

# 데이터 로드
data = Dataset.from_json("/data/ONTHEIT/DATA/data_without_ontheit/train.jsonl")

# Hugging Face Hub에 업로드
data.push_to_hub("nlpai-lab/ko-triplet-v1.0", private=False)  # private 옵션은 선택 사항