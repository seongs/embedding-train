# KoE5: 한국어 특화 임베딩 모델

KoE5는 한국어에 특화된 자연어 처리를 위한 임베딩 모델입니다. 이 모델은 [multilingual-e5-large](https://huggingface.co/intfloat/multilingual-e5-large)를 기반으로 하여, 국립국어원 및 AIHub 등에서 제공하는 대규모 한국어 데이터셋을 사용하여 특화 학습을 거쳤습니다.

## 사용

- TBD

## 벤치마크

- mteb 실험 결과

- 벤치마크 사용법

```python
# pip install sentence_transformers
# pip install mteb
from sentence_transformers import SentenceTransformer
from mteb import MTEB

model_name = "/data/ONTHEIT/MODELS/KoE5-base-epoch=3/"
model = SentenceTransformer(model_name)

evaluation = MTEB(tasks=["Ko-StrategyQA"],task_langs=["ko"])
evaluation.run(model, output_folder=f"results/{model_name}")
```

## 학습

모델을 파인 튜닝하고 학습시키기 위한 단계는 다음과 같습니다:

1. 의존성 패키지를 설치 합니다.
```bash
pip install -r requirements.txt
```
2. `scripts/finetune.sh` 스크립트를 수정하여 학습 파라미터를 설정합니다.
3. 설정한 파라미터에 따라 스크립트를 실행하여 모델을 학습시킵니다.

### 학습 데이터

- 학습 데이터 구조에 대한 설명
- 학습 데이터 위치 설정 방법
