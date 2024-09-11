# embedding-train
**embedding-train** 프로젝트는 **multilingual-e5** 모델을 한국어 데이터로 fine-tuning하여 한국어에 특화된 임베딩 모델을 만드는 것을 목표로 합니다. 이를 통해 한국어 질의와 문서 간의 의미적 유사성을 잘 반영할 수 있는 고성능 임베딩 모델을 구축합니다.

## 프로젝트 버전
- `v1`: InfoNCE loss를 `huggingface Trainer`의 `compute_loss` 함수에 직접 구현하여 학습하는 파이프라인입니다. 이를 통해 기본적인 한국어 임베딩 학습이 가능합니다. [관련 블로그](https://yjoonjang.medium.com/koe5-%EC%B5%9C%EC%B4%88%EC%9D%98-%ED%95%9C%EA%B5%AD%EC%96%B4-%EC%9E%84%EB%B2%A0%EB%94%A9-%EB%AA%A8%EB%8D%B8-multilingual-e5-finetune-22fa7e56d220)
- `v1.1`: InfoNCE loss가 [배치 크기에 큰 영향을 받는다](https://yjoonjang.medium.com/%EB%B0%B0%EC%B9%98-%EC%82%AC%EC%9D%B4%EC%A6%88%EB%A5%BC-%EB%AF%B8%EC%B9%9C%EB%93%AF%EC%9D%B4-%ED%82%A4%EC%9A%B0%EB%8A%94-%EB%B2%95-gradient-cache-60e066907b69)는 사실을 기반으로, **gradient cache**를 적용하여 큰 배치 크기에서도 안정적으로 학습할 수 있도록 개선된 파이프라인입니다.

## 학습

모델을 파인 튜닝하고 학습시키기 위한 단계는 다음과 같습니다:

1. 의존성 패키지를 설치 합니다.
```bash
pip install -r requirements.txt
```
2. `v1.1/scripts/finetune.sh` 스크립트를 수정하여 학습 파라미터를 설정합니다.
3. 설정한 파라미터에 따라 스크립트를 실행하여 모델을 학습시킵니다.
```bash
bash v1.1/scripts/finetune.sh
```

## 데이터 구성
모델 학습에 사용된 데이터는 다음과 같은 구조를 가지고 있습니다:
```json
{
    "query": "사용자가 입력한 질의",
    "document": "질의와 관련된 문서",
    "hard_negative": "질의와 관련성이 적지만, 유사해 보이는 문서"
}
```
데이터는 AIHUB, KorQUAD, KommonGen, Exobrain, KLUE, KoBEST, NIKL로부터 총 10종의 오픈 데이터를 수집하였으며, 총 데이터 통계는 다음과 같습니다.

<p align="center">
  <img src="https://github.com/user-attachments/assets/57ae64ac-2fec-4d5f-a0ad-45b8c54b12d1" alt="data-image">
</p>

## 평가
- `python evaluate.py`명령어로 평가를 수행합니다.
- `streamlit run leaderboard.py`명령어로 평가에 대한 리더보드를 확인합니다.

## 결과
Ko-strategyQA, AutoRAG-embedding-benchmark, PublicHealthQA의 총 3가지 평가 데이터셋으로 평가를 진행했으며, 결과는 다음과 같습니다.
<p align="center">
  <img src="https://github.com/user-attachments/assets/a8645bda-f9f0-443f-931c-1c311ff86736" alt="eval-image">
</p>
